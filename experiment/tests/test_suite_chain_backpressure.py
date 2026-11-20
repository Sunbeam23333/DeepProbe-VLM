"""Whole-DAG capacity regression using the observed November 21 byte budgets.

These are synthetic task completions, never production training acceptance.
The byte model deliberately counts only retained finals after each archive;
even that optimistic model reproduces the old eager-alignment deadlock.
The real FleetState is then exercised with one to four competing claimants.
No model weights, cluster, bucket, or multi-GB fixture files are needed.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import threading

import pytest

from deepprobe_vlm import experiment_suite, suite_fleet, suite_runner as runner
from deepprobe_vlm.suite_storage import StorageError


# Measured baseline: smol-old-checkpoint-retention-20261121-v1/apply-v1,
# and TRAINING_RESERVE_RECOMMENDATION.json SHA b3533a647c9fab4c3d1743c546fcbbe105aa5063bc02345f90f464440854eec1.
BASE_BYTES = 62_504_292_995
LIMIT_BYTES = 150_000_000_000
OTHER_BYTES = 2_000_000_000 + 268_435_456 + 1_048_576  # safety, CPU producer, supervisor
OURO_FINAL = 7_482_814_704
SMOL_FINAL = 8_589_657_473
R4_NATIVE = 22_317_153_372
R4_ALIGN = 'train__ouro_r4_s17__alignment'
R4_SFT = 'train__ouro_r4_s17__sft'
FROZEN_SFT = 'train__ouro_r4_frozen_vision_s17__sft'
SMOL_ALIGN = 'train__smollm2_s29__alignment'
FIRST_COMPONENT = {R4_ALIGN, R4_SFT, FROZEN_SFT}
ACCEPTED = {R4_ALIGN, SMOL_ALIGN} | {
    'train__' + model + '__' + stage for model in ('ouro_r1_s17', 'ouro_r2_s17')
    for stage in ('alignment', 'sft')}
REUSED_EVALUATIONS = {'evaluate__' + model + '__' + benchmark
    for model in ('ouro_r1_s17', 'ouro_r2_s17') for benchmark in experiment_suite.BENCHMARKS}
PRIORITY = {'cleanup': 0, 'archive': 1, 'evaluate': 2, 'latency': 2, 'compare_latency': 2}


def final_bytes(task):
    return SMOL_FINAL if task['model_id'].startswith('smollm2_') else OURO_FINAL


def reserve_bytes(task):
    kind = task['kind']
    if kind == 'train':
        if task['id'] in ACCEPTED:
            return final_bytes(task) + 128 * 1024 ** 2
        smol = task['model_id'].startswith('smollm2_')
        if task['stage'] == 'sft':
            return 54_525_859_992 if smol else 47_884_490_928
        if smol:
            return 20_284_228_248
        return 25_605_657_840 if task['model_id'].startswith(('ouro_r1_', 'ouro_r2_')) else 18_070_438_560
    if kind == 'evaluate':
        if task['id'] in REUSED_EVALUATIONS:
            return 128 * 1024 ** 2
        png_caps = {'videomme899': {4: 4_000_000_000, 8: 8_000_000_000, 16: 16_000_000_000},
                    'tempcompass_mc': {4: 2_000_000_000, 8: 4_000_000_000, 16: 8_000_000_000}}
        return png_caps[task['benchmark']][task['frames']] + 32 * 1024 ** 2 + 3_000_000_000
    return {'archive': 64 * 1024 ** 2, 'cleanup': 100_000_000, 'compare_latency': 100_000_000,
            'gate': 1_000_000_000, 'latency': 1_000_000_000, 'aggregate': 1_000_000_000}[kind]


def peak_bytes(task):
    # The native cp600 restore and the full future output peak coexist. This
    # models the action's second admission after restore, without lowering it.
    return reserve_bytes(task) + (R4_NATIVE if task['id'] == R4_SFT else 0)


def checkpoint_effect(task, resident):
    if task['kind'] == 'train':
        resident[task['id']] = final_bytes(task)
    elif task['kind'] == 'cleanup':
        del resident[task['checkpoint_task_id']]


def unfinished_components(plan, started, completed):
    """Independent semantic oracle, not the implementation's grouping helper."""
    trains = {task['id']: task for task in plan['tasks'] if task['kind'] == 'train'}
    neighbors = {task_id: set() for task_id in trains}
    for task_id, task in trains.items():
        parent = task['initializer'].get('checkpoint_task_id')
        if parent:
            neighbors[task_id].add(parent)
            neighbors[parent].add(task_id)
    cleanup = {task['checkpoint_task_id']: task['id'] for task in plan['tasks'] if task['kind'] == 'cleanup'}
    found, seen = [], set()
    for task_id in trains:
        if task_id in seen:
            continue
        component, todo = set(), [task_id]
        while todo:
            current = todo.pop()
            if current not in component:
                component.add(current)
                todo.extend(neighbors[current] - component)
        seen.update(component)
        if component & started and not all(cleanup[t] in completed for t in component):
            found.append(component)
    return found


def test_previous_eager_order_deadlocks_even_after_every_cache_becomes_ready():
    """Executable counterexample; adding READY alone cannot cure this state."""
    plan = experiment_suite.default_plan()
    completed, resident = set(), {}
    all_caches_ready = False
    stalls = []
    for _ in range(2 * len(plan['tasks'])):
        ready = sorted((t for t in plan['tasks'] if t['id'] not in completed
                        and set(t['dependencies']) <= completed), key=lambda t: PRIORITY.get(t['kind'], 3))
        usage = BASE_BYTES + OTHER_BYTES + sum(resident.values())
        for task in ready:
            if task['kind'] == 'evaluate' and task['id'] not in REUSED_EVALUATIONS and not all_caches_ready:
                continue
            if usage + peak_bytes(task) >= LIMIT_BYTES:
                continue
            completed.add(task['id'])
            checkpoint_effect(task, resident)
            break
        else:
            stalls.append({'completed': len(completed), 'usage': usage, 'resident': deepcopy(resident)})
            if all_caches_ready:
                break
            all_caches_ready = True
    assert len(stalls) == 2 and stalls[0]['completed'] < stalls[1]['completed'] < 185
    assert 130_000_000_000 < stalls[-1]['usage'] < LIMIT_BYTES
    assert len(resident) == 9 and all(task_id.endswith('__alignment') for task_id in resident)
    assert all(LIMIT_BYTES - stalls[-1]['usage'] < peak_bytes(task)
               for task in plan['tasks'] if task['id'] not in completed and set(task['dependencies']) <= completed)


class CapacityFleet:
    """Run actual atomic claims, with synthetic foreground task completions."""
    def __init__(self, root, workers):
        self.root = root.resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.plan = experiment_suite.default_plan()
        self.tasks = {task['id']: task for task in self.plan['tasks']}
        self.profile = {'schema_version': 2, 'fleet_id': 'synthetic-chain-test', 'priority': 'normal',
            'gpu_product': 'NVIDIA-B300', 'node_selector': {'sslaunch-pool': 'stream',
                'aliyun.accelerator/nvidia_name': 'NVIDIA-L20D'},
            'workers': [{'worker_id': 'worker-0' + str(i), 'job_name': 'jinhaopeng-dpvlm-chain-test-' + str(i),
                         'namespace': 'default'} for i in range(1, workers + 1)]}
        storage = {'schema_version': 2, 'roots': [str(self.root)], 'ledger_root': str(self.root / 'ledger'),
            'home_limit_bytes': LIMIT_BYTES, 'oss_limit_bytes': None,
            'oss_prefixes': ['oss://ss-devbox-stream/project/synthetic-chain/'],
            'home_safety_bytes': 2_000_000_000, 'oss_safety_bytes': 0, 'free_guard_bytes': 1_000_000_000}
        self.binding = {'schema_version': 2, 'suite_id': self.profile['fleet_id'],
            'allocation': suite_fleet.profile_allocation(self.profile), 'execution_profile': self.profile,
            'storage': storage, 'source_files': [], 'tasks': {tid: {'reserve_bytes': reserve_bytes(task)}
                                                         for tid, task in self.tasks.items()}}
        self.state_root = self.root / 'state'
        self.state = suite_fleet.FleetState(self.plan, self.binding, self.state_root,
            plan_sha256=runner.digest(self.plan), binding_sha256=runner.digest(self.binding))
        self.state.initialize()
        self.completed, self.started, self.resident, self.reserved = set(), set(), {}, {}
        self.all_caches_ready = False
        self.maximum_projected = 0
        self.maximum_unfinished_components = 0
        self.maximum_active_training = 0
        self.accepted_sequence = []

    def reserve(self, task, owner):
        amount = peak_bytes(task)
        projected = BASE_BYTES + OTHER_BYTES + sum(self.resident.values()) + sum(self.reserved.values()) + amount
        if projected >= LIMIT_BYTES:
            raise StorageError('Synthetic full retained-final and measured-peak capacity bound')
        token = owner['claim_id']
        self.reserved[token] = amount
        self.maximum_projected = max(self.maximum_projected, projected)
        return token

    def readiness(self, task):
        if task['kind'] == 'evaluate' and task['id'] not in REUSED_EVALUATIONS and not self.all_caches_ready:
            return None
        return {}

    def claim_batch(self):
        workers = self.profile['workers']
        barrier = threading.Barrier(len(workers))
        def claim(worker):
            barrier.wait(timeout=10)
            return self.state.claim(worker['worker_id'], 'job-' + worker['worker_id'], self.reserve,
                                    readiness=self.readiness)
        with ThreadPoolExecutor(max_workers=len(workers)) as pool:
            results = list(pool.map(claim, workers))
        claimed = [result for result in results if result['status'] == 'claimed']
        active_training = [result['task']['id'] for result in claimed if result['task']['kind'] == 'train']
        unarchived = [task for task in self.plan['tasks'] if task['kind'] == 'archive'
                      and task.get('archive_scope') == 'checkpoint'
                      and task['checkpoint_task_id'] in self.completed and task['id'] not in self.completed]
        assert not (active_training and unarchived), 'Next training began while previous intermediates awaited archive/cleanup'
        self.maximum_active_training = max(self.maximum_active_training, len(active_training))
        self.started.update(active_training)
        components = unfinished_components(self.plan, self.started, self.completed)
        self.maximum_unfinished_components = max(self.maximum_unfinished_components, len(components))
        assert len(active_training) <= 1, 'Concurrent workers opened multiple training stages'
        assert len(components) <= 1, 'Concurrent workers opened independent unfinished training chains'
        return results, claimed

    def finish(self, selected):
        task, claim = selected['task'], selected['claim']
        tid = task['id']; attempt = Path(claim['attempt'])
        assert set(task['dependencies']) <= self.completed
        runner.write_once(attempt / 'SYNTHETIC_RECEIPT.json', {'status': 'SYNTHETIC_ONLY', 'task': tid})
        completion = {'status': 'TASK_ACCEPTED', 'task_id': tid, 'plan_sha256': self.state.plan_sha,
            'binding_sha256': self.state.binding_sha, 'receipts': [runner.artifact(attempt / 'SYNTHETIC_RECEIPT.json')]}
        self.state.finish(task, claim, completion)
        checkpoint_effect(task, self.resident)
        del self.reserved[claim['storage_token']]
        self.completed.add(tid)
        self.accepted_sequence.append(tid)

    def drain_until_wait_or_complete(self):
        for _ in range(2 * len(self.plan['tasks'])):
            results, claimed = self.claim_batch()
            if not claimed:
                return results
            # Reverse completion order to vary interleavings for consumer work;
            # dependencies only become eligible in the next actual claim batch.
            for selected in reversed(claimed):
                self.finish(selected)
        raise AssertionError('Fleet made no bounded progress')


@pytest.mark.parametrize('workers', [1, 2, 3, 4])
def test_full185_waits_for_cache_then_completes_without_capacity_or_chain_deadlock(tmp_path, workers):
    fleet = CapacityFleet(tmp_path, workers)
    initial_plan = runner.digest(fleet.plan)
    waiting = fleet.drain_until_wait_or_complete()
    assert all(row['status'] == 'wait' for row in waiting)
    assert fleet.started == FIRST_COMPONENT
    assert FIRST_COMPONENT <= fleet.completed
    assert not fleet.reserved
    assert set(fleet.resident) == FIRST_COMPONENT
    # Shared R4 alignment cannot be freed until the frozen-vision branch and
    # both models' declared consumers finish, even with multiple claimants.
    assert 'cleanup__ouro_r4_s17__alignment' not in fleet.completed
    assert not list(fleet.state_root.rglob('FAILED.json'))
    fleet.all_caches_ready = True
    completed = fleet.drain_until_wait_or_complete()
    assert all(row['status'] == 'complete' for row in completed)
    assert len(fleet.completed) == len(fleet.accepted_sequence) == len(set(fleet.accepted_sequence)) == 185
    assert not fleet.resident and not fleet.reserved
    assert fleet.maximum_projected < LIMIT_BYTES
    assert fleet.maximum_unfinished_components == fleet.maximum_active_training == 1
    assert runner.digest(fleet.plan) == initial_plan
    assert len(list((fleet.state_root / 'tasks').glob('*/COMPLETED.json'))) == 185
    assert all(len(list((fleet.state_root / 'tasks' / task_id / 'attempts').iterdir())) == 1 for task_id in fleet.completed)


def test_generic_fixture_tasks_without_scientific_initializers_keep_their_parallelism(tmp_path):
    fleet = CapacityFleet(tmp_path, 4)
    fleet.plan['tasks'] = [{'id': 'one', 'kind': 'train', 'dependencies': []},
                           {'id': 'two', 'kind': 'train', 'dependencies': []}]
    root = tmp_path / 'generic'; root.mkdir()
    # A separate project root avoids inheriting the scientific fleet lease.
    binding = deepcopy(fleet.binding)
    binding['storage']['roots'] = [str(root)]
    binding['storage']['ledger_root'] = str(root / 'ledger')
    state = suite_fleet.FleetState(fleet.plan, binding, root / 'state',
        plan_sha256=runner.digest(fleet.plan), binding_sha256=runner.digest(binding))
    state.initialize()
    reserve = lambda task, owner: owner['claim_id']
    first = state.claim('worker-01', 'job-one', reserve)
    second = state.claim('worker-02', 'job-two', reserve)
    assert {first['task']['id'], second['task']['id']} == {'one', 'two'}


def test_frozen_vision_branch_waits_for_prior_sft_archive_completion(tmp_path):
    fleet = CapacityFleet(tmp_path, 4)
    for _ in range(20):
        _, claimed = fleet.claim_batch()
        assert claimed, 'Initial R4 chain must be runnable before benchmark caches'
        for selected in claimed:
            fleet.finish(selected)
        if R4_SFT in fleet.completed:
            break
    else:
        raise AssertionError('Initial R4 SFT never completed')
    archive_id = 'archive__ouro_r4_s17__sft'
    assert archive_id not in fleet.completed and FROZEN_SFT not in fleet.started
    _, claimed = fleet.claim_batch()
    archive = next(row for row in claimed if row['task']['id'] == archive_id)
    assert all(row['task']['kind'] != 'train' for row in claimed)
    for row in claimed:
        if row is not archive:
            fleet.finish(row)
    # Archive is still active; another worker cannot run the sibling SFT while
    # its predecessor's periodic optimizer checkpoints remain on the machine.
    _, more = fleet.claim_batch()
    assert all(row['task']['kind'] != 'train' for row in more)
    for row in more:
        fleet.finish(row)
    fleet.finish(archive)
    for _ in range(20):
        _, more = fleet.claim_batch()
        if any(row['task']['id'] == FROZEN_SFT for row in more):
            break
        assert more
        for row in more:
            fleet.finish(row)
    else:
        raise AssertionError('Archived R4 SFT failed to unblock its frozen-vision sibling')
