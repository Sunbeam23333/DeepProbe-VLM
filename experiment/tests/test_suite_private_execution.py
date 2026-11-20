"""Private disk scheduling and retention using synthetic local-only fixtures."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import shutil

import pytest

from deepprobe_vlm import experiment_suite, suite_fleet, suite_runner as runner
from deepprobe_vlm.suite_scratch import DEFAULT_SCRATCH, worker_run_root
from deepprobe_vlm.suite_scratch_budget import PrivateScratchBudget, TieredTaskBudget
from deepprobe_vlm.suite_durable_metadata import retain_pin
from deepprobe_vlm.suite_storage import StorageError


def profile(root, workers=4):
    return {'schema_version': 3, 'fleet_id': 'synthetic-private', 'priority': 'low',
        'gpu_product': 'NVIDIA-B300', 'node_selector': {'sslaunch-pool': 'stream', 'aliyun.accelerator/nvidia_name': 'NVIDIA-L20D'},
        'scratch': {**DEFAULT_SCRATCH, 'mount_path': str(root / 'private')},
        'workers': [{'worker_id': f'worker-0{i}', 'job_name': f'jinhaopeng-dpvlm-private-{i}', 'namespace': 'default'}
                    for i in range(1, workers + 1)]}


def fleet(root):
    root = root.resolve(); home = root / 'shared'; home.mkdir()
    plan = experiment_suite.default_plan()
    binding = {'schema_version': 2, 'suite_id': 'synthetic-private', 'execution_profile': profile(root),
               'storage': {'roots': [str(home)]}, 'source_files': []}
    state = suite_fleet.FleetState(plan, binding, home / 'state', plan_sha256=runner.digest(plan), binding_sha256=runner.digest(binding))
    state.initialize()
    return state


def finish(state, selected):
    task, claim = selected['task'], selected['claim']
    receipt = Path(claim['attempt']) / 'SYNTHETIC_ONLY.json'
    runner.write_once(receipt, {'status': 'SYNTHETIC_NOT_PRODUCTION', 'task_id': task['id']})
    state.finish(task, claim, {'status': 'TASK_ACCEPTED', 'task_id': task['id'], 'plan_sha256': state.plan_sha,
        'binding_sha256': state.binding_sha, 'receipts': [runner.artifact(receipt)]})


def claim(state, worker):
    return state.claim(worker, 'job-' + worker, lambda task, owner: owner['claim_id'])


def test_entire185_dag_runs_four_sfts_with_chain_affinity_and_no_repeated_tasks(tmp_path):
    state = fleet(tmp_path)
    original = runner.digest(state.plan)
    workers = [row['worker_id'] for row in state.profile['workers']]
    completed, model_owner = set(), {}
    max_training = max_sft = 0
    for _ in range(2 * len(state.plan['tasks'])):
        with ThreadPoolExecutor(max_workers=4) as pool:
            selected = list(pool.map(lambda worker: claim(state, worker), workers))
        accepted = [row for row in selected if row['status'] == 'claimed']
        if not accepted:
            assert all(row['status'] == 'complete' for row in selected)
            break
        max_training = max(max_training, sum(row['task']['kind'] == 'train' for row in accepted))
        max_sft = max(max_sft, sum(row['task'].get('stage') == 'sft' for row in accepted))
        for row in reversed(accepted):
            task, owner = row['task'], row['claim']['owner']
            assert task['id'] not in completed and set(task['dependencies']) <= completed
            if task.get('model_id'):
                model = task['model_id']
                assert model_owner.setdefault(model, owner['worker_id']) == owner['worker_id']
            if task['kind'] == 'train' and task['initializer'].get('checkpoint_task_id'):
                ancestor = task['initializer']['checkpoint_task_id']
                assert ancestor.replace('train__', 'archive__', 1) in completed
            finish(state, row); completed.add(task['id'])
    assert len(completed) == 185 and max_training == max_sft == 4
    assert model_owner['ouro_r4_s17'] == model_owner['ouro_r4_frozen_vision_s17']
    assert runner.digest(state.plan) == original
    assert len(list((state.state / 'chain-affinity').glob('*.json'))) == 17


def test_lost_job_cannot_adopt_private_chain_and_duplicate_worker_cannot_claim(tmp_path):
    state = fleet(tmp_path)
    gate = claim(state, 'worker-01'); finish(state, gate)
    training = claim(state, 'worker-01')
    with pytest.raises(ValueError, match='unfinished task claim'):
        claim(state, 'worker-01')
    finish(state, training)
    with pytest.raises(ValueError, match='cannot adopt'):
        state.claim('worker-01', 'replacement-job', lambda task, owner: owner['claim_id'])


def private_budget(root, *, used=0):
    root = root.resolve(); p = profile(root, workers=1)
    worker = 'worker-01'; run_root = worker_run_root(p, worker); run_root.mkdir(parents=True)
    state = root / 'shared-state'; (state / 'workers' / worker).mkdir(parents=True)
    live = {'worker_id': worker, 'job_uid': 'job-1', 'pod_uid': 'pod-1', 'node': 'node-1'}
    budget = PrivateScratchBudget(p, worker, live, state, scanner=lambda roots: {'bytes': used})
    return budget, p, state, live


def test_private_actual_plus_reserved_and_uncertain_release_fail_closed(tmp_path):
    budget, p, state, live = private_budget(tmp_path, used=40 * 1024 ** 3)
    owner = {'worker_id': 'worker-01', 'job_uid': 'job-1', 'task_id': 'train'}
    token = budget.reserve(owner, 50 * 1024 ** 3)
    with pytest.raises(StorageError, match='capacity gated'):
        budget.reserve({**owner, 'task_id': 'second'}, 10 * 1024 ** 3)
    with pytest.raises(StorageError, match='not proven quiescent'):
        budget.release(token, lambda exact: {'owner': exact, 'active_writers': [1], 'active_uploads': []})
    assert budget.snapshot()['active_reservations'] == 1
    budget.release(token, lambda exact: {'owner': exact, 'active_writers': [], 'active_uploads': []})
    assert budget.snapshot()['active_reservations'] == 0
    with pytest.raises(StorageError, match='another exact Pod'):
        PrivateScratchBudget(p, 'worker-01', {**live, 'pod_uid': 'pod-2'}, state)


def test_home_reservation_failure_rolls_back_undispatched_private_token(tmp_path):
    budget, _, _, _ = private_budget(tmp_path)
    class BlockedShared:
        def reserve(self, *args):
            raise StorageError('home capacity gated')
    tiered = TieredTaskBudget(BlockedShared(), budget, {'train': {'kind': 'train'}})
    with pytest.raises(StorageError, match='home capacity gated'):
        tiered.reserve({'worker_id': 'worker-01', 'job_uid': 'job-1', 'task_id': 'train'}, 1024, 0)
    assert budget.snapshot()['active_reservations'] == 0


def test_receipt_bytes_survive_private_file_loss_and_weights_never_copy_to_home(tmp_path):
    tmp_path = tmp_path.resolve(); private = tmp_path / 'private'; private.mkdir()
    receipt = private / 'MANIFEST.json'; receipt.write_text('{"synthetic":true}')
    old = runner.artifact(receipt)
    durable = retain_pin(old, private, tmp_path / 'shared')
    receipt.unlink()
    assert runner.verify_pin(durable)['sha256'] == old['sha256']
    weights = private / 'weights.safetensors'; weights.write_bytes(b'synthetic')
    with pytest.raises(ValueError, match='Only bounded metadata'):
        retain_pin(runner.artifact(weights), private, tmp_path / 'shared')


def test_private_archive_register_consume_cleanup_and_receipts_survive_pod_disk_loss():
    import test_suite_actions_retention as fixtures
    from deepprobe_vlm.suite_storage import SharedBudget
    case = fixtures.ActionRetentionTest()
    case.setUp()
    try:
        private = case.base / 'private-pod'; private.mkdir()
        case.worker.private_scratch = True
        case.worker.run_root = private
        storage = {'schema_version': 2, 'roots': [str(case.project), str(case.staging)],
            'ledger_root': str(case.project / 'ledger'), 'home_limit_bytes': 150_000_000_000,
            'oss_limit_bytes': None, 'oss_prefixes': [case.environment['archive_root'] + '/'],
            'home_safety_bytes': 1_000_000, 'oss_safety_bytes': 0, 'free_guard_bytes': 0}
        case.worker.env['storage'] = storage
        budget = SharedBudget(storage, lambda: {'complete': True, 'objects': [], 'multipart': []})
        case.worker._shared_budget = budget
        token = budget.reserve({'fixture': 'synthetic-private-actions'}, 1_000_000, 0)
        case.worker.budget = lambda extra, oss_extra=0: budget.extend(token, home_bytes=extra, oss_bytes=oss_extra)
        trained = case.execute(case.stage)
        archived = case.execute(case.archive)
        ledger = runner.read(budget.path)
        assert ledger['artifacts']
        assert all(location['path'].startswith('oss://') for artifact in ledger['artifacts'].values()
                   for location in artifact['locations'])
        assert Path(archived['manifest']['path']).is_relative_to(case.state)
        case.accept_consumer()
        case.accept_fixture_dependency('archive__ouro_r1_s17__results')
        cleaned = case.execute(case.cleanup)
        assert not (Path(trained['checkpoint']) / 'projector.safetensors').exists()
        assert any(uri.endswith('projector.safetensors') for uri in case.store.objects)
        shutil.rmtree(private)  # Synthetic Pod emptyDir loss after all consumers/cleanup.
        for task_id in (case.stage, case.archive, case.cleanup):
            case.worker.completed(task_id)
        cleanup = runner.read(cleaned['outputs'][0]['path'])
        runner.verify_pin(cleanup['receipt'])
        assert fixtures.actions.inventory(case.old) == case.old_files
    finally:
        case.doCleanups()
