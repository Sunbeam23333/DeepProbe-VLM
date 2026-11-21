"""Run the complete runner/DAG with synthetic science and real local budgets.

Only the hardware boundary and task adapter are substituted. No GPU, model,
benchmark result, remote archive, real mount or cross-node filesystem passes
because this CPU regression passes.
"""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

from deepprobe_vlm import experiment_suite as suite
from deepprobe_vlm import suite_fleet as fleet
from deepprobe_vlm import suite_isolated_pod as isolated_pod
from deepprobe_vlm import suite_isolated_storage as isolated
from deepprobe_vlm import suite_runner as runner
from deepprobe_vlm import suite_scratch as scratch
from deepprobe_vlm import suite_storage as storage


def _old_home_inventory(home, namespace):
    result = {}
    for directory, directories, names in os.walk(home):
        directories[:] = [name for name in directories if Path(directory) / name != namespace]
        for name in names:
            path = Path(directory) / name
            result[str(path.relative_to(home))] = path.read_bytes()
    return result


def test_single_isolated_worker_completes_unchanged_full185_with_real_budget_and_retention_order(tmp_path, monkeypatch):
    for key in ('DP_SUITE_WORKER_ID', 'DP_SUITE_FLEET_STATE', 'DP_SUITE_STORAGE_LEDGER',
                'DP_SUITE_STORAGE_TOKEN', 'DP_SUITE_SCRATCH_TOKEN'):
        monkeypatch.delenv(key, raising=False)
    root = tmp_path.resolve()
    home, namespace, mount = root / 'home', root / 'home/new-worker', root / 'private-pod'
    namespace.mkdir(parents=True); mount.mkdir()
    old_lease = home / 'locks/fleet-leases/old-active-project.json'
    old_lease.parent.mkdir(parents=True)
    old_lease.write_bytes(b'{"old_project_lease":"must-not-read-or-change"}\n')
    (home / 'locks/experiment-suite.lock').write_bytes(b'old coordinator lock sentinel\n')
    source = home / 'fixture_source.py'; source.write_bytes(b'# synthetic executor source pin\n')
    plan = suite.default_plan()
    suite.validate_plan(plan, config_root=suite.EXPERIMENT_ROOT)
    original_plan, plan_sha = deepcopy(plan), runner.digest(plan)
    tasks = {task['id']: task for task in plan['tasks']}
    assert len(tasks) == 185
    profile = {'schema_version': 3, 'fleet_id': 'isolated-full185', 'priority': 'low',
        'gpu_product': 'NVIDIA-B300',
        'node_selector': {'sslaunch-pool': 'stream', 'aliyun.accelerator/nvidia_name': 'NVIDIA-L20D'},
        'scratch': {**scratch.DEFAULT_SCRATCH, 'mount_path': str(mount)},
        'workers': [{'worker_id': 'worker-01', 'job_name': 'jinhaopeng-dpvlm-isolated-full185', 'namespace': 'default'}]}
    config = {'schema_version': 2, 'roots': [str(home)], 'ledger_root': str(home / 'old-parent-ledger'),
        'home_limit_bytes': 4 * 1024 ** 3, 'oss_limit_bytes': None,
        'oss_prefixes': ['oss://ss-devbox-stream/fixture/isolated-full185/'],
        'home_safety_bytes': 1024 * 1024, 'oss_safety_bytes': 0, 'free_guard_bytes': 0}
    def no_remote():
        raise AssertionError('Synthetic isolated execution must not query a remote inventory')
    parent = storage.SharedBudget(config, no_remote)
    old_cpu_token = parent.reserve({'task': 'old-cpu-still-authorized'}, 1024 * 1024, 0)
    environment = {'suite_id': profile['fleet_id'], 'execution_profile': profile,
        'source_files': [runner.artifact(source)], 'storage': config, 'runtime_environment': {}}
    hold_pin = isolated.create_lifetime_hold(environment, namespace_root=str(namespace),
        home_limit_bytes=2 * 1024 ** 3, home_safety_bytes=1024 * 1024,
        plan_sha256=plan_sha, output=namespace / 'hold', budget=parent)
    environment = isolated.isolate_environment(environment, hold_pin)
    env_path = namespace / 'ENVIRONMENT.json'
    runner.write_once(env_path, environment)
    state = namespace / 'state'
    live = {'worker_id': 'worker-01', 'job_uid': 'synthetic-job', 'pod_uid': 'synthetic-pod',
            'node': 'synthetic-node', 'other_active_project_jobs': []}
    allocation = fleet.profile_allocation(profile)
    binding = {'schema_version': 2, 'suite_id': profile['fleet_id'], 'plan_sha256': plan_sha,
        'allocation': allocation, 'execution_profile': profile, 'source_files': environment['source_files'],
        'storage': config, 'isolated_storage': environment['isolated_storage'],
        'environment': environment['runtime_environment'], 'startup': {}, 'tasks': {}}
    def command(task_id, receipt):
        return {'mode': 'execute', 'argv': [sys.executable, '-B', str(source),
            '--environment', str(env_path), '--environment-sha256', runner.sha(env_path), '--task', task_id],
            'cwd': str(namespace), 'inputs': [], 'reserve_bytes': 64 * 1024, 'oss_reserve_bytes': 0,
            'timeout_seconds': 10, 'receipts': [{'path': receipt, 'checks': [
                {'pointer': '/status', 'equals': 'SUITE_STARTUP_ACCEPTED' if task_id == 'startup' else 'SYNTHETIC_TASK_EXECUTED'},
                {'pointer': '/synthetic_only', 'equals': True}]}]}
    binding['startup'] = command('startup', '{invocation_dir}/STARTUP.json')
    binding['startup'].pop('mode')
    binding['tasks'] = {task_id: command(task_id, '{task_attempt_dir}/SYNTHETIC_RECEIPT.json') for task_id in tasks}
    binding_sha = runner.digest(binding)
    old_home, parent_bytes = _old_home_inventory(home, namespace), parent.path.read_bytes()
    hold = runner.read(hold_pin['path'])
    executed, resident, archives, results, child_ledgers = [], {}, {}, {}, set()
    private_runs = scratch.worker_run_root(profile, 'worker-01')
    transactions, mount_checks, live_calls = [], [], 0
    real_transaction, real_replace = fleet.project_transaction, storage._replace

    def synthetic_mount_check(env, mounts):
        assert env == environment and mounts == [{'synthetic_mount_fixture': True}]
        assert not state.exists() and not (namespace / 'locks').exists()
        assert not isolated.local_ledger_root(environment).exists()
        assert parent.path.read_bytes() == parent_bytes
        proof = {'status': 'SYNTHETIC_ISOLATED_MOUNTS_ONLY', 'synthetic_only': True}
        mount_checks.append(proof)
        return proof

    @contextmanager
    def isolated_transaction(policy):
        assert len(mount_checks) == 1, 'Kernel mount admission must precede the first state transaction'
        assert policy['roots'] == [str(namespace)]
        transactions.append(tuple(policy['roots']))
        with real_transaction(policy) as locked:
            yield locked

    def isolated_replace(path, document):
        assert Path(path) != parent.path, 'The GPU path must never write the coordinator ledger'
        return real_replace(path, document)

    def bounded_live_check():
        nonlocal live_calls
        live_calls += 1
        assert live_calls <= 2 * len(tasks) + 16, 'Isolated runner stalled instead of completing its DAG'
        return deepcopy(live)

    def synthetic_command(command, current_binding, context, attempt, *, extra_environment, storage_snapshot):
        assert current_binding == binding
        assert extra_environment['DP_SUITE_WORKER_ID'] == 'worker-01'
        assert extra_environment['DP_SUITE_FLEET_STATE'] == str(state)
        child_root = isolated.local_ledger_root(environment) / 'ledger'
        assert extra_environment['DP_SUITE_STORAGE_LEDGER'] == str(child_root)
        assert child_root.is_relative_to(mount)
        assert storage_snapshot['shared']['parent_ledger_opened'] is False
        task_id = context.get('task_id', 'startup')
        # A separately constructed adapter budget must observe the runner's
        # real reservation through the same private child ledger.
        with monkeypatch.context() as process_environment:
            for key, value in extra_environment.items():
                process_environment.setenv(key, value)
            adapter_budget = isolated.budget_for_isolated(environment)
            child_ledgers.add(str(adapter_budget.path))
            before = runner.read(adapter_budget.path)
            token = extra_environment['DP_SUITE_STORAGE_TOKEN']
            row = before['reservations'][token]
            assert row['state'] == 'ACTIVE' and row['owner']['task_id'] == task_id
            adapter_budget.extend(token, home_bytes=row['home_bytes'], oss_bytes=row['oss_bytes'])
        if task_id == 'startup':
            assert 'DP_SUITE_SCRATCH_TOKEN' not in extra_environment
            value = {'status': 'SUITE_STARTUP_ACCEPTED', 'synthetic_only': True,
                'allocation': allocation, 'source_identity_sha256': runner.source_identity(binding),
                **live, 'gpu_count': 8, 'gpu_uuids': ['synthetic-gpu-' + str(i) for i in range(8)],
                'gpu_product': 'NVIDIA-B300', 'hardware_preflight_passed': True}
        else:
            task = tasks[task_id]
            assert task_id not in executed
            assert set(task['dependencies']) <= set(executed)
            for dependency in task['dependencies']:
                assert runner.completion_for(state, dependency)['task_id'] == dependency
            assert 'DP_SUITE_SCRATCH_TOKEN' in extra_environment
            if task['kind'] == 'train':
                initializer = task['initializer'].get('checkpoint_task_id')
                if initializer:
                    assert initializer in resident and resident[initializer].is_file()
                    assert initializer in archives and archives[initializer].is_file()
                checkpoint = private_runs / task_id / 'checkpoint.bin'
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(('synthetic checkpoint ' + task_id).encode())
                resident[task_id] = checkpoint
            elif task['kind'] in {'evaluate', 'latency'}:
                checkpoint_task = task['checkpoint_task_id']
                assert checkpoint_task in resident and resident[checkpoint_task].is_file()
                assert checkpoint_task in archives
            elif task['kind'] == 'archive' and task.get('archive_scope') == 'checkpoint':
                checkpoint_task = task['checkpoint_task_id']
                assert resident[checkpoint_task].is_file()
                archive = namespace / 'synthetic-archive' / (checkpoint_task + '.bin')
                archive.parent.mkdir(exist_ok=True)
                archive.write_bytes(resident[checkpoint_task].read_bytes())
                archives[checkpoint_task] = archive
            elif task['kind'] == 'archive' and task.get('archive_scope') == 'results':
                output = namespace / 'synthetic-results' / (task['model_id'] + '.json')
                output.parent.mkdir(exist_ok=True)
                runner.write_once(output, {'synthetic_only': True, 'dependencies': task['dependencies']})
                results[task['model_id']] = output
            elif task['kind'] == 'cleanup':
                checkpoint_task = task['checkpoint_task_id']
                assert set(task['required_consumers']) <= set(executed)
                assert checkpoint_task in archives and archives[checkpoint_task].is_file()
                assert task['model_id'] in results and results[task['model_id']].is_file()
                checkpoint = resident.pop(checkpoint_task)
                assert checkpoint.is_relative_to(private_runs)
                checkpoint.unlink()
            elif task['kind'] == 'aggregate':
                assert not resident
            executed.append(task_id)
            value = {'status': 'SYNTHETIC_TASK_EXECUTED', 'task_id': task_id, 'kind': task['kind'],
                     'synthetic_only': True, 'real_model_executed': False}
        output = Path(runner.expand(command['receipts'][0]['path'], context))
        runner.write_once(output, value)
        runner.write_once(attempt / 'COMMAND_EXIT.json', {'returncode': 0,
            'group_cleanup_verified': True, 'synthetic_executor_no_process_created': True})
        assert parent.path.read_bytes() == parent_bytes
        return runner.verify_receipts(command['receipts'], context)

    monkeypatch.setattr(runner, 'run_command', synthetic_command)
    monkeypatch.setattr(isolated_pod, 'inspect_isolated_mounts', synthetic_mount_check)
    monkeypatch.setattr(scratch, '_mounts', lambda: [{'synthetic_mount_fixture': True}])
    monkeypatch.setattr(fleet, 'project_transaction', isolated_transaction)
    monkeypatch.setattr(storage, '_replace', isolated_replace)
    monkeypatch.setattr(scratch, 'inspect_scratch', lambda env, observed: {
        'status': 'SYNTHETIC_SCRATCH_ONLY', 'synthetic_only': True, 'live': deepcopy(observed)})
    result = runner.run_fleet_worker(plan, binding, plan_sha256=plan_sha,
        binding_sha256=binding_sha, state_root=state, worker_id='worker-01',
        poll_seconds=0.001, live_check=bounded_live_check)
    assert result['status'] == 'EXPERIMENT_SUITE_COMPLETE'
    assert result['completed_tasks'] == 185 and result['model_quality_or_speedup_claim'] is False
    assert len(executed) == len(set(executed)) == 185 and set(executed) == set(tasks)
    assert plan == original_plan and runner.digest(plan) == plan_sha
    assert Counter(tasks[task_id]['kind'] for task_id in executed) == {
        'gate': 1, 'train': 34, 'archive': 52, 'evaluate': 36, 'latency': 15,
        'compare_latency': 12, 'cleanup': 34, 'aggregate': 1}
    indexes = {task_id: index for index, task_id in enumerate(executed)}
    for task_id, task in tasks.items():
        assert all(indexes[dependency] < indexes[task_id] for dependency in task['dependencies'])
        completion = runner.completion_for(state, task_id)
        assert completion['plan_sha256'] == plan_sha and completion['kind'] == task['kind']
        assert completion['binding_sha256'] == binding_sha
        assert completion['dependencies'] == [runner.artifact(runner.completion_path(state, dependency))
                                             for dependency in task['dependencies']]
        assert len(list((state / 'tasks' / task_id / 'attempts').iterdir())) == 1
        receipt = runner.read(completion['receipts'][0]['path'])
        assert receipt['task_id'] == task_id and receipt['kind'] == task['kind']
        assert receipt['synthetic_only'] is True and receipt['real_model_executed'] is False
        assert runner.read(Path(completion['attempt']) / 'COMMAND_EXIT.json')['group_cleanup_verified'] is True
    assert not resident and len(archives) == 34 and len(results) == 18
    assert all(path.is_file() for path in [*archives.values(), *results.values()])
    assert len(list((state / 'chain-affinity').glob('*.json'))) == 17
    assert (namespace / 'locks/fleet-leases/isolated-full185.json').is_file()
    assert len(transactions) >= 2 * len(tasks)
    assert len(mount_checks) == 1
    assert len(runner.read(state / 'FLEET_RENDEZVOUS.json')['workers']) == 1
    assert not list(state.rglob('FAILED.json')) and not list(state.rglob('STOPPED.json'))
    assert len(child_ledgers) == 1
    child = runner.read(next(iter(child_ledgers)))
    assert len(child['reservations']) == 186
    assert all(row['state'] == 'RELEASED' for row in child['reservations'].values())
    scratch_ledger = runner.read(state / 'workers/worker-01/SCRATCH_BUDGET.json')
    assert len(scratch_ledger['reservations']) == 185
    assert all(row['state'] == 'RELEASED' for row in scratch_ledger['reservations'].values())
    assert parent.path.read_bytes() == parent_bytes
    final_parent = runner.read(parent.path)
    assert final_parent['reservations'][hold['reservation_token']]['state'] == 'ACTIVE'
    assert final_parent['reservations'][hold['reservation_token']]['home_bytes'] == 2 * 1024 ** 3
    assert final_parent['reservations'][old_cpu_token]['state'] == 'ACTIVE'
    assert _old_home_inventory(home, namespace) == old_home
