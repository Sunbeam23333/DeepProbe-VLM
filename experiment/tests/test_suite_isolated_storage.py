"""Real temporary ledgers for one-Pod delegation; no mount/GPU/cloud is certified."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from deepprobe_vlm import checkpoint_retention as retention
from deepprobe_vlm import suite_isolated_storage as isolated
from deepprobe_vlm import suite_runner as runner
from deepprobe_vlm import suite_scratch as scratch
from deepprobe_vlm import suite_storage as storage


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(runner.canonical(value) + b'\n')
    return retention.file_pin(path)


def files(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob('*') if path.is_file()}


def quiet(owner):
    return {'owner': owner, 'active_writers': [], 'active_uploads': []}


def no_remote():
    raise AssertionError('An unlimited fixture must never call a remote inventory')


@pytest.fixture
def delegation(tmp_path, monkeypatch):
    for name in ('DP_SUITE_WORKER_ID', 'DP_SUITE_FLEET_STATE', 'DP_SUITE_STORAGE_LEDGER',
                 'DP_SUITE_STORAGE_TOKEN', 'DP_SUITE_SCRATCH_TOKEN'):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path.resolve()
    home, mount = root / 'home', root / 'private-scratch'
    home.mkdir(); mount.mkdir()
    namespace = home / 'new-isolated-namespace'
    namespace.mkdir()
    source = home / 'source.py'; source.write_bytes(b'# immutable fixture source\n')
    profile = {'schema_version': 3, 'fleet_id': 'isolated-storage-fixture', 'priority': 'low',
        'gpu_product': 'NVIDIA-B300',
        'node_selector': {'sslaunch-pool': 'stream', 'aliyun.accelerator/nvidia_name': 'NVIDIA-L20D'},
        'workers': [{'worker_id': 'worker-01', 'job_name': 'jinhaopeng-dpvlm-isolated-storage-fixture',
                     'namespace': 'default'}],
        'scratch': {**scratch.DEFAULT_SCRATCH, 'mount_path': str(mount)}}
    configuration = {'schema_version': 2, 'roots': [str(home)], 'ledger_root': str(home / 'parent-ledger'),
        'home_limit_bytes': 50 * 1024 * 1024, 'oss_limit_bytes': None,
        'oss_prefixes': ['oss://ss-devbox-stream/fixture/isolated-storage/'],
        'home_safety_bytes': 32 * 1024, 'oss_safety_bytes': 0, 'free_guard_bytes': 0}
    environment = {'suite_id': profile['fleet_id'], 'execution_profile': profile,
                   'source_files': [retention.file_pin(source)], 'storage': configuration,
                   'runtime_environment': {}}
    parent = storage.SharedBudget(configuration, no_remote)
    cpu_token = parent.reserve({'operation': 'existing-cpu-producer', 'pid': 'fixture'}, 1024 * 1024, 0)
    plan_sha = runner.digest({'fixture': '185-task-plan-identity'})
    cap, safety = 4 * 1024 * 1024, 64 * 1024
    hold_pin = isolated.create_lifetime_hold(environment, namespace_root=str(namespace),
        home_limit_bytes=cap, home_safety_bytes=safety, plan_sha256=plan_sha,
        output=namespace / 'hold', budget=parent)
    environment = isolated.isolate_environment(environment, hold_pin)
    hold = runner.read(hold_pin['path'])
    live = {'worker_id': 'worker-01', 'job_uid': 'job-original', 'pod_uid': 'pod-original',
            'node': 'node-original'}
    state = namespace / 'state'
    write_json(state / 'workers' / 'worker-01' / 'PRIVATE_SCRATCH_PREPARED.json', {'live': live})
    return SimpleNamespace(root=root, home=home, mount=mount, namespace=namespace,
        env=environment, parent=parent, cpu_token=cpu_token, hold=hold, cap=cap, safety=safety,
        live=live, state=state, parent_bytes=parent.path.read_bytes(),
        local=isolated.local_ledger_root(environment))


def assert_parent_unchanged(value):
    assert value.parent.path.read_bytes() == value.parent_bytes
    ledger = json.loads(value.parent_bytes)
    row = ledger['reservations'][value.hold['reservation_token']]
    assert row['state'] == 'ACTIVE' and row['home_bytes'] == value.cap and row['oss_bytes'] == 0
    assert row['owner'] == value.hold['reservation_owner']
    assert ledger['reservations'][value.cpu_token]['state'] == 'ACTIVE'
    assert retention.file_pin(value.env['isolated_storage']['coordinator_hold']['path']) == value.env['isolated_storage']['coordinator_hold']
    assert retention.file_pin(value.hold['parent_ledger_reserved']['path']) == value.hold['parent_ledger_reserved']


def test_static_parent_hold_is_exact_and_cpu_view_is_entirely_read_only(delegation):
    value = delegation
    before = files(value.root)
    view = isolated.IsolatedCpuView(value.env)
    assert isolated.budget_for_isolated(value.env).__class__ is isolated.IsolatedCpuView
    for _ in range(2):
        snapshot = view.snapshot()
        actual = storage.local_inventory([str(value.namespace)])
        assert snapshot['home']['bytes'] == actual['bytes']
        assert snapshot['projected_bytes'] == {'home': actual['bytes'] + value.safety, 'oss': None}
        assert snapshot['reserved_bytes'] == {'home': 0, 'oss': 0}
        assert snapshot['child_ledger_created'] is False
        assert snapshot['coordinator_hold'] == value.env['isolated_storage']['coordinator_hold']
    with pytest.raises(storage.StorageError, match='read-only'):
        view.reserve({'task': 'not-allowed'}, 1, 0)
    assert not value.local.exists()
    assert not (value.namespace / 'budget-observations').exists()
    assert files(value.root) == before
    assert_parent_unchanged(value)


def test_parent_and_child_instances_share_only_private_ledger_and_preserve_observations(delegation):
    value = delegation
    parent_worker = isolated.IsolatedWorkerBudget(value.env, value.live, value.state)
    expected_path = value.local / 'ledger' / 'LEDGER.json'
    assert parent_worker.path == expected_path
    assert parent_worker.path.is_relative_to(value.mount)
    assert parent_worker.config['roots'] == [str(value.namespace), str(value.local)]
    assert parent_worker.config['oss_limit_bytes'] is None
    owner = {'worker_id': 'worker-01', 'job_uid': 'job-original', 'task_id': 'training', 'claim_id': 'claim-1'}
    token = parent_worker.reserve(owner, 200_000, 999_999_999)
    old_observations = files(parent_worker.observations)
    child = isolated.IsolatedWorkerBudget(deepcopy(value.env), deepcopy(value.live), value.state)
    child.extend(token, home_bytes=300_000, oss_bytes=1_999_999_999)
    observed = parent_worker.snapshot()
    assert observed['reserved_bytes'] == {'home': 300_000, 'oss': 1_999_999_999}
    assert observed['parent_ledger_opened'] is False
    assert observed['delegated_home_limit_bytes'] == value.cap
    assert observed['active_reservations'] == 1
    ledger_before = expected_path.read_bytes()
    with pytest.raises(storage.StorageError, match='not been proven quiescent'):
        child.release(token, lambda exact: quiet({**exact, 'job_uid': 'wrong-job'}))
    assert expected_path.read_bytes() == ledger_before
    parent_worker.release(token, quiet)
    assert child.snapshot()['active_reservations'] == 0
    current = json.loads(expected_path.read_bytes())
    assert current['reservations'][token]['state'] == 'RELEASED'
    assert [event['kind'] for event in current['events']] == ['RESERVED', 'EXTENDED', 'RELEASED']
    final_observations = files(parent_worker.observations)
    assert len(final_observations) == 4
    assert all(final_observations[name] == content for name, content in old_observations.items())
    for name, content in final_observations.items():
        assert name == runner.digest(json.loads(content)) + '.json'
    assert_parent_unchanged(value)


def test_cumulative_namespace_files_and_active_reservations_share_one_lifetime_cap(delegation):
    value = delegation
    existing = value.namespace / 'accepted-old-results.bin'
    existing.write_bytes(b'R' * (1024 * 1024))
    budget = isolated.IsolatedWorkerBudget(value.env, value.live, value.state)
    owner = {'worker_id': 'worker-01', 'job_uid': 'job-original', 'task_id': 'later-task'}
    token = budget.reserve(owner, 1024 * 1024, 0)
    snapshot = budget.snapshot()
    inventory = storage.local_inventory(budget.config['roots'])
    assert snapshot['projected_bytes']['home'] == inventory['bytes'] + 1024 * 1024 + value.safety
    assert snapshot['home']['bytes'] >= existing.stat().st_size
    remaining = value.cap - snapshot['projected_bytes']['home']
    assert 0 < remaining < value.cap
    before_ledger, before_observations = budget.path.read_bytes(), files(budget.observations)
    with pytest.raises(storage.StorageError, match='home capacity gated'):
        budget.reserve({**owner, 'task_id': 'would-overbook'}, remaining, 0)
    with pytest.raises(storage.StorageError, match='home capacity gated'):
        budget.extend(token, home_bytes=1024 * 1024 + remaining)
    assert budget.path.read_bytes() == before_ledger
    assert files(budget.observations) == before_observations
    budget.release(token, quiet)
    after = budget.snapshot()
    assert after['reserved_bytes']['home'] == 0
    assert after['home']['bytes'] >= existing.stat().st_size
    assert after['projected_bytes']['home'] == after['home']['bytes'] + value.safety
    assert existing.read_bytes() == b'R' * (1024 * 1024)
    assert_parent_unchanged(value)


@pytest.mark.parametrize('field', ['worker_id', 'job_uid', 'pod_uid', 'node'])
def test_different_allocation_cannot_adopt_existing_local_ledger(delegation, field):
    value = delegation
    original = isolated.IsolatedWorkerBudget(value.env, value.live, value.state)
    token = original.reserve({'task': 'still-active'}, 1000, 0)
    before = files(value.local)
    with pytest.raises(ValueError):
        isolated.IsolatedWorkerBudget(value.env, {**value.live, field: 'replacement'}, value.state)
    assert files(value.local) == before
    assert json.loads(original.path.read_bytes())['reservations'][token]['state'] == 'ACTIVE'
    assert_parent_unchanged(value)


def test_existing_identity_cannot_silently_recreate_a_missing_active_ledger(delegation):
    value = delegation
    budget = isolated.IsolatedWorkerBudget(value.env, value.live, value.state)
    budget.reserve({'task': 'must-not-forget'}, 1000, 0)
    observations = files(budget.observations)
    budget.path.unlink()  # Only this synthetic fixture's private ledger.
    with pytest.raises(ValueError):
        isolated.IsolatedWorkerBudget(value.env, value.live, value.state)
    assert not budget.path.exists()
    assert files(budget.observations) == observations
    assert_parent_unchanged(value)


def repinned_hold(value, hold_change=None, ledger_change=None):
    environment, hold = deepcopy(value.env), deepcopy(value.hold)
    if ledger_change:
        ledger = runner.read(hold['parent_ledger_reserved']['path'])
        ledger_change(ledger, hold)
        hold['parent_ledger_reserved'] = write_json(value.namespace / 'invalid-fixture' / 'SNAPSHOT.json', ledger)
    if hold_change:
        hold_change(hold)
    environment['isolated_storage']['coordinator_hold'] = write_json(
        value.namespace / 'invalid-fixture' / 'HOLD.json', hold)
    return environment


@pytest.mark.parametrize('field', ['suite_id', 'namespace_root', 'worker', 'profile_sha256',
    'parent_storage_sha256', 'source_identity_sha256', 'home_limit_bytes', 'home_safety_bytes',
    'release_policy', 'automatic_release', 'reservation_token'])
def test_repinned_hold_with_wrong_bound_fields_is_rejected(delegation, field):
    value = delegation
    changes = {'worker': {**value.hold['worker'], 'job_name': 'jinhaopeng-dpvlm-other'},
        'home_limit_bytes': value.cap - 1, 'home_safety_bytes': value.safety + 1,
        'automatic_release': True}
    environment = repinned_hold(value, lambda hold: hold.__setitem__(field, changes.get(field, 'wrong')))
    with pytest.raises(ValueError):
        isolated.IsolatedCpuView(environment)
    assert not value.local.exists()
    assert_parent_unchanged(value)


@pytest.mark.parametrize('change', ['config', 'missing-token', 'released', 'owner', 'amount'])
def test_repinned_parent_snapshot_still_requires_full_active_reservation(delegation, change):
    def corrupt(ledger, hold):
        row = ledger['reservations'][hold['reservation_token']]
        if change == 'config': ledger['config_sha256'] = '0' * 64
        elif change == 'missing-token': del ledger['reservations'][hold['reservation_token']]
        elif change == 'released': row['state'] = 'RELEASED'
        elif change == 'owner': row['owner'] = {'task': 'someone-else'}
        else: row['home_bytes'] -= 1
    environment = repinned_hold(delegation, ledger_change=corrupt)
    with pytest.raises(ValueError):
        isolated.validate_isolation(environment)
    assert_parent_unchanged(delegation)


def test_hold_and_snapshot_cannot_jointly_rebind_reservation_owner(delegation):
    def corrupt(ledger, hold):
        owner = {**hold['reservation_owner'], 'job_name': 'jinhaopeng-dpvlm-different-job'}
        ledger['reservations'][hold['reservation_token']]['owner'] = owner
        hold['reservation_owner'] = deepcopy(owner)
    environment = repinned_hold(delegation, ledger_change=corrupt)
    with pytest.raises(ValueError):
        isolated.validate_isolation(environment)
    assert_parent_unchanged(delegation)


@pytest.mark.parametrize('change', ['hold-sha', 'hold-size', 'hold-bytes', 'snapshot-bytes', 'source-pins', 'job-name'])
def test_changed_pins_or_environment_identity_are_rejected(delegation, change):
    value = delegation
    environment = deepcopy(value.env)
    if change == 'hold-sha': environment['isolated_storage']['coordinator_hold']['sha256'] = '0' * 64
    elif change == 'hold-size': environment['isolated_storage']['coordinator_hold']['bytes'] += 1
    elif change == 'hold-bytes': Path(value.env['isolated_storage']['coordinator_hold']['path']).write_bytes(b'{}\n')
    elif change == 'snapshot-bytes': Path(value.hold['parent_ledger_reserved']['path']).write_bytes(b'{}\n')
    elif change == 'source-pins': environment['source_files'][0]['sha256'] = '0' * 64
    else: environment['execution_profile']['workers'][0]['job_name'] = 'jinhaopeng-dpvlm-wrong-job'
    with pytest.raises(ValueError):
        isolated.IsolatedCpuView(environment)
    assert not value.local.exists()
    assert value.parent.path.read_bytes() == value.parent_bytes


@pytest.mark.parametrize('field,value', [('schema_version', True), ('schema_version', 2),
    ('home_limit_bytes', True), ('home_limit_bytes', 0), ('home_safety_bytes', 0)])
def test_contract_rejects_invalid_version_or_amounts(delegation, field, value):
    environment = deepcopy(delegation.env)
    environment['isolated_storage'][field] = value
    with pytest.raises(ValueError):
        isolated.validate_isolation(environment, verify_hold=False)


@pytest.mark.parametrize('change', ['two-workers', 'schema2', 'high-priority', 'wrong-pool'])
def test_contract_requires_complete_single_worker_private_profile(delegation, change):
    environment = deepcopy(delegation.env)
    profile = environment['execution_profile']
    if change == 'two-workers': profile['workers'].append({**profile['workers'][0], 'worker_id': 'worker-02'})
    elif change == 'schema2': profile['schema_version'] = 2
    elif change == 'high-priority': profile['priority'] = 'high'
    else: profile['node_selector']['sslaunch-pool'] = 'other'
    with pytest.raises(ValueError):
        isolated.validate_isolation(environment, verify_hold=False)


@pytest.mark.parametrize('key', ['HF_HOME', 'HF_HUB_CACHE', 'HF_MODULES_CACHE', 'TMPDIR',
                                'TORCH_EXTENSIONS_DIR', 'TRITON_CACHE_DIR', 'TRANSFORMERS_CACHE',
                                'DP_SUITE_KUBECTL_CACHE', 'TORCHINDUCTOR_CACHE_DIR'])
def test_runtime_cache_cannot_escape_to_original_home(delegation, key):
    environment = deepcopy(delegation.env)
    environment['runtime_environment'][key] = str(delegation.home / 'legacy-runtime-cache')
    with pytest.raises(ValueError, match='cache'):
        isolated.IsolatedCpuView(environment)
    assert not delegation.local.exists()
    assert_parent_unchanged(delegation)


def test_worker_factory_requires_worker_state_and_private_ledger_routing(delegation, monkeypatch):
    value = delegation
    monkeypatch.setenv('DP_SUITE_WORKER_ID', 'worker-01')
    monkeypatch.setenv('DP_SUITE_FLEET_STATE', str(value.state))
    monkeypatch.setenv('DP_SUITE_STORAGE_LEDGER', str(value.local / 'ledger'))
    first = isolated.budget_for_isolated(value.env)
    token = first.reserve({'task': 'factory-child'}, 1024, 0)
    child = isolated.budget_for_isolated(value.env)
    child.extend(token, 2048, 0)
    first.release(token, quiet)
    assert child.snapshot()['active_reservations'] == 0
    monkeypatch.setenv('DP_SUITE_STORAGE_LEDGER', value.env['storage']['ledger_root'])
    with pytest.raises(ValueError, match='private local ledger'):
        isolated.budget_for_isolated(value.env)
    monkeypatch.setenv('DP_SUITE_WORKER_ID', 'worker-02')
    with pytest.raises(ValueError, match='worker identity'):
        isolated.budget_for_isolated(value.env)
    assert_parent_unchanged(value)


@pytest.mark.parametrize('outside', ['namespace', 'parent-home', 'parent-ledger'])
def test_worker_outputs_cannot_escape_or_replace_namespace_root(delegation, outside):
    value = delegation
    path = {'namespace': value.namespace, 'parent-home': value.home,
            'parent-ledger': Path(value.env['storage']['ledger_root'])}[outside]
    with pytest.raises(ValueError, match='escapes'):
        isolated.require_output(value.env, path)
    assert isolated.require_output(value.env, value.state) == value.state
    assert_parent_unchanged(value)


def test_lifetime_hold_is_once_only_and_never_implicitly_released(delegation, monkeypatch):
    value = delegation
    environment = {key: item for key, item in value.env.items() if key != 'isolated_storage'}
    arguments = {'namespace_root': str(value.namespace), 'home_limit_bytes': value.cap,
        'home_safety_bytes': value.safety, 'plan_sha256': value.hold['plan_sha256'],
        'output': value.namespace / 'hold', 'budget': value.parent}
    with pytest.raises(ValueError, match='once-only'):
        isolated.create_lifetime_hold(environment, **arguments)
    monkeypatch.setenv('DP_SUITE_WORKER_ID', 'worker-01')
    with pytest.raises(ValueError, match='cannot create'):
        isolated.create_lifetime_hold(environment, **arguments)
    assert_parent_unchanged(value)


def test_hold_publication_failure_preserves_parent_reservation(delegation, monkeypatch):
    value = delegation
    environment = {key: deepcopy(item) for key, item in value.env.items() if key != 'isolated_storage'}
    namespace = value.home / 'another-namespace'; namespace.mkdir()
    original_write = isolated.write_once
    def interrupted(path, document):
        if Path(path).name == 'LIFETIME_HOLD.json':
            raise RuntimeError('synthetic hold publication failure')
        return original_write(path, document)
    monkeypatch.setattr(isolated, 'write_once', interrupted)
    with pytest.raises(RuntimeError, match='publication failure'):
        isolated.create_lifetime_hold(environment, namespace_root=str(namespace),
            home_limit_bytes=1024 * 1024, home_safety_bytes=4096,
            plan_sha256=value.hold['plan_sha256'], output=namespace / 'hold', budget=value.parent)
    ledger = json.loads(value.parent.path.read_bytes())
    orphan = [row for row in ledger['reservations'].values()
              if row['owner'].get('namespace_root') == str(namespace)]
    assert len(orphan) == 1 and orphan[0]['state'] == 'ACTIVE'
    assert orphan[0]['home_bytes'] == 1024 * 1024
    assert not (namespace / 'hold' / 'LIFETIME_HOLD.json').exists()
    assert len(list((namespace / 'hold').glob('PARENT_LEDGER_RESERVED.*.json'))) == 1
