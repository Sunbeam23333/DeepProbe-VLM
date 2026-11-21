"""One-Pod storage delegation under a durable coordinator lifetime hold.

The coordinator and old CPU producers alone use the parent home ledger. The
GPU worker accounts only a disjoint new namespace and keeps its mutable child
ledger on verified private disk. Shared receipts are immutable observations,
never a cross-node current-state pointer. A hold is not automatically released.
"""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path

from . import suite_runner as runner
from .checkpoint_retention import check_pin, digest, file_pin, write_once
from .suite_storage import SharedBudget, StorageError, local_inventory, require, validate_config

RELEASE_POLICY = 'explicit_after_exact_worker_quiescence_and_final_namespace_inventory'


def contract(environment):
    value = environment.get('isolated_storage')
    require(isinstance(value, dict) and set(value) == {
        'schema_version', 'namespace_root', 'home_limit_bytes', 'home_safety_bytes', 'coordinator_hold'}
        and type(value['schema_version']) is int and value['schema_version'] == 1, 'Exact isolated storage contract required')
    profile = environment.get('execution_profile', {})
    from .suite_fleet import validate_execution_profile
    validate_execution_profile(profile, environment['suite_id'])
    require(profile.get('schema_version') == 3 and len(profile.get('workers', [])) == 1,
            'Isolated delegation permits exactly one private-disk worker')
    validate_config(environment['storage'])
    namespace = runner.safe(value['namespace_root'])
    roots = [runner.safe(p) for p in environment['storage']['roots']]
    require(str(namespace) == value['namespace_root'] and Path(value['namespace_root']).is_absolute()
            and any(namespace != p and namespace.is_relative_to(p) for p in roots),
            'Isolated namespace must be a new canonical child of an accounted home root')
    old_ledger = runner.safe(environment['storage']['ledger_root'])
    require(not namespace.is_relative_to(old_ledger) and not old_ledger.is_relative_to(namespace),
            'Isolated namespace and parent ledger must be disjoint')
    require(type(value['home_limit_bytes']) is int and 0 < value['home_limit_bytes'] <= 150_000_000_000
            and type(value['home_safety_bytes']) is int and 0 < value['home_safety_bytes'] < value['home_limit_bytes'],
            'Isolated lifetime byte bound and positive safety reserve required')
    return value


def validate_isolation(environment, *, verify_hold=True):
    value = contract(environment)
    if not verify_hold:
        return value
    hold = runner.read(check_pin(value['coordinator_hold']))
    worker = environment['execution_profile']['workers'][0]
    require(type(hold.get('schema_version')) is int and hold['schema_version'] == 1
            and hold.get('status') == 'ISOLATED_WORKER_LIFETIME_HOLD_ACTIVE'
            and hold.get('namespace_root') == value['namespace_root']
            and hold.get('home_limit_bytes') == value['home_limit_bytes']
            and hold.get('home_safety_bytes') == value['home_safety_bytes']
            and hold.get('suite_id') == environment['suite_id']
            and hold.get('worker') == worker
            and hold.get('profile_sha256') == digest(environment['execution_profile'])
            and hold.get('parent_storage_sha256') == digest(environment['storage'])
            and hold.get('release_policy') == RELEASE_POLICY
            and hold.get('automatic_release') is False,
            'Lifetime hold does not bind this exact isolated worker and parent policy')
    ledger = runner.read(check_pin(hold['parent_ledger_reserved']))
    row = ledger.get('reservations', {}).get(hold.get('reservation_token'))
    require(ledger.get('config_sha256') == digest(environment['storage'])
            and row and row.get('state') == 'ACTIVE'
            and row.get('owner') == hold.get('reservation_owner')
            and row.get('home_bytes') == value['home_limit_bytes'],
            'Coordinator did not durably reserve the full isolated lifetime bound')
    source = runner.source_identity({'source_files': environment['source_files']})
    require(hold.get('source_identity_sha256') == source,
            'Lifetime hold source identity differs')
    require(isinstance(hold.get('plan_sha256'), str) and runner.SHA.fullmatch(hold['plan_sha256'])
            and hold['reservation_owner'] == {
                'operation': 'isolated-worker-lifetime', 'suite_id': environment['suite_id'],
                'namespace_root': value['namespace_root'], 'job_name': worker['job_name'],
                'worker_id': worker['worker_id'], 'plan_sha256': hold['plan_sha256'],
                'source_identity_sha256': source} and row.get('oss_bytes') == 0,
            'Lifetime reservation owner differs from the exact plan/source/worker')
    runtime = environment.get('runtime_environment', {})
    namespace = runner.safe(value['namespace_root'])
    for key in ('HF_HOME', 'HF_MODULES_CACHE', 'TMPDIR', 'TORCH_EXTENSIONS_DIR', 'TRITON_CACHE_DIR',
                'TORCHINDUCTOR_CACHE_DIR',
                'DP_SUITE_KUBECTL_CACHE'):
        require(key in runtime and runner.safe(runtime[key]).is_relative_to(namespace),
                'Isolated runtime cache escapes its lifetime namespace: ' + key)
    readonly = hold.get('read_only_backbone_cache')
    if readonly is not None:
        require(set(readonly) == {'hub_cache', 'backbones_sha256'}
                and readonly['backbones_sha256'] == digest(environment['assets']['backbones'])
                and runtime.get('HF_HUB_CACHE') == readonly['hub_cache'],
                'Read-only Hub exception differs from the held backbone catalog')
    else:
        require(not environment.get('assets', {}).get('backbones')
                and runner.safe(runtime['HF_HUB_CACHE']).is_relative_to(namespace),
                'Production backbone inputs need the exact held read-only Hub cache catalog')
    if 'TRANSFORMERS_CACHE' in runtime:
        require(runtime['TRANSFORMERS_CACHE'] == runtime['HF_HUB_CACHE'],
                'Transformers cache must be the same explicitly bound Hub')
    return value


def isolate_environment(environment, coordinator_hold):
    """Build a runtime candidate without mutating legacy cache or input bindings."""
    require('isolated_storage' not in environment, 'Do not replace an existing isolation contract')
    hold = runner.read(check_pin(coordinator_hold))
    result = deepcopy(environment)
    result['isolated_storage'] = {'schema_version': 1, 'coordinator_hold': coordinator_hold,
        **{key: hold[key] for key in ('namespace_root', 'home_limit_bytes', 'home_safety_bytes')}}
    root = runner.safe(hold['namespace_root']) / 'runtime-cache'
    for key, suffix in {'HF_HOME': 'hf', 'HF_HUB_CACHE': 'hf/hub', 'HF_MODULES_CACHE': 'hf/modules', 'TMPDIR': 'tmp',
                        'TORCH_EXTENSIONS_DIR': 'extensions', 'TRITON_CACHE_DIR': 'triton',
                        'TORCHINDUCTOR_CACHE_DIR': 'tmp/torch-inductor',
                        'DP_SUITE_KUBECTL_CACHE': 'kubectl'}.items():
        result['runtime_environment'][key] = str(root / suffix)
    if hold.get('read_only_backbone_cache') is not None:
        result['runtime_environment']['HF_HUB_CACHE'] = hold['read_only_backbone_cache']['hub_cache']
    if 'TRANSFORMERS_CACHE' in result['runtime_environment']:
        result['runtime_environment']['TRANSFORMERS_CACHE'] = result['runtime_environment']['HF_HUB_CACHE']
    validate_isolation(result)
    return result


def create_lifetime_hold(environment, *, namespace_root, home_limit_bytes, home_safety_bytes,
                         plan_sha256, output, budget=None):
    """Coordinator-only explicit reservation; never called by GPU/preflight code.

    Caller proves that all previous GPU writers are terminal and runs this on
    the same devbox filesystem client as existing CPU producers. A failure
    after reserve intentionally retains the token for explicit reconciliation.
    """
    require(not os.environ.get('DP_SUITE_WORKER_ID'), 'GPU worker cannot create a parent lifetime hold')
    require('isolated_storage' not in environment and runner.SHA.fullmatch(plan_sha256),
            'Create hold from the pre-delegation environment and exact plan')
    proposed = {**environment, 'isolated_storage': {'schema_version': 1,
        'namespace_root': namespace_root, 'home_limit_bytes': home_limit_bytes,
        'home_safety_bytes': home_safety_bytes, 'coordinator_hold': {}}}
    contract(proposed)
    namespace, output = runner.safe(namespace_root), runner.safe(output)
    require(namespace.is_dir() and namespace.stat().st_uid == os.getuid()
            and output.is_relative_to(namespace), 'Create a new owned namespace before its hold')
    require(not (output / 'LIFETIME_HOLD.json').exists(), 'Lifetime hold is once-only; inspect the existing receipt')
    source = runner.source_identity({'source_files': environment['source_files']})
    worker = deepcopy(environment['execution_profile']['workers'][0])
    readonly = None
    if environment.get('assets', {}).get('backbones'):
        runtime = environment['runtime_environment']
        hub = runtime.get('HF_HUB_CACHE') or str(runner.safe(runtime['HF_HOME']) / 'hub')
        require(not runtime.get('TRANSFORMERS_CACHE') or runtime['TRANSFORMERS_CACHE'] == hub,
                'Original Transformers and Hub caches differ')
        readonly = {'hub_cache': str(runner.safe(hub)), 'backbones_sha256': digest(environment['assets']['backbones'])}
    owner = {'operation': 'isolated-worker-lifetime', 'suite_id': environment['suite_id'],
             'namespace_root': namespace_root, 'job_name': worker['job_name'],
             'worker_id': worker['worker_id'], 'plan_sha256': plan_sha256, 'source_identity_sha256': source}
    if budget is None:
        from .suite_storage import budget_for
        budget = budget_for(environment)
    require(budget.config == environment['storage'], 'Lifetime hold must use the unchanged parent ledger')
    token = budget.reserve(owner, home_limit_bytes, 0)
    output.mkdir(parents=True, exist_ok=True)
    with budget.lock():
        ledger = budget._read()
        require(ledger['reservations'].get(token, {}).get('state') == 'ACTIVE', 'Fresh parent hold disappeared')
        snapshot = budget._admit(ledger)
        saved = output / ('PARENT_LEDGER_RESERVED.' + digest(ledger) + '.json')
        write_once(saved, ledger)
    value = {'schema_version': 1, 'status': 'ISOLATED_WORKER_LIFETIME_HOLD_ACTIVE',
             'utc': runner.utc(), 'suite_id': environment['suite_id'], 'namespace_root': namespace_root,
             'home_limit_bytes': home_limit_bytes, 'home_safety_bytes': home_safety_bytes,
             'worker': worker, 'profile_sha256': digest(environment['execution_profile']),
             'parent_storage_sha256': digest(environment['storage']), 'plan_sha256': plan_sha256,
             'source_identity_sha256': source, 'reservation_token': token, 'reservation_owner': owner,
             'read_only_backbone_cache': readonly,
             'parent_ledger_reserved': file_pin(saved), 'parent_admission_snapshot': snapshot,
             'automatic_release': False, 'release_policy': RELEASE_POLICY}
    write_once(output / 'LIFETIME_HOLD.json', value)
    return file_pin(output / 'LIFETIME_HOLD.json')


def coordination_storage(binding):
    """Keep project leases and all scheduler journals out of the old namespace."""
    if 'isolated_storage' not in binding:
        return binding['storage']
    root = runner.safe(binding['isolated_storage']['namespace_root'])
    return {**binding['storage'], 'roots': [str(root)]}


def require_output(environment, path):
    if 'isolated_storage' in environment:
        namespace = runner.safe(contract(environment)['namespace_root'])
        path = runner.safe(path)
        require(path != namespace and path.is_relative_to(namespace), 'GPU output escapes its isolated home namespace')
    return path


def local_ledger_root(environment):
    worker = environment['execution_profile']['workers'][0]['worker_id']
    return runner.safe(environment['execution_profile']['scratch']['mount_path']) / 'isolated-budget' / worker


def _configuration(environment):
    value = validate_isolation(environment)
    root = local_ledger_root(environment)
    return {**deepcopy(environment['storage']), 'roots': [value['namespace_root'], str(root)],
            'ledger_root': str(root / 'ledger'), 'home_limit_bytes': value['home_limit_bytes'],
            'home_safety_bytes': value['home_safety_bytes'], 'oss_limit_bytes': None}


class IsolatedCpuView:
    """Read-only admission view; CPU preparation cannot create the child ledger."""
    def __init__(self, environment):
        self.environment = environment
        self.config = _configuration(environment)

    def snapshot(self):
        value = validate_isolation(self.environment)
        home = local_inventory([value['namespace_root']])
        return {'utc': runner.utc(), 'home': home,
                'oss': {'bytes': None, 'status': 'NOT_MEASURED_UNLIMITED_QUOTA'},
                'active_reservations': 0, 'reserved_bytes': {'home': 0, 'oss': 0},
                'projected_bytes': {'home': home['bytes'] + value['home_safety_bytes'], 'oss': None},
                'delegated_home_limit_bytes': value['home_limit_bytes'],
                'coordinator_hold': value['coordinator_hold'], 'child_ledger_created': False,
                'accounting_scope': 'isolated_namespace_under_coordinator_lifetime_hold'}

    def reserve(self, *args, **kwargs):
        raise StorageError('Isolated CPU admission is read-only; asset producers keep their original ledger')


class IsolatedWorkerBudget(SharedBudget):
    def __init__(self, environment, live, state_root):
        self.environment = environment
        value = validate_isolation(environment)
        require_output(environment, state_root)
        worker = environment['execution_profile']['workers'][0]
        require(live.get('worker_id') == worker['worker_id'] and all(live.get(key) for key in ('job_uid', 'pod_uid', 'node')),
                'Isolated budget requires the exact verified Pod allocation')
        self.local_root = local_ledger_root(environment)
        self.local_root.mkdir(parents=True, mode=0o700, exist_ok=True)
        existing_identity = (self.local_root / 'IDENTITY.json').exists()
        require(existing_identity or not any(self.local_root.iterdir()), 'Unidentified existing isolated child budget')
        require(not existing_identity or (self.local_root / 'ledger/LEDGER.json').is_file(),
                'Existing isolated identity has a missing ledger; explicit recovery is required')
        identity = {'isolation_sha256': digest(value), 'profile_sha256': digest(environment['execution_profile']),
                    **{key: live[key] for key in ('worker_id', 'job_uid', 'pod_uid', 'node')}}
        write_once(self.local_root / 'IDENTITY.json', identity)
        self.observations = runner.safe(value['namespace_root']) / 'budget-observations'
        self.observations.mkdir(mode=0o700, exist_ok=True)
        super().__init__(_configuration(environment), lambda: (_ for _ in ()).throw(
            StorageError('Isolated worker must not scan parent OSS quota')))

    def _write(self, ledger):
        super()._write(ledger)
        write_once(self.observations / (digest(ledger) + '.json'), ledger)

    def _snapshot(self, ledger):
        snapshot = super()._snapshot(ledger)
        value = self.environment['isolated_storage']
        return {**snapshot, 'delegated_home_limit_bytes': value['home_limit_bytes'],
                'coordinator_hold': value['coordinator_hold'],
                'accounting_scope': 'isolated_namespace_and_local_control_under_coordinator_lifetime_hold',
                'parent_ledger_opened': False}


def budget_for_isolated(environment):
    if not os.environ.get('DP_SUITE_WORKER_ID'):
        return IsolatedCpuView(environment)
    require(os.environ['DP_SUITE_WORKER_ID'] == environment['execution_profile']['workers'][0]['worker_id'],
            'Isolated worker identity differs from its single allocation')
    state = require_output(environment, os.environ['DP_SUITE_FLEET_STATE'])
    live = runner.read(state / 'workers' / os.environ['DP_SUITE_WORKER_ID'] / 'PRIVATE_SCRATCH_PREPARED.json')['live']
    require(os.environ.get('DP_SUITE_STORAGE_LEDGER') == str(local_ledger_root(environment) / 'ledger'),
            'Isolated child must inherit its private local ledger, never the parent ledger')
    return IsolatedWorkerBudget(environment, live, state)
