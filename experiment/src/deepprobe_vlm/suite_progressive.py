"""Immutable CPU cache readiness for a frozen, progressively admitted fleet.

Only publication performs remote cache verification. Resolution checks local
evidence and never fetches checkpoint weights or constructs benchmark media.
An absent receipt means pending; invalid evidence is always an error.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from . import suite_runner as runner


BENCHMARKS = {'videomme899': 'videomme', 'tempcompass_mc': 'tempcompass'}
FRAMES = (4, 8, 16)
STATUS = 'SUITE_BENCHMARK_CACHE_READY'
# These modules define the legacy frame protocol, normalization, pin checks,
# complete remote readback and hydration. A new runtime may reuse a frozen
# publisher only when their real bytes still implement that exact protocol.
# Preflight/workflow also contain unrelated storage/resume admission: their
# complete ASTs are compared after removing only the reviewed additions below.
LEGACY_CACHE_SOURCE_MODULES = tuple('src/deepprobe_vlm/' + name + '.py' for name in (
    'suite_legacy_cache', 'suite_media', 'formal_integrity', 'manifest', 'data',
    'evaluate', 'checkpoint_retention', 'suite_workflow', 'suite_preflight'))


def _function(tree, name):
    values = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    runner.require(len(values) == 1, 'Legacy cache source must retain one function: ' + name)
    return values[0]


def _normalize_exact(nodes, added, original=''):
    """Undo an explicitly reviewed statement change, never an arbitrary body.

    Everything not matched byte-for-byte at the AST level survives the full
    module comparison, including signatures, imports, constants and surrounding
    checks. An altered or duplicated private branch therefore still fails.
    """
    needle = ast.parse(added).body
    replacements = ast.parse(original).body
    expected = [ast.dump(node, include_attributes=False) for node in needle]
    matches = [index for index in range(len(nodes) - len(needle) + 1)
               if [ast.dump(node, include_attributes=False) for node in nodes[index:index + len(needle)]] == expected]
    runner.require(len(matches) <= 1, 'Repeated private compatibility statements are not allowed')
    if matches:
        index = matches[0]
        nodes[index:index + len(needle)] = replacements


def _cache_preflight_identity(path):
    """Keep all frame, pixel, identity and existing admission implementations."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    _normalize_exact(tree.body, 'from .suite_run_root import inspect_run_root')
    _normalize_exact(_function(tree, 'inspect').body, """
check('run_root_directory', lambda: inspect_run_root(environment))
if failures:
    return result()
""")
    history = _function(tree, '_history_gate')
    for node in history.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == 'modern':
            _normalize_exact(node.body, """
verify_history_binding(plan, task, h, verify_files=True)
if environment['execution_profile'].get('schema_version') == 3 and h['mode'] == 'resume':
    from .suite_checkpoint_resume import verify_private_resume
    verify_private_resume(task, h, environment)
""", 'verify_history_binding(plan, task, h, verify_files=True)')
    _normalize_exact(_function(tree, '_initial_storage_gate').body, """
if environment.get('execution_profile', {}).get('schema_version') == 3:
    return _chain_storage_gate(plan, binding, environment, estimates)
""")
    _normalize_exact(_function(tree, '_chain_storage_gate').body, """
if environment.get('execution_profile', {}).get('schema_version') == 3:
    from .suite_chains import private_capacity_requirements
    requirements = private_capacity_requirements(plan, binding, environment, estimates)
    observed = runner.capacity(binding, requirements['shared_home_additional_bytes'])
    return {**observed, **requirements, 'future_capacity_rechecked_per_task': True,
            'private_node_storage_observed_on_coordinator': False}
""")
    return ast.dump(tree, include_attributes=False)


def _cache_workflow_identity(path):
    """Exclude only reviewed run-container preparation, not cache preparation."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    prepare = _function(tree, 'prepare_workflow')
    _normalize_exact(prepare.body, """
if 'isolated_storage' in env:
    from .suite_isolated_storage import require_output
    require_output(env, output)
    require_output(env, candidate['state_root'])
""")
    _normalize_exact(prepare.body, """
if env.get('run_root'):
    declared_root = runner.safe(env['run_root'])
    runner.require(output != declared_root and not output.is_relative_to(declared_root)
                   and not declared_root.is_relative_to(output),
                   'Workflow reports must be disjoint from the declared suite run root')
""")
    _normalize_exact(prepare.body, """
from .suite_run_root import prepare_run_root, validate_run_root_location
def run_root_failure(exc):
    result = {'status': 'SUITE_WORKFLOW_RUN_ROOT_GATED', 'error_type': type(exc).__name__,
              'error': str(exc), 'cpu_admitted': False, 'gpu_submitted': False,
              'asset_preparation_started': False, 'automatic_deletion': False}
    save(attempt / 'WORKFLOW_GATED.json', result)
    return result
try:
    validate_run_root_location(env)
except (ValueError, OSError, KeyError) as exc:
    return run_root_failure(exc)
""")
    _normalize_exact(prepare.body, """
try:
    run_root = prepare_run_root(env)
except (ValueError, OSError, KeyError) as exc:
    return run_root_failure(exc)
run_root_pin = save(attempt / 'RUN_ROOT_PREPARATION.json', run_root)
""")
    _normalize_exact(prepare.body, """
result = {'status': 'SUITE_WORKFLOW_BOUND_NOT_ADMITTED', 'plan': plan_pin, 'environment': environment_pin,
          'binding': binding_pin, 'cache_preparations': cache_results, 'gpu_submitted': False,
          'state_root': candidate['state_root'], 'cpu_admitted': False,
          'run_root_preparation': run_root_pin}
""", """
result = {'status': 'SUITE_WORKFLOW_BOUND_NOT_ADMITTED', 'plan': plan_pin, 'environment': environment_pin,
          'binding': binding_pin, 'cache_preparations': cache_results, 'gpu_submitted': False,
          'state_root': candidate['state_root'], 'cpu_admitted': False}
""")
    _normalize_exact(_function(tree, 'main').body, """
unsuccessful = result.get('status') in {'SUITE_WORKFLOW_CAPACITY_GATED', 'SUITE_WORKFLOW_HISTORY_GATED',
                                     'SUITE_WORKFLOW_RUN_ROOT_GATED'}
""", """
unsuccessful = result.get('status') in {'SUITE_WORKFLOW_CAPACITY_GATED', 'SUITE_WORKFLOW_HISTORY_GATED'}
""")
    return ast.dump(tree, include_attributes=False)


def _cache_runner_identity(path):
    """Retain the runner except reviewed workload and scratch plumbing.

    Hash/path/pin primitives, constants, imports and all other definitions stay
    exact. Even the private worker body remains protected: only the explicit
    token routing and scratch initialization statements below are normalized.
    """
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    # Only the explicit cache variable is added; frame/scoring code stays exact.
    _normalize_exact(tree.body, '''
ENVIRONMENT_KEYS = set(OFFLINE) | {
    "PATH", "PYTHONPATH", "HF_HOME", "HF_HUB_CACHE", "HF_MODULES_CACHE", "TRANSFORMERS_CACHE",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TMPDIR", "NCCL_DEBUG",
    "NCCL_IB_TIMEOUT", "NCCL_NET_PLUGIN", "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "NCCL_ASYNC_ERROR_HANDLING", "TOKENIZERS_PARALLELISM", "DP_SUITE_KUBECTL_CACHE",
}
''', '''
ENVIRONMENT_KEYS = set(OFFLINE) | {
    "PATH", "PYTHONPATH", "HF_HOME", "HF_HUB_CACHE", "HF_MODULES_CACHE", "TRANSFORMERS_CACHE",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "TMPDIR", "NCCL_DEBUG",
    "NCCL_IB_TIMEOUT", "NCCL_NET_PLUGIN", "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "NCCL_ASYNC_ERROR_HANDLING", "TOKENIZERS_PARALLELISM", "DP_SUITE_KUBECTL_CACHE",
}
''')
    _normalize_exact(tree.body, '''
ENVIRONMENT_KEYS = set(OFFLINE) | {
    "PATH", "PYTHONPATH", "HF_HOME", "HF_HUB_CACHE", "HF_MODULES_CACHE", "TRANSFORMERS_CACHE",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "TMPDIR", "NCCL_DEBUG",
    "NCCL_IB_TIMEOUT", "NCCL_NET_PLUGIN", "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "NCCL_ASYNC_ERROR_HANDLING", "TOKENIZERS_PARALLELISM", "DP_SUITE_KUBECTL_CACHE",
}
''', '''
ENVIRONMENT_KEYS = set(OFFLINE) | {
    "PATH", "PYTHONPATH", "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "TMPDIR", "NCCL_DEBUG",
    "NCCL_IB_TIMEOUT", "NCCL_NET_PLUGIN", "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "NCCL_ASYNC_ERROR_HANDLING", "TOKENIZERS_PARALLELISM",
}
''')
    validation = _function(tree, 'validate_binding')
    _normalize_exact(validation.body, '''
require(isinstance(binding, dict) and set(binding) == {
    "schema_version", "suite_id", "plan_sha256", "allocation", "source_files", "storage",
    "environment", "startup", "tasks"} | ({"execution_profile"} if binding.get("schema_version") == 2 else set())
    | ({"isolated_storage"} if 'isolated_storage' in binding else set()), "invalid execution binding fields")
''', '''
require(isinstance(binding, dict) and set(binding) == {
    "schema_version", "suite_id", "plan_sha256", "allocation", "source_files", "storage",
    "environment", "startup", "tasks"} | ({"execution_profile"} if binding.get("schema_version") == 2 else set()), "invalid execution binding fields")
''')
    _normalize_exact(validation.body, '''
if 'isolated_storage' in binding:
    require(binding['schema_version'] == 2, 'Isolated storage requires a private fleet binding')
    from .suite_isolated_storage import contract
    contract(binding)
''')
    _normalize_exact(tree.body, '''
def completion_path(state_root, task_id):
    """Keep a typed import distinct from a task executed by this fleet."""
    require(isinstance(task_id, str) and ID.fullmatch(task_id), "unsafe task id")
    root = safe(state_root)
    executed = root / "tasks" / task_id / "COMPLETED.json"
    imported = root / "task-imports" / (task_id + ".json")
    require(not (executed.exists() and imported.exists()), "Task has conflicting executed/imported completion records")
    return imported if imported.exists() else executed
''')
    _normalize_exact(_function(tree, 'completion_for').body, '''
path = completion_path(state_root, task_id)
if path == safe(state_root) / "task-imports" / (task_id + ".json"):
    from .suite_task_import import get_task_import
    return get_task_import(state_root, task_id)
value = read(path)
''', 'value = read(safe(state_root) / "tasks" / task_id / "COMPLETED.json")')
    _function(tree, '_fleet_live').body = [ast.Pass()]
    worker = _function(tree, 'run_fleet_worker')
    _normalize_exact(worker.body, '''
if 'isolated_storage' in environment:
    from .suite_isolated_storage import require_output, validate_isolation
    require(environment['isolated_storage'] == binding.get('isolated_storage'), 'Isolated environment/binding differ')
    validate_isolation(environment)
    require(read(environment['isolated_storage']['coordinator_hold']['path'])['plan_sha256'] == plan_sha256,
            'Lifetime hold belongs to another scientific plan')
    require_output(environment, state_root)
    from .suite_isolated_pod import inspect_isolated_mounts
    from .suite_scratch import _mounts
    inspect_isolated_mounts(environment, _mounts())
''')
    _normalize_exact(_function(worker, 'execute').body, """
tokens = ({'DP_SUITE_STORAGE_TOKEN': token['home'],
           **({'DP_SUITE_SCRATCH_TOKEN': token['scratch']} if token['scratch'] else {})}
          if isinstance(token, dict) else {'DP_SUITE_STORAGE_TOKEN': token})
return run_command(command, binding, task_context, attempt,
    extra_environment={**worker_environment, **tokens,
                       **({'DP_SUITE_TASK_ID': task_context['task_id']} if 'task_id' in task_context else {})},
    storage_snapshot=budget.snapshot())
""", """
return run_command(command, binding, task_context, attempt,
    extra_environment={**worker_environment, 'DP_SUITE_STORAGE_TOKEN': token},
    storage_snapshot=budget.snapshot())
""")
    release = _function(worker, 'release_if_quiet')
    _normalize_exact(release.body, """
token_key = digest(token) if isinstance(token, dict) else token
if token_key in released_tokens:
    return True
""", """
if token in released_tokens:
    return True
""")
    for node in release.body:
        if isinstance(node, ast.If):
            _normalize_exact(node.body, 'released_tokens.add(token_key)', 'released_tokens.add(token)')
    for node in worker.body:
        if isinstance(node, ast.Try):
            for private in node.body:
                if isinstance(private, ast.If):
                    _normalize_exact(private.body, '''
if 'isolated_storage' in environment:
    from .suite_isolated_storage import IsolatedWorkerBudget
    budget = IsolatedWorkerBudget(environment, scratch_live, state_root)
    worker_environment['DP_SUITE_STORAGE_LEDGER'] = str(budget.root)
''')
            for handler in node.handlers:
                _normalize_exact(handler.body, 'from .suite_isolated_storage import coordination_storage')
                for guarded in handler.body:
                    if isinstance(guarded, ast.With):
                        for item in guarded.items:
                            if ast.dump(item.context_expr, include_attributes=False) == ast.dump(
                                    ast.parse("project_transaction(coordination_storage(binding))", mode='eval').body,
                                    include_attributes=False):
                                item.context_expr = ast.parse("project_transaction(binding['storage'])", mode='eval').body
            _normalize_exact(node.body, """
if binding['execution_profile'].get('schema_version') == 3:
    from .suite_scratch import inspect_scratch, worker_run_root
    from .suite_scratch_budget import PrivateScratchBudget, TieredTaskBudget
    scratch_live = live_check()
    scratch_proof = inspect_scratch(environment, scratch_live)
    run_root = worker_run_root(binding['execution_profile'], worker_id)
    safe(run_root.parent).mkdir(mode=0o700, exist_ok=True)
    safe(run_root).mkdir(mode=0o700)
    require(not any(run_root.iterdir()), 'A new private worker must have an empty run container')
    _fsync_directory(run_root.parent)
    write_once(state_root / 'workers' / worker_id / 'PRIVATE_SCRATCH_PREPARED.json', {
        'live': scratch_live, 'inspection': scratch_proof, 'run_root': str(run_root),
        'profile_sha256': digest(binding['execution_profile'])})
    private_budget = PrivateScratchBudget(binding['execution_profile'], worker_id, scratch_live, state_root)
    budget = TieredTaskBudget(budget, private_budget, {task['id']: task for task in plan['tasks']})
    caches = safe(binding['execution_profile']['scratch']['mount_path']) / 'runtime-cache' / worker_id
    for key, name in {'TMPDIR': 'tmp', 'CUDA_CACHE_PATH': 'cuda', 'TRITON_CACHE_DIR': 'triton',
                      'TORCH_EXTENSIONS_DIR': 'torch-extensions'}.items():
        path = safe(caches / name)
        path.mkdir(parents=True, mode=0o700, exist_ok=True)
        worker_environment[key] = str(path)
""")
            for loop in node.body:
                if isinstance(loop, ast.While):
                    _normalize_exact(loop.body, '''
dependency_pins = [artifact(completion_path(state_root, item)) for item in task['dependencies']]
''', '''
dependency_pins = [artifact(state_root / 'tasks' / item / 'COMPLETED.json') for item in task['dependencies']]
''')
    return ast.dump(tree, include_attributes=False)


def _legacy_publisher_compatible(environment, producer):
    if producer['source_root'] == environment['experiment_root']:
        return
    # A different directory is allowed only with a live, complete old source
    # attestation and the current runtime's complete, byte-verified source pins.
    # Nothing here rebases or rewrites the old contract or READY proof.
    _producer(environment, producer, live=True)
    from .formal_integrity import source_inventory
    old_root = _accounted(environment, producer['source_root'])
    current_root = _accounted(environment, environment['experiment_root'])
    pins = environment.get('source_files')
    runner.require(isinstance(pins, list) and pins, 'Legacy cache reuse requires current runtime source pins')
    current = {}
    for pin in pins:
        runner.require(isinstance(pin, dict) and set(pin) == {'path', 'bytes', 'sha256'}
                       and type(pin['bytes']) is int and pin['bytes'] > 0,
                       'Legacy cache runtime source pins must include path/size/SHA')
        path = runner.safe(pin['path'])
        runner.require(path != current_root and path.is_relative_to(current_root),
                       'Legacy cache runtime source pin escapes current source root')
        name = path.relative_to(current_root).as_posix()
        runner.require(name not in current, 'Duplicate legacy cache runtime source pin')
        runner.verify_pin(pin)
        current[name] = pin['sha256']
    runner.require(current == source_inventory(current_root),
                   'Legacy cache runtime source pins differ from its complete source tree')
    old = runner.read(producer['source_manifest']['path'])['files']
    identities = {'src/deepprobe_vlm/' + module + '.py': identity for module, identity in (
        ('suite_runner', _cache_runner_identity), ('suite_preflight', _cache_preflight_identity),
        ('suite_workflow', _cache_workflow_identity))}
    for name in LEGACY_CACHE_SOURCE_MODULES:
        runner.require(name in old and name in current
                       and (name in identities or old[name] == current[name]),
                       'Legacy cache validation source differs or is unpinned: ' + name)
    for name, identity in identities.items():
        label = Path(name).stem.removeprefix('suite_')
        runner.require(name in old and name in current, 'Legacy cache ' + label + ' source is unpinned')
        runner.require(identity(old_root / name) == identity(current_root / name),
                       'Legacy cache ' + label + ' validation implementation differs')
        # Recheck the exact bytes consumed by the semantic comparison.
        runner.verify_pin({'path': str(old_root / name), 'sha256': old[name]})
        runner.verify_pin({'path': str(current_root / name), 'sha256': current[name]})
    runner.verify_pin(producer['source_manifest'])


def key(benchmark, frames):
    runner.require(benchmark in BENCHMARKS and type(frames) is int and frames in FRAMES,
                   'Unsupported progressive benchmark shape')
    return benchmark + ':f' + str(frames)


def _context(environment):
    from .suite_fleet import validate_execution_profile
    from .suite_storage import validate_config
    runner.require(isinstance(environment, dict) and environment.get('execution_profile', {}).get('schema_version') in {2, 3}
                   and environment.get('storage', {}).get('schema_version') == 2,
                   'Progressive caches require a schema-v2/v3 fleet and schema-v2 storage')
    validate_execution_profile(environment['execution_profile'], suite_id=environment['suite_id'])
    validate_config(environment['storage'])


def _accounted(environment, value):
    runner.require(isinstance(value, str), 'Accounted cache paths must be strings')
    path = runner.safe(value)
    runner.require(str(path) == value and any(path != runner.safe(root) and path.is_relative_to(runner.safe(root))
                   for root in environment['storage']['roots']), 'Progressive cache path escapes accounted roots')
    return path


def _metadata_pin(environment, pin):
    from .suite_media import MAX_METADATA_BYTES
    runner.require(isinstance(pin, dict) and set(pin) == {'path', 'bytes', 'sha256'}
                   and type(pin['bytes']) is int and 0 < pin['bytes'] <= MAX_METADATA_BYTES,
                   'Complete bounded metadata pin required')
    _accounted(environment, pin['path'])
    runner.verify_pin(pin)


def _request(environment, request):
    from .suite_media import _key, PROJECT_PREFIX
    fields = {'benchmark', 'frames', 'source_manifest', 'accepted_decode_receipt', 'output_dir', 'oss_prefix',
              'media_root', 'max_png_bytes', 'metadata_reserve_bytes'}
    runner.require(isinstance(request, dict) and fields <= set(request) <= fields | {'archive_audit', 'legacy_reuse'},
                   'Invalid progressive benchmark request')
    identity = key(request['benchmark'], request['frames'])
    for name in ('source_manifest', 'accepted_decode_receipt', 'archive_audit'):
        if name in request:
            _metadata_pin(environment, request[name])
    if 'legacy_reuse' in request:
        from .suite_legacy_cache import input_inventory
        runner.require(isinstance(request['legacy_reuse'], dict)
                       and set(request['legacy_reuse']) == {'inventory', 'acceptance'}, 'Invalid legacy reuse evidence')
        for pin in request['legacy_reuse'].values():
            _metadata_pin(environment, pin)
        input_inventory(request)
    runner.require(type(request['max_png_bytes']) is int and request['max_png_bytes'] > 0
                   and type(request['metadata_reserve_bytes']) is int
                   and request['metadata_reserve_bytes'] >= 64 * 1024 * 1024, 'Invalid progressive cache byte reserves')
    output = _accounted(environment, request['output_dir'])
    _accounted(environment, str(output.with_name(output.name + '-inputs')))
    _accounted(environment, request['media_root'])
    prefix = _key(request['oss_prefix'])
    runner.require(prefix.endswith('/') and prefix != PROJECT_PREFIX
                   and all(part and part not in {'.', '..'} for part in prefix[:-1].split('/'))
                   and any(('oss://ss-devbox-stream/' + prefix).startswith(root)
                           for root in environment['storage']['oss_prefixes']),
                   'Cache prefix must be an exact accounted stream prefix')
    return identity


def _producer(environment, producer, *, live=False):
    runner.require(isinstance(producer, dict) and set(producer) == {'source_root', 'source_manifest'},
                   'Exact cache producer source required')
    root = _accounted(environment, producer['source_root'])
    pin = producer['source_manifest']
    _metadata_pin(environment, pin)
    runner.require(not Path(pin['path']).is_relative_to(root), 'Producer source manifest must be outside its source tree')
    document = runner.read(pin['path'])
    runner.require(isinstance(document, dict) and document.get('schema_version') == 1
                   and document.get('purpose') == 'DEEPPROBE_FROZEN_P1_SOURCE'
                   and isinstance(document.get('files'), dict) and document['files'],
                   'Invalid frozen cache producer source manifest')
    if live:
        from .formal_integrity import verify_frozen_source_manifest
        return verify_frozen_source_manifest(root, Path(pin['path']), pin['sha256'])
    return None


def _declaration(environment, declaration):
    _context(environment)
    runner.require(isinstance(declaration, dict) and set(declaration) == {'contract', 'contract_sha256', 'ready_path'},
                   'Invalid progressive cache declaration')
    contract = declaration['contract']
    runner.require(isinstance(contract, dict) and set(contract) == {'schema_version', 'suite_id', 'request', 'producer'}
                   and type(contract['schema_version']) is int and contract['schema_version'] == 1
                   and contract['suite_id'] == environment['suite_id']
                   and declaration['contract_sha256'] == runner.digest(contract), 'Progressive cache contract differs')
    identity = _request(environment, contract['request'])
    _producer(environment, contract['producer'])
    if 'legacy_reuse' in contract['request']:
        _legacy_publisher_compatible(environment, contract['producer'])
    ready = _accounted(environment, declaration['ready_path'])
    request = contract['request']
    output = Path(request['output_dir'])
    owned_dirs = (output, output.with_name(output.name + '-inputs'), Path(request['media_root']),
                  Path(contract['producer']['source_root']))
    runner.require(not any(ready == root or ready.is_relative_to(root) or root.is_relative_to(ready)
                           for root in owned_dirs), 'READY path overlaps cache or producer directories')
    runner.require(ready.suffix == '.json', 'READY must use an exact JSON artifact path')
    runner.require(all(ready != Path(request[name]['path']) for name in ('source_manifest', 'accepted_decode_receipt', 'archive_audit')
                       if name in request) and ready != Path(contract['producer']['source_manifest']['path']),
                   'READY path overlaps pinned input evidence')
    return identity


def make_declaration(environment, request, producer_source_manifest, ready_path, producer_source_root=None):
    declaration = {'contract': {'schema_version': 1, 'suite_id': environment['suite_id'], 'request': deepcopy(request),
        'producer': {'source_root': str(producer_source_root or environment['experiment_root']),
                     'source_manifest': deepcopy(producer_source_manifest)}}, 'ready_path': str(ready_path)}
    declaration['contract_sha256'] = runner.digest(declaration['contract'])
    _declaration(environment, declaration)
    _producer(environment, declaration['contract']['producer'], live=True)
    return declaration


def validate_declarations(environment, plan):
    """Validate all six frozen shapes without waiting for any READY receipt."""
    _context(environment)
    declarations = environment.get('benchmark_preparations')
    expected = {key(benchmark, frames) for benchmark in BENCHMARKS for frames in FRAMES}
    runner.require(isinstance(declarations, dict) and set(declarations) == expected,
                   'Progressive admission requires all six benchmark declarations')
    shapes = {key(task['benchmark'], task['frames']) for task in plan['tasks'] if task['kind'] == 'evaluate'}
    runner.require(shapes == expected, 'Scientific plan must retain all six benchmark shapes')
    paths, ready_paths, prefixes = [], [], []
    for identity, declaration in declarations.items():
        runner.require(_declaration(environment, declaration) == identity, 'Benchmark declaration key differs')
        request = declaration['contract']['request']
        output = Path(request['output_dir'])
        paths.extend((output, output.with_name(output.name + '-inputs'), Path(request['media_root'])))
        ready_paths.append(Path(declaration['ready_path']))
        prefixes.append(request['oss_prefix'])
    runner.require(all(a != b and not a.is_relative_to(b) and not b.is_relative_to(a)
                       for i, a in enumerate(paths) for b in paths[i + 1:]), 'Progressive cache directories overlap')
    runner.require(len(set(ready_paths)) == len(ready_paths) and all(
        not (ready == root or ready.is_relative_to(root) or root.is_relative_to(ready))
        for ready in ready_paths for root in paths), 'Progressive READY paths overlap')
    runner.require(all(not a.startswith(b) and not b.startswith(a)
                       for i, a in enumerate(prefixes) for b in prefixes[i + 1:]), 'Progressive cache prefixes overlap')
    return declarations


def validate_admission(preflight, environment, plan):
    """Accept only a complete, explicit progressive CPU admission snapshot.

    A deferred cache may have become ready since this snapshot was produced.
    Caches claimed ready by the snapshot must still have the exact same pin.
    Launcher separately verifies plan/environment/binding hashes and freshness.
    """
    declarations = validate_declarations(environment, plan)
    runner.require(isinstance(preflight, dict) and preflight.get('status') == 'SUITE_CPU_PROGRESSIVE_ADMITTED'
                   and preflight.get('benchmark_preparation_mode') == 'external_progressive'
                   and preflight.get('failures') == [], 'Explicit progressive CPU admission required')
    checks = preflight.get('checks')
    runner.require(isinstance(checks, list) and checks, 'Progressive CPU checks are missing')
    expected = {'benchmark/' + benchmark + '/' + str(frames): key(benchmark, frames)
                for benchmark in BENCHMARKS for frames in FRAMES}
    seen, deferred = set(), []
    for row in checks:
        runner.require(isinstance(row, dict) and isinstance(row.get('check'), str)
                       and row.get('status') in {'passed', 'deferred'}, 'Invalid progressive CPU check')
        name = row['check']
        if name not in expected:
            runner.require(not name.startswith('benchmark/') and row['status'] == 'passed',
                           'Only declared benchmark checks may be deferred')
            continue
        runner.require(name not in seen, 'Duplicate progressive benchmark check')
        seen.add(name)
        declaration = declarations[expected[name]]
        request = declaration['contract']['request']
        result = row.get('result')
        if row['status'] == 'deferred':
            wanted = {'status': 'DEFERRED_BENCHMARK', 'benchmark': request['benchmark'], 'frames': request['frames'],
                      'contract_sha256': declaration['contract_sha256'], 'ready_path': declaration['ready_path']}
            runner.require(result == wanted, 'Deferred benchmark differs from frozen declaration')
            deferred.append(wanted)
        else:
            runner.require(isinstance(result, dict) and set(result) == {'ready_pin', 'contract_sha256', 'cache_binding_sha256'},
                           'Ready benchmark admission must pin its exact receipt')
            resolved = resolve_ready(environment, request['benchmark'], request['frames'], expected_pin=result['ready_pin'])
            runner.require(resolved is not None and result == {name: resolved[name] for name in result},
                           'Admitted READY receipt differs from current evidence')
    runner.require(seen == set(expected), 'Progressive CPU admission omits benchmark shapes')
    supplied = preflight.get('deferred_benchmarks')
    runner.require(isinstance(supplied, list) and all(isinstance(item, dict) for item in supplied)
                   and sorted(map(runner.canonical, supplied)) == sorted(map(runner.canonical, deferred)),
                   'Deferred benchmark inventory differs from CPU checks')
    return True


def _asset(environment, declaration, asset):
    from . import suite_media as media
    request = declaration['contract']['request']
    runner.require(isinstance(asset, dict) and set(asset) == {'cache', 'media_root', 'max_bytes'}
                   and asset['media_root'] == request['media_root'] and type(asset['max_bytes']) is int
                   and asset['max_bytes'] == request['max_png_bytes'], 'READY asset differs from frozen request')
    if 'legacy_reuse' in request:
        from .suite_legacy_cache import validate_asset
        for name in ('source_manifest', 'source_binding', 'manifest', 'receipt'):
            _metadata_pin(environment, asset['cache'][name])
        return validate_asset(request, asset)
    cache = asset['cache']
    runner.require(isinstance(cache, dict) and set(cache) == {'kind', 'source_manifest', 'source_binding', 'manifest',
                   'receipt', 'complete', 'plan', 'prefix'} and cache['kind'] == 'suite'
                   and cache['prefix'] == request['oss_prefix'] and cache['source_manifest'] == request['source_manifest'],
                   'READY cache prefix/source differs from frozen request')
    output = Path(request['output_dir'])
    inputs = output.with_name(output.name + '-inputs')
    names = {'source_binding': inputs / 'SOURCE_BINDING.json', 'manifest': output / 'manifest.jsonl',
             'receipt': output / 'FRAME_CACHE_RECEIPT.json', 'complete': output / 'CACHE_COMPLETE.json',
             'plan': output / 'BUILD_PLAN.json'}
    for name, path in names.items():
        _metadata_pin(environment, cache[name])
        runner.require(cache[name]['path'] == str(path), 'READY cache metadata path differs from frozen request')
    rows, binding, inventory, frames, _, _ = media._bound_cache_inventory(cache)
    runner.require(frames == request['frames'] and binding['dataset'] == BENCHMARKS[request['benchmark']],
                   'READY benchmark or frame shape differs')
    runner.require(binding == runner.read(cache['source_binding']['path']), 'Cache source binding evidence differs')
    evidence = binding.get('normalization_evidence')
    runner.require(isinstance(evidence, dict) and set(evidence) == {'original_decode_receipt', 'archive_audit', 'archive_bound_receipt'}
                   and evidence['original_decode_receipt'] == request['accepted_decode_receipt']
                   and evidence['archive_audit'] == request.get('archive_audit'), 'Cache decode/archive evidence differs from request')
    normalized = evidence['archive_bound_receipt']
    _metadata_pin(environment, normalized)
    normalized_value = runner.read(normalized['path'])
    runner.require(normalized['path'] == str(inputs / 'ARCHIVE_BOUND_SOURCE_RECEIPT.json')
                   and binding.get('accepted_original_decode_receipt_sha256') == media.digest(media.encode(normalized_value)),
                   'Cache normalized decode receipt differs')
    catalog_path = inputs / 'ARCHIVE_CATALOG.json'
    _metadata_pin(environment, runner.artifact(catalog_path))
    catalog = runner.read(catalog_path)
    runner.require(isinstance(catalog, list)
                   and binding.get('archive_catalog_sha256') == media.digest(media.encode(catalog)),
                   'Cache original archive catalog differs')
    from .suite_workflow import bind_legacy_archive_identities
    runner.require(normalized_value == bind_legacy_archive_identities(
        runner.read(request['accepted_decode_receipt']['path']), catalog),
        'Cache normalized decode evidence differs from frozen input')
    runner.require(inventory and 0 < sum(item['bytes'] for item in inventory) <= asset['max_bytes'],
                   'READY cache exceeds frozen hydration reserve')
    return rows, inventory


def _remote_result(asset, declaration, rows, inventory):
    result = {'qa': len(rows), 'frames': declaration['contract']['request']['frames'],
        'acceptance': asset['cache'].get('complete', asset['cache']['receipt']),
        'cache_binding_sha256': runner.digest(asset['cache']), 'frame_objects': len(inventory),
        'frame_bytes': sum(item['bytes'] for item in inventory), 'all_png_full_get_and_rgb_verified': True,
        'local_media_created': False, 'local_hydration_still_required_before_evaluation': True,
        'formal_evaluation_completed': False, 'network_or_hydration_requested': True}
    if 'legacy_reuse' in declaration['contract']['request']:
        result.update(cache_preparation='accepted_legacy_f8_reuse', original_videos_decoded=False,
                      new_pngs_uploaded=False, exact_legacy_prefix_verified=True)
    return result


def _registered(environment, declaration):
    identity = _declaration(environment, declaration)
    runner.require(environment.get('benchmark_preparations', {}).get(identity) == declaration,
                   'READY declaration is not frozen in this environment')
    return identity


def resolve_ready(environment, benchmark, frames, expected_pin=None):
    """Read the exact declared READY; missing claimed evidence is never pending."""
    identity = key(benchmark, frames)
    declaration = environment.get('benchmark_preparations', {}).get(identity)
    runner.require(declaration is not None, 'Benchmark has no frozen progressive declaration')
    _registered(environment, declaration)
    path = runner.safe(declaration['ready_path'])
    failure = path.with_name(path.stem + '.FAILED.json')
    if failure.exists():
        failed_pin = runner.artifact(failure)
        failed = runner.read(failure)
        runner.require(failed.get('status') == 'SUITE_BENCHMARK_PREPARATION_FAILED'
                       and failed.get('contract_sha256') == declaration['contract_sha256']
                       and failed.get('contract') == declaration['contract'], 'Malformed cache producer failure marker')
        for item in failed.get('evidence', []):
            runner.verify_pin(item)
        runner.verify_pin(failed_pin)
        raise ValueError('CPU cache preparation failed; preserve evidence and reconcile before recovery: ' + str(failure))
    if expected_pin is not None:
        runner.require(isinstance(expected_pin, dict) and expected_pin.get('path') == str(path), 'Claimed READY path differs')
        runner.verify_pin(expected_pin)
    try:
        path.stat()
    except FileNotFoundError:
        return None
    pin = runner.artifact(path)
    _metadata_pin(environment, pin)
    value = runner.read(path)
    fields = {'schema_version', 'status', 'generated_utc', 'contract', 'contract_sha256', 'asset',
              'cache_binding_sha256', 'inventory_sha256', 'remote_verification', 'producer_source_verification'}
    runner.require(isinstance(value, dict) and set(value) == fields and type(value['schema_version']) is int
                   and value['schema_version'] == 1 and value['status'] == STATUS
                   and value['contract'] == declaration['contract']
                   and value['contract_sha256'] == declaration['contract_sha256'], 'Published READY contract differs')
    generated = datetime.fromisoformat(value['generated_utc'].replace('Z', '+00:00'))
    runner.require(generated.tzinfo is not None, 'READY timestamp must include a timezone')
    producer = _producer(environment, declaration['contract']['producer'], live=True)
    runner.require(value['producer_source_verification'] == producer, 'READY producer source verification differs')
    rows, inventory = _asset(environment, declaration, value['asset'])
    runner.require(value['cache_binding_sha256'] == runner.digest(value['asset']['cache'])
                   and value['inventory_sha256'] == runner.digest(inventory)
                   and value['remote_verification'] == _remote_result(value['asset'], declaration, rows, inventory),
                   'READY full GET/RGB verification or inventory differs')
    runner.verify_pin(pin)
    if expected_pin is not None:
        runner.verify_pin(expected_pin)
    return {'asset': value['asset'], 'ready_pin': pin, 'contract_sha256': declaration['contract_sha256'],
            'cache_binding_sha256': value['cache_binding_sha256']}


def publish_ready(environment, declaration, asset):
    """Publish only after actual complete remote verification; reuse is read-only."""
    from .suite_preflight import _remote_benchmark_gate
    _registered(environment, declaration)
    request = declaration['contract']['request']
    existing = resolve_ready(environment, request['benchmark'], request['frames'])
    if existing is not None:
        runner.require(existing['asset'] == asset, 'An immutable READY already binds a different asset')
        return existing
    producer = _producer(environment, declaration['contract']['producer'], live=True)
    rows, inventory = _asset(environment, declaration, asset)
    checked = _remote_benchmark_gate(asset, request['benchmark'], request['frames'], environment)
    runner.require(checked == _remote_result(asset, declaration, rows, inventory), 'Remote cache acceptance is incomplete')
    final_rows, final_inventory = _asset(environment, declaration, asset)
    runner.require(rows == final_rows and inventory == final_inventory
                   and producer == _producer(environment, declaration['contract']['producer'], live=True),
                   'Cache evidence or producer changed during acceptance')
    value = {'schema_version': 1, 'status': STATUS, 'generated_utc': runner.utc(),
             'contract': deepcopy(declaration['contract']), 'contract_sha256': declaration['contract_sha256'],
             'asset': deepcopy(asset), 'cache_binding_sha256': runner.digest(asset['cache']),
             'inventory_sha256': runner.digest(inventory), 'remote_verification': checked,
             'producer_source_verification': producer}
    path = runner.safe(declaration['ready_path'])
    path.parent.mkdir(parents=True, exist_ok=True)
    runner.write_once(path, value)
    return resolve_ready(environment, request['benchmark'], request['frames'])


def task_readiness(environment, plan, task):
    """Return claim evidence, or None only when a new evaluation awaits cache.

    Accepted historical evaluation results need no new frames. Authenticate
    their full existing receipt before bypassing the cache gate; a hint or a
    malformed import never becomes permission to run or silently wait.
    """
    if task['kind'] != 'evaluate' or 'benchmark_preparations' not in environment:
        return {}
    history = environment.get('history', {})
    if task['id'] in history:
        from .suite_bindings import verify_history_binding
        proof = verify_history_binding(plan, task, history[task['id']])
        runner.require(proof.get('eligible') is True and proof.get('mode') == 'accepted',
                       'Only verified accepted evaluation history bypasses the cache gate')
        return {}
    ready = resolve_ready(environment, task['benchmark'], task['frames'])
    return None if ready is None else {name: value for name, value in ready.items() if name != 'asset'}
