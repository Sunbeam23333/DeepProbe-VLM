"""Private worker compatibility preserves the original F8 cache contract."""
import ast
from copy import deepcopy
from pathlib import Path
import tarfile

import pytest

from deepprobe_vlm import formal_integrity
from deepprobe_vlm import suite_legacy_cache as legacy
from deepprobe_vlm import suite_progressive as progressive, suite_runner as runner
from deepprobe_vlm.suite_scratch import DEFAULT_SCRATCH
from test_suite_legacy_cache import old_cache, _freeze_publisher_implementation, _new_runtime, _pin_runtime
from test_suite_progressive import declared


def _private_profile(env):
    env['execution_profile'] = {**deepcopy(env['execution_profile']), 'schema_version': 3,
        'scratch': {**DEFAULT_SCRATCH, 'mount_path': str(Path(env['storage']['roots'][0]).with_name('private-scratch'))}}


def test_schema3_declares_all_six_shapes_without_constructing_media(declared):
    env, plan, _ = declared
    before = runner.canonical(env['benchmark_preparations'])
    _private_profile(env)
    assert len(progressive.validate_declarations(env, plan)) == 6
    for benchmark in progressive.BENCHMARKS:
        for frames in progressive.FRAMES:
            assert progressive.resolve_ready(env, benchmark, frames) is None
    assert runner.canonical(env['benchmark_preparations']) == before
    assert not Path(env['execution_profile']['scratch']['mount_path']).exists()


@pytest.mark.parametrize('damage', ['priority', 'memory_disk', 'gpu_shape', 'version'])
def test_schema3_does_not_bypass_execution_profile_validation(declared, damage):
    env, plan, _ = declared
    _private_profile(env)
    profile = env['execution_profile']
    if damage == 'priority': profile['priority'] = 'high'
    elif damage == 'memory_disk': profile['scratch']['medium'] = 'Memory'
    elif damage == 'gpu_shape': profile['gpu_product'] = 'unexpected-model'
    else: profile['schema_version'] = 4
    with pytest.raises(ValueError):
        progressive.validate_declarations(env, plan)


def test_schema3_reuses_ready_without_republishing_or_fetching(old_cache):
    env, plan, declaration, store = old_cache
    _freeze_publisher_implementation(env)
    request = declaration['contract']['request']
    asset = legacy.prepare_asset(env, request, store=store)
    accepted = progressive.publish_ready(env, declaration, asset)
    before = (runner.canonical(declaration), Path(declaration['ready_path']).read_bytes(), list(store.gets))
    _new_runtime(env)
    _private_profile(env)
    assert len(progressive.validate_declarations(env, plan)) == 6
    assert progressive.resolve_ready(env, request['benchmark'], 8) == accepted
    assert progressive.publish_ready(env, declaration, asset) == accepted
    assert (runner.canonical(declaration), Path(declaration['ready_path']).read_bytes(), store.gets) == before
    assert not store.puts


@pytest.mark.parametrize(('module', 'function'), [
    ('suite_preflight', '_cached_media'),
    ('suite_preflight', '_remote_benchmark_gate'),
    ('suite_preflight', '_history_gate'),
    ('suite_preflight', '_source_gate'),
    ('suite_preflight', '_initial_storage_gate'),
    ('suite_preflight', 'inspect'),
    ('suite_workflow', 'source_catalog'),
    ('suite_workflow', 'bind_legacy_archive_identities'),
    ('suite_workflow', 'prepare_benchmark'),
    ('suite_workflow', 'prepare_workflow'),
    ('suite_runner', 'verify_pin'),
    ('suite_runner', 'safe'),
    ('suite_runner', 'completion_path'),
    ('suite_runner', 'completion_for'),
    ('suite_runner', 'run_fleet_worker'),
])
def test_repinned_function_changes_remain_in_the_compatibility_comparison(old_cache, module, function):
    env, _, declaration, store = old_cache
    _freeze_publisher_implementation(env)
    current = _new_runtime(env)
    path = current / 'src/deepprobe_vlm' / (module + '.py')
    tree = ast.parse(path.read_text())
    target = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function)
    target.body.insert(0, ast.Return(value=ast.Constant(value=True)))
    path.write_text(ast.unparse(ast.fix_missing_locations(tree)) + '\n')
    _pin_runtime(env)
    _private_profile(env)
    with pytest.raises(ValueError, match='validation implementation differs'):
        progressive.resolve_ready(env, declaration['contract']['request']['benchmark'], 8)
    assert not store.gets and not store.puts


@pytest.mark.parametrize(('module', 'original', 'changed'), [
    ('suite_preflight', "requirements['shared_home_additional_bytes']", "requirements['different_bytes']"),
    ('suite_preflight', "verify_private_resume(task, h, environment)", "verify_private_resume(task, h, {})"),
    ('suite_preflight', 'from .suite_run_root import inspect_run_root', 'from .suite_run_root import inspect_run_root as altered'),
    ('suite_workflow', 'validate_run_root_location(env)', 'validate_run_root_location({})'),
    ('suite_runner', "token['scratch']", "token['different_scratch']"),
    ('suite_runner', "'TMPDIR': 'tmp'", "'TMPDIR': 'frames'"),
    ('suite_runner', '"TORCHINDUCTOR_CACHE_DIR", "TMPDIR"', '"LD_PRELOAD", "TMPDIR"'),
    ('suite_runner', 'return get_task_import(state_root, task_id)', 'return {}'),
    ('suite_runner', 'artifact(completion_path(state_root, item))', 'artifact(completion_path(state_root, task_id))'),
])
def test_private_exceptions_accept_only_the_reviewed_statements(old_cache, module, original, changed):
    env, _, declaration, store = old_cache
    _freeze_publisher_implementation(env)
    current = _new_runtime(env)
    path = current / 'src/deepprobe_vlm' / (module + '.py')
    source = path.read_text()
    assert original in source
    path.write_text(source.replace(original, changed, 1))
    _pin_runtime(env)
    with pytest.raises(ValueError, match='validation implementation differs'):
        progressive.resolve_ready(env, declaration['contract']['request']['benchmark'], 8)
    assert not store.gets and not store.puts


@pytest.mark.private_evidence
def test_actual_frozen_v5_source_reuses_ready_in_current_private_runtime(old_cache):
    """Use real pinned source bytes with synthetic F8 PNGs and an in-memory OSS."""
    env, plan, declaration, store = old_cache
    repository = Path(__file__).resolve().parents[1]
    evidence = repository / 'docs/operations/evidence/suite-progressive-fleet-20261121-v3'
    manifest_path = evidence / 'P1_SOURCE_MANIFEST.json'
    assert runner.sha(manifest_path) == '205ec15613ee156606f9e3d33bd4e0b893cd3f6723083be0025d64edaa3776c0'
    manifest = runner.read(manifest_path)
    root = Path(env['storage']['roots'][0])
    old_root = root / 'actual-frozen-v5'; old_root.mkdir()
    with tarfile.open(evidence / 'source-and-control.tar.gz', 'r:gz') as bundle:
        members = bundle.getmembers()
        for name, digest in manifest['files'].items():
            matches = [member for member in members if member.isfile() and member.name.endswith('/experiment/' + name)]
            assert len(matches) == 1
            target = old_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.extractfile(matches[0]).read())
            assert runner.sha(target) == digest
    retained_manifest = root / 'ACTUAL_V5_SOURCE.json'
    retained_manifest.write_bytes(manifest_path.read_bytes())
    assert formal_integrity.source_inventory(old_root) == manifest['files']
    env['experiment_root'] = str(old_root)
    for item in env['benchmark_preparations'].values():
        item['contract']['producer'] = {'source_root': str(old_root), 'source_manifest': runner.artifact(retained_manifest)}
        item['contract_sha256'] = runner.digest(item['contract'])
    request = declaration['contract']['request']
    asset = legacy.prepare_asset(env, request, store=store)
    accepted = progressive.publish_ready(env, declaration, asset)
    before = (runner.canonical(declaration), Path(declaration['ready_path']).read_bytes(), list(store.gets))
    current = root / 'actual-current-source'; current.mkdir()
    for name in formal_integrity.source_inventory(repository):
        target = current / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repository / name).read_bytes())
    env['experiment_root'] = str(current)
    _pin_runtime(env)
    _private_profile(env)
    assert len(progressive.validate_declarations(env, plan)) == 6
    assert progressive.resolve_ready(env, request['benchmark'], 8) == accepted
    assert progressive.publish_ready(env, declaration, asset) == accepted
    assert (runner.canonical(declaration), Path(declaration['ready_path']).read_bytes(), store.gets) == before
    assert not store.puts
