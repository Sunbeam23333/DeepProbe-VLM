"""Isolated cache routing and reader lifetimes; no cloud, GPU or child process."""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from deepprobe_vlm import suite_actions as actions
from deepprobe_vlm import suite_media as media
from deepprobe_vlm import suite_progressive as progressive
from deepprobe_vlm import suite_runner as runner
from deepprobe_vlm.suite_isolated_assets import isolate_benchmark_asset
from deepprobe_vlm.suite_scratch import DEFAULT_SCRATCH
from deepprobe_vlm.suite_storage import SharedBudget


def profile():
    return {'schema_version': 2, 'fleet_id': 'isolated-fixture', 'priority': 'low',
        'gpu_product': 'NVIDIA-B300',
        'node_selector': {'sslaunch-pool': 'stream', 'aliyun.accelerator/nvidia_name': 'NVIDIA-L20D'},
        'workers': [{'worker_id': 'worker-01', 'job_name': 'jinhaopeng-dpvlm-isolated-fixture',
                     'namespace': 'default'}]}


def tree(root):
    """Capture tiny synthetic files only, including unchanged source receipts."""
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob('*') if path.is_file()}


@pytest.fixture
def routing(tmp_path):
    root = tmp_path.resolve()
    namespace = root / 'isolated'
    environment = {'suite_id': 'isolated-fixture', 'execution_profile': profile(),
        'isolated_storage': {'schema_version': 1, 'namespace_root': str(namespace)}}
    task = {'id': 'evaluation-one', 'benchmark': 'videomme899', 'frames': 8}
    data = {'cache': {'kind': 'synthetic-fixture', 'manifest': {'sha256': 'a' * 64}},
        'media_root': str(root / 'legacy' / 'media'), 'max_bytes': 1024,
        'acceptance': {'path': str(root / 'legacy' / 'READY.json'), 'sha256': 'b' * 64},
        'other_metadata': {'values': ['keep-original']}}
    return SimpleNamespace(root=root, namespace=namespace, environment=environment,
                           task=task, data=data)


def test_absent_isolation_preserves_legacy_object_without_new_validation():
    data = {'cache': None, 'media_root': 'legacy-relative-value'}
    assert isolate_benchmark_asset({'execution_profile': None}, None, data) is data


@pytest.mark.parametrize('benchmark', ['videomme899', 'tempcompass_mc'])
@pytest.mark.parametrize('frames', [4, 8, 16])
@pytest.mark.parametrize('schema_version', [2, 3])
def test_cached_asset_copies_only_media_root_without_creating_files(routing, benchmark, frames, schema_version):
    value = routing
    value.task.update(benchmark=benchmark, frames=frames)
    value.environment['execution_profile']['schema_version'] = schema_version
    if schema_version == 3:
        value.environment['execution_profile']['scratch'] = {
            **DEFAULT_SCRATCH, 'mount_path': str(value.root / 'private-scratch')}
    before_data, before_env, before_tree = deepcopy(value.data), deepcopy(value.environment), tree(value.root)
    result = isolate_benchmark_asset(value.environment, value.task, value.data)
    assert result == {**before_data, 'media_root': str(value.namespace / 'media' / benchmark / f'f{frames}')}
    assert result is not value.data
    assert result['cache'] is not value.data['cache']
    assert result['cache']['manifest'] is not value.data['cache']['manifest']
    assert result['other_metadata'] is not value.data['other_metadata']
    assert value.data == before_data
    assert value.environment == before_env
    assert tree(value.root) == before_tree
    assert not value.namespace.exists()
    assert not Path(value.data['media_root']).exists()
    result['cache']['manifest']['sha256'] = 'c' * 64
    result['other_metadata']['values'].append('isolated-copy-only')
    assert value.data == before_data


def test_existing_original_ready_bytes_and_pin_are_unchanged(routing):
    value = routing
    ready = value.root / 'legacy' / 'READY.json'
    ready.parent.mkdir()
    ready.write_bytes(runner.canonical({'asset': value.data}) + b'\n')
    pin = runner.artifact(ready)
    value.data['acceptance'] = pin
    before = ready.read_bytes()
    result = isolate_benchmark_asset(value.environment, value.task, value.data)
    assert result['acceptance'] == pin
    assert runner.verify_pin(pin) == pin
    assert ready.read_bytes() == before
    assert not value.namespace.exists()


@pytest.mark.parametrize('invalid', [None, False, [], 'isolated', 1])
def test_present_malformed_isolation_is_not_silently_disabled(routing, invalid):
    routing.environment['isolated_storage'] = invalid
    with pytest.raises(ValueError):
        isolate_benchmark_asset(routing.environment, routing.task, routing.data)
    assert not routing.namespace.exists()


@pytest.mark.parametrize('version', [None, False, True, '1', 1.0, 0, 2])
def test_isolation_requires_exact_integer_schema_one(routing, version):
    routing.environment['isolated_storage']['schema_version'] = version
    with pytest.raises(ValueError):
        isolate_benchmark_asset(routing.environment, routing.task, routing.data)


@pytest.mark.parametrize('invalid', [None, False, [], '', '/', 'relative/cache', '//double/root',
                                    '/with\\backslash', '/with\x00nul', '/with\nnewline', '/with\x7fdelete'])
def test_namespace_rejects_unsafe_path_values(routing, invalid):
    routing.environment['isolated_storage']['namespace_root'] = invalid
    with pytest.raises(ValueError):
        isolate_benchmark_asset(routing.environment, routing.task, routing.data)


@pytest.mark.parametrize('suffix', ['/', '/.', '/../elsewhere', '//child'])
def test_namespace_rejects_noncanonical_spelling(routing, suffix):
    routing.environment['isolated_storage']['namespace_root'] = str(routing.namespace) + suffix
    with pytest.raises(ValueError):
        isolate_benchmark_asset(routing.environment, routing.task, routing.data)


@pytest.mark.parametrize('location', ['namespace', 'namespace-parent', 'destination-parent', 'destination'])
@pytest.mark.parametrize('dangling', [False, True])
def test_namespace_and_generated_media_path_reject_symlinks(routing, location, dangling):
    target = routing.root / 'symlink-target'
    if not dangling:
        target.mkdir()
    if location == 'namespace':
        link = routing.namespace
    elif location == 'namespace-parent':
        link = routing.root / 'namespace-link'
        routing.environment['isolated_storage']['namespace_root'] = str(link / 'isolated')
    elif location == 'destination-parent':
        link = routing.namespace / 'media'
    else:
        link = routing.namespace / 'media' / 'videomme899' / 'f8'
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        isolate_benchmark_asset(routing.environment, routing.task, routing.data)
    assert link.is_symlink()


@pytest.mark.parametrize('location', ['namespace', 'namespace-parent', 'destination-parent', 'destination', 'source'])
def test_existing_non_directory_blocks_routing(routing, location):
    paths = {'namespace': routing.namespace, 'namespace-parent': routing.root / 'blocked-parent',
        'destination-parent': routing.namespace / 'media',
        'destination': routing.namespace / 'media' / 'videomme899' / 'f8',
        'source': Path(routing.data['media_root'])}
    path = paths[location]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'preserve this file')
    if location == 'namespace-parent':
        routing.environment['isolated_storage']['namespace_root'] = str(path / 'isolated')
    with pytest.raises(ValueError):
        isolate_benchmark_asset(routing.environment, routing.task, routing.data)
    assert path.read_bytes() == b'preserve this file'


@pytest.mark.parametrize('relationship', ['same', 'source-inside-namespace', 'namespace-inside-source'])
def test_original_media_root_must_not_overlap_namespace(routing, relationship):
    source = {'same': routing.namespace, 'source-inside-namespace': routing.namespace / 'original-cache',
              'namespace-inside-source': routing.root}[relationship]
    routing.data['media_root'] = str(source)
    with pytest.raises(ValueError, match='overlap'):
        isolate_benchmark_asset(routing.environment, routing.task, routing.data)
    assert not routing.namespace.exists()


@pytest.mark.parametrize('invalid', [None, '', '/', 'relative/media', '/original/../media', '/original//media'])
def test_cached_original_media_root_must_also_be_canonical(routing, invalid):
    routing.data['media_root'] = invalid
    with pytest.raises(ValueError):
        isolate_benchmark_asset(routing.environment, routing.task, routing.data)


@pytest.mark.parametrize('dangling', [False, True])
def test_cached_original_media_root_rejects_symlinks(routing, dangling):
    target = routing.root / 'original-target'
    if not dangling:
        target.mkdir()
    link = routing.root / 'original-link'
    link.symlink_to(target, target_is_directory=True)
    routing.data['media_root'] = str(link / 'media')
    with pytest.raises(ValueError):
        isolate_benchmark_asset(routing.environment, routing.task, routing.data)
    assert link.is_symlink()


@pytest.mark.parametrize('change', [
    {'schema_version': 1}, {'priority': 'high'}, {'gpu_product': 'NVIDIA-A100'},
    {'node_selector': {'sslaunch-pool': 'embody'}}, {'workers': []}, {'fleet_id': 'different-fleet'}])
def test_complete_execution_profile_is_validated(routing, change):
    routing.environment['execution_profile'].update(change)
    with pytest.raises(ValueError):
        isolate_benchmark_asset(routing.environment, routing.task, routing.data)


def test_missing_profile_is_rejected(routing):
    del routing.environment['execution_profile']
    with pytest.raises(ValueError):
        isolate_benchmark_asset(routing.environment, routing.task, routing.data)


def test_two_valid_workers_are_not_allowed_to_share_one_isolated_namespace(routing):
    routing.environment['execution_profile']['workers'].append({
        'worker_id': 'worker-02', 'job_name': 'jinhaopeng-dpvlm-isolated-fixture-two', 'namespace': 'default'})
    with pytest.raises(ValueError, match='one worker'):
        isolate_benchmark_asset(routing.environment, routing.task, routing.data)


@pytest.mark.parametrize('task', [None, {}, {'benchmark': 'mvbench', 'frames': 8},
    {'benchmark': '../videomme899', 'frames': 8}, {'benchmark': None, 'frames': 8},
    *({'benchmark': 'videomme899', 'frames': frames} for frames in [None, True, False, '8', 8.0, 0, 1, 32])])
def test_only_declared_benchmarks_and_exact_integer_frame_counts_are_accepted(routing, task):
    with pytest.raises(ValueError):
        isolate_benchmark_asset(routing.environment, task, routing.data)


def test_direct_media_is_read_only_and_does_not_require_a_distinct_cache_root(routing):
    data = {'media_root': str(routing.namespace), 'annotation': {'sha256': 'a' * 64}}
    before = deepcopy(data)
    assert isolate_benchmark_asset(routing.environment, routing.task, data) is data
    assert data == before
    assert not routing.namespace.exists()


@pytest.mark.parametrize('invalid_part', ['isolation', 'namespace', 'profile', 'task'])
def test_direct_media_does_not_bypass_isolation_admission(routing, invalid_part):
    if invalid_part == 'isolation':
        routing.environment['isolated_storage']['schema_version'] = True
    elif invalid_part == 'namespace':
        routing.environment['isolated_storage']['namespace_root'] = '/'
    elif invalid_part == 'profile':
        routing.environment['execution_profile']['priority'] = 'high'
    else:
        routing.task['frames'] = True
    with pytest.raises(ValueError):
        isolate_benchmark_asset(routing.environment, routing.task, {'media_root': 'unused-direct-root'})


@pytest.fixture
def lifecycle(routing, monkeypatch):
    value = routing
    monkeypatch.delenv('DP_SUITE_SCRATCH_TOKEN', raising=False)
    old = Path(value.data['media_root'])
    old_frame = old / 'frames' / '0.png'
    old_frame.parent.mkdir(parents=True)
    old_frame.write_bytes(b'original cache must remain untouched')
    old_metadata = old.parent / (old.name + '.acceptance')
    old_metadata.mkdir()
    (old_metadata / 'ACTIVE_READERS.json').write_bytes(b'{"task_id":"original-live-reader"}\n')
    (old_metadata / 'READY.json').write_bytes(b'{"synthetic":"original-ready"}\n')
    value.data['acceptance'] = runner.artifact(old_metadata / 'READY.json')
    original_tree = tree(value.root / 'legacy')
    configuration = {'schema_version': 2, 'roots': [str(value.root)],
        'ledger_root': str(value.namespace / 'ledger'), 'home_limit_bytes': 150_000_000_000,
        'oss_limit_bytes': None, 'home_safety_bytes': 0, 'oss_safety_bytes': 0,
        'free_guard_bytes': 0, 'oss_prefixes': ['oss://ss-devbox-stream/synthetic/']}
    budget = SharedBudget(configuration, Mock(side_effect=AssertionError('Unexpected remote inventory')))
    worker = object.__new__(actions.Actions)
    worker.private_scratch = False
    worker.root = value.root
    worker.state = value.namespace / 'state'
    worker.tasks = {value.task['id']: value.task}
    worker.env = {**value.environment, 'storage': configuration, 'runtime_environment': {},
        'tools': {'oss_config': '/synthetic/config'},
        'assets': {'benchmarks': {'videomme899': {'8': value.data}}}}
    worker._shared_budget = budget
    worker.budget = Mock(return_value={'synthetic_reservation': True})
    lock = Mock(wraps=budget.asset_lock)
    monkeypatch.setattr(budget, 'asset_lock', lock)
    frame = b'isolated synthetic frame'
    items = [{'path': 'frames/0.png', 'bytes': len(frame), 'sha256': hashlib.sha256(frame).hexdigest()}]
    monkeypatch.setattr(media, '_bound_cache_inventory', lambda cache:
                        (None, None, deepcopy(items), None, None, None))
    store = object()
    monkeypatch.setattr(media, 'OSSStore', Mock(return_value=store))

    def hydrate(cache, *, media_root, max_bytes, store, resume):
        destination = Path(media_root)
        destination.mkdir(parents=True, exist_ok=True)
        media._write_once(destination / '.suite-media-owner.json', media.encode({
            'schema_version': 1, 'kind': 'suite_benchmark_media',
            'cache_binding_sha256': runner.digest(cache), 'media_root': str(destination),
            'max_bytes': max_bytes, 'png_bytes': len(frame)}))
        local_frame = destination / items[0]['path']
        local_frame.parent.mkdir(exist_ok=True)
        media._write_once(local_frame, frame)
        metadata = destination.parent / (destination.name + '.acceptance')
        metadata.mkdir(exist_ok=True)
        media._write_once(metadata / 'READY.json', b'{"synthetic":"isolated-ready"}\n')
        return {'media_root': str(destination), 'acceptance': runner.artifact(metadata / 'READY.json')}

    hydrate_mock = Mock(side_effect=hydrate)
    monkeypatch.setattr(media, 'hydrate_frame_cache', hydrate_mock)
    destination = value.namespace / 'media' / 'videomme899' / 'f8'
    return SimpleNamespace(value=value, worker=worker, destination=destination,
        active=destination.parent / 'f8.acceptance' / 'ACTIVE_READERS.json', old=old,
        original_tree=original_tree, items=items, hydrate=hydrate_mock, lock=lock, store=store)


@pytest.mark.parametrize('use_progressive_ready', [False, True])
def test_actions_hydrate_lease_and_cleanup_use_only_isolated_media(lifecycle, monkeypatch, use_progressive_ready):
    value, worker = lifecycle.value, lifecycle.worker
    ready = {'ready_pin': value.data['acceptance'], 'contract_sha256': 'd' * 64,
             'cache_binding_sha256': runner.digest(value.data['cache']), 'asset': value.data}
    original_ready = deepcopy(ready)
    resolve = Mock(return_value=ready)
    if use_progressive_ready:
        worker.env['benchmark_preparations'] = {'synthetic': True}
        # No static fallback is available in this branch.
        worker.env.pop('assets')
        attempt = worker.state / 'tasks' / value.task['id'] / 'attempts' / 'attempt-0001'
        proof = {key: item for key, item in ready.items() if key != 'asset'}
        claim = {'task_id': value.task['id'], 'task_sha256': runner.digest(value.task),
                 'attempt': str(attempt), 'asset_readiness': proof}
        actions.save(worker.state / 'tasks' / value.task['id'] / 'CLAIM.json', claim)
        actions.save(attempt / 'TASK_INTENT.json', {'task_id': value.task['id'], 'claim': claim})
        monkeypatch.setattr(progressive, 'resolve_ready', resolve)
    original_environment = deepcopy(worker.env)
    with worker.benchmark_asset(value.task) as asset:
        assert asset['data']['media_root'] == str(lifecycle.destination)
        assert (lifecycle.destination / lifecycle.items[0]['path']).is_file()
        assert runner.read(lifecycle.active)['task_id'] == value.task['id']
        assert tree(value.root / 'legacy') == lifecycle.original_tree
        assert ready == original_ready
    lifecycle.hydrate.assert_called_once_with(value.data['cache'], media_root=str(lifecycle.destination),
        max_bytes=value.data['max_bytes'], store=lifecycle.store, resume=True)
    lifecycle.lock.assert_called_once_with({'media_root': str(lifecycle.destination)})
    assert not (lifecycle.destination / lifecycle.items[0]['path']).exists()
    assert not lifecycle.active.exists()
    assert (lifecycle.destination / '.suite-media-owner.json').is_file()
    release = runner.read(asset['release_receipt']['path'])
    assert release['media_root'] == str(lifecycle.destination)
    assert release['status'] == 'OWNED_LOCAL_BENCHMARK_PNGS_RELEASED'
    assert release['released_home_bytes'] == lifecycle.items[0]['bytes']
    assert release['reader_processes_quiescent'] is True
    assert release['oss_deletion_performed'] is False
    assert tree(value.root / 'legacy') == lifecycle.original_tree
    assert worker.env == original_environment
    assert ready == original_ready
    runner.verify_pin(value.data['acceptance'])
    if use_progressive_ready:
        resolve.assert_called_once_with(worker.env, 'videomme899', 8, expected_pin=value.data['acceptance'])


def test_unreconciled_new_reader_blocks_instead_of_falling_back_to_original_cache(lifecycle):
    lifecycle.active.parent.mkdir(parents=True)
    lifecycle.active.write_bytes(b'{"task_id":"unreconciled-isolated-reader"}\n')
    before = lifecycle.active.read_bytes()
    frame = lifecycle.destination / lifecycle.items[0]['path']
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b'existing isolated reader still uses this PNG')
    with pytest.raises(ValueError, match='Unreconciled benchmark reader lease'):
        with lifecycle.worker.benchmark_asset(lifecycle.value.task):
            pytest.fail('An existing isolated reader must prevent evaluation')
    lifecycle.hydrate.assert_not_called()
    assert lifecycle.active.read_bytes() == before
    assert frame.read_bytes() == b'existing isolated reader still uses this PNG'
    assert tree(lifecycle.value.root / 'legacy') == lifecycle.original_tree


def test_unknown_descendants_preserve_only_new_cache_and_new_reader_lease(lifecycle):
    with pytest.raises(ValueError, match='cache and reader lease preserved'):
        with lifecycle.worker.benchmark_asset(lifecycle.value.task):
            lifecycle.worker._benchmark_lease['readers_quiet'] = False
    assert (lifecycle.destination / lifecycle.items[0]['path']).is_file()
    assert runner.read(lifecycle.active)['task_id'] == lifecycle.value.task['id']
    assert tree(lifecycle.value.root / 'legacy') == lifecycle.original_tree
    assert not hasattr(lifecycle.worker, '_benchmark_lease')


def test_invalid_isolation_is_rejected_before_cache_lock_or_hydration(lifecycle):
    lifecycle.worker.env['isolated_storage']['namespace_root'] = str(lifecycle.old)
    with pytest.raises(ValueError, match='overlap'):
        with lifecycle.worker.benchmark_asset(lifecycle.value.task):
            pytest.fail('An overlapping original cache must not be used')
    lifecycle.lock.assert_not_called()
    lifecycle.hydrate.assert_not_called()
    assert tree(lifecycle.value.root / 'legacy') == lifecycle.original_tree


def test_actions_validate_isolation_before_direct_media_early_return(lifecycle):
    lifecycle.value.data.pop('cache')
    lifecycle.worker.env['isolated_storage']['schema_version'] = True
    with pytest.raises(ValueError):
        with lifecycle.worker.benchmark_asset(lifecycle.value.task):
            pytest.fail('Direct-media evaluation must still validate isolation')
    lifecycle.lock.assert_not_called()
    lifecycle.hydrate.assert_not_called()
    assert tree(lifecycle.value.root / 'legacy') == lifecycle.original_tree
