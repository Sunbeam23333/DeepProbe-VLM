"""Budget races and crash recovery use synthetic roots/metadata, never a bucket."""
from contextlib import nullcontext
import copy
import json
import multiprocessing
from pathlib import Path
import time

import pytest

from deepprobe_vlm import checkpoint_retention as retention
from deepprobe_vlm import suite_storage as storage


def empty_remote():
    return {'complete': True, 'objects': [], 'multipart': []}


def zero_local(roots):
    return {'bytes': 0, 'files': 0, 'inventory_sha256': 'a' * 64}


def config(tmp_path):
    root = tmp_path / 'owned'
    root.mkdir()
    return {'schema_version': 2, 'roots': [str(root)], 'ledger_root': str(root / 'ledger'),
            'home_limit_bytes': 100, 'oss_limit_bytes': 100,
            'oss_prefixes': ['oss://ss-devbox-stream/fixture/project/'],
            'home_safety_bytes': 0, 'oss_safety_bytes': 0, 'free_guard_bytes': 0}


def race_reserve(configuration, barrier, queue, worker):
    budget = storage.SharedBudget(configuration, empty_remote, local_scanner=zero_local)
    barrier.wait(timeout=10)
    try:
        queue.put(('accepted', budget.reserve({'worker_id': worker}, 60, 60)))
    except storage.StorageError:
        queue.put(('gated', worker))


def race_cache(configuration, barrier, counter):
    budget = storage.SharedBudget(configuration, empty_remote, local_scanner=zero_local)
    barrier.wait(timeout=10)
    with budget.asset_lock({'media_root': 'same-cache'}):
        path = Path(counter)
        value = int(path.read_text()) if path.exists() else 0
        time.sleep(0.05)
        path.write_text(str(value + 1))


def test_four_processes_cannot_overbook_either_domain(tmp_path):
    configuration = config(tmp_path)
    context = multiprocessing.get_context('spawn')
    barrier, queue = context.Barrier(4), context.Queue()
    processes = [context.Process(target=race_reserve, args=(configuration, barrier, queue, str(i))) for i in range(4)]
    for process in processes:
        process.start()
    results = [queue.get(timeout=20) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert [status for status, _ in results].count('accepted') == 1
    restarted = storage.SharedBudget(configuration, empty_remote, local_scanner=zero_local)
    assert restarted.snapshot()['reserved_bytes'] == {'home': 60, 'oss': 60}


def test_same_cache_publication_is_serialized_across_workers(tmp_path):
    configuration = config(tmp_path)
    context = multiprocessing.get_context('spawn')
    barrier, counter = context.Barrier(4), tmp_path / 'cache-publication-count'
    processes = [context.Process(target=race_cache, args=(configuration, barrier, str(counter))) for _ in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    assert counter.read_text() == '4'


def test_home_and_oss_are_separate_not_combined(tmp_path):
    budget = storage.SharedBudget(config(tmp_path), empty_remote, local_scanner=zero_local)
    token = budget.reserve({'worker': 'one'}, 60, 60)
    assert token
    assert budget.snapshot()['projected_bytes'] == {'home': 60, 'oss': 60}
    with pytest.raises(storage.StorageError, match='oss capacity'):
        budget.reserve({'worker': 'two'}, 1, 40)
    with pytest.raises(storage.StorageError, match='home capacity'):
        budget.reserve({'worker': 'three'}, 40, 1)


def test_explicit_null_oss_quota_skips_inventory_but_keeps_home_and_reservations(tmp_path):
    configuration = config(tmp_path)
    configuration.update(oss_limit_bytes=None, home_safety_bytes=10)
    def forbidden_inventory():
        raise AssertionError('Unlimited OSS must not scan the project prefix for quota')
    budget = storage.SharedBudget(configuration, forbidden_inventory, local_scanner=zero_local)
    token = budget.reserve({'worker': 'one'}, 60, 10**15)
    value = budget.snapshot()
    assert value['oss']['status'] == 'NOT_MEASURED_UNLIMITED_QUOTA'
    assert value['oss']['bytes'] is None and value['oss']['objects'] is None
    assert value['oss']['inventory_sha256'] is None
    assert value['projected_bytes'] == {'home': 70, 'oss': None}
    assert value['reserved_bytes']['oss'] == 10**15
    budget.extend(token, 75, 2 * 10**15)
    with pytest.raises(storage.StorageError, match='home capacity'):
        budget.reserve({'worker': 'two'}, 15, 1)
    budget.release(token, lambda owner: {'owner': owner, 'active_writers': [], 'active_uploads': []})
    assert budget.snapshot()['active_reservations'] == 0
    for field, value in [('home_limit_bytes', None), ('home_limit_bytes', True),
                         ('oss_limit_bytes', True), ('oss_limit_bytes', 10**15)]:
        with pytest.raises(storage.StorageError):
            storage.validate_config({**configuration, field: value})


def test_four_processes_cannot_overbook_home_with_unlimited_oss(tmp_path):
    configuration = config(tmp_path)
    configuration['oss_limit_bytes'] = None
    context = multiprocessing.get_context('spawn')
    barrier, queue = context.Barrier(4), context.Queue()
    processes = [context.Process(target=race_reserve, args=(configuration, barrier, queue, str(i))) for i in range(4)]
    for process in processes:
        process.start()
    results = [queue.get(timeout=20) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert [status for status, _ in results].count('accepted') == 1


def migration_fixture(tmp_path):
    old = config(tmp_path)
    old['home_limit_bytes'] = 10_000_000
    budget = storage.SharedBudget(old, empty_remote, local_scanner=zero_local)
    result = Path(old['roots'][0]) / 'kept-result.json'
    result.write_text('{"real_fixture":true}')
    budget.register_artifact('kept-result', kind='result', locations=[retention.file_pin(result)])
    token = budget.reserve({'worker': 'completed'}, 10, 20)
    budget.release(token, lambda owner: {'owner': owner, 'active_writers': [], 'active_uploads': []})
    proof_file = tmp_path / 'quiescence.json'
    proof_file.write_text(json.dumps({'active_project_jobs': [], 'active_writers': [], 'active_uploads': []}))
    def proof(scope):
        return {'observed_utc': storage.utc(), 'active_project_jobs': [], 'active_writers': [],
                'active_uploads': [], 'evidence': [retention.file_pin(proof_file)]}
    return budget, old, {**old, 'oss_limit_bytes': None}, proof


def test_policy_migration_preserves_exact_ledger_and_is_explicit_idempotent(tmp_path):
    budget, old, new, proof = migration_fixture(tmp_path)
    original = budget.path.read_bytes()
    before = json.loads(original)
    files_before = sorted(str(p) for p in budget.root.rglob('*'))
    proposal = storage.disable_oss_quota(old, new, quiescence=proof, authorization='User removed OSS quota')
    assert proposal['applied'] is False and budget.path.read_bytes() == original
    assert sorted(str(p) for p in budget.root.rglob('*')) == files_before
    value = storage.disable_oss_quota(old, new, quiescence=proof, authorization='User removed OSS quota', apply=True)
    assert Path(value['ledger_backup']['path']).read_bytes() == original
    current = json.loads(budget.path.read_text())
    assert current['config_sha256'] == retention.digest(new)
    assert current['artifacts'] == before['artifacts'] and current['reservations'] == before['reservations']
    assert current['events'][:-1] == before['events']
    with pytest.raises(storage.StorageError, match='same shared storage'):
        budget.snapshot()  # stale finite-policy workers cannot keep mutating
    assert storage.disable_oss_quota(old, new, quiescence=proof,
        authorization='User removed OSS quota', apply=True) == value
    updated = storage.SharedBudget(new, lambda: pytest.fail('unlimited quota scan'), local_scanner=zero_local)
    with pytest.raises(storage.StorageError, match='permanently'):
        with updated.deletion_guard(['kept-result']):
            pytest.fail('Policy migration must not lose permanent result protection')


@pytest.mark.parametrize('phase', ['before_replace', 'after_replace'])
def test_policy_migration_recovers_both_sides_of_atomic_replacement(tmp_path, monkeypatch, phase):
    budget, old, new, proof = migration_fixture(tmp_path)
    before = budget.path.read_bytes()
    original_replace = storage._replace
    def interrupted(path, value):
        if value.get('config_sha256') == retention.digest(new):
            if phase == 'after_replace':
                original_replace(path, value)
            raise RuntimeError('simulated process interruption')
        return original_replace(path, value)
    monkeypatch.setattr(storage, '_replace', interrupted)
    with pytest.raises(RuntimeError, match='interruption'):
        storage.disable_oss_quota(old, new, quiescence=proof, authorization='Explicit request', apply=True)
    backups = list(budget.root.glob('policy-migrations/*/LEDGER.before.*.json'))
    assert len(backups) == 1 and backups[0].read_bytes() == before
    assert not list(budget.root.glob('policy-migrations/*/COMPLETED.json'))
    monkeypatch.setattr(storage, '_replace', original_replace)
    result = storage.disable_oss_quota(old, new, quiescence=proof, authorization='Explicit request', apply=True)
    assert result['applied'] is True
    current = json.loads(budget.path.read_text())
    assert len([e for e in current['events'] if e['kind'] == 'OSS_QUOTA_DISABLED']) == 1
    assert current['artifacts'] == json.loads(before)['artifacts']


def test_policy_migration_rejects_active_stale_or_broader_changes(tmp_path):
    budget, old, new, proof = migration_fixture(tmp_path)
    before = budget.path.read_bytes()
    for changed in ({**new, 'home_limit_bytes': 99}, {**new, 'home_safety_bytes': 1}):
        with pytest.raises(storage.StorageError, match='Only OSS finite-to-null'):
            storage.disable_oss_quota(old, changed, quiescence=proof, authorization='Explicit request', apply=True)
    for changed_proof in (lambda scope: {**proof(scope), 'active_project_jobs': ['running']},
                          lambda scope: {**proof(scope), 'observed_utc': '2000-01-01T00:00:00Z'}):
        with pytest.raises(storage.StorageError, match='fresh|stale'):
            storage.disable_oss_quota(old, new, quiescence=changed_proof, authorization='Explicit request', apply=True)
    assert budget.path.read_bytes() == before and not (budget.root / 'policy-migrations').exists()
    budget.reserve({'worker': 'unreconciled'}, 1, 1)
    with pytest.raises(storage.StorageError, match='no active or unreconciled'):
        storage.disable_oss_quota(old, new, quiescence=proof, authorization='Explicit request', apply=True)
    assert not (budget.root / 'policy-migrations').exists()


def test_policy_migration_rejects_wrong_old_policy_and_home_backup_overflow(tmp_path):
    budget, old, new, proof = migration_fixture(tmp_path)
    with pytest.raises(storage.StorageError, match='policy SHA differs'):
        storage.disable_oss_quota({**old, 'oss_limit_bytes': 99}, new,
            quiescence=proof, authorization='Explicit request', apply=True)
    # The original file already exhausts this tiny synthetic home budget.
    # Migration cannot create a retained copy by bypassing the local cap.
    small = tmp_path / 'small'
    small.mkdir()
    policy = config(small)
    tiny = storage.SharedBudget(policy, empty_remote, local_scanner=zero_local)
    before = tiny.path.read_bytes()
    with pytest.raises(storage.StorageError, match='Home capacity cannot hold'):
        storage.disable_oss_quota(policy, {**policy, 'oss_limit_bytes': None},
            quiescence=proof, authorization='Explicit request', apply=True)
    assert tiny.path.read_bytes() == before and not (tiny.root / 'policy-migrations').exists()


def test_old_versions_multipart_and_scratch_count_without_reclaim_credit(tmp_path):
    configuration = config(tmp_path)
    remote = empty_remote()
    uri = configuration['oss_prefixes'][0] + 'model'
    remote['objects'] = [{'uri': uri, 'version_id': str(n), 'bytes': 20} for n in range(2)]
    remote['multipart'] = [{'uri': uri, 'upload_id': 'interrupted', 'part_number': 1, 'bytes': 35}]
    budget = storage.SharedBudget(configuration, lambda: remote, local_scanner=lambda roots: {'bytes': 80})
    snapshot = budget.snapshot()
    assert snapshot['oss']['bytes'] == 75
    with pytest.raises(storage.StorageError, match='oss capacity'):
        budget.reserve({'worker': 'one'}, 1, 25)
    with pytest.raises(storage.StorageError, match='home capacity'):
        budget.reserve({'worker': 'two'}, 20, 1)


def test_crashed_reservation_never_expires_and_release_requires_exact_quiescence(tmp_path):
    configuration = config(tmp_path)
    budget = storage.SharedBudget(configuration, empty_remote, local_scanner=zero_local)
    owner = {'worker_id': 'worker-1', 'job_uid': 'exact-uid', 'task_id': 'train'}
    token = budget.reserve(owner, 80, 20)
    restart = storage.SharedBudget(configuration, empty_remote, local_scanner=zero_local)
    with pytest.raises(storage.StorageError, match='already exists'):
        restart.reserve(owner, 1, 1)
    for proof in ({'owner': owner, 'active_writers': [], 'active_uploads': ['upload-1']},
                  {'owner': {'job_uid': 'other'}, 'active_writers': [], 'active_uploads': []}):
        with pytest.raises(storage.StorageError, match='quiescent'):
            restart.release(token, lambda value: proof)
    restart.release(token, lambda value: {'owner': value, 'active_writers': [], 'active_uploads': []})
    assert restart.snapshot()['active_reservations'] == 0
    assert restart.reserve(owner, 80, 20)


def test_changed_configuration_and_incomplete_inventory_fail_closed(tmp_path):
    configuration = config(tmp_path)
    budget = storage.SharedBudget(configuration, empty_remote, local_scanner=zero_local)
    altered = copy.deepcopy(configuration)
    altered['oss_limit_bytes'] = 99
    with pytest.raises(storage.StorageError, match='same shared storage'):
        storage.SharedBudget(altered, empty_remote, local_scanner=zero_local)
    budget.remote_inventory = lambda: {'complete': False, 'objects': [], 'multipart': []}
    with pytest.raises(storage.StorageError, match='Full object-version'):
        budget.reserve({'task': 'x'}, 1, 1)


def test_public_results_and_pending_recovery_references_block_gc(tmp_path):
    configuration = config(tmp_path)
    budget = storage.SharedBudget(configuration, empty_remote, local_scanner=zero_local)
    result = Path(configuration['roots'][0]) / 'RESULT.json'
    result.write_text('{"coverage": 2697}')
    pinned = retention.file_pin(result)
    budget.register_artifact('public-result', kind='result', locations=[pinned])
    with pytest.raises(storage.StorageError, match='permanently'):
        with budget.deletion_guard(['public-result']):
            pytest.fail('permanent results must not enter cleanup')
    budget.register_artifact('checkpoint', kind='checkpoint', locations=[pinned], references=['eval', 'recovery'])
    budget.complete_reference('checkpoint', 'eval', pinned)
    with pytest.raises(storage.StorageError, match='recovery'):
        with budget.deletion_guard(['checkpoint']):
            pytest.fail('pending recovery reference must retain weights')
    budget.complete_reference('checkpoint', 'recovery', pinned)
    with budget.deletion_guard(['checkpoint']):
        pass
    result.write_text('changed')
    with pytest.raises(storage.StorageError, match='evidence changed'):
        with budget.deletion_guard(['checkpoint']):
            pass


def test_local_inventory_includes_partial_upload_and_cache_files(tmp_path):
    root = tmp_path / 'root'
    (root / 'tmp').mkdir(parents=True)
    (root / 'cache').mkdir()
    (root / 'tmp' / 'part.pending').write_bytes(b'a' * 5000)
    (root / 'cache' / 'model.suite-restore').write_bytes(b'b' * 5000)
    value = storage.local_inventory([str(root)])
    assert value['files'] == 2 and value['bytes'] >= 10000


def test_local_inventory_records_atomic_rename_between_walk_and_lstat(tmp_path, monkeypatch):
    root = tmp_path / 'root'
    root.mkdir()
    pending = root / 'COMPLETED.json.pending-123'
    pending.write_bytes(b'receipt')
    final = root / 'COMPLETED.json'
    real_walk = storage.os.walk
    def publish_before_lstat(*args, **kwargs):
        for directory, dirs, files in real_walk(*args, **kwargs):
            if pending.name in files:
                pending.rename(final)
            yield directory, dirs, files
    monkeypatch.setattr(storage.os, 'walk', publish_before_lstat)
    value = storage.local_inventory([str(root)])
    assert value['vanished_during_scan'] == [str(pending)]
    assert final.read_bytes() == b'receipt'
    # The next observation sees the published name; the in-flight worker's
    # reservation covers the first observation's concurrent publication.
    assert storage.local_inventory([str(root)])['bytes'] >= len(b'receipt')


def test_local_inventory_rejects_walk_permission_errors(tmp_path, monkeypatch):
    root = tmp_path / 'root'
    root.mkdir()
    def denied_walk(*args, **kwargs):
        kwargs['onerror'](PermissionError(13, 'permission denied', str(root / 'unreadable')))
        return iter(())
    monkeypatch.setattr(storage.os, 'walk', denied_walk)
    with pytest.raises(storage.StorageError, match='traversal failed'):
        storage.local_inventory([str(root)])


def test_bound_huggingface_alias_counts_link_without_duplicating_target(tmp_path):
    root = tmp_path / 'owned'
    (root / 'blobs').mkdir(parents=True)
    weight = root / 'blobs' / 'weight'
    weight.write_bytes(b'a' * 10000)
    baseline = storage.local_inventory([str(root)])['bytes']
    (root / 'snapshot').mkdir()
    alias = root / 'snapshot' / 'model.safetensors'
    alias.symlink_to(weight)
    observed = storage.local_inventory([str(root)])
    assert observed['files'] == 2
    assert observed['bytes'] == baseline + max(alias.lstat().st_size, alias.lstat().st_blocks * 512)
    external = tmp_path / 'outside'
    external.write_bytes(b'foreign')
    (root / 'unbound').symlink_to(external)
    actual = storage.local_inventory([str(root)])
    assert actual['files'] == 3
    assert actual['external_symlinks_not_followed'][0]['target'] == str(external)


def test_oss_inventory_lists_old_versions_and_every_multipart_page(tmp_path):
    provider = storage.OssutilInventory('/bin/true', '/dev/null', ['oss://ss-devbox-stream/fixture/project/'])
    calls = []
    def fake_api(args):
        calls.append(args)
        common = {'IsTruncated': False}
        if args[0] == 'list-object-versions':
            return {**common, 'Name': 'ss-devbox-stream', 'Prefix': 'fixture/project/',
                    'Version': [{'Key': 'fixture/project/model', 'Size': '7', 'VersionId': 'old', 'ETag': 'one'},
                                {'Key': 'fixture/project/model', 'Size': '8', 'VersionId': 'new', 'ETag': 'two'}]}
        if args[0] == 'list-multipart-uploads':
            return {**common, 'Bucket': 'ss-devbox-stream', 'Prefix': 'fixture/project/',
                    'Upload': {'Key': 'fixture/project/partial', 'UploadId': 'u'}}
        return {**common, 'Bucket': 'ss-devbox-stream', 'Key': 'fixture/project/partial', 'UploadId': 'u',
                'Part': {'PartNumber': '1', 'Size': '9'}}
    provider._json = fake_api
    observed = storage.remote_usage(provider(), provider.prefixes)
    assert observed['bytes'] == 24 and observed['object_bytes'] == 15 and observed['multipart_bytes'] == 9
    assert [a[0] for a in calls] == ['list-object-versions', 'list-multipart-uploads', 'list-parts']


def test_shared_bucket_permission_fallback_reports_unknown_historic_versions():
    provider = storage.OssutilInventory('/bin/true', '/dev/null', ['oss://ss-devbox-stream/fixture/project/'])
    def fake_api(args):
        if args[0] == 'list-object-versions':
            raise storage.InventoryAccessDenied('AccessDenied 403')
        if args[0] == 'list-objects-v2':
            return {'Name': 'ss-devbox-stream', 'Prefix': 'fixture/project/', 'IsTruncated': False,
                    'Contents': {'Key': 'fixture/project/model', 'Size': '25', 'ETag': 'fresh'}}
        return {'Bucket': 'ss-devbox-stream', 'Prefix': 'fixture/project/', 'IsTruncated': False}
    provider._json = fake_api
    inventory = provider()
    usage = storage.remote_usage(inventory, provider.prefixes)
    assert usage['bytes'] == 25
    assert usage['accounting_scope'] == 'live_objects_and_uploaded_multipart'
    assert usage['legacy_version_bytes_unknown'] is True
    assert usage['version_inventory_status'] == 'AccessDenied:403'
