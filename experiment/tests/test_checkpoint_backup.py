"""Complete final backups are distinct from result publication and use byte readback."""
from pathlib import Path
from types import SimpleNamespace
import hashlib

import pytest

from deepprobe_vlm import checkpoint_backup as backup
from deepprobe_vlm import checkpoint_retention as retention
import test_checkpoint_retention as retention_fixture


class FakeHub:
    def __init__(self):
        self.objects, self.uploads, self.downloads = {}, [], []
        self.legacy = self

    def list_repo_files(self, repo_id, repo_type, **kwargs):
        return [SimpleNamespace(path=name, size=len(value), type='blob',
                                blob_id=hashlib.sha256(value).hexdigest()) for name, value in self.objects.items()]

    def upload_file(self, repo_id, repo_type, path, path_in_repo, **kwargs):
        assert isinstance(path, Path), 'SDK upload must read the original path, not buffer an entire model'
        assert path_in_repo not in self.objects
        self.objects[path_in_repo] = path.read_bytes()
        self.uploads.append(path_in_repo)

    def download_stream(self, repo_id, repo_type, key, revision):
        self.downloads.append(key)
        value = self.objects[key]
        class Response:
            def iter_content(self, chunk_size):
                assert chunk_size == 8 * 1024 * 1024
                yield value[:2]
                yield value[2:]
            def close(self):
                pass
        return Response()


@pytest.fixture
def case():
    value = retention_fixture.RetentionTest()
    value.setUp()
    yield value
    value.doCleanups()


def test_complete_final_backup_streams_all_files_and_resumes_identical_prefix(case):
    api = FakeHub()
    store = backup.ModelScopeCheckpointStore(api, run_id=case.owner['run_id'],
        inventory_sha256=case.manifest['checkpoints']['training/final'])
    directory = case.root / '.retention' / 'modelscope'
    result = backup.backup_final(case.manifest, store, state_dir=directory, guard=case.guard)
    assert result['full_final_recoverable'] and not result['second_local_copy_created']
    assert len(api.objects) == len(result['files']) + 1
    assert all('checkpoint-3' not in name for name in api.objects)
    assert any(name.endswith('config.json') for name in api.objects)
    assert set(api.downloads) == set(api.objects)
    count = len(api.uploads)
    backup.backup_final(case.manifest, store, state_dir=directory, guard=case.guard)
    assert len(api.uploads) == count
    final_plan = retention.plan_cleanup(case.manifest, case.receipts, checkpoints=['training/final'])
    verified = retention.verify_external_weight_backup(final_plan,
        {'prefix': store.prefix, 'files': final_plan['targets']}, store)
    assert verified['status'] == 'EXTERNAL_WEIGHT_BACKUP_FULL_GET_SHA256_VERIFIED'


def test_corrupt_existing_model_backup_never_overwrites_or_deletes_local(case):
    api = FakeHub()
    store = backup.ModelScopeCheckpointStore(api, run_id=case.owner['run_id'],
        inventory_sha256=case.manifest['checkpoints']['training/final'])
    api.objects[store.path_prefix + 'projector.safetensors'] = b'bad'
    with pytest.raises(ValueError, match='readback differs'):
        backup.backup_final(case.manifest, store, state_dir=case.root / '.retention' / 'ms', guard=case.guard)
    assert api.objects[store.path_prefix + 'projector.safetensors'] == b'bad'
    case.assert_weights_present()


def test_curated_result_or_other_inventory_cannot_be_checkpoint_backup(case):
    api = FakeHub()
    store = backup.ModelScopeCheckpointStore(api, run_id=case.owner['run_id'], inventory_sha256='f' * 64)
    with pytest.raises(ValueError, match='complete final checkpoint inventory'):
        backup.backup_final(case.manifest, store, state_dir=case.root / '.retention' / 'ms', guard=case.guard)
    assert not api.uploads
