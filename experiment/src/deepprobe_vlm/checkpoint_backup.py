"""Immutable ModelScope final-checkpoint backup with streaming full readback.

This is separate from curated result publication. Only the complete final from
an accepted training manifest is eligible. It is uploaded directly from its
existing local files; verification streams bytes without a second disk copy.
No ModelScope remote deletion, visibility change or destructive sync is used.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import uuid

from . import checkpoint_retention as r


class ModelScopeCheckpointStore:
    def __init__(self, api, *, run_id, inventory_sha256, repo_id='JasonJin/DeepProbe-VLM'):
        r.require(repo_id == 'JasonJin/DeepProbe-VLM' and r.valid_sha(inventory_sha256)
                  and isinstance(run_id, str) and run_id and all(c.isalnum() or c in '-_' for c in run_id),
                  'Exact authorized model backup repository/run identity required')
        self.api, self.repo_id = api, repo_id
        self.path_prefix = f'artifacts/{run_id}/final/{inventory_sha256}/'
        self.prefix = 'modelscope://datasets/' + repo_id + '/' + self.path_prefix

    def _key(self, uri):
        r.require(uri.startswith(self.prefix), 'Object escapes exact ModelScope final backup')
        suffix = uri.removeprefix(self.prefix)
        r.relative(suffix)
        # Keep the original manifest's paths in receipts while storing a normal
        # loadable checkpoint tree under artifacts/run/final/inventory/.
        return self.path_prefix + suffix.removeprefix('training/final/')

    def list_prefix(self, prefix):
        r.require(prefix == self.prefix, 'Model backup listing must use its exact final prefix')
        result = []
        for item in self.api.list_repo_files(self.repo_id, 'dataset', revision='master', recursive=True):
            if item.type == 'tree' or not item.path.startswith(self.path_prefix):
                continue
            suffix = item.path.removeprefix(self.path_prefix)
            r.relative(suffix)
            relative = suffix if suffix == 'BACKUP_MANIFEST.json' else 'training/final/' + suffix
            result.append({'uri': self.prefix + relative, 'bytes': item.size,
                           'blob_id': getattr(item, 'blob_id', None)})
        r.require(len({v['uri'] for v in result}) == len(result), 'Duplicate ModelScope backup entries')
        return sorted(result, key=lambda item: item['uri'])

    def iter_object(self, uri):
        # Low-level SDK stream performs no body caching and adds no retries.
        response = self.api.legacy.download_stream(self.repo_id, 'dataset', self._key(uri), 'master')
        try:
            for block in response.iter_content(chunk_size=8 * 1024 * 1024):
                if block:
                    yield block
        finally:
            response.close()

    def put_if_absent(self, uri, source, expected_sha256):
        pinned = r.file_pin(source)
        r.require(pinned['sha256'] == expected_sha256, 'Local final changed before ModelScope upload')
        r.require(not any(row['uri'] == uri for row in self.list_prefix(self.prefix)),
                  'Immutable ModelScope final object already exists')
        self.api.upload_file(self.repo_id, 'dataset', Path(source), self._key(uri), revision='master',
                             buffer_size_mb=16, disable_tqdm=True,
                             commit_message='Archive verified DeepProbe-VLM final checkpoint')
        r.check_pin(pinned)


@contextmanager
def configured_api(state_dir):
    """Load credentials only at execution, outside serializable suite metadata."""
    r.require(importlib.metadata.version('modelscope-hub') == '0.7.1',
              'Checkpoint backup requires the reviewed modelscope-hub==0.7.1 runtime')
    token = os.environ.get('MODELSCOPE_API_TOKEN')
    credentials = os.environ.get('MODELSCOPE_CREDENTIALS_DIR')
    r.require(token or credentials, 'ModelScope token environment/private credential directory required')
    from modelscope_hub import HubApi
    from modelscope_hub.config import HubConfig
    directory = r.safe(state_dir)
    directory.mkdir(parents=True, exist_ok=True)
    credential_path = r.safe(credentials) if credentials else directory / 'private-credentials'
    credential_path.mkdir(parents=True, exist_ok=True, mode=0o700)
    r.require(credential_path.stat().st_mode & 0o077 == 0, 'ModelScope credential directory must be private')
    config = HubConfig(endpoint='https://modelscope.ai', token=token,
                       config_dir=credential_path, cache_dir=directory / 'sdk-cache')
    yield HubApi(config=config)


def backup_final(manifest, store, *, state_dir, guard):
    r.validate_manifest(manifest)
    r.verify_training_completion(manifest)
    owner = manifest['owner']
    r._quiescent(owner, guard)
    rows = [row for row in manifest['files'] if row['path'].startswith('training/final/')]
    r.require(rows and store.prefix.endswith('/' + manifest['checkpoints']['training/final'] + '/'),
              'Backup prefix must bind the complete final checkpoint inventory')
    directory = r.safe(state_dir)
    directory.mkdir(parents=True, exist_ok=True)
    document = {'schema_version': 1, 'kind': 'COMPLETE_FINAL_CHECKPOINT_BACKUP',
                'run_id': owner['run_id'], 'training_manifest_sha256': r.digest(manifest),
                'checkpoint_inventory_sha256': manifest['checkpoints']['training/final'], 'files': rows}
    document_path = r.write_once(directory / 'BACKUP_MANIFEST.json', document)
    document_pin = r.file_pin(document_path)
    expected = {store.prefix + row['path']: row for row in rows}
    expected[store.prefix + 'BACKUP_MANIFEST.json'] = {**document_pin, 'path': 'BACKUP_MANIFEST.json'}
    intent = {'backup_prefix': store.prefix, 'files': list(expected.values()), 'overwrite': False,
              'local_second_copy': False, 'readback': 'streaming_full_sha256'}
    r.write_once(directory / 'INTENT.json', intent)
    before = r._listing(store, store.prefix)
    r.require(set(before) <= set(expected), 'Unexpected file in immutable ModelScope final backup')
    for uri, row in expected.items():
        r._quiescent(owner, guard)
        source = document_path if row['path'] == 'BACKUP_MANIFEST.json' else Path(owner['run_root']) / row['path']
        if uri not in before:
            r.check_pin({**row, 'path': str(source)})
            store.put_if_absent(uri, source, row['sha256'])
        r.require(r._object_digest(store, uri) == {'bytes': row['bytes'], 'sha256': row['sha256']},
                  'ModelScope final backup full readback differs')
        if source.exists():
            r.check_pin({**row, 'path': str(source)})
        else:
            r.require(r.cleanable(row['path']) and r._recorded_deletion(Path(owner['run_root']), row, manifest),
                      'Missing local final weight has no prior verified cleanup intent')
    listed = r._listing(store, store.prefix)
    r.require(set(listed) == set(expected), 'Incomplete or extra ModelScope final backup files')
    for uri, row in expected.items():
        r.require(listed[uri]['bytes'] == row['bytes'] and r._object_digest(store, uri) == {
            'bytes': row['bytes'], 'sha256': row['sha256']}, 'ModelScope backup changed during full verification')
    r.require(listed == r._listing(store, store.prefix), 'ModelScope backup listing changed during readback')
    r._quiescent(owner, guard)
    result = {'status': 'MODELSCOPE_COMPLETE_FINAL_FULL_GET_SHA256_VERIFIED', 'utc': r.utc(),
              'backup_prefix': store.prefix, 'training_manifest_sha256': r.digest(manifest),
              'checkpoint_inventory_sha256': manifest['checkpoints']['training/final'],
              'files': rows, 'bytes': sum(row['bytes'] for row in rows),
              'full_final_recoverable': True, 'second_local_copy_created': False}
    path = directory / ('VERIFIED-' + uuid.uuid4().hex + '.json')
    r.write_once(path, result)
    return {**result, 'receipt': r.file_pin(path)}
