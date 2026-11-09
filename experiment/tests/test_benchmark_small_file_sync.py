"""Offline tests for exact, non-destructive partial benchmark synchronization."""
import pytest
import base64
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts/operations' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SYNC = script('sync_benchmark_small_files')
COORD = script('coordinate_benchmark_small_file_sync')


class MemoryStore:
    def __init__(self, objects=None, mode=None):
        self.objects = dict(objects or {})
        self.mode, self.puts, self.lock = mode, 0, threading.Lock()

    def keys(self, prefix):
        return [key for key in self.objects if key.startswith(prefix)]

    def exists(self, key):
        return key in self.objects

    def open(self, key):
        return io.BytesIO(self.objects[key])

    def put_new(self, key, stream, item):
        assert hasattr(stream, 'read')  # Remote payload is a stream, not a copied ZIP/file.
        with self.lock:
            self.puts += 1
            if self.mode == 'fail':
                raise RuntimeError('https://credential.invalid/?secret=DO_NOT_LOG')
            if key in self.objects:
                raise RuntimeError('forbid overwrite')
            self.objects[key] = b'bad' if self.mode == 'corrupt' else stream.read()
            if self.mode == 'lost_response':
                raise RuntimeError('https://credential.invalid/?secret=DO_NOT_LOG')


def item(data=b'example', key=None, path=None):
    return {'source_host': 'haopengjin2ali' if path else 'local',
            'source_path': str(path) if path else '/unused/local/source',
            'bucket_key': key or SYNC.META_PREFIX + 'synthetic.json',
            'size_bytes': len(data), 'sha256': SYNC.sha(data)}


class SmallFileSyncTests(unittest.TestCase):
    def test_new_local_payload_is_uploaded_and_read_back(self):
        entry, store = item(), MemoryStore()
        result = SYNC.sync_one(store, entry, {entry['bucket_key']: b'example'})
        self.assertEqual(result['publication'], 'uploaded')
        self.assertEqual(store.objects[entry['bucket_key']], b'example')

    def test_existing_identical_object_is_verified_without_put(self):
        entry = item()
        store = MemoryStore({entry['bucket_key']: b'example'})
        result = SYNC.sync_one(store, entry, {entry['bucket_key']: b'example'})
        self.assertEqual(result['publication'], 'already_present')
        self.assertEqual(store.puts, 0)

    def test_existing_conflicting_object_is_never_overwritten(self):
        entry = item()
        store = MemoryStore({entry['bucket_key']: b'xxxxxxx'})
        with self.assertRaisesRegex(SYNC.SafeError, 'oss_sha_conflict'):
            SYNC.sync_one(store, entry, {entry['bucket_key']: b'example'})
        self.assertEqual(store.objects[entry['bucket_key']], b'xxxxxxx')
        self.assertEqual(store.puts, 0)

    def test_oversized_or_short_oss_object_fails(self):
        entry = item()
        for content in (b'exampleX', b'exampl'):
            with self.subTest(content=content), self.assertRaisesRegex(SYNC.SafeError, 'oss_size_conflict'):
                SYNC.sync_one(MemoryStore({entry['bucket_key']: content}), entry,
                              {entry['bucket_key']: b'example'})

    def test_source_sha_change_fails_before_put(self):
        entry, store = item(), MemoryStore()
        with self.assertRaisesRegex(SYNC.SafeError, 'source_sha_changed'):
            SYNC.sync_one(store, entry, {entry['bucket_key']: b'changed'})
        self.assertEqual(store.puts, 0)

    def test_corrupt_put_is_detected_by_full_readback(self):
        entry, store = item(data=b'good'), MemoryStore(mode='corrupt')
        with self.assertRaisesRegex(SYNC.SafeError, 'oss_size_conflict'):
            SYNC.sync_one(store, entry, {entry['bucket_key']: b'good'})
        self.assertEqual(store.puts, 1)

    def test_lost_put_response_resolves_exact_key_without_retry(self):
        entry, store = item(), MemoryStore(mode='lost_response')
        result = SYNC.sync_one(store, entry, {entry['bucket_key']: b'example'})
        self.assertEqual(result['sha256'], entry['sha256'])
        self.assertEqual(result['publication'], 'reconciled')
        self.assertEqual(store.puts, 1)

    def test_reconciled_put_is_not_misreported_as_preexisting(self):
        entry, store = item(), MemoryStore(mode='lost_response')
        result = SYNC.synchronize(store, [entry], {entry['bucket_key']: b'example'}, lambda value: None)
        self.assertEqual(result['reconciled_objects'], 1)
        self.assertEqual(result['already_present_objects'], 0)
        self.assertEqual(result['uploaded_objects'], 0)

    def test_failed_put_is_not_blindly_retried_or_secret_rendered(self):
        entry, store = item(), MemoryStore(mode='fail')
        with self.assertRaisesRegex(SYNC.SafeError, '^write_outcome_unconfirmed$') as error:
            SYNC.sync_one(store, entry, {entry['bucket_key']: b'example'})
        self.assertNotIn('DO_NOT_LOG', str(error.exception))
        self.assertEqual(store.puts, 1)

    def test_remote_file_stream_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d).resolve() / 'frame.png'
            path.write_bytes(b'example')
            before = SYNC.fingerprint(path.stat())
            entry, store = item(path=path), MemoryStore()
            SYNC.sync_one(store, entry, {})
            self.assertEqual(path.read_bytes(), b'example')
            self.assertEqual(SYNC.fingerprint(path.stat()), before)

    def test_remote_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root / 'real').write_bytes(b'example')
            (root / 'link').symlink_to(root / 'real')
            with self.assertRaisesRegex(SYNC.SafeError, 'unsafe_source_path'):
                SYNC.sync_one(MemoryStore(), item(path=root / 'link'), {})

    def test_source_replacement_during_upload_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d).resolve() / 'frame.png'
            path.write_bytes(b'example')
            entry, store = item(path=path), MemoryStore()
            original_put = store.put_new
            def changed_put(*args):
                original_put(*args)
                path.write_bytes(b'changed')
            store.put_new = changed_put
            with self.assertRaisesRegex(SYNC.SafeError, 'source_changed_during_sync'):
                SYNC.sync_one(store, entry, {})

    def test_unexpected_prefix_object_blocks_all_mutation(self):
        entry = item()
        store = MemoryStore({SYNC.META_PREFIX + 'unplanned': b'old'})
        with self.assertRaisesRegex(SYNC.SafeError, 'unexpected_existing_prefix_object'):
            SYNC.synchronize(store, [entry], {entry['bucket_key']: b'example'}, lambda value: None)
        self.assertEqual(store.puts, 0)

    def test_partial_batch_receipt_never_claims_full_benchmark_ready(self):
        entries = [item(key=SYNC.META_PREFIX + 'one'), item(key=SYNC.PNG_PREFIX + 'two')]
        payloads = {entry['bucket_key']: b'example' for entry in entries}
        store = MemoryStore()
        result = SYNC.synchronize(store, entries, payloads, lambda value: None, workers=2)
        self.assertEqual(result['verified_objects'], 2)
        self.assertEqual(result['verified_bytes'], 14)
        self.assertFalse(result['formal_evaluation_ready'])
        self.assertFalse(result['all_benchmark_assets_ready'])
        self.assertTrue(result['full_source_sha_checked'] and result['full_oss_sha_checked'])
        repeat = SYNC.synchronize(store, entries, payloads, lambda value: None, workers=2)
        self.assertEqual(result['verification_manifest_sha256'], repeat['verification_manifest_sha256'])
        self.assertEqual(repeat['uploaded_objects'], 0)

    def test_no_new_work_is_scheduled_after_failure(self):
        entries = [item(key=SYNC.META_PREFIX + str(i)) for i in range(5)]
        payloads = {entry['bucket_key']: b'example' for entry in entries}
        store = MemoryStore(mode='fail')
        with self.assertRaises(SYNC.SafeError):
            SYNC.synchronize(store, entries, payloads, lambda value: None, workers=1)
        self.assertEqual(store.puts, 1)

    def test_exact_local_payload_set_and_hash_required(self):
        entry = item()
        payload = {entry['bucket_key']: base64.b64encode(b'example').decode()}
        self.assertEqual(SYNC.local_payloads(payload, [entry])[entry['bucket_key']], b'example')
        for invalid in ({}, {**payload, 'unplanned': ''}, {entry['bucket_key']: base64.b64encode(b'changed').decode()}):
            with self.assertRaises(SYNC.SafeError):
                SYNC.local_payloads(invalid, [entry])

    def test_oversized_input_and_invalid_base64_are_rejected(self):
        with patch.object(SYNC, 'MAX_REQUEST', 8), self.assertRaisesRegex(SYNC.SafeError, 'request_oversized'):
            SYNC.read_request(io.BytesIO(b'x' * 9))
        with self.assertRaisesRegex(SYNC.SafeError, 'invalid_base64'):
            SYNC.decode64('????', 10)

    def test_changed_plan_bytes_fail_before_parsing_or_source_reads(self):
        with self.assertRaisesRegex(SYNC.SafeError, 'plan_sha_changed'):
            SYNC.parse_plans(b'not json', b'not json')

    def test_duplicate_worker_lock_fails_closed(self):
        with tempfile.TemporaryDirectory() as d, patch.object(SYNC, 'CONTROL', Path(d).resolve()):
            with SYNC.sync_lock():
                with self.assertRaisesRegex(SYNC.SafeError, 'sync_already_active'):
                    with SYNC.sync_lock():
                        self.fail('second worker acquired lock')

    def test_default_worker_does_not_load_oss_or_take_write_lock(self):
        entry = item()
        output = io.StringIO()
        with patch.object(SYNC.sys, 'argv', ['worker']), patch.object(SYNC, 'read_request', return_value=([entry], {})), \
                patch.object(SYNC, 'inspect_source_sizes'), patch.object(SYNC, 'OSSStore') as store, \
                patch.object(SYNC, 'sync_lock') as lock, contextlib.redirect_stdout(output):
            SYNC.main()
        store.assert_not_called()
        lock.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())['status'], 'PREPARED_NO_MUTATIONS')

    def test_oss_adapter_forbids_overwrite(self):
        class Bucket:
            def put_object(self, key, stream, headers):
                self.headers = headers
        adapter = SYNC.OSSStore.__new__(SYNC.OSSStore)
        adapter.bucket = Bucket()
        entry = item()
        adapter.put_new(entry['bucket_key'], io.BytesIO(b'example'), entry)
        self.assertEqual(adapter.bucket.headers['x-oss-forbid-overwrite'], 'true')
        self.assertEqual(adapter.bucket.headers['x-oss-meta-sha256'], entry['sha256'])

    def test_coordinator_discards_raw_errors_and_sensitive_fields(self):
        self.assertIsNone(COORD.safe_event('Traceback https://secret.invalid/?token=secret'))
        result = COORD.safe_event(json.dumps({'status': 'SYNC_FAILED', 'error_code': 'https://secret.invalid',
                                             'token': 'DO_NOT_LOG'}))
        self.assertEqual(result, {'status': 'SYNC_FAILED'})

    @pytest.mark.private_evidence
    def test_authentic_pinned_plans_validate_when_present(self):
        root = ROOT / 'docs/operations/evidence/benchmark-bucket-cache-20261013'
        meta = root / 'BENCHMARK_METADATA_BUCKET_SYNC_PLAN.json'
        png = root / 'VIDEOMME_EXISTING547_PNG_BUCKET_SYNC_PLAN.json'
        if not meta.exists() or not png.exists():
            self.skipTest('release-specific immutable evidence is not present')
        entries = SYNC.parse_plans(meta.read_bytes(), png.read_bytes())
        self.assertEqual(len(entries), 5010)
        self.assertEqual(sum(e['size_bytes'] for e in entries), 1997181374)


if __name__ == '__main__':
    unittest.main()
