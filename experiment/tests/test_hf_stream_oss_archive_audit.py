"""Offline contracts for the final 31-archive metadata audit. No network or ZIPs."""
import pytest
import copy
import fcntl
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('hf_archive_audit',
        ROOT / 'experiment/scripts/operations/audit_hf_stream_oss_archives.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
PLAN_RAW = None
PLAN = None


def load_pinned_plan():
    global PLAN_RAW, PLAN
    if PLAN_RAW is None:
        PLAN_RAW = (ROOT / 'experiment/docs/operations/evidence/benchmark-bucket-cache-20261013/'
                    'HF_OSS_ARCHIVE_CACHE_PLAN.json').read_bytes()
        PLAN = AUDIT.load_plan(PLAN_RAW)


def fixture():
    load_pinned_plan()
    rows, heads, small, entries = [], {}, {}, []

    def add_small(key, value):
        raw = AUDIT.canonical(value)
        etag = hashlib.md5(raw).hexdigest().upper()  # Object identity only, not content SHA proof.
        rows.append({'key': key, 'size_bytes': len(raw), 'etag': etag, 'last_modified': 1791858000})
        small[key] = {'raw_utf8': raw.decode(), 'size_bytes': len(raw),
                      'sha256': AUDIT.digest(raw), 'etag': etag}

    add_small(AUDIT.PLAN_KEY, {'plan_sha256': AUDIT.PLAN_SHA256, 'plan': PLAN})
    modes = ['pinned_hf_range_stream', 'pinned_hf_bounded_range_stream', 'recovered_pinned_final_object']
    for index, asset in enumerate(PLAN['assets']):
        key = AUDIT.payload_key(asset)
        etag = f'{index + 1:032x}'.upper()
        rows.append({'key': key, 'size_bytes': asset['size_bytes'], 'etag': etag,
                     'last_modified': 1791858000})
        heads[key] = {'size_bytes': asset['size_bytes'], 'etag': etag,
                      'sha256_metadata': asset['sha256'], 'source_revision_metadata': asset['revision']}
        mode = 'verified_local_candidate' if asset.get('local_source') else modes[index % 3]
        receipt = {'binding': AUDIT.expected_binding(asset), 'source_stream_sha256': asset['sha256'],
                   'source_bytes': asset['size_bytes'], 'source_mode': mode,
                   'oss_readback': {'bytes': asset['size_bytes'], 'sha256': asset['sha256'],
                                    'verification': 'complete_OSS_GET_SHA256_and_size'},
                   'formal_evaluation_ready': False}
        rkey = AUDIT.receipt_key(asset)
        add_small(rkey, receipt)
        entries.append({**AUDIT.expected_binding(asset), 'receipt_sha256': small[rkey]['sha256']})
    add_small(AUDIT.CATALOG_KEY, {'plan_sha256': AUDIT.PLAN_SHA256, 'archive_count': 31,
              'total_bytes': AUDIT.TOTAL_BYTES, 'entries': entries,
              'all_pinned_zip_archives_cached': True, 'ntu_media_included': False,
              'formal_evaluation_ready': False, 'all_benchmark_assets_ready': False})
    rows.sort(key=lambda row: row['key'])
    return {'schema_version': 1, 'bucket': AUDIT.BUCKET, 'prefix': AUDIT.PREFIX,
            'plan_sha256': AUDIT.PLAN_SHA256, 'objects_before': rows,
            'objects_after': copy.deepcopy(rows), 'payload_heads': heads,
            'small_objects': small, 'multipart_uploads': [],
            'worker_lock_acquired': True, 'zip_body_get_count': 0}


def edit_small(snapshot, key, fn):
    record = snapshot['small_objects'][key]
    value = json.loads(record['raw_utf8'])
    fn(value)
    raw = AUDIT.canonical(value)
    record.update(raw_utf8=raw.decode(), size_bytes=len(raw), sha256=AUDIT.digest(raw))
    for rows in [snapshot['objects_before'], snapshot['objects_after']]:
        next(row for row in rows if row['key'] == key)['size_bytes'] = len(raw)


class FakeStore:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.small_gets = []
        self.heads = []
        self.lists = 0

    def objects(self):
        self.lists += 1
        return copy.deepcopy(self.snapshot['objects_before' if self.lists == 1 else 'objects_after'])

    def unfinished_uploads(self):
        return self.snapshot['multipart_uploads']

    def small_object(self, key):
        if key.endswith('.zip'):
            raise AssertionError('ZIP GET forbidden in independent audit')
        self.small_gets.append(key)
        return copy.deepcopy(self.snapshot['small_objects'][key])

    def payload_head(self, key):
        self.heads.append(key)
        return copy.deepcopy(self.snapshot['payload_heads'][key])


@pytest.mark.private_evidence
class MetadataAuditTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = fixture()
        self.asset = PLAN['assets'][0]
        self.key = AUDIT.payload_key(self.asset)
        self.rkey = AUDIT.receipt_key(self.asset)

    def reject(self, code=None):
        with self.assertRaisesRegex(AUDIT.AuditError, code or '.'):
            AUDIT.audit_snapshot(PLAN_RAW, self.snapshot)

    def test_complete_exact_set_accepts_and_keeps_all_readiness_false(self):
        result = AUDIT.audit_snapshot(PLAN_RAW, self.snapshot)
        self.assertEqual((result['exact_object_count'], result['archive_count']), (64, 31))
        self.assertEqual(result['archive_bytes'], 118322902217)
        self.assertEqual(len(result['checked_archives']), 31)
        for key in ['all_benchmark_assets_ready', 'formal_evaluation_ready', 'ntu_media_included',
                    'source_or_payload_bytes_rehashed_by_this_audit']:
            self.assertIs(result[key], False)
        self.assertEqual(result['zip_body_get_count'], 0)

    def test_collection_gets_only_33_small_objects_and_heads_31_zip_keys(self):
        store = FakeStore(self.snapshot)
        collected = AUDIT.collect_snapshot(PLAN_RAW, store)
        AUDIT.audit_snapshot(PLAN_RAW, collected)
        self.assertEqual(len(store.small_gets), 33)
        self.assertEqual(len(store.heads), 31)
        self.assertEqual(store.lists, 2)
        self.assertTrue(all(not key.endswith('.zip') for key in store.small_gets))

    def test_raw_plan_change_is_rejected(self):
        with self.assertRaisesRegex(AUDIT.AuditError, 'frozen_plan_sha_changed'):
            AUDIT.audit_snapshot(PLAN_RAW + b' ', self.snapshot)

    def test_missing_object(self):
        self.snapshot['objects_before'].pop()
        self.reject('exact_64')

    def test_extra_object(self):
        self.snapshot['objects_before'].append(copy.deepcopy(self.snapshot['objects_before'][0]))
        self.reject('exact_64')

    def test_foreign_key_replaces_expected_key(self):
        self.snapshot['objects_before'][0]['key'] = AUDIT.PREFIX + 'unexpected.txt'
        self.reject('exact_prefix_keys')

    def test_duplicate_key_with_64_rows(self):
        self.snapshot['objects_before'][0] = copy.deepcopy(self.snapshot['objects_before'][1])
        self.reject('duplicate_listed_key')

    def test_stable_prefix_required(self):
        self.snapshot['objects_after'][0]['etag'] = 'CHANGED'
        self.reject('prefix_changed')

    def test_unfinished_upload(self):
        self.snapshot['multipart_uploads'] = [{'key': self.key, 'upload_id': 'unfinished'}]
        self.reject('unfinished_prefix')
        store = FakeStore(self.snapshot)
        with self.assertRaisesRegex(AUDIT.AuditError, 'unfinished_prefix'):
            AUDIT.collect_snapshot(PLAN_RAW, store)
        self.assertEqual(store.small_gets, [])

    def test_missing_lock_evidence(self):
        self.snapshot['worker_lock_acquired'] = False
        self.reject('worker_lock_not_verified')

    def test_wrong_payload_size(self):
        for rows in [self.snapshot['objects_before'], self.snapshot['objects_after']]:
            next(row for row in rows if row['key'] == self.key)['size_bytes'] += 1
        self.reject('payload_listed_size')

    def test_head_sha_revision_size_or_etag_conflict(self):
        for field in ['sha256_metadata', 'source_revision_metadata', 'size_bytes', 'etag']:
            with self.subTest(field=field):
                self.snapshot = fixture()
                self.snapshot['payload_heads'][self.key][field] = 'wrong'
                self.reject('payload_head_source_binding')

    def test_missing_head(self):
        self.snapshot['payload_heads'].pop(self.key)
        self.reject('payload_head_set')

    def test_small_sha_or_size_or_etag_conflict(self):
        for field, replacement in [('sha256', '0' * 64), ('size_bytes', 1), ('etag', 'bad')]:
            with self.subTest(field=field):
                self.snapshot = fixture()
                self.snapshot['small_objects'][self.rkey][field] = replacement
                self.reject('small_object_')

    def test_receipt_binding_every_field(self):
        for field in AUDIT.expected_binding(self.asset):
            with self.subTest(field=field):
                self.snapshot = fixture()
                edit_small(self.snapshot, self.rkey, lambda r: r['binding'].update({field: 'wrong'}))
                self.reject('receipt_binding_or_readback')

    def test_receipt_source_sha_or_source_bytes_conflict(self):
        for field in ['source_stream_sha256', 'source_bytes']:
            self.snapshot = fixture()
            edit_small(self.snapshot, self.rkey, lambda r: r.update({field: 'wrong'}))
            self.reject('receipt_binding_or_readback')

    def test_receipt_full_oss_readback_fields(self):
        for field in ['bytes', 'sha256', 'verification']:
            self.snapshot = fixture()
            edit_small(self.snapshot, self.rkey, lambda r: r['oss_readback'].update({field: 'wrong'}))
            self.reject('receipt_binding_or_readback')

    def test_receipt_readiness_bool_strict(self):
        for value in [True, 0, 'false', None]:
            self.snapshot = fixture()
            edit_small(self.snapshot, self.rkey, lambda r: r.update(formal_evaluation_ready=value))
            self.reject('receipt_binding_or_readback')

    def test_catalog_all_flags_strict(self):
        for field, bad in [('all_pinned_zip_archives_cached', 1), ('all_benchmark_assets_ready', 0),
                           ('formal_evaluation_ready', True), ('ntu_media_included', True)]:
            self.snapshot = fixture()
            edit_small(self.snapshot, AUDIT.CATALOG_KEY, lambda c: c.update({field: bad}))
            self.reject('catalog_binding')

    def test_catalog_missing_duplicate_or_bad_receipt_reference(self):
        mutators = [lambda c: c['entries'].pop(),
                    lambda c: c['entries'].__setitem__(1, c['entries'][0]),
                    lambda c: c['entries'][0].update(receipt_sha256='0' * 64)]
        for mutate in mutators:
            self.snapshot = fixture()
            edit_small(self.snapshot, AUDIT.CATALOG_KEY, mutate)
            self.reject('catalog_binding')

    def test_unknown_or_wrong_local_source_mode(self):
        for value in ['unknown', 'verified_local_candidate']:
            self.snapshot = fixture()
            edit_small(self.snapshot, self.rkey, lambda r: r.update(source_mode=value))
            self.reject('receipt_source_mode')

    def test_changed_cache_plan(self):
        edit_small(self.snapshot, AUDIT.PLAN_KEY, lambda r: r.update(plan_sha256='0' * 64))
        self.reject('cache_plan_bytes')


    def test_missing_catalog_blocks_collection_before_get(self):
        self.snapshot['objects_before'] = [r for r in self.snapshot['objects_before']
                                           if r['key'] != AUDIT.CATALOG_KEY]
        store = FakeStore(self.snapshot)
        with self.assertRaisesRegex(AUDIT.AuditError, 'exact_64'):
            AUDIT.collect_snapshot(PLAN_RAW, store)
        self.assertEqual(store.small_gets, [])


class IndependentMetadataTests(unittest.TestCase):
    def setUp(self):
        self.key = AUDIT.PREFIX + 'fixture.zip'
        self.rkey = AUDIT.PREFIX + 'fixture.receipt.json'

    def test_duplicate_json_member_is_rejected(self):
        with self.assertRaisesRegex(AUDIT.AuditError, 'duplicate_json_key'):
            AUDIT.decode_json(b'{"a":1,"a":2}')


    def test_reader_refuses_zip_get_before_bucket_access(self):
        store = AUDIT.ReadOnlyOSS.__new__(AUDIT.ReadOnlyOSS)
        store.payloads, store.small_keys = {self.key}, {self.rkey}
        with self.assertRaisesRegex(AUDIT.AuditError, 'zip_body_get_forbidden'):
            store.small_object(self.key)


    def test_small_reader_accumulates_short_reads_and_closes(self):
        class ShortReader(io.BytesIO):
            content_length, etag = 10, 'E'
            def read(self, n=-1):
                return super().read(min(n, 3))
        stream = ShortReader(b'0123456789')
        store = AUDIT.ReadOnlyOSS.__new__(AUDIT.ReadOnlyOSS)
        store.payloads, store.small_keys = {self.key}, {self.rkey}
        store.bucket = SimpleNamespace(get_object=lambda _: stream)
        record = store.small_object(self.rkey)
        self.assertEqual(record['raw_utf8'], '0123456789')
        self.assertTrue(stream.closed)


    def test_small_reader_rejects_truncation_and_oversized_header(self):
        for length, body in [(10, b'short'), (AUDIT.SMALL_LIMIT + 1, b'')]:
            stream = io.BytesIO(body)
            stream.content_length, stream.etag = length, 'E'
            store = AUDIT.ReadOnlyOSS.__new__(AUDIT.ReadOnlyOSS)
            store.payloads, store.small_keys = {self.key}, {self.rkey}
            store.bucket = SimpleNamespace(get_object=lambda _: stream)
            with self.assertRaises(AUDIT.AuditError):
                store.small_object(self.rkey)
            self.assertTrue(stream.closed)


    def test_list_pagination_stall_rejected(self):
        page = SimpleNamespace(object_list=[], is_truncated=True, next_marker='')
        store = AUDIT.ReadOnlyOSS.__new__(AUDIT.ReadOnlyOSS)
        store.bucket = SimpleNamespace(list_objects=lambda **kwargs: page)
        with self.assertRaisesRegex(AUDIT.AuditError, 'pagination_stalled'):
            store.objects()


    def test_existing_worker_lock_blocks_without_altering_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / '.cache.lock'
            path.write_bytes(b'unchanged')
            with path.open('rb') as writer_lock:
                fcntl.flock(writer_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(AUDIT.AuditError, 'archive_worker_still_active'):
                    with AUDIT.inactive_worker_lock(path):
                        self.fail('active worker accepted')
            with AUDIT.inactive_worker_lock(path):
                self.assertEqual(path.read_bytes(), b'unchanged')


if __name__ == '__main__':
    unittest.main(verbosity=2)
