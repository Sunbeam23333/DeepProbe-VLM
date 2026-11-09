#!/usr/bin/env python3
"""Read-only final audit of the frozen 31-archive cache, without ZIP-body GETs.

The parent must first establish that the coordinator is terminal. Remote mode
also refuses an active worker lock, an incomplete exact object set, or remaining
multipart uploads. It never invokes the worker and has no mutation API.

Remote stdout contains the small-metadata snapshot and its audit. It can later be
checked offline with --snapshot, without OSS credentials or network access.
"""
from __future__ import annotations

import argparse
import configparser
from contextlib import contextmanager
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys

BUCKET = 'ss-devbox-stream'
PREFIX = 'jinhaopeng/Paper/DeepProbe-VLM/benchmark-cache/20261013-v1/hf-archives/'
ENDPOINT = 'https://oss-cn-wulanchabu.aliyuncs.com'
PLAN_SHA256 = '89cf4eddb4ffc62a89b46a4b7966a68c3fce3af46dcb8c6fb32fa35ea96b1512'
TOTAL_BYTES = 118_322_902_217
LOCK_PATH = Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/benchmark-stream-cache-20261013/control/.cache.lock')
OSS_CONFIG = Path('/home/jinhaopeng/.ossutilconfig')
SMALL_LIMIT = 2_000_000
PLAN_KEY = PREFIX + 'CACHE_PLAN.json'
CATALOG_KEY = PREFIX + 'ARCHIVE_CATALOG_COMPLETE.json'
MV_NAMES = {'clevrer', 'data0613', 'FunQA_test', 'Moments_in_Time_Raw',
            'perception', 'scene_qa', 'ssv2_video', 'sta', 'star', 'tvqa', 'vlnqa'}
REVISIONS = {'OpenGVLab/MVBench': '230a2d4fac8900333c61754641c7a13e069ac9c6',
             'lmms-lab/Video-MME': 'ead1408f75b618502df9a1d8e0950166bf0a2a0b'}


class AuditError(Exception):
    pass


def require(value, code):
    if not value:
        raise AuditError(code)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def decode_json(raw):
    def unique_pairs(pairs):
        obj = {}
        for key, value in pairs:
            require(key not in obj, 'duplicate_json_key')
            obj[key] = value
        return obj
    try:
        return json.loads(raw, object_pairs_hook=unique_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(AuditError('nonfinite_json')))
    except (ValueError, UnicodeError):
        raise AuditError('invalid_json') from None


def load_plan(raw):
    require(digest(raw) == PLAN_SHA256, 'frozen_plan_sha_changed')
    plan = decode_json(raw)
    require(plan.get('bucket') == BUCKET and plan.get('prefix') == PREFIX and
            plan.get('endpoint') == ENDPOINT and plan.get('region') == 'cn-wulanchabu',
            'unapproved_destination')
    assets = plan.get('assets', [])
    expected = {('OpenGVLab/MVBench', 'video/' + name + '.zip') for name in MV_NAMES}
    expected |= {('lmms-lab/Video-MME', f'videos_chunked_{n:02d}.zip') for n in range(1, 21)}
    require(len(assets) == 31 and {(a['repo_id'], a['path']) for a in assets} == expected,
            'archive_set_changed')
    require(len({a['asset_id'] for a in assets}) == 31, 'duplicate_asset_id')
    for asset in assets:
        require(asset['revision'] == REVISIONS[asset['repo_id']], 'revision_changed')
        require(type(asset['size_bytes']) is int and asset['size_bytes'] > 0, 'invalid_asset_size')
        require(re.fullmatch('[0-9a-f]{64}', asset['sha256']), 'invalid_asset_sha')
    require(plan.get('archive_count') == 31 and plan.get('total_bytes') == TOTAL_BYTES and
            sum(a['size_bytes'] for a in assets) == TOTAL_BYTES, 'plan_totals_changed')
    require(plan.get('formal_evaluation_ready') is False and plan.get('ntu_media_included') is False,
            'plan_readiness_changed')
    return plan


def payload_key(asset):
    return PREFIX + asset['repo_id'] + '/' + asset['revision'] + '/' + asset['path']


def receipt_key(asset):
    return payload_key(asset) + '.CACHE_VERIFIED.json'


def expected_binding(asset):
    return {'plan_sha256': PLAN_SHA256, 'bucket': BUCKET, 'key': payload_key(asset),
            'repo_id': asset['repo_id'], 'revision': asset['revision'], 'source_path': asset['path'],
            'size_bytes': asset['size_bytes'], 'sha256': asset['sha256']}


def expected_keys(plan):
    payloads = {payload_key(a) for a in plan['assets']}
    small = {PLAN_KEY, CATALOG_KEY} | {receipt_key(a) for a in plan['assets']}
    require(len(payloads) == 31 and len(small) == 33, 'duplicate_expected_key')
    return payloads, small


def validate_listing(rows, expected):
    require(type(rows) is list and len(rows) == 64, 'exact_64_object_count_required')
    require(all(type(row) is dict and set(row) == {'key', 'size_bytes', 'etag', 'last_modified'}
                for row in rows), 'invalid_listing_shape')
    require(len({row['key'] for row in rows}) == 64, 'duplicate_listed_key')
    require({row['key'] for row in rows} == expected, 'exact_prefix_keys_mismatch')
    for row in rows:
        require(type(row['size_bytes']) is int and row['size_bytes'] > 0, 'invalid_listed_size')
        require(type(row['last_modified']) is int and row['last_modified'] >= 0, 'invalid_listed_time')
        require(isinstance(row['etag'], str) and bool(row['etag']), 'missing_listed_etag')
    return {row['key']: row for row in rows}


def audit_snapshot(plan_raw, snapshot):
    """Validate only captured listing, HEADs and small JSON bytes; no I/O."""
    plan = load_plan(plan_raw)
    payloads, small_keys = expected_keys(plan)
    require(snapshot.get('schema_version') == 1 and snapshot.get('bucket') == BUCKET and
            snapshot.get('prefix') == PREFIX and snapshot.get('plan_sha256') == PLAN_SHA256,
            'snapshot_binding_mismatch')
    require(snapshot.get('zip_body_get_count') == 0, 'unexpected_zip_body_get')
    require(snapshot.get('worker_lock_acquired') is True, 'worker_lock_not_verified')
    require(snapshot.get('multipart_uploads') == [], 'unfinished_prefix_multipart_uploads')
    before = validate_listing(snapshot.get('objects_before'), payloads | small_keys)
    after = validate_listing(snapshot.get('objects_after'), payloads | small_keys)
    require(before == after, 'prefix_changed_during_audit')
    heads = snapshot.get('payload_heads')
    small = snapshot.get('small_objects')
    require(type(heads) is dict and set(heads) == payloads, 'payload_head_set_mismatch')
    require(type(small) is dict and set(small) == small_keys, 'small_object_set_mismatch')
    parsed = {}
    for key, record in small.items():
        require(type(record) is dict and set(record) == {'raw_utf8', 'size_bytes', 'sha256', 'etag'},
                'invalid_small_record')
        require(isinstance(record['raw_utf8'], str), 'small_object_not_text')
        raw = record['raw_utf8'].encode('utf-8')
        require(len(raw) <= SMALL_LIMIT and len(raw) == record['size_bytes'] == before[key]['size_bytes'],
                'small_object_size_mismatch')
        require(record['etag'] == before[key]['etag'], 'small_object_etag_mismatch')
        require(record['sha256'] == digest(raw), 'small_object_sha_mismatch')
        value = decode_json(raw)
        require(canonical(value) == raw, 'noncanonical_metadata_bytes')
        parsed[key] = value
    require(small[PLAN_KEY]['raw_utf8'].encode() == canonical({'plan_sha256': PLAN_SHA256, 'plan': plan}),
            'cache_plan_bytes_mismatch')
    entries, checked = [], []
    for asset in plan['assets']:
        key, rkey = payload_key(asset), receipt_key(asset)
        require(before[key]['size_bytes'] == asset['size_bytes'], 'payload_listed_size_mismatch')
        expected_head = {'size_bytes': asset['size_bytes'], 'etag': before[key]['etag'],
                         'sha256_metadata': asset['sha256'], 'source_revision_metadata': asset['revision']}
        require(heads[key] == expected_head, 'payload_head_source_binding_mismatch')
        receipt = parsed[rkey]
        require(type(receipt) is dict, 'invalid_receipt')
        mode = receipt.get('source_mode')
        allowed_modes = {'recovered_pinned_final_object'}
        if asset.get('local_source'):
            allowed_modes.add('verified_local_candidate')
        else:
            allowed_modes |= {'pinned_hf_range_stream', 'pinned_hf_bounded_range_stream'}
        require(mode in allowed_modes, 'receipt_source_mode_mismatch')
        expected_receipt = {'binding': expected_binding(asset), 'source_stream_sha256': asset['sha256'],
                            'source_bytes': asset['size_bytes'], 'source_mode': mode,
                            'oss_readback': {'bytes': asset['size_bytes'], 'sha256': asset['sha256'],
                                             'verification': 'complete_OSS_GET_SHA256_and_size'},
                            'formal_evaluation_ready': False}
        require(canonical(receipt) == canonical(expected_receipt), 'receipt_binding_or_readback_mismatch')
        entries.append({**expected_binding(asset), 'receipt_sha256': small[rkey]['sha256']})
        checked.append({'asset_id': asset['asset_id'], 'key': key, 'size_bytes': asset['size_bytes'],
                        'source_sha256': asset['sha256'], 'receipt_sha256': small[rkey]['sha256'],
                        'source_mode': mode, 'etag': before[key]['etag']})
    expected_catalog = {'plan_sha256': PLAN_SHA256, 'archive_count': 31, 'total_bytes': TOTAL_BYTES,
                        'entries': entries, 'all_pinned_zip_archives_cached': True,
                        'ntu_media_included': False, 'formal_evaluation_ready': False,
                        'all_benchmark_assets_ready': False}
    require(canonical(parsed[CATALOG_KEY]) == canonical(expected_catalog), 'catalog_binding_mismatch')
    metadata_bytes = sum(before[k]['size_bytes'] for k in small_keys)
    return {'schema_version': 1, 'status': 'ARCHIVE_METADATA_INDEPENDENT_AUDIT_PASSED',
            'plan_sha256': PLAN_SHA256, 'bucket': BUCKET, 'prefix': PREFIX,
            'exact_object_count': 64, 'archive_count': 31, 'archive_bytes': TOTAL_BYTES,
            'small_object_count': 33, 'small_object_bytes': metadata_bytes,
            'exact_prefix_total_bytes': TOTAL_BYTES + metadata_bytes,
            'catalog_sha256': small[CATALOG_KEY]['sha256'], 'checked_archives': checked,
            'all_pinned_zip_archives_cached': True, 'all_benchmark_assets_ready': False,
            'formal_evaluation_ready': False, 'ntu_media_included': False,
            'remaining_media_gate': 'MVBench fine_grained_pose: 200 QA require authorized NTU media',
            'zip_body_get_count': 0, 'source_or_payload_bytes_rehashed_by_this_audit': False,
            'verification_scope': 'exact stable LIST + 31 HEAD source metadata + 33 complete small JSON GETs; '
                                  '31 existing worker receipts attest full source and OSS payload SHA; '
                                  'the complete catalog binds those receipts after worker finalize',
            'worker_or_coordinator_invoked': False, 'oss_objects_written_or_deleted': False}


def safe_path(path):
    path = Path(path)
    require(all(not item.is_symlink() for item in [path, *path.parents]), 'symlink_path')
    return path


@contextmanager
def inactive_worker_lock(path=LOCK_PATH):
    """Open the existing lock read-only; do not create or modify any state."""
    fd = os.open(safe_path(path), os.O_RDONLY | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise AuditError('archive_worker_still_active') from None
        yield
    finally:
        os.close(fd)


class ReadOnlyOSS:
    """Restricted adapter: LIST, HEAD and allowlisted small-object GET only."""
    def __init__(self, small_keys, payloads):
        import oss2
        require(oss2.__version__ == '2.19.1', 'unreviewed_oss_sdk')
        config = configparser.ConfigParser()
        config.read(safe_path(OSS_CONFIG))
        require(config.has_section('default') and config['default'].get('region') == 'cn-wulanchabu',
                'oss_region_changed')
        auth = oss2.Auth(config['default']['accesskeyid'], config['default']['accesskeysecret'])
        self.bucket = oss2.Bucket(auth, ENDPOINT, BUCKET, connect_timeout=60, enable_crc=True)
        self.small_keys = small_keys
        self.payloads = payloads

    def objects(self):
        marker, rows = '', []
        while True:
            page = self.bucket.list_objects(prefix=PREFIX, marker=marker, max_keys=1000)
            for item in page.object_list:
                rows.append({'key': item.key, 'size_bytes': item.size, 'etag': item.etag.strip('"'),
                             'last_modified': int(item.last_modified)})
            require(len(rows) <= 64, 'unexpected_extra_prefix_object')
            if not page.is_truncated:
                return sorted(rows, key=lambda row: row['key'])
            require(page.next_marker and page.next_marker > marker, 'object_pagination_stalled')
            marker = page.next_marker

    def unfinished_uploads(self):
        page = self.bucket.list_multipart_uploads(prefix=PREFIX, max_uploads=1)
        require(not page.upload_list and not page.is_truncated, 'unfinished_prefix_multipart_uploads')
        return []

    def payload_head(self, key):
        require(key in self.payloads, 'unapproved_head_key')
        result = self.bucket.head_object(key)
        headers = {str(k).lower(): str(v) for k, v in result.headers.items()}
        return {'size_bytes': result.content_length, 'etag': result.etag.strip('"'),
                'sha256_metadata': headers.get('x-oss-meta-sha256'),
                'source_revision_metadata': headers.get('x-oss-meta-source-revision')}

    def small_object(self, key):
        require(key in self.small_keys and key not in self.payloads, 'zip_body_get_forbidden')
        response = self.bucket.get_object(key)
        try:
            require(type(response.content_length) is int and 0 < response.content_length <= SMALL_LIMIT,
                    'invalid_small_content_length')
            parts, count = [], 0
            while count <= SMALL_LIMIT:
                part = response.read(min(64 * 1024, SMALL_LIMIT + 1 - count))
                if not part:
                    break
                parts.append(part)
                count += len(part)
            require(count == response.content_length and count <= SMALL_LIMIT, 'small_body_length_mismatch')
            raw = b''.join(parts)
            return {'raw_utf8': raw.decode('utf-8'), 'size_bytes': count,
                    'sha256': digest(raw), 'etag': response.etag.strip('"')}
        finally:
            response.close()


def collect_snapshot(plan_raw, store):
    """Caller must hold inactive_worker_lock throughout this read-only snapshot."""
    plan = load_plan(plan_raw)
    payloads, small_keys = expected_keys(plan)
    before = store.objects()
    validate_listing(before, payloads | small_keys)
    uploads = store.unfinished_uploads()
    require(uploads == [], 'unfinished_prefix_multipart_uploads')
    small = {key: store.small_object(key) for key in sorted(small_keys)}
    heads = {key: store.payload_head(key) for key in sorted(payloads)}
    after = store.objects()
    require(store.unfinished_uploads() == [], 'unfinished_prefix_multipart_uploads')
    snapshot = {'schema_version': 1, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'bucket': BUCKET, 'prefix': PREFIX, 'plan_sha256': PLAN_SHA256,
                'objects_before': before, 'objects_after': after, 'small_objects': small,
                'payload_heads': heads, 'multipart_uploads': uploads,
                'worker_lock_acquired': True, 'zip_body_get_count': 0}
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--remote-read-only', action='store_true')
    mode.add_argument('--snapshot', type=Path)
    args = parser.parse_args()
    plan_raw = safe_path(args.plan).read_bytes()
    plan = load_plan(plan_raw)
    if args.remote_read_only:
        payloads, small_keys = expected_keys(plan)
        with inactive_worker_lock():
            snapshot = collect_snapshot(plan_raw, ReadOnlyOSS(small_keys, payloads))
            audit = audit_snapshot(plan_raw, snapshot)
        output = {'snapshot': snapshot, 'snapshot_sha256': digest(canonical(snapshot)), 'audit': audit}
    else:
        raw = safe_path(args.snapshot).read_bytes()
        require(len(raw) <= 100_000_000, 'snapshot_file_oversized')
        capture = decode_json(raw)
        if 'snapshot' in capture:
            snapshot = capture['snapshot']
            require(capture.get('snapshot_sha256') == digest(canonical(snapshot)), 'snapshot_sha_mismatch')
        else:
            snapshot = capture
        output = audit_snapshot(plan_raw, snapshot)
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        code = str(error) if isinstance(error, AuditError) else 'audit_' + type(error).__name__
        print(json.dumps({'status': 'FAILED', 'error_code': code,
                          'all_benchmark_assets_ready': False, 'formal_evaluation_ready': False}), flush=True)
        sys.exit(2)
