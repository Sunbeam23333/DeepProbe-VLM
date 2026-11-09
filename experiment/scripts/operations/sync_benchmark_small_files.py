#!/usr/bin/env python3
"""Sync two exact benchmark plans to OSS; stdin carries local metadata in RAM.

Default mode validates plans, local payload hashes, and remote file sizes only.
--execute reads OSS credentials in memory and verifies full source/OSS SHA256.
This partial cache batch never opens the benchmark evaluation gate.
"""
from __future__ import annotations
import argparse
import base64
import configparser
import contextlib
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

BUCKET = 'ss-devbox-stream'
ENDPOINT = 'https://oss-cn-wulanchabu.aliyuncs.com'
BASE = 'jinhaopeng/Paper/DeepProbe-VLM/benchmark-cache/20261013-v1/'
META_PREFIX = BASE + 'metadata/'
REVISION = 'ead1408f75b618502df9a1d8e0950166bf0a2a0b'
PNG_PREFIX = BASE + 'frame-cache/videomme/' + REVISION + '/uniform8/'
META_SHA = '88c5d6483701bc2b24cc1dfc19b5ae36c1e1ee612f93990c2883a1ad6574eddd'
PNG_SHA = 'fdeec64adbc6f2d2a19d3db16482df2906f8ad1895e1f779123efda2c89f603e'
MEDIA = Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/benchmark-media-20261012/videomme')
CONTROL = Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/benchmark-small-file-sync-20261013')
OSS_CONFIG = Path('/home/jinhaopeng/.ossutilconfig')
MAX_REQUEST = 64 * 1024 * 1024
MAX_LOCAL_BYTES = 20 * 1024 * 1024
WORKERS = 4
CHUNK = 1024 * 1024


class SafeError(Exception):
    pass


def require(value, code):
    if not value:
        raise SafeError(code)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def safe_path(path):
    path = Path(path)
    require(path.is_absolute() and all(not p.is_symlink() for p in [path, *path.parents]),
            'unsafe_source_path')
    return path


def parse_plans(metadata_raw, png_raw):
    require(sha(metadata_raw) == META_SHA and sha(png_raw) == PNG_SHA, 'plan_sha_changed')
    metadata, png = json.loads(metadata_raw), json.loads(png_raw)
    for plan, prefix, count, total in [(metadata, META_PREFIX, 634, 29571214),
                                        (png, PNG_PREFIX, 4376, 1967610160)]:
        require(plan.get('bucket') == BUCKET and plan.get('endpoint') == ENDPOINT and
                plan.get('prefix') == prefix, 'unapproved_destination')
        require(plan.get('entry_count') == count and len(plan['entries']) == count and
                plan.get('total_bytes') == total == sum(x['size_bytes'] for x in plan['entries']),
                'plan_totals_changed')
        for item in plan['entries']:
            relative = PurePosixPath(item['relative_key'])
            require(not relative.is_absolute() and '..' not in relative.parts and
                    str(relative) == item['relative_key'], 'unsafe_destination_key')
            require(item['bucket_key'] == prefix + str(relative), 'destination_key_changed')
            require(type(item['size_bytes']) is int and 0 < item['size_bytes'] <= 20 * 1024 * 1024,
                    'invalid_source_size')
            require(bool(re.fullmatch('[0-9a-f]{64}', item['sha256'])), 'invalid_source_sha')
            require(item['source_host'] in {'local', 'haopengjin2ali'}, 'unapproved_source_host')
    require(png.get('metadata_plan_sha256') == META_SHA and png.get('source_count') == 547,
            'frame_plan_binding_changed')
    receipts = {e['bucket_key']: e for e in metadata['entries'] if e['category'] == 'frame_source_receipt'}
    require(len(receipts) == 547, 'source_receipt_set_changed')
    for item in png['entries']:
        require(item['source_host'] == 'haopengjin2ali' and
                bool(re.fullmatch('[A-Za-z0-9_-]{11}/0[0-7]\\.png', item['relative_key'])),
                'frame_identity_changed')
        require(Path(item['source_path']) == MEDIA / 'frames' / item['relative_key'],
                'frame_source_path_changed')
        receipt = receipts.get(item['source_receipt_bucket_key'])
        require(receipt and receipt['sha256'] == item['source_receipt_sha256'], 'frame_receipt_binding_changed')
    entries = metadata['entries'] + png['entries']
    require(len({e['bucket_key'] for e in entries}) == len(entries), 'duplicate_destination_key')
    require(sum(e['source_host'] == 'local' for e in entries) == 7, 'local_payload_set_changed')
    return entries


def decode64(value, maximum):
    require(isinstance(value, str) and len(value) <= 4 * ((maximum + 2) // 3), 'encoded_payload_oversized')
    try:
        data = base64.b64decode(value, validate=True)
    except Exception:
        raise SafeError('invalid_base64') from None
    require(len(data) <= maximum, 'decoded_payload_oversized')
    return data


def local_payloads(encoded, entries):
    expected = {e['bucket_key']: e for e in entries if e['source_host'] == 'local'}
    require(isinstance(encoded, dict) and set(encoded) == set(expected), 'local_payload_keys_changed')
    require(sum(e['size_bytes'] for e in expected.values()) <= MAX_LOCAL_BYTES, 'local_payload_budget_exceeded')
    result = {}
    for key, item in expected.items():
        data = decode64(encoded[key], item['size_bytes'])
        require(len(data) == item['size_bytes'] and sha(data) == item['sha256'], 'local_payload_sha_changed')
        result[key] = data
    return result


def read_request(stream):
    raw = stream.read(MAX_REQUEST + 1)
    require(len(raw) <= MAX_REQUEST, 'request_oversized')
    request = json.loads(raw)
    require(set(request) == {'metadata_plan_b64', 'png_plan_b64', 'local_payloads'}, 'request_fields_changed')
    entries = parse_plans(decode64(request['metadata_plan_b64'], 1024 * 1024),
                          decode64(request['png_plan_b64'], 8 * 1024 * 1024))
    return entries, local_payloads(request['local_payloads'], entries)


def fingerprint(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


@contextlib.contextmanager
def open_source(item, payloads):
    if item['source_host'] == 'local':
        with io.BytesIO(payloads[item['bucket_key']]) as stream:
            yield stream
        return
    path = safe_path(item['source_path'])
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size == item['size_bytes'], 'source_size_changed')
        yield stream
        after = os.fstat(stream.fileno())
        require(fingerprint(before) == fingerprint(after) == fingerprint(path.stat()), 'source_changed_during_sync')


def hash_stream(stream, expected_size, size_code):
    digest, count = hashlib.sha256(), 0
    while True:
        data = stream.read(min(CHUNK, expected_size - count + 1))
        if not data:
            break
        count += len(data)
        require(count <= expected_size, size_code)
        digest.update(data)
    require(count == expected_size, size_code)
    return digest.hexdigest()


def inspect_source_sizes(entries):
    for item in entries:
        if item['source_host'] == 'haopengjin2ali':
            st = safe_path(item['source_path']).stat()
            require(stat.S_ISREG(st.st_mode) and st.st_size == item['size_bytes'], 'source_size_changed')


class OSSStore:
    def __init__(self):
        import oss2
        require(oss2.__version__ == '2.19.1', 'unreviewed_oss_sdk')
        config = configparser.ConfigParser()
        config.read(safe_path(OSS_CONFIG))
        require(config.has_section('default') and config['default'].get('region') == 'cn-wulanchabu',
                'oss_region_changed')
        auth = oss2.Auth(config['default']['accesskeyid'], config['default']['accesskeysecret'])
        self.bucket = oss2.Bucket(auth, ENDPOINT, BUCKET, connect_timeout=60, enable_crc=True)
        self.oss2 = oss2

    def exists(self, key):
        try:
            self.bucket.head_object(key)
            return True
        except self.oss2.exceptions.NoSuchKey:
            return False

    def open(self, key):
        return self.bucket.get_object(key)

    def put_new(self, key, stream, item):
        return self.bucket.put_object(key, stream, headers={
            'x-oss-forbid-overwrite': 'true', 'x-oss-meta-sha256': item['sha256']})

    def keys(self, prefix):
        marker = ''
        while True:
            page = self.bucket.list_objects(prefix=prefix, marker=marker, max_keys=1000)
            yield from (item.key for item in page.object_list)
            if not page.is_truncated:
                return
            require(page.next_marker and page.next_marker != marker, 'oss_pagination_stalled')
            marker = page.next_marker


def verify_remote(store, item):
    stream = store.open(item['bucket_key'])
    try:
        observed = hash_stream(stream, item['size_bytes'], 'oss_size_conflict')
    finally:
        stream.close()
    require(observed == item['sha256'], 'oss_sha_conflict')


def sync_one(store, item, payloads):
    with open_source(item, payloads) as stream:
        require(hash_stream(stream, item['size_bytes'], 'source_size_changed') == item['sha256'],
                'source_sha_changed')
        stream.seek(0)
        publication = 'already_present'
        if not store.exists(item['bucket_key']):
            try:
                store.put_new(item['bucket_key'], stream, item)
                publication = 'uploaded'
            except Exception:
                # Resolve a race/lost response by exact readback, never overwrite.
                require(store.exists(item['bucket_key']), 'write_outcome_unconfirmed')
                publication = 'reconciled'
        verify_remote(store, item)
    return {'bucket_key': item['bucket_key'], 'size_bytes': item['size_bytes'],
            'sha256': item['sha256'], 'publication': publication}


def synchronize(store, entries, payloads, emit, workers=WORKERS):
    expected = {item['bucket_key'] for item in entries}
    for prefix in (META_PREFIX, PNG_PREFIX):
        require(set(store.keys(prefix)) <= expected, 'unexpected_existing_prefix_object')
    iterator, verified, total_bytes = iter(entries), [], 0
    publications = {'uploaded': 0, 'already_present': 0, 'reconciled': 0}
    pool = ThreadPoolExecutor(max_workers=workers)
    pending = set()
    try:
        for _ in range(workers):
            item = next(iterator, None)
            if item is not None:
                pending.add(pool.submit(sync_one, store, item, payloads))
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            batch = [future.result() for future in done]
            for result in batch:
                publications[result.pop('publication')] += 1
                total_bytes += result['size_bytes']
                verified.append(result)
            if len(verified) % 25 < len(batch) or not pending:
                emit({'status': 'SYNC_PROGRESS', 'verified_objects': len(verified),
                      'total_objects': len(entries), 'verified_bytes': total_bytes})
            for _ in batch:
                item = next(iterator, None)
                if item is not None:
                    pending.add(pool.submit(sync_one, store, item, payloads))
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
    require(len(verified) == len(entries), 'sync_incomplete')
    return {'status': 'EXISTING_PARTIAL_BATCH_VERIFIED', 'verified_objects': len(verified),
            'verified_bytes': total_bytes, 'uploaded_objects': publications['uploaded'],
            'already_present_objects': publications['already_present'],
            'reconciled_objects': publications['reconciled'],
            'verification_manifest_sha256': sha(canonical(sorted(verified, key=lambda e: e['bucket_key']))),
            'metadata_plan_sha256': META_SHA, 'png_plan_sha256': PNG_SHA,
            'full_source_sha_checked': True, 'full_oss_sha_checked': True,
            'local_sources_preserved': True, 'formal_evaluation_ready': False,
            'all_benchmark_assets_ready': False, 'frame_sources_in_this_batch': 547,
            'required_full_frame_sources': 899}


@contextlib.contextmanager
def sync_lock():
    safe_path(CONTROL).mkdir(parents=True, exist_ok=True)
    fd = os.open(CONTROL / '.sync.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SafeError('sync_already_active') from None
        yield
    finally:
        os.close(fd)


def emit(value):
    print(json.dumps(value, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    entries, payloads = read_request(sys.stdin.buffer)
    inspect_source_sizes(entries)
    if not args.execute:
        emit({'status': 'PREPARED_NO_MUTATIONS', 'total_objects': len(entries),
              'total_bytes': sum(x['size_bytes'] for x in entries)})
        return
    with sync_lock():
        result = synchronize(OSSStore(), entries, payloads, emit)
    emit(result)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        emit({'status': 'SYNC_FAILED', 'error_code': str(error) if isinstance(error, SafeError) else 'operation_failed'})
        sys.exit(2)
