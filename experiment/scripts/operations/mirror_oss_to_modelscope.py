#!/usr/bin/env python3
"""Mirror a pinned OSS transfer plan as bounded, verified tar shards.

stdin: JSON with plan and token. Credentials never appear in arguments or logs.
Each source range is ETag-bound; each published shard gets a full SHA256 readback.
Existing shards must match before they are reused. No remote deletions, source
mutations, training submissions, or repository visibility changes are performed.
"""
import concurrent.futures
import configparser
import contextlib
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from datetime import datetime, timezone


def utc():
    return datetime.now(timezone.utc).isoformat()


def emit(event, **values):
    print(json.dumps({'utc': utc(), 'event': event, **values}), flush=True)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    import oss2
    from modelscope_hub import HubApi
    from modelscope_hub.config import HubConfig
    request = json.load(sys.stdin)
    plan, token = request['plan'], request.pop('token')
    assert plan['repo_id'] == 'JasonJin/DeepProbe-VLM'
    assert plan['repo_type'] == 'model' and plan['bucket'] == 'ss-devbox-stream'
    assert os.getuid() == 83980
    cfg = configparser.ConfigParser()
    cfg.read('/home/jinhaopeng/.ossutilconfig')
    credentials = cfg['default']
    bucket = oss2.Bucket(oss2.Auth(credentials['accesskeyid'], credentials['accesskeysecret']),
                         'https://oss-cn-wulanchabu-internal.aliyuncs.com', plan['bucket'], connect_timeout=60, enable_crc=True)
    logging.disable(logging.CRITICAL)
    with tempfile.TemporaryDirectory(prefix='deepprobe-migration-', dir='/tmp') as td:
        td = Path(td)
        api = HubApi(config=HubConfig(endpoint='https://modelscope.ai', token=token, config_dir=td/'credentials', cache_dir=td/'cache'))
        del token, request
        # The repo was explicitly selected by the user. It must already exist.
        api.get_repo(plan['repo_id'], 'model', revision='master')
        existing = {x.path: x for x in api.list_repo_files(plan['repo_id'], 'model', revision='master') if x.type != 'tree'}
        receipts = []
        emit('STARTED', pid=os.getpid(), plan_sha256=digest(json.dumps(plan, sort_keys=True).encode()), shards=len(plan['shards']), source_bytes=plan['source_bytes'])
        for shard in plan['shards']:
            target = plan['prefix'] + '/shards/' + shard['name']
            emit('BUILDING', name=shard['name'], category=shard['category'], source_bytes=sum(r['length'] for r in shard['pieces']))
            def read(row):
                assert row['key'].startswith('jinhaopeng/Paper/DeepProbe-VLM/')
                headers = {'If-Match': row['etag']}
                start, end = row['offset'], row['offset'] + row['length'] - 1
                if row['length']:
                    stream = bucket.get_object(row['key'], byte_range=(start, end), headers=headers)
                    try:
                        if row['length'] != row['object_bytes']:
                            assert stream.status == 206
                            assert stream.headers.get('Content-Range', stream.headers.get('content-range')) == f"bytes {start}-{end}/{row['object_bytes']}"
                        raw = stream.read()
                    finally:
                        stream.close()
                else:
                    raw = b''
                assert len(raw) == row['length']
                if row.get('sha256') and row['offset'] == 0 and row['length'] == row['object_bytes']:
                    assert digest(raw) == row['sha256'], 'Pinned source SHA256 differs'
                return raw
            stream = io.BytesIO()
            entries = []
            with tarfile.open(fileobj=stream, mode='w') as tar, concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                for row, raw in zip(shard['pieces'], pool.map(read, shard['pieces'])):
                    info = tarfile.TarInfo(row['member'])
                    info.size, info.mode, info.mtime = len(raw), 0o644, 0
                    tar.addfile(info, io.BytesIO(raw))
                    entries.append({**row, 'piece_sha256': digest(raw)})
                manifest = json.dumps({'pieces': entries}, sort_keys=True).encode()
                info = tarfile.TarInfo('SHARD_MANIFEST.json')
                info.size, info.mode = len(manifest), 0o644
                tar.addfile(info, io.BytesIO(manifest))
            blob = stream.getvalue()
            pin = {'path': target, 'bytes': len(blob), 'sha256': digest(blob)}
            emit('UPLOAD_OR_RECONCILE', **pin)
            if target in existing:
                assert existing[target].size == len(blob), 'Existing shard has a different size'
            else:
                with contextlib.redirect_stdout(sys.stderr):
                    api.upload_file(plan['repo_id'], 'model', blob, target, revision='master', disable_tqdm=True,
                                    commit_message='Preserve pinned DeepProbe-VLM migration data')
            # SDK full-file readback with expected SHA; at most one shard on disk.
            with contextlib.redirect_stdout(sys.stderr):
                downloaded = api.download_file(plan['repo_id'], 'model', target, revision='master', local_dir=td/'readback', force=True,
                                               expected_sha256=pin['sha256'], file_size=pin['bytes'])
            h = hashlib.sha256()
            with Path(downloaded).open('rb') as f:
                for block in iter(lambda: f.read(8*1024**2), b''):
                    h.update(block)
            assert h.hexdigest() == pin['sha256']
            Path(downloaded).unlink()
            receipts.append(pin)
            emit('SHARD_VERIFIED', **pin, completed=len(receipts), total=len(plan['shards']))
            del blob, stream
        complete = {'status': 'ALL_PLANNED_SHARDS_FULL_READBACK_VERIFIED', 'utc': utc(), 'shards': receipts,
                    'source_bytes': plan['source_bytes'], 'source_objects': plan['source_objects'], 'scientific_suite_complete': False}
        raw = json.dumps(complete, indent=2).encode()
        with contextlib.redirect_stdout(sys.stderr):
            api.upload_file(plan['repo_id'], 'model', raw, plan['prefix']+'/TRANSFER_COMPLETE.json', disable_tqdm=True,
                            commit_message='Accept verified migration data transfer')
        emit('COMPLETE', **complete)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # SDK exceptions may contain signed URLs; do not dump their text.
        import traceback
        emit('FAILED', error_type=type(exc).__name__, frames=traceback.format_tb(exc.__traceback__))
        raise SystemExit(1)
