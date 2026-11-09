#!/usr/bin/env python3
"""Send exact local metadata in stdin RAM; remote files remain on their host."""
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

PYTHON = '/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/benchmark-oss-stream-runtime-20261013/venv/bin/python'
REMOTE_WORKER = '/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/benchmark-small-file-sync-20261013/release-v2/sync_benchmark_small_files.py'
BOOTSTRAP = ('import hashlib,runpy,sys; p=sys.argv.pop(1); expected=sys.argv.pop(1); '
             'assert hashlib.sha256(open(p,"rb").read()).hexdigest()==expected; '
             'sys.argv[0]=p; runpy.run_path(p,run_name="__main__")')
STATUS = {'PREPARED_NO_MUTATIONS', 'SYNC_PROGRESS', 'SYNC_FAILED', 'EXISTING_PARTIAL_BATCH_VERIFIED'}
COUNTS = {'verified_objects', 'total_objects', 'total_bytes', 'verified_bytes', 'uploaded_objects',
          'already_present_objects', 'reconciled_objects', 'frame_sources_in_this_batch', 'required_full_frame_sources'}
SHAS = {'verification_manifest_sha256', 'metadata_plan_sha256', 'png_plan_sha256'}
BOOLS = {'full_source_sha_checked', 'full_oss_sha_checked', 'local_sources_preserved',
         'formal_evaluation_ready', 'all_benchmark_assets_ready'}


class SafeError(Exception):
    pass


def require(value, code):
    if not value:
        raise SafeError(code)


def checked_file(path, expected=None):
    path = Path(path)
    require(path.is_file() and all(not p.is_symlink() for p in [path, *path.parents]), 'unsafe_local_file')
    data = path.read_bytes()
    if expected:
        require(hashlib.sha256(data).hexdigest() == expected, 'local_file_sha_changed')
    return data


def prepare(config):
    require(config['python'] == PYTHON and config['worker'] == REMOTE_WORKER, 'remote_execution_path_changed')
    worker = Path(__file__).with_name('sync_benchmark_small_files.py')
    checked_file(worker, config['worker_sha256'])
    spec = importlib.util.spec_from_file_location('small_sync_validator', worker)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = checked_file(config['metadata_plan'], config['metadata_plan_sha256'])
    png = checked_file(config['png_plan'], config['png_plan_sha256'])
    entries = module.parse_plans(metadata, png)
    payloads = {}
    for item in entries:
        if item['source_host'] == 'local':
            payloads[item['bucket_key']] = base64.b64encode(checked_file(item['source_path'], item['sha256'])).decode()
    module.local_payloads(payloads, entries)
    request = {'metadata_plan_b64': base64.b64encode(metadata).decode(),
               'png_plan_b64': base64.b64encode(png).decode(), 'local_payloads': payloads}
    data = json.dumps(request, separators=(',', ':')).encode()
    require(len(data) <= module.MAX_REQUEST, 'request_budget_exceeded')
    return entries, data


def safe_event(line):
    try:
        event = json.loads(line)
    except Exception:
        return None
    if not isinstance(event, dict) or event.get('status') not in STATUS:
        return None
    result = {'status': event['status']}
    for key in COUNTS:
        if type(event.get(key)) is int and 0 <= event[key] <= 10**13:
            result[key] = event[key]
    for key in SHAS:
        if isinstance(event.get(key), str) and re.fullmatch('[0-9a-f]{64}', event[key]):
            result[key] = event[key]
    for key in BOOLS:
        if type(event.get(key)) is bool:
            result[key] = event[key]
    code = event.get('error_code')
    if isinstance(code, str) and re.fullmatch('[a-z_]{1,80}', code):
        result['error_code'] = code
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--remote-check', action='store_true')
    args = parser.parse_args()
    config = json.loads(checked_file(args.config, args.config_sha256))
    entries, data = prepare(config)
    if not args.execute and not args.remote_check:
        print(json.dumps({'status': 'PREPARED_NO_MUTATIONS', 'total_objects': len(entries),
                          'total_bytes': sum(x['size_bytes'] for x in entries)}), flush=True)
        return
    command = ['env', 'PYTHONNOUSERSITE=1', 'PYTHONDONTWRITEBYTECODE=1', config['python'], '-c',
               BOOTSTRAP, config['worker'], config['worker_sha256']]
    if args.execute:
        command.append('--execute')
    process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
        '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3', 'haopengjin2ali',
        shlex.join(command)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        process.stdin.write(data)
        process.stdin.close()
        final = None
        for raw in process.stdout:
            event = safe_event(raw)
            if event:
                print(json.dumps(event, sort_keys=True), flush=True)
                if event['status'] != 'SYNC_PROGRESS':
                    final = event
        rc = process.wait()
    finally:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
    expected = 'EXISTING_PARTIAL_BATCH_VERIFIED' if args.execute else 'PREPARED_NO_MUTATIONS'
    require(rc == 0 and final and final['status'] == expected, 'remote_outcome_requires_exact_recheck')
    if args.execute:
        require(final.get('verified_objects') == 5010 and final.get('verified_bytes') == 1997181374 and
                sum(final.get(key, -10000) for key in ('uploaded_objects', 'already_present_objects', 'reconciled_objects')) == 5010 and
                final.get('full_source_sha_checked') is True and final.get('full_oss_sha_checked') is True and
                final.get('formal_evaluation_ready') is False and final.get('all_benchmark_assets_ready') is False,
                'incomplete_verification_summary')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        code = str(error) if isinstance(error, SafeError) else 'coordinator_failed'
        print(json.dumps({'status': 'SYNC_FAILED', 'error_code': code}), flush=True)
        sys.exit(2)
