#!/usr/bin/env python3
"""One-shot compressed MVBench acquisition; the HF token is stdin/in-memory only."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request

BOOTSTRAP = ('import hashlib,runpy,sys; p=sys.argv.pop(1); expected=sys.argv.pop(1); '
             'assert hashlib.sha256(open(p,"rb").read()).hexdigest()==expected; '
             'sys.argv[0]=p; runpy.run_path(p,run_name="__main__")')
SAFE_FIELDS = {'status', 'path', 'sha256', 'size_bytes', 'receipt_sha256', 'receipt_path',
               'before_bytes', 'partial_bytes', 'projected_bytes', 'reserve_bytes', 'limit_bytes',
               'archives', 'bytes', 'ntu_media_included', 'error_type', 'media_reserved_as_total',
               'measured_dual_root_bytes', 'measured_media_subtree_bytes'}


def read_token(stream):
    token = stream.readline(4098).strip()
    if not re.fullmatch(r'hf_[A-Za-z0-9]{1,4093}', token): raise ValueError('invalid stdin credential')
    return token


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def resolve_signed_url(opener, spec, token):
    url = ('https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/'
           '230a2d4fac8900333c61754641c7a13e069ac9c6/' + spec['path'])
    request = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + token,
        'Range': 'bytes=0-0', 'Accept-Encoding': 'identity', 'Cache-Control': 'no-cache'})
    try:
        response = opener.open(request, timeout=45)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        if response.code != 302: raise RuntimeError('hub_access_failed')
        return response.headers['Location']


def main():
    config_path = Path(sys.argv[1])
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != sys.argv[2]:
        raise ValueError('coordinator config changed')
    config = json.loads(config_path.read_text())
    if not re.fullmatch(r'[0-9a-f]{64}', config.get('worker_sha256', '')):
        raise ValueError('worker SHA binding required')
    token = read_token(sys.stdin)
    raw_plan = Path(config['local_plan']).read_bytes()
    if hashlib.sha256(raw_plan).hexdigest() != config['plan_sha256']: raise ValueError('local plan changed')
    plan = json.loads(raw_plan)
    names = {'clevrer', 'data0613', 'FunQA_test', 'Moments_in_Time_Raw', 'perception',
             'scene_qa', 'ssv2_video', 'sta', 'star', 'tvqa', 'vlnqa'}
    if (plan.get('repository') != 'OpenGVLab/MVBench' or plan.get('revision') != '230a2d4fac8900333c61754641c7a13e069ac9c6'
            or len(plan.get('archives', [])) != 11
            or {s.get('path') for s in plan['archives']} != {'video/' + name + '.zip' for name in names}):
        raise ValueError('unexpected dataset or archive set')
    base = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
            '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3', 'haopengjin2ali']

    def call(mode, name=None, payload=None):
        command = [config['python'], '-c', BOOTSTRAP, config['worker'], config['worker_sha256'],
                   mode, '--plan', config['remote_plan'], '--plan-sha256', config['plan_sha256']]
        if name is not None: command += ['--archive', name]
        response = subprocess.run(base + [shlex.join(command)], input=json.dumps(payload) if payload else '',
                                  capture_output=True, text=True, timeout=2100)
        # Do not relay raw SSH/worker output, which might contain an exception URL.
        result = json.loads(response.stdout)
        if not isinstance(result, dict): raise ValueError('invalid remote response')
        clean = {key: value for key, value in result.items() if key in SAFE_FIELDS}
        print(json.dumps(clean), flush=True)
        if response.returncode not in (0, 75, 2): raise RuntimeError('remote_asset_operation_failed')
        return response.returncode, result

    # urllib never reads .netrc; disable redirects and environment proxy auth.
    opener = urllib.request.build_opener(NoRedirect(), urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 24 * 3600
    for spec in sorted(plan['archives'], key=lambda x: x['size_bytes']):
        failures = 0
        while True:
            if time.monotonic() > deadline: raise TimeoutError('asset_wait_exceeded_24h')
            rc, state = call('status', spec['path'])
            if state['status'] == 'VERIFIED_EXISTING': break
            if rc == 75:
                time.sleep(60); continue
            if rc != 0 or state['status'] not in {'CAPACITY_READY', 'VERIFIED_TARGET_RECEIPT_PENDING'}:
                raise RuntimeError('asset_status_failed')
            signed = None
            if state['status'] == 'CAPACITY_READY':
                for attempt in range(8):
                    try:
                        signed = resolve_signed_url(opener, spec, token)
                        break
                    except Exception:
                        if attempt == 7: raise RuntimeError('hub_link_resolution_failed') from None
                        time.sleep(min(30, 2 ** attempt))
            rc, result = call('download', spec['path'], {'signed_url': signed} if signed else None)
            signed = None
            if rc == 75:
                time.sleep(60); continue
            if rc == 2 and result.get('status') == 'DOWNLOAD_INCOMPLETE':
                failures += 1
                if failures >= 6: raise RuntimeError('bounded_download_retry_exhausted')
                # Recheck the exact intent/partial state before every continuation.
                time.sleep(min(30, 2 ** failures)); continue
            if rc != 0 or result['status'] not in ('ARCHIVE_SHA_VERIFIED', 'VERIFIED_EXISTING'):
                raise RuntimeError('unverified_download')
            break
    rc, result = call('finalize')
    if rc != 0 or result.get('status') != 'ALL_ARCHIVES_SHA_VERIFIED':
        raise RuntimeError('final_archive_receipt_missing')
    print(json.dumps({'event': 'MVBENCH_AVAILABLE_ARCHIVES_COMPLETE', 'archives': 11,
                      'bytes': sum(x['size_bytes'] for x in plan['archives']), 'ntu_media_included': False,
                      'receipt_path': result['receipt_path'], 'receipt_sha256': result['receipt_sha256']}), flush=True)


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        print(json.dumps({'event': 'MVBENCH_ASSET_COORDINATOR_FAILED', 'error_type': type(exc).__name__}), flush=True)
        raise SystemExit(1)
