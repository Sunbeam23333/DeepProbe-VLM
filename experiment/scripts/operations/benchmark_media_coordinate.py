#!/usr/bin/env python3
"""Resolve public pinned Hub CDN URLs locally and run bounded Ali acquisition."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'docs/operations/evidence/benchmark-media-acquisition-20261012'
PLAN = EVIDENCE / 'videomme_archive_directory.json'
PLAN_SHA = 'ca3c186c375286500e0576b80eac0012e6fcf61abe9dd8714be731fe88292ac0'
PYTHON = '/home/jinhaopeng/.cache/Paper/DeepProbe-VLM/env/bin/python'
WORKER = '/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/benchmark-media-20261012/videomme/benchmark_media_cache_v4.py'
BASE_URL = 'https://huggingface.co/datasets/lmms-lab/Video-MME/resolve/ead1408f75b618502df9a1d8e0950166bf0a2a0b/'


def invoke(mode, payload=None):
    cmd = shlex.join([PYTHON, '-u', WORKER, mode])
    process = subprocess.Popen(['ssh', '-o', 'ConnectTimeout=15', 'haopengjin2ali', cmd],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    process.stdin.write(json.dumps(payload) if payload else '')
    process.stdin.close()
    for line in process.stdout:
        print(line, end='', flush=True)
    if process.wait() != 0:
        raise RuntimeError('Ali media worker failed; preserved evidence requires inspection before retry')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true',
                        help='Explicitly run the frozen remote acquisition plan')
    parser.add_argument('--only', help='Optional exact zip name for a bounded pilot')
    parser.add_argument('--after-archive', help='Wait for the already running pilot archive receipt before continuing')
    args = parser.parse_args()
    if not args.execute:
        parser.error('--execute is required; this historical coordinator writes remote state')
    if hashlib.sha256(PLAN.read_bytes()).hexdigest() != PLAN_SHA:
        raise ValueError('immutable media plan SHA mismatch')
    specs = json.loads(PLAN.read_text())
    if args.after_archive:
        if args.after_archive not in {spec['path'] for spec in specs}:
            raise ValueError('unknown prerequisite archive')
        receipt = str(Path(WORKER).parent / 'archives' / (args.after_archive + '.complete.json'))
        check = 'from pathlib import Path; print(Path(' + repr(receipt) + ').is_file())'
        print(json.dumps({'event': 'waiting_for_running_pilot', 'archive': args.after_archive}), flush=True)
        deadline = time.monotonic() + 10800
        while True:
            result = subprocess.run(['ssh', '-o', 'ConnectTimeout=15', 'haopengjin2ali',
                                     shlex.join([PYTHON, '-c', check])],
                                    text=True, capture_output=True, timeout=30, check=True)
            if result.stdout.strip() == 'True':
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('pilot prerequisite not completed within three hours')
            time.sleep(15)
    # Smallest archive first establishes measured transfer and decode behavior.
    specs.sort(key=lambda spec: (spec['path'] != 'videos_chunked_20.zip', spec['path']))
    if args.only:
        specs = [spec for spec in specs if spec['path'] == args.only]
        if len(specs) != 1:
            raise ValueError('unknown archive name')
    for spec in specs:
        print(json.dumps({'event': 'starting_archive', 'path': spec['path'], 'bytes': spec['size_bytes']}), flush=True)
        request = urllib.request.Request(BASE_URL + spec['path'], headers={'Range': 'bytes=0-0'})
        with urllib.request.urlopen(request, timeout=60) as response:
            signed_url = response.url
            if response.status != 206:
                raise ValueError('Hub CDN range capability missing')
        # The signed URL remains in memory/stdin only, never in files or logs.
        invoke('archive', {'archive': spec, 'signed_url': signed_url})
    if not args.only:
        invoke('finalize')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Persist a small safe failure marker for the downstream one-shot flow.
        # Do not serialize exception text, which may include a signed URL.
        failure = {'complete': False, 'error_type': type(error).__name__,
                   'action': 'inspect acquisition log and preserved temporary files before resuming'}
        marker = str(Path(WORKER).parent / 'COORDINATOR_FAILURE.json')
        code = ('import json,sys; from pathlib import Path; p=Path(' + repr(marker)
                + '); value=json.load(sys.stdin); p.open("x").write(json.dumps(value,indent=2)+"\\n")')
        subprocess.run(['ssh', '-o', 'ConnectTimeout=15', 'haopengjin2ali',
                        shlex.join([PYTHON, '-c', code])], input=json.dumps(failure),
                       text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        print(json.dumps({'event': 'coordinator_failure', **failure}), flush=True)
        sys.exit(1)
