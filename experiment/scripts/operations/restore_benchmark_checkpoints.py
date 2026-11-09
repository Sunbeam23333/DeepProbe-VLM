#!/usr/bin/env python3
"""Restore only the two accepted SFT finals into a new evaluation asset root.

Run on the authorized Ali devbox. Old runs and OSS objects are read-only.
Every restored object is bound to an already accepted archive SHA inventory.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess


BASE = Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM')
P1 = Path(str(BASE) + '-p1')
OUT = P1 / 'benchmark-eval-assets-20261012' / 'checkpoints'
LIMIT = 99_000_000_000
RESERVE = 3_000_000_000
OSS = '/home/jinhaopeng/.local/bin/ossutil'
CONF = '/home/jinhaopeng/.ossutilconfig'
RUNS = {
    'r1': ('dp-p1-ouro-r1-f8-g4-s17-sft-a1',
           '3f5bdb5d011576164dd6bee85f1589c5ecddec948c5dc5af8555b1d30b739dfa'),
    'r2': ('dp-p1-ouro-r2-f8-g4-s17-sft-a1',
           'e99959054a1b8d5fdb9f604a95a8cd87f78fc242e8c22dd8155f3514dbc4d502'),
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def measured_bytes():
    roots = [p.resolve(strict=True) for p in (BASE, P1)]
    total = 0
    def fail(error):
        raise error
    for root in roots:
        for directory, dirs, files in os.walk(root, followlinks=False, onerror=fail):
            parent = Path(directory)
            for name in list(dirs):
                path = parent / name
                if path.is_symlink():
                    target = path.resolve(strict=True)
                    require(any(target.is_relative_to(r) for r in roots),
                            'Directory symlink escapes project roots: ' + str(path))
                    dirs.remove(name)
            for name in files:
                path = parent / name
                if path.is_symlink():
                    target = path.resolve(strict=True)
                    require(target.is_file(), 'Unexpected symlink: ' + str(path))
                    if not any(target.is_relative_to(r) for r in roots):
                        total += target.stat().st_size
                else:
                    require(stat.S_ISREG(path.stat().st_mode), 'Not regular: ' + str(path))
                    total += path.stat().st_size
    return total


def write_new(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def main():
    require(not OUT.exists() and not OUT.is_symlink(), 'Evaluation checkpoints root already exists; inspect it, do not overwrite')
    entries = {}
    for label, (run_id, expected) in RUNS.items():
        path = BASE / 'runs' / run_id / 'ARCHIVE_MANIFEST.json'
        require(path.is_file() and not path.is_symlink() and sha(path) == expected,
                'Accepted manifest differs for ' + label)
        manifest = json.loads(path.read_text())
        require(manifest['run_id'] == run_id and manifest['stage'] == 'sft', 'Manifest identity mismatch')
        rows = [x for x in manifest['files'] if x['path'].startswith('training/final/')]
        require(len(rows) == 20, 'Expected full 20-file final')
        names = [x['path'][len('training/final/'):] for x in rows]
        require(len(set(names)) == 20 and all(not Path(x).is_absolute() and '..' not in Path(x).parts for x in names), 'Unsafe or duplicate final path')
        prefix = 'oss://ss-devbox-stream/jinhaopeng/Paper/DeepProbe-VLM/runs/' + run_id + '/'
        require(manifest['uri'].rstrip('/') + '/' == prefix, 'Manifest OSS prefix differs')
        entries[label] = {'run_id': run_id, 'archive_manifest_sha256': expected,
                          'prefix': prefix, 'files': rows}
    required = sum(x['bytes'] for e in entries.values() for x in e['files'])
    before = measured_bytes()
    require(before + required + RESERVE < LIMIT, 'Insufficient combined project capacity')
    OUT.mkdir(parents=True, mode=0o700)
    write_new(OUT / 'RESTORE_INTENT.json', {
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'combined_bytes_before': before, 'required_bytes': required,
        'working_reserve_bytes': RESERVE, 'limit_bytes': LIMIT,
        'source_archives': entries,
    })
    inventories = {}
    for label, entry in entries.items():
        destination = OUT / label
        destination.mkdir()
        def copy_verified(source, target, expected_sha, expected_bytes=None):
            require(not target.exists() and not target.is_symlink(), 'Restore destination already exists')
            target.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run([OSS, 'cp', source, str(target), '-c', CONF],
                                    capture_output=True, text=True, timeout=1800)
            with (OUT / 'restore.log').open('a') as log:
                log.write(source + '\n' + result.stdout + result.stderr)
            require(result.returncode == 0, 'OSS read failed: ' + source)
            require(target.is_file() and not target.is_symlink(), 'Unexpected restored path')
            require(expected_bytes is None or target.stat().st_size == expected_bytes, 'Restored byte size mismatch')
            require(sha(target) == expected_sha, 'Restored SHA mismatch: ' + source)
            target.chmod(0o444)
        copy_verified(entry['prefix'] + 'ARCHIVE_MANIFEST.json',
                      destination / 'ARCHIVE_MANIFEST.json', entry['archive_manifest_sha256'])
        rows = []
        for record in entry['files']:
            relative = record['path'][len('training/final/'):]
            target = destination / 'final' / relative
            copy_verified(entry['prefix'] + record['path'], target,
                          record['sha256'], record['bytes'])
            rows.append({'path': relative, 'bytes': target.stat().st_size, 'sha256': sha(target)})
            print(json.dumps({'checkpoint': label, 'verified': relative, 'bytes': record['bytes']}), flush=True)
        inventories[label] = {'path': str(destination / 'final'),
                              'run_id': entry['run_id'],
                              'archive_manifest_sha256': entry['archive_manifest_sha256'],
                              'files': rows, 'file_count': len(rows),
                              'total_bytes': sum(x['bytes'] for x in rows)}
        for path in sorted((destination / 'final').rglob('*'), reverse=True):
            if path.is_dir():
                path.chmod(0o555)
        (destination / 'final').chmod(0o555)
    after = measured_bytes()
    require(after + RESERVE < LIMIT, 'Combined project capacity exceeded after restore')
    write_new(OUT / 'CHECKPOINT_RESTORE_VERIFIED.json', {
        'status': 'two_accepted_sft_finals_restored_full_sha256',
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'combined_bytes_after': after, 'restored_bytes': required,
        'checkpoints': inventories, 'old_runs_modified': False, 'oss_modified': False,
    })
    print(json.dumps({'status': 'complete', 'receipt': str(OUT / 'CHECKPOINT_RESTORE_VERIFIED.json'),
                      'sha256': sha(OUT / 'CHECKPOINT_RESTORE_VERIFIED.json')}), flush=True)


if __name__ == '__main__':
    main()
