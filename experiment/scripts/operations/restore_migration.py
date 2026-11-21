#!/usr/bin/env python3
"""Restore verified migration shards from ModelScope without the old machine.

Uses MODELSCOPE_TOKEN only for access; public repositories need no credential.
Never launches training or changes a scheduler. Output paths preserve original
OSS keys below --output/oss, with old-to-new mapping in TRANSFER_PLAN.json.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile


def main():
    from modelscope_hub import HubApi
    from modelscope_hub.config import HubConfig
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default='JasonJin/DeepProbe-VLM')
    p.add_argument('--prefix', default='migration/2026-11-21')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--category', action='append', help='Repeat to restore selected categories only')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    output = a.output.resolve()
    with tempfile.TemporaryDirectory(prefix='deepprobe-restore-', dir=output) as temp:
        temp = Path(temp)
        api = HubApi(config=HubConfig(endpoint='https://modelscope.ai', token=os.environ.get('MODELSCOPE_TOKEN'),
                                     config_dir=temp/'credentials', cache_dir=temp/'cache'))
        plan_file = api.download_file(a.repo, 'model', a.prefix+'/TRANSFER_PLAN.json', revision='master', local_dir=temp)
        plan = json.loads(Path(plan_file).read_bytes())
        available = {x.path for x in api.list_repo_files(a.repo, 'model', revision='master') if x.type != 'tree'}
        receipts = output/'restore-receipts'
        receipts.mkdir(exist_ok=True)
        objects = {}
        for shard in plan['shards']:
            if a.category and shard['category'] not in a.category:
                continue
            remote = a.prefix+'/shards/'+shard['name']
            if remote not in available:
                raise RuntimeError(f'Transfer still incomplete: {remote}')
            saved = receipts/(shard['name']+'.json')
            file = api.download_file(a.repo, 'model', remote, revision='master', local_dir=temp/'shards')
            with tarfile.open(file, 'r:') as archive:
                members = archive.getmembers()
                assert all(m.isfile() for m in members)
                assert len({m.name for m in members}) == len(members)
                manifest = json.load(archive.extractfile('SHARD_MANIFEST.json'))
                assert [{k: r[k] for k in p} for r,p in zip(manifest['pieces'], shard['pieces'])] == shard['pieces']
                assert len(manifest['pieces']) == len(shard['pieces'])
                assert {m.name for m in members} == {'SHARD_MANIFEST.json'} | {r['member'] for r in manifest['pieces']}
                for row in manifest['pieces']:
                    objects[row['key']] = (row['object_bytes'], row.get('sha256'))
                    key = PurePosixPath(row['key'])
                    assert not key.is_absolute() and '..' not in key.parts
                    target = output/'oss'/key
                    target.parent.mkdir(parents=True, exist_ok=True)
                    assert target.resolve().is_relative_to(output)
                    h = hashlib.sha256()
                    # Reconcile bytes before skipping an already restored piece.
                    same = False
                    if target.exists():
                        assert target.stat().st_size <= row['object_bytes'], 'Existing output is larger than planned'
                        with target.open('rb') as f:
                            f.seek(row['offset'])
                            same = hashlib.sha256(f.read(row['length'])).hexdigest() == row['piece_sha256']
                    if same:
                        continue
                    source = archive.extractfile(row['member'])
                    mode = 'r+b' if target.exists() else 'w+b'
                    with target.open(mode) as f:
                        f.seek(row['offset'])
                        length = 0
                        for block in iter(lambda: source.read(8*1024**2), b''):
                            h.update(block); length += len(block); f.write(block)
                        f.flush(); os.fsync(f.fileno())
                    assert length == row['length'] and h.hexdigest() == row['piece_sha256']
            saved.write_text(json.dumps({'status': 'RESTORED_PIECES_SHA256_VERIFIED', 'shard': remote})+'\n')
            Path(file).unlink()
            print('Restored', shard['name'], flush=True)
        for key, (size, expected_sha) in objects.items():
            target = output/'oss'/key
            assert target.stat().st_size == size, 'Reassembled object size differs'
            if expected_sha:
                h = hashlib.sha256()
                with target.open('rb') as f:
                    for block in iter(lambda: f.read(8*1024**2), b''):
                        h.update(block)
                assert h.hexdigest() == expected_sha, 'Accepted checkpoint SHA differs'
        shutil.copyfile(plan_file, output/'TRANSFER_PLAN.json')


if __name__ == '__main__':
    main()
