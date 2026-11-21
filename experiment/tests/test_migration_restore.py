"""Exercise real shard reassembly and protection of existing local data."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import types

import pytest


@pytest.fixture
def migration(tmp_path, monkeypatch):
    source = Path(__file__).parents[1]/'scripts/operations/restore_migration.py'
    spec = importlib.util.spec_from_file_location('restore_migration', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = {}
    key = 'jinhaopeng/Paper/DeepProbe-VLM/test/model.bin'
    payload = b'first-piece-and-second-piece'
    pieces = []
    for i, (start, end) in enumerate(((0, 11), (11, len(payload)))):
        pieces.append(dict(key=key, etag='pinned-etag', object_bytes=len(payload), offset=start,
                           length=end-start, member=f'payload/{i}', sha256=hashlib.sha256(payload).hexdigest()))
    plan = {'shards': [{'name': 'test.tar', 'category': 'test', 'pieces': pieces}]}
    pp = tmp_path/'plan.json'; pp.write_text(json.dumps(plan))
    files['migration/2026-11-21/TRANSFER_PLAN.json'] = pp

    def make_archive(corrupt=False):
        path = tmp_path/'test.tar'
        with tarfile.open(path, 'w') as archive:
            rows = []
            for row in pieces:
                raw = payload[row['offset']:row['offset']+row['length']]
                digest = hashlib.sha256(raw).hexdigest()
                if corrupt:
                    raw = b'X'*len(raw)
                info = tarfile.TarInfo(row['member']); info.size = len(raw)
                archive.addfile(info, io.BytesIO(raw))
                rows.append({**row, 'piece_sha256': digest})
            raw = json.dumps({'pieces': rows}).encode()
            info = tarfile.TarInfo('SHARD_MANIFEST.json'); info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
        files['migration/2026-11-21/shards/test.tar'] = path

    make_archive()

    class Api:
        def __init__(self, **kwargs):
            pass

        def download_file(self, repo, kind, name, *, local_dir, **kwargs):
            target = Path(local_dir)/Path(name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(files[name], target)
            return target

        def list_repo_files(self, *args, **kwargs):
            return [types.SimpleNamespace(path=k, type='blob') for k in files]

    monkeypatch.setitem(sys.modules, 'modelscope_hub', types.SimpleNamespace(HubApi=Api))
    monkeypatch.setitem(sys.modules, 'modelscope_hub.config', types.SimpleNamespace(HubConfig=lambda **kw: kw))
    output = tmp_path/'output'
    monkeypatch.setattr(sys, 'argv', ['restore_migration.py', '--output', str(output)])
    return module, output/'oss'/key, payload, make_archive


def test_reassembly_and_verified_resume(migration):
    module, target, payload, _ = migration
    module.main()
    assert target.read_bytes() == payload
    first_mtime = target.stat().st_mtime_ns
    module.main()
    assert target.read_bytes() == payload
    assert target.stat().st_mtime_ns == first_mtime


def test_rejects_corrupt_piece(migration):
    module, _, _, make_archive = migration
    make_archive(corrupt=True)
    with pytest.raises(AssertionError):
        module.main()


def test_refuses_oversized_existing_output(migration):
    module, target, payload, _ = migration
    target.parent.mkdir(parents=True)
    target.write_bytes(payload+b'unrelated-trailing-data')
    before = target.read_bytes()
    with pytest.raises(AssertionError, match='larger'):
        module.main()
    assert target.read_bytes() == before
