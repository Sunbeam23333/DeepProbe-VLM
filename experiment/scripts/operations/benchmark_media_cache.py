#!/usr/bin/env python3
"""Acquire pinned Video-MME archives into an exact, lossless uniform-eight cache.

The original archive is verified against its immutable Hub LFS SHA before any
member is used. Only temporary files created by this command are discarded.
Signed download URLs arrive on stdin and are never written to receipts/logs.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import fcntl
import os
from pathlib import Path
import re
import sys
import threading
import time
import urllib.request
import zipfile

ROOTS = [Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM'),
         Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1')]
REVISION = 'ead1408f75b618502df9a1d8e0950166bf0a2a0b'
REGISTRY = ROOTS[1] / 'formal-source-final-20261009-v1/benchmark_source_registry.json'
REGISTRY_SHA = 'a143d6a364221738b727d3944527039060315e04373599c5829ab3ea60eb0226'
PLAN_SHA = 'ca3c186c375286500e0576b80eac0012e6fcf61abe9dd8714be731fe88292ac0'
MEDIA = ROOTS[1] / 'benchmark-media-20261012/videomme'
FRAME_CAP = 18_000_000_000
LIMIT = 99_000_000_000
MARGIN = 2_000_000_000
DECODE_WORKERS = 6
LOCK = threading.Lock()
FRAME_BYTES = 0


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')


def save_atomic(path, value):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError('refusing to replace completed artifact: ' + str(path))
    temporary = path.with_name(path.name + '.writing-' + str(os.getpid()))
    with temporary.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temporary.rename(path)


def tree_bytes(path):
    size = 0
    errors = []
    for directory, _, files in os.walk(path, followlinks=False, onerror=errors.append):
        for name in files:
            p = Path(directory) / name
            if p.is_symlink():
                if not p.exists():
                    raise ValueError('broken symlink: ' + str(p))
                continue
            size += p.stat().st_size
    if errors:
        raise OSError(str(errors))
    return size


def sources():
    if sha(REGISTRY) != REGISTRY_SHA:
        raise ValueError('immutable source registry changed')
    ids = {e['source_video_id'] for e in json.loads(REGISTRY.read_text())['entries']
           if e['dataset'] == 'videomme'}
    if len(ids) != 899:
        raise ValueError('expected exact 899 Video-MME sources')
    return ids


def verify_entry(entry):
    if entry['decode_complete'] is not True or len(entry['frames']) != 8:
        raise ValueError('incomplete source entry')
    for f in entry['frames']:
        p = MEDIA / f['relative_path']
        if p.is_symlink() or p.stat().st_size != f['size_bytes'] or sha(p) != f['sha256']:
            raise ValueError('cache frame changed: ' + str(p))


def download(spec, url, workers):
    target = MEDIA / 'temporary' / spec['path']
    receipt = MEDIA / 'archives' / (spec['path'] + '.verified.json')
    if receipt.exists() and target.exists():
        if target.stat().st_size != spec['size_bytes'] or sha(target) != spec['sha256']:
            raise ValueError('existing archive verification mismatch')
        return target
    if target.exists():
        raise FileExistsError('unverified final archive exists: ' + str(target))
    partial = target.with_suffix('.partial')
    if partial.exists():
        raise FileExistsError('previous partial requires evidence-preserving handling: ' + str(partial))
    # Charge external library links conservatively in addition to both trees.
    current = sum(tree_bytes(p) for p in ROOTS) + 100_000_000
    frame_used = tree_bytes(MEDIA / 'frames')
    max_source = max(m['size_bytes'] for m in spec['members'])
    projected = current + spec['size_bytes'] + DECODE_WORKERS * max_source + max(0, FRAME_CAP - frame_used) + MARGIN
    if projected >= LIMIT:
        raise ValueError(f'capacity gate: projected {projected} >= {LIMIT}')
    save(MEDIA / 'archives' / (spec['path'] + '.admission.json'),
         {'combined_bytes_including_external_allowance': current, 'projected_bytes': projected,
          'limit_bytes': LIMIT, 'frame_cap_bytes': FRAME_CAP, 'decode_workers': DECODE_WORKERS,
          'archive': {k: v for k, v in spec.items() if k != 'members'}})
    started = time.time()
    fd = os.open(partial, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.ftruncate(fd, spec['size_bytes'])
    piece = 32 * 1024 * 1024
    chunks = [(start, min(spec['size_bytes'], start + piece) - 1)
              for start in range(0, spec['size_bytes'], piece)]
    def fetch(bounds):
        start, end = bounds
        last_error = None
        for retry in range(4):
            try:
                req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(req, timeout=60) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{spec["size_bytes"]}':
                        raise ValueError('incorrect HTTP range response')
                    data = response.read(end - start + 2)
                if len(data) != end - start + 1:
                    raise ValueError('short or excess HTTP range payload')
                view = memoryview(data)
                offset = start
                while view:
                    n = os.pwrite(fd, view, offset)
                    if n <= 0:
                        raise OSError('short pwrite')
                    offset += n
                    view = view[n:]
                return len(data)
            except Exception as e:
                # Error class only: exception strings can contain signed URLs.
                last_error = type(e).__name__
                time.sleep(min(8, 1 + retry * 2))
        raise RuntimeError('range download failed: ' + str(last_error))
    total = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for n in pool.map(fetch, chunks):
                total += n
                if total % (256 * 1024 * 1024) < piece:
                    print(json.dumps({'event': 'download_progress', 'archive': spec['path'], 'bytes': total,
                                      'elapsed_seconds': round(time.time() - started, 2)}), flush=True)
        os.fsync(fd)
    finally:
        os.close(fd)
    digest = sha(partial)
    if total != spec['size_bytes'] or digest != spec['sha256']:
        raise ValueError('immutable LFS size/SHA verification failed')
    partial.rename(target)
    save(receipt, {'archive_path': spec['path'], 'archive_sha256': digest,
                   'size_bytes': total, 'repo_id': 'lmms-lab/Video-MME', 'revision': REVISION,
                   'download_seconds': time.time() - started, 'verified': True})
    print(json.dumps({'event': 'archive_verified', 'archive': spec['path'], 'bytes': total,
                      'elapsed_seconds': round(time.time() - started, 2)}), flush=True)
    return target


def decode_selected(path, *, declared_count_override=None):
    """One complete pass when container count is exact; otherwise two passes.

    The optional override is only for equivalence tests of incorrect metadata.
    The observed full decode count always controls the accepted frame indices.
    """
    import av
    selected = {}
    with av.open(str(path), options={'err_detect': 'explode'}) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        declared = stream.frames if declared_count_override is None else declared_count_override
        indices = [round(i * (declared - 1) / 7) for i in range(8)] if declared > 0 else []
        wanted = set(indices)
        count = 0
        for index, frame in enumerate(container.decode(video=0)):
            count += 1
            if index in wanted:
                selected[index] = frame.to_image().convert('RGB')
    if count <= 0:
        raise ValueError('source has no decoded frames')
    if count == declared and set(selected) == wanted:
        return selected, count, indices, declared, 1
    indices = [round(i * (count - 1) / 7) for i in range(8)]
    wanted = set(indices)
    selected = {}
    second_count = 0
    with av.open(str(path), options={'err_detect': 'explode'}) as container:
        container.streams.video[0].thread_count = 1
        for index, frame in enumerate(container.decode(video=0)):
            second_count += 1
            if index in wanted:
                selected[index] = frame.to_image().convert('RGB')
    if second_count != count or set(selected) != wanted:
        raise ValueError('source changed or decode pass mismatch')
    return selected, count, indices, declared, 2


def decode_member(spec, archive, member):
    global FRAME_BYTES
    import av
    from PIL import Image
    member_name = member['path']
    match = re.fullmatch(r'data/([A-Za-z0-9_-]{11})\.mp4', member_name)
    if not match:
        raise ValueError('unexpected selected member path: ' + member_name)
    identity = match.group(1)
    receipt = MEDIA / 'entries' / (identity + '.json')
    if receipt.exists():
        entry = json.loads(receipt.read_text())
        verify_entry(entry)
        return entry
    temp = MEDIA / 'temporary' / (identity + '.mp4')
    started = time.time()
    digest = hashlib.sha256()
    with zipfile.ZipFile(archive) as bundle, bundle.open(member_name) as src, temp.open('xb') as dest:
        info = bundle.getinfo(member_name)
        if info.file_size != member['size_bytes'] or info.CRC != member['crc32']:
            raise ValueError('member metadata differs from pinned central directory')
        for data in iter(lambda: src.read(8 * 1024 * 1024), b''):
            digest.update(data)
            dest.write(data)
    original_sha = digest.hexdigest()
    if temp.stat().st_size != member['size_bytes']:
        raise ValueError('source extracted length mismatch')
    selected, count, indices, declared, passes = decode_selected(temp)
    if sha(temp) != original_sha:
        raise ValueError('source bytes changed during decode')
    output = MEDIA / 'frames' / identity
    output.mkdir()
    frames = []
    for slot, index in enumerate(indices):
        image = selected[index]
        # Bound even an incompressible PNG before the actual write.
        with LOCK:
            if FRAME_BYTES + image.width * image.height * 4 + 1_000_000 > FRAME_CAP:
                raise ValueError('lossless PNG cache capacity cap exceeded')
            path = output / f'{slot:02d}.png'
            image.save(path, format='PNG', compress_level=6)
            with Image.open(path) as loaded:
                if loaded.mode != 'RGB' or loaded.size != image.size or loaded.tobytes() != image.tobytes():
                    raise ValueError('lossless frame readback changed RGB bytes')
            frames.append({'relative_path': path.relative_to(MEDIA).as_posix(),
                           'size_bytes': path.stat().st_size, 'sha256': sha(path),
                           'rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                           'width': image.width, 'height': image.height, 'source_frame_index': index})
            FRAME_BYTES += path.stat().st_size
    entry = {'source_video_id': 'youtube:' + identity, 'original_sha256': original_sha,
             'original_size_bytes': member['size_bytes'], 'decoded_frame_count': count,
             'selected_indices': indices, 'image_paths': [f['relative_path'] for f in frames],
             'frames': frames, 'decode_complete': True,
             'container_declared_frame_count': declared, 'full_decode_passes': passes,
             'second_pass_frame_count': count if passes == 2 else None,
             'archive_path': spec['path'], 'archive_sha256': spec['sha256'], 'member_path': member_name,
             'repo_id': 'lmms-lab/Video-MME', 'revision': REVISION,
             'cache_implementation_sha256': sha(Path(__file__)),
             'decode_implementation': 'PyAV full sequential decode; verified count fast path or second pass; thread_count=1',
             'av_version': av.__version__, 'decode_seconds': time.time() - started}
    verify_entry(entry)
    save(receipt, entry)
    # The exact source is reproducible from the verified immutable archive.
    temp.unlink()
    print(json.dumps({'event': 'source_cached', 'source_video_id': entry['source_video_id'],
                      'decoded_frame_count': count, 'png_bytes': sum(f['size_bytes'] for f in frames),
                      'seconds': round(entry['decode_seconds'], 2)}), flush=True)
    return entry


def process(payload, workers):
    expected = sources()
    spec = payload['archive']
    plan_path = MEDIA / 'videomme_archive_directory.json'
    if sha(plan_path) != PLAN_SHA:
        raise ValueError('pinned media metadata plan changed')
    pinned_specs = {item['path']: item for item in json.loads(plan_path.read_text())}
    if pinned_specs.get(spec['path']) != spec:
        raise ValueError('incoming archive does not equal immutable media plan')
    done = MEDIA / 'archives' / (spec['path'] + '.complete.json')
    selected = [m for m in spec['members'] if 'youtube:' + Path(m['path']).stem in expected]
    if done.exists():
        for member in selected:
            verify_entry(json.loads((MEDIA / 'entries' / (Path(member['path']).stem + '.json')).read_text()))
        return
    archive = download(spec, payload['signed_url'], workers)
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
        entries = list(pool.map(lambda m: decode_member(spec, archive, m), selected))
    archive.unlink()
    save(done, {'archive_path': spec['path'], 'archive_sha256': spec['sha256'],
                'source_count': len(entries), 'source_video_ids': sorted(e['source_video_id'] for e in entries),
                'source_receipt_sha256': {Path(m['path']).stem: sha(MEDIA / 'entries' / (Path(m['path']).stem + '.json')) for m in selected},
                'decode_seconds': time.time() - started, 'temporary_archive_deleted_after_verified_cache': True})
    print(json.dumps({'event': 'archive_complete', 'archive': spec['path'], 'sources': len(entries),
                      'decode_seconds': time.time() - started}), flush=True)


def finalize():
    expected = sources()
    entries = [json.loads(p.read_text()) for p in sorted((MEDIA / 'entries').glob('*.json'))]
    actual = {e['source_video_id'] for e in entries}
    if actual != expected or len(entries) != 899:
        raise ValueError(f'incomplete exact source set: actual={len(actual)}, missing={len(expected-actual)}')
    for entry in entries:
        verify_entry(entry)
    if any((MEDIA / 'temporary').iterdir()):
        raise ValueError('unresolved acquisition temporary files')
    result = {'schema_version': 1, 'kind': 'videomme_uniform8_frame_cache', 'complete': True,
              'num_frames': 8, 'sampling': 'deepprobe_vlm.data.uniform_indices', 'failures': [],
              'repo_id': 'lmms-lab/Video-MME', 'revision': REVISION,
              'source_registry_sha256': REGISTRY_SHA, 'entries': entries,
              'media_archive_plan_sha256': PLAN_SHA,
              'archive_directory_sha256': PLAN_SHA,
              'cache_implementation_sha256': sha(Path(__file__)),
              'source_cache_implementation_sha256s': sorted({e['cache_implementation_sha256'] for e in entries}),
              'originals_retained': False, 'provenance': 'Pinned LFS archive SHA plus per-member CRC and source SHA',
              'cache_bytes': sum(f['size_bytes'] for e in entries for f in e['frames']),
              'limitations': ['Exact eight uniform RGB frames retained, no audio/subtitles.',
                              'Original bytes were fully decoded, exact frame-count sampling verified, then discarded from task-scoped temporary storage.']}
    save_atomic(MEDIA / 'FRAME_CACHE_RECEIPT.json', result)
    save_atomic(MEDIA / 'FRAME_CACHE_COMPLETE.json',
                {'schema_version': 1, 'receipt_sha256': sha(MEDIA / 'FRAME_CACHE_RECEIPT.json'),
                 'source_count': 899, 'frame_count': 7192, 'complete': True})
    print(json.dumps({'event': 'complete', 'sources': 899, 'frames': 7192,
                      'receipt_sha256': sha(MEDIA / 'FRAME_CACHE_RECEIPT.json')}), flush=True)


def main():
    global FRAME_BYTES
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['archive', 'finalize'])
    parser.add_argument('--download-workers', type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.download_workers <= 16:
        raise ValueError('download workers must be 1..16')
    os.nice(10)
    for sub in ['temporary', 'archives', 'frames', 'entries']:
        (MEDIA / sub).mkdir(parents=True, exist_ok=True)
    instance_lock = (MEDIA / '.acquisition.lock').open('a')
    fcntl.flock(instance_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    FRAME_BYTES = tree_bytes(MEDIA / 'frames')
    if args.mode == 'archive':
        process(json.load(sys.stdin), args.download_workers)
    else:
        finalize()


if __name__ == '__main__':
    main()
