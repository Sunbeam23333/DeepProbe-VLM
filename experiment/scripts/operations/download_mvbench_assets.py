#!/usr/bin/env python3
"""Bounded, resumable downloads of pinned compressed MVBench ZIPs only.

Signed URLs arrive on stdin and never enter argv, receipts, or diagnostics.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
import urllib.parse
import urllib.request

BASE = Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM')
P1 = Path(str(BASE) + '-p1')
OUT = P1 / 'benchmark-mvbench-assets-20261013'
MEDIA_DONE = P1 / 'benchmark-media-20261012/videomme/FRAME_CACHE_COMPLETE.json'
REVISION = '230a2d4fac8900333c61754641c7a13e069ac9c6'
LIMIT = 99_000_000_000
# v4 maximum conservative additional occupancy: largest ZIP + six largest
# sources + 18GB PNG cap + 2GB margin + 0.1GB allowance = 30,974,216,988 B.
MEDIA_RESERVE = 31_000_000_000
PRE_MEDIA_MAX_ARCHIVE = 100_000_000
CDN_HOSTS = {'cas-bridge.xethub.hf.co', 'cdn-lfs.hf.co', 'cdn-lfs-us-1.hf.co', 'cdn-lfs-eu-1.hf.co',
             'us.aws.cdn.hf.co'}
ARCHIVE_NAMES = {'clevrer', 'data0613', 'FunQA_test', 'Moments_in_Time_Raw', 'perception',
                 'scene_qa', 'ssv2_video', 'sta', 'star', 'tvqa', 'vlnqa'}
SHA_RE = re.compile(r'[0-9a-f]{64}')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def safe_path(path):
    path = Path(path)
    for component in (path, *path.parents):
        if component.is_symlink(): raise ValueError('symlink in state path')
    return path


def read_json(path, maximum=20_000_000):
    safe_path(path)
    if not path.is_file() or path.stat().st_size > maximum: raise ValueError('invalid JSON state')
    return json.loads(path.read_text())


def save(path, value):
    safe_path(path)
    temporary = path.with_name(path.name + '.writing')
    safe_path(temporary)
    data = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    with temporary.open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path)  # publish complete bytes without replacing anything
    temporary.unlink()
    directory = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)


@contextmanager
def download_lock():
    safe_path(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    path = safe_path(OUT / '.download.lock')
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def measured_bytes(*, media_total_budget=False):
    roots = [safe_path(p).resolve(strict=True) for p in (BASE, P1)]
    total = 0
    media = safe_path(MEDIA_DONE.parent)
    media_bytes = frame_bytes = metadata_bytes = 0
    archives, sources = [], []
    def fail(error): raise error
    for root in roots:
        for directory, dirs, files in os.walk(root, followlinks=False, onerror=fail):
            parent = Path(directory)
            for name in list(dirs):
                path = parent / name
                if path.is_symlink():
                    target = path.resolve(strict=True)
                    if not any(target.is_relative_to(r) for r in roots): raise ValueError('external directory symlink')
                    if media_total_budget and path.is_relative_to(media) and not target.is_relative_to(media):
                        raise ValueError('excluded media directory link escapes subtree')
                    dirs.remove(name)
            for name in files:
                path = parent / name
                try:
                    if path.is_symlink():
                        target = path.resolve(strict=True)
                        if not target.is_file(): raise ValueError('nonfile symlink')
                        if media_total_budget and path.is_relative_to(media) and not target.is_relative_to(media):
                            raise ValueError('excluded media file link escapes subtree')
                        size = target.stat().st_size if not any(target.is_relative_to(r) for r in roots) else 0
                    else:
                        info = path.stat()
                        if not stat.S_ISREG(info.st_mode): raise ValueError('nonregular file')
                        size = info.st_size
                    total += size
                    if media_total_budget and path.is_relative_to(media):
                        relative = path.relative_to(media)
                        media_bytes += size
                        if relative.parts[0] == 'frames': frame_bytes += size
                        elif relative.parts[0] == 'temporary':
                            if len(relative.parts) != 2: raise ValueError('unexpected media scratch layout')
                            if re.fullmatch(r'videos_chunked_(?:0[1-9]|1[0-9]|20)\.(?:zip|partial)', relative.name): archives.append(size)
                            elif re.fullmatch(r'[A-Za-z0-9_-]{11}\.mp4', relative.name): sources.append(size)
                            else: raise ValueError('unexpected media scratch file')
                        else: metadata_bytes += size
                except FileNotFoundError:
                    if path.is_symlink(): raise ValueError('dangling link in capacity scan')
                    # Concurrent Video-MME deletes reproducible source scratch;
                    # missing bytes decrease usage. Its additions have a 31GB cap.
                    continue
    if not media_total_budget: return total
    if (len(archives) > 1 or any(size > 5_343_301_110 for size in archives)
            or len(sources) > 6 or any(size > 921_819_313 for size in sources)
            or frame_bytes > 18_000_000_000 or metadata_bytes > 1_000_000_000):
        raise ValueError('media total-budget assumptions exceeded')
    return total - media_bytes, {'media_reserved_as_total': True, 'excluded_media_root': str(media),
        'measured_dual_root_bytes': total, 'measured_media_subtree_bytes': media_bytes,
        'measured_media_frame_bytes': frame_bytes, 'measured_media_metadata_bytes': metadata_bytes,
        'media_temporary_archive_count': len(archives), 'media_temporary_source_count': len(sources),
        'excluded_media_links_contained': True}


def validate_plan(plan):
    if plan.get('repository') != 'OpenGVLab/MVBench' or plan.get('revision') != REVISION:
        raise ValueError('unexpected dataset')
    specs = plan.get('archives', [])
    expected = {'video/' + name + '.zip' for name in ARCHIVE_NAMES}
    if len(specs) != 11 or {s.get('path') for s in specs} != expected: raise ValueError('unexpected archive set')
    for spec in specs:
        if type(spec.get('size_bytes')) is not int or not 0 < spec['size_bytes'] <= 12_000_000_000:
            raise ValueError('invalid pinned size')
        if not SHA_RE.fullmatch(spec.get('sha256', '')): raise ValueError('invalid pinned SHA')
    return {spec['path']: spec for spec in specs}


def identity(spec):
    # Do not copy arbitrary incoming metadata into persistent receipts.
    return {'repository': 'OpenGVLab/MVBench', 'revision': REVISION,
            'file': {key: spec[key] for key in ('path', 'size_bytes', 'sha256')}}


def paths(spec):
    target = safe_path(OUT / spec['path'])
    partial = safe_path(target.with_suffix(target.suffix + '.partial'))
    intent = safe_path(target.with_suffix(target.suffix + '.INTENT.json'))
    verified = safe_path(target.with_suffix(target.suffix + '.VERIFIED.json'))
    return target, partial, intent, verified


def media_complete():
    if not MEDIA_DONE.exists(): return False
    done = read_json(MEDIA_DONE)
    receipt = safe_path(MEDIA_DONE.with_name('FRAME_CACHE_RECEIPT.json'))
    if not (done.get('complete') is True and done.get('source_count') == 899 and done.get('frame_count') == 7192
            and receipt.is_file() and sha(receipt) == done.get('receipt_sha256')):
        raise ValueError('inconsistent Video-MME completion')
    value = read_json(receipt)
    if not (value.get('complete') is True and value.get('kind') == 'videomme_uniform8_frame_cache'
            and value.get('repo_id') == 'lmms-lab/Video-MME'
            and value.get('revision') == 'ead1408f75b618502df9a1d8e0950166bf0a2a0b'
            and value.get('num_frames') == 8 and len(value.get('entries', [])) == 899
            and value.get('failures') == []):
        raise ValueError('unexpected Video-MME receipt')
    return True


def verify_receipt(spec, receipt):
    value = read_json(receipt)
    if (value.get('identity') != identity(spec) or value.get('sha256') != spec['sha256']
            or value.get('size_bytes') != spec['size_bytes'] or value.get('status') != 'ARCHIVE_SHA_VERIFIED'):
        raise ValueError('archive receipt changed')


def status(spec):
    target, partial, intent, verified = paths(spec)
    if intent.exists() and read_json(intent).get('identity') != identity(spec): raise ValueError('partial source changed')
    if partial.exists() and not intent.exists(): raise ValueError('unattested partial')
    if target.exists():
        if not target.is_file() or target.stat().st_size != spec['size_bytes'] or sha(target) != spec['sha256']:
            raise ValueError('existing pinned asset changed')
        if verified.exists(): verify_receipt(spec, verified)
        return {'status': 'VERIFIED_EXISTING' if verified.exists() else 'VERIFIED_TARGET_RECEIPT_PENDING',
                'path': spec['path'], 'sha256': spec['sha256'], 'size_bytes': spec['size_bytes']}
    if verified.exists(): raise ValueError('receipt without archive')
    partial_bytes = partial.stat().st_size if partial.exists() else 0
    if partial_bytes > spec['size_bytes']: raise ValueError('oversize partial')
    done = media_complete()
    if not done and spec['size_bytes'] > PRE_MEDIA_MAX_ARCHIVE:
        return {'status': 'WAITING_FOR_VIDEOMME_COMPLETE', 'path': spec['path']}
    reserve = 2_000_000_000 if done else MEDIA_RESERVE
    if done:
        before = measured_bytes()
        basis = {'media_reserved_as_total': False}
    else:
        before, basis = measured_bytes(media_total_budget=True)
    projected = before + spec['size_bytes'] - partial_bytes + reserve
    return {'status': 'CAPACITY_READY' if projected < LIMIT else 'WAITING_FOR_CAPACITY',
            'path': spec['path'], 'before_bytes': before, 'partial_bytes': partial_bytes,
            'projected_bytes': projected, 'reserve_bytes': reserve, 'limit_bytes': LIMIT, **basis}


def validate_url(url):
    if not isinstance(url, str) or len(url) > 16384: raise ValueError('invalid signed CDN URL')
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != 'https' or parsed.hostname not in CDN_HOSTS or parsed.port not in (None, 443)
            or parsed.username or parsed.password or parsed.fragment or any(ord(c) <= 32 or ord(c) >= 127 for c in url)):
        raise ValueError('unexpected signed CDN URL')
    return url


class PinnedCDNRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def bounded_download(spec, url, partial, *, opener=None):
    """Never write a byte beyond the pinned total, including on broken servers."""
    validate_url(url)
    start = partial.stat().st_size if partial.exists() else 0
    expected = spec['size_bytes']
    if start == expected: return
    if not 0 <= start < expected: raise ValueError('invalid partial size')
    opener = opener or urllib.request.build_opener(PinnedCDNRedirect())
    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{expected - 1}', 'Accept-Encoding': 'identity'})
    with opener.open(request, timeout=45) as response:
        if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{expected - 1}/{expected}':
            raise ValueError('range identity mismatch')
        flags = os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW
        flags |= 0 if partial.exists() else os.O_CREAT | os.O_EXCL
        fd = os.open(partial, flags, 0o600)
        deadline = time.monotonic() + 1800
        with os.fdopen(fd, 'ab') as stream:
            if os.fstat(stream.fileno()).st_size != start: raise ValueError('partial changed')
            written = start
            while True:
                if time.monotonic() > deadline: raise TimeoutError('bounded download deadline')
                block = response.read(min(8 * 1024 * 1024, expected - written + 1))
                if not block: break
                if len(block) > expected - written: raise ValueError('response exceeds pinned byte limit')
                stream.write(block); written += len(block)
            stream.flush(); os.fsync(stream.fileno())
            if written != expected: raise ValueError('incomplete response')


def publish_verified(spec):
    target, partial, intent, receipt = paths(spec)
    source = target if target.exists() else partial
    if source.stat().st_size != spec['size_bytes'] or sha(source) != spec['sha256']:
        raise ValueError('archive SHA or size differs')
    if not target.exists():
        os.link(partial, target)
    if partial.exists():
        if not os.path.samefile(partial, target): raise ValueError('separate partial beside verified target')
        partial.unlink()
    target.chmod(0o444)
    if receipt.exists():
        verify_receipt(spec, receipt)
    else:
        save(receipt, {'status': 'ARCHIVE_SHA_VERIFIED', 'identity': identity(spec),
                       'verified_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       'target': str(target), 'size_bytes': spec['size_bytes'], 'sha256': spec['sha256']})
    return {'status': 'ARCHIVE_SHA_VERIFIED', 'path': spec['path'], 'sha256': spec['sha256'],
            'size_bytes': spec['size_bytes'], 'receipt_sha256': sha(receipt)}


def finalize(specs, plan_sha):
    receipts = []
    for spec in specs.values():
        result = status(spec)
        if result['status'] not in {'VERIFIED_EXISTING', 'VERIFIED_TARGET_RECEIPT_PENDING'}:
            raise ValueError('archive set incomplete')
        receipts.append(publish_verified(spec))
    output = OUT / 'ALL_ARCHIVES_SHA_VERIFIED.json'
    value = {'complete': True, 'repository': 'OpenGVLab/MVBench', 'revision': REVISION,
             'plan_sha256': plan_sha, 'archive_count': 11, 'compressed_bytes': sum(s['size_bytes'] for s in specs.values()),
             'archives': sorted(receipts, key=lambda row: row['path']), 'ntu_media_included': False,
             'media_decode_verified': False, 'formal_evaluation_ready': False}
    if output.exists():
        if read_json(output) != value: raise ValueError('final asset receipt changed')
    else: save(output, value)
    return {'status': 'ALL_ARCHIVES_SHA_VERIFIED', 'receipt_path': str(output), 'receipt_sha256': sha(output),
            'archives': 11, 'bytes': value['compressed_bytes'], 'ntu_media_included': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('status', 'download', 'finalize'))
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--archive')
    args = parser.parse_args()
    if sha(safe_path(args.plan)) != args.plan_sha256: raise ValueError('plan changed')
    specs = validate_plan(read_json(args.plan))
    try:
        with download_lock():
            if args.mode == 'finalize': result = finalize(specs, args.plan_sha256)
            else:
                spec = specs[args.archive]
                result = status(spec)
                if args.mode == 'download' and result['status'] == 'VERIFIED_TARGET_RECEIPT_PENDING':
                    result = publish_verified(spec)
                elif args.mode == 'download' and result['status'] == 'CAPACITY_READY':
                    raw = sys.stdin.read(20001)
                    if len(raw) > 20000: raise ValueError('oversize signed URL input')
                    payload = json.loads(raw)
                    if set(payload) != {'signed_url'}: raise ValueError('invalid signed URL fields')
                    url = validate_url(payload['signed_url'])
                    target, partial, intent, receipt = paths(spec)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not intent.exists(): save(intent, {'identity': identity(spec), 'capacity_admission': result})
                    try:
                        bounded_download(spec, url, partial)
                    except Exception as error:
                        print(json.dumps({'status': 'DOWNLOAD_INCOMPLETE', 'path': spec['path'],
                                          'error_type': type(error).__name__}), flush=True)
                        return 2
                    result = publish_verified(spec)
    except BlockingIOError:
        result = {'status': 'WAITING_FOR_MVBENCH_LOCK'}
    print(json.dumps(result), flush=True)
    return 75 if result['status'].startswith('WAITING_') else 0


if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({'status': 'FAILED', 'error_type': type(exc).__name__}), flush=True)
        raise SystemExit(2)
