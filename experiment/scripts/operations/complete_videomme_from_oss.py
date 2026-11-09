#!/usr/bin/env python3
"""Complete the fixed Video-MME uniform8 cache from independently accepted OSS ZIPs.

No HF requests, ZIP/video disk files, deletes, GPU work, or old-file overwrites.
One bounded member per CPU worker lives in BytesIO through complete PyAV decode.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import configparser
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import zipfile

P0 = Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM')
P1 = Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1')
MEDIA = P1 / 'benchmark-media-20261012/videomme'
RUNTIME = P1 / 'benchmark-videomme-oss-decode-20261013'
BUCKET = 'ss-devbox-stream'
ENDPOINT = 'https://oss-cn-wulanchabu.aliyuncs.com'
BASE = 'jinhaopeng/Paper/DeepProbe-VLM/benchmark-cache/20261013-v1/'
REVISION = 'ead1408f75b618502df9a1d8e0950166bf0a2a0b'
ARCHIVE_PREFIX = BASE + 'hf-archives/lmms-lab/Video-MME/' + REVISION + '/'
PNG_PREFIX = BASE + 'frame-cache/videomme/' + REVISION + '/uniform8/'
META_PREFIX = BASE + 'metadata/videomme-frame-cache/'
DIRECTORY_SHA = 'ca3c186c375286500e0576b80eac0012e6fcf61abe9dd8714be731fe88292ac0'
REGISTRY_SHA = 'a143d6a364221738b727d3944527039060315e04373599c5829ab3ea60eb0226'
PNG_PLAN_SHA = 'fdeec64adbc6f2d2a19d3db16482df2906f8ad1895e1f779123efda2c89f603e'
META_PLAN_SHA = '88c5d6483701bc2b24cc1dfc19b5ae36c1e1ee612f93990c2883a1ad6574eddd'
ARCHIVE_PLAN_SHA = '89cf4eddb4ffc62a89b46a4b7966a68c3fce3af46dcb8c6fb32fa35ea96b1512'
LIMIT = 99_000_000_000
MAX_MEMBER = 921_819_313
MAX_FRAME_PIXELS = 3840 * 2160
BLOCK = 8 * 1024 * 1024


class SafeError(Exception): pass


def require(value, code):
    if not value: raise SafeError(code)


def digest(data): return hashlib.sha256(data).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(BLOCK), b''): h.update(data)
    return h.hexdigest()


def encode(value): return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def safe(path):
    p = Path(path)
    require(p.is_absolute() and '..' not in p.parts and not any(x.is_symlink() for x in [p, *p.parents]), 'unsafe_local_path')
    return p


def put_file(path, raw):
    path = safe(path)
    if path.exists():
        require(path.is_file() and path.read_bytes() == raw, 'local_immutable_conflict')
        return
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def tree_bytes(root):
    count = 0
    def error(exc): raise exc
    for directory, folders, names in os.walk(root, followlinks=False, onerror=error):
        for name in folders:
            path = Path(directory) / name
            if path.is_symlink():
                target = path.resolve(strict=True)
                require(any(target.is_relative_to(base) for base in (P0, P1, Path(root))),
                        'unaccounted_external_directory_symlink')
        for name in names:
            p = Path(directory) / name
            if not p.is_symlink(): count += p.stat().st_size; continue
            target = p.resolve(strict=True)
            if any(target.is_relative_to(base) for base in (P0, P1, Path(root))): continue
            require(target.is_file(), 'unaccounted_external_symlink')
            # Match the established dual-root gate: charge every external
            # file link's target bytes, even repeated interpreter aliases.
            count += target.stat().st_size
    return count


def load_pinned(pin, expected=None):
    require(set(pin) == {'path', 'sha256'}, 'pin_fields_changed')
    require(expected is None or pin['sha256'] == expected, 'immutable_input_pin_changed')
    p = safe(pin['path']); raw = p.read_bytes()
    require(digest(raw) == pin['sha256'], 'input_sha_changed')
    return json.loads(raw)


def load_inputs(config_path, config_sha):
    cfg = load_pinned({'path': str(config_path), 'sha256': config_sha})
    require(set(cfg) == {'schema_version', 'directory', 'registry', 'png_plan', 'metadata_plan', 'archive_audit',
                         'workers', 'frame_cap_bytes', 'runtime_library_path'}, 'config_fields_changed')
    require(cfg['schema_version'] == 1 and type(cfg['workers']) is int and 1 <= cfg['workers'] <= 6,
            'worker_limit_changed')
    require(type(cfg['frame_cap_bytes']) is int and 1_967_610_160 <= cfg['frame_cap_bytes'] <= 18_000_000_000,
            'frame_cap_changed')
    require(cfg['runtime_library_path'] == str(RUNTIME / 'python-libs'), 'runtime_library_path_changed')
    directory = load_pinned(cfg['directory'], DIRECTORY_SHA)
    require(Path(cfg['directory']['path']) == MEDIA / 'videomme_archive_directory.json', 'directory_path_changed')
    registry = load_pinned(cfg['registry'], REGISTRY_SHA)
    png = load_pinned(cfg['png_plan'], PNG_PLAN_SHA)
    metadata = load_pinned(cfg['metadata_plan'], META_PLAN_SHA)
    audit = load_pinned(cfg['archive_audit'])
    require(audit.get('status') == 'ARCHIVE_METADATA_INDEPENDENT_AUDIT_PASSED'
            and audit.get('plan_sha256') == ARCHIVE_PLAN_SHA and audit.get('archive_count') == 31
            and audit.get('all_pinned_zip_archives_cached') is True, 'archive_acceptance_missing')
    sources = {r['source_video_id'] for r in registry['entries'] if r['dataset'] == 'videomme'}
    require(len(sources) == 899, 'selected_source_set_changed')
    existing = {r['source_video_id'] for r in png['entries']}
    require(len(existing) == 547 and existing <= sources and len(png['entries']) == 4376, 'existing_source_set_changed')
    remaining = sources - existing
    require(len(remaining) == 352, 'remaining_source_set_changed')
    return cfg, directory, sources, existing, png, metadata, audit


class OSSStore:
    def __init__(self, library_path):
        sys.path.insert(0, str(safe(library_path)))
        import oss2
        require(oss2.__version__ == '2.19.1', 'oss_sdk_changed')
        config = configparser.ConfigParser(); config.read(safe('/home/jinhaopeng/.ossutilconfig'))
        require(config['default'].get('region') == 'cn-wulanchabu', 'oss_region_changed')
        self.oss2 = oss2
        self.bucket = oss2.Bucket(oss2.Auth(config['default']['accesskeyid'], config['default']['accesskeysecret']),
                                  ENDPOINT, BUCKET, connect_timeout=60, enable_crc=True)

    def head(self, key): return self.bucket.head_object(key)

    def get(self, key, maximum=32_000_000):
        try: stream = self.bucket.get_object(key)
        except self.oss2.exceptions.NoSuchKey: return None
        try:
            data = stream.read(maximum + 1)
            require(len(data) <= maximum, 'oss_small_object_oversized')
            return data
        finally: stream.close()

    def range(self, key, start, end, etag, total):
        stream = self.bucket.get_object(key, byte_range=(start, end), headers={'If-Match': etag})
        try:
            header = stream.headers.get('Content-Range', stream.headers.get('content-range'))
            require(stream.status == 206 and header == f'bytes {start}-{end}/{total}', 'oss_range_binding_failed')
            data = stream.read(end - start + 2)
            require(len(data) == end - start + 1, 'oss_range_length_failed')
            return data
        finally: stream.close()

    def put_verified(self, key, raw):
        old = self.get(key)
        if old is not None:
            require(old == raw, 'oss_immutable_conflict'); return
        try:
            self.bucket.put_object(key, raw, headers={'x-oss-forbid-overwrite': 'true', 'x-oss-meta-sha256': digest(raw)})
        except Exception:
            require(self.get(key) == raw, 'oss_write_outcome_unconfirmed'); return
        require(self.get(key) == raw, 'oss_readback_mismatch')

    def keys(self, prefix):
        marker = ''
        while True:
            page = self.bucket.list_objects(prefix=prefix, marker=marker, max_keys=1000)
            for row in page.object_list: yield row.key
            if not page.is_truncated: return
            require(page.next_marker and page.next_marker != marker, 'oss_listing_stalled'); marker = page.next_marker


class RangeReader(io.RawIOBase):
    """Seekable bounded block cache; If-Match binds every ZIP range to audited ETag."""
    def __init__(self, store, key, size, etag):
        self.store, self.key, self.size, self.etag = store, key, size, etag
        self.position, self.start, self.cached = 0, 0, b''

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        pos = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset if whence == 2 else -1
        require(0 <= pos <= self.size, 'zip_seek_out_of_bounds'); self.position = pos; return pos

    def read(self, size=-1):
        if size is None or size < 0: size = self.size - self.position
        require(size <= 2 * BLOCK, 'unbounded_zip_read')
        end = min(self.size, self.position + size); out = bytearray()
        while self.position < end:
            if not self.start <= self.position < self.start + len(self.cached):
                self.start = self.position
                last = min(self.size, self.start + BLOCK) - 1
                self.cached = self.store.range(self.key, self.start, last, self.etag, self.size)
            n = min(end - self.position, self.start + len(self.cached) - self.position)
            require(n > 0, 'empty_range_cache')
            begin = self.position - self.start; out.extend(self.cached[begin:begin+n]); self.position += n
        return bytes(out)


def extract_member(store, checked, spec, member):
    require(0 < member['size_bytes'] <= MAX_MEMBER, 'member_memory_limit')
    reader = RangeReader(store, checked['key'], spec['size_bytes'], checked['etag'])
    target, h, size = io.BytesIO(), hashlib.sha256(), 0
    try:
        with zipfile.ZipFile(reader) as bundle:
            matches = [i for i in bundle.infolist() if i.filename == member['path']]
            require(len(matches) == 1, 'zip_member_missing_or_duplicate')
            info = matches[0]
            require(info.file_size == member['size_bytes'] and info.compress_size == member['compressed_bytes']
                    and info.CRC == member['crc32'] and not info.flag_bits & 1, 'zip_member_metadata_changed')
            with bundle.open(info) as source:
                for data in iter(lambda: source.read(BLOCK), b''):
                    size += len(data); require(size <= member['size_bytes'], 'member_size_exceeded')
                    h.update(data); target.write(data)
        require(size == member['size_bytes'], 'member_size_short')
        target.seek(0); return target, h.hexdigest()
    except BaseException:
        target.close(); raise


def decode_selected(source, *, declared_count_override=None):
    import av
    selected = {}; source.seek(0)
    with av.open(source, options={'err_detect': 'explode'}) as container:
        container.streams.video[0].thread_count = 1
        declared = container.streams.video[0].frames if declared_count_override is None else declared_count_override
        indices = [round(i * (declared - 1) / 7) for i in range(8)] if declared > 0 else []
        wanted = set(indices); count = 0
        for index, frame in enumerate(container.decode(video=0)):
            count += 1
            if index in wanted:
                require(frame.width * frame.height <= MAX_FRAME_PIXELS, 'frame_pixel_memory_limit')
                selected[index] = frame.to_image().convert('RGB')
    require(count > 0, 'source_has_no_frames')
    if count == declared and set(selected) == wanted: return selected, count, indices, declared, 1
    indices = [round(i * (count - 1) / 7) for i in range(8)]; wanted = set(indices); selected = {}; second = 0
    source.seek(0)
    with av.open(source, options={'err_detect': 'explode'}) as container:
        container.streams.video[0].thread_count = 1
        for index, frame in enumerate(container.decode(video=0)):
            second += 1
            if index in wanted:
                require(frame.width * frame.height <= MAX_FRAME_PIXELS, 'frame_pixel_memory_limit')
                selected[index] = frame.to_image().convert('RGB')
    require(second == count and set(selected) == wanted, 'full_decode_count_mismatch')
    return selected, count, indices, declared, 2


def verify_entry(entry):
    from PIL import Image
    identity = entry['source_video_id'].removeprefix('youtube:')
    require(re.fullmatch('[A-Za-z0-9_-]{11}', identity) and entry['decode_complete'] is True, 'source_identity_changed')
    count = entry['decoded_frame_count']; indices = [round(i * (count - 1) / 7) for i in range(8)]
    require(type(count) is int and count > 0 and entry['selected_indices'] == indices and len(entry['frames']) == 8,
            'source_sampling_changed')
    require(entry['image_paths'] == [f['relative_path'] for f in entry['frames']], 'ordered_frames_changed')
    for slot, frame in enumerate(entry['frames']):
        require(frame['relative_path'] == f'frames/{identity}/{slot:02d}.png'
                and frame['source_frame_index'] == indices[slot], 'frame_path_or_index_changed')
        path = safe(MEDIA / frame['relative_path'])
        require(path.stat().st_size == frame['size_bytes'] and sha(path) == frame['sha256'], 'frame_local_sha_changed')
        with Image.open(path) as image:
            image.load()
            require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (frame['width'], frame['height'])
                    and digest(image.tobytes()) == frame['rgb_sha256'], 'frame_rgb_changed')


def validate_archive(store, checked, spec):
    require(checked['key'] == ARCHIVE_PREFIX + spec['path'] and checked['size_bytes'] == spec['size_bytes']
            and checked['source_sha256'] == spec['sha256'], 'audited_archive_binding_changed')
    raw = store.get(checked['key'] + '.CACHE_VERIFIED.json')
    require(raw is not None and digest(raw) == checked['receipt_sha256'], 'archive_receipt_changed')
    head = store.head(checked['key'])
    require(head.content_length == spec['size_bytes'] and head.etag.strip('"') == checked['etag'].strip('"'), 'archive_head_changed')
    return raw


class FrameBudget:
    def __init__(self, cap): self.cap, self.used, self.lock = cap, tree_bytes(MEDIA / 'frames'), threading.Lock()

    def source_admit(self):
        with self.lock:
            current = sum(tree_bytes(root) for root in (P0, P1))
            require(current + max(0, self.cap - self.used) + 100_000_000 + 2_000_000_000 < LIMIT,
                    'source_boundary_root_capacity_failed')

    def write(self, path, raw):
        with self.lock:
            if path.exists(): put_file(path, raw); return
            require(self.used + len(raw) <= self.cap, 'png_total_capacity_limit')
            put_file(path, raw); self.used += len(raw)


def sync_entry(store, entry):
    identity = entry['source_video_id'].removeprefix('youtube:')
    for frame in entry['frames']:
        raw = safe(MEDIA / frame['relative_path']).read_bytes()
        require(digest(raw) == frame['sha256'], 'frame_changed_before_upload')
        store.put_verified(PNG_PREFIX + frame['relative_path'].removeprefix('frames/'), raw)
    raw = safe(MEDIA / 'entries' / (identity + '.json')).read_bytes()
    require(json.loads(raw) == entry, 'source_receipt_changed_before_upload')
    store.put_verified(META_PREFIX + 'entries/' + identity + '.json', raw)


def validate_source_binding(entry, spec, member, checked):
    require(entry.get('source_video_id') == 'youtube:' + Path(member['path']).stem
            and entry.get('original_size_bytes') == member['size_bytes']
            and entry.get('archive_path') == spec['path'] and entry.get('archive_sha256') == spec['sha256']
            and entry.get('member_path') == member['path'] and entry.get('repo_id') == 'lmms-lab/Video-MME'
            and entry.get('revision') == REVISION and entry.get('oss_archive_etag') == checked['etag']
            and entry.get('oss_archive_receipt_sha256') == checked['receipt_sha256']
            and entry.get('cache_implementation_sha256') == sha(__file__)
            and isinstance(entry.get('original_sha256'), str) and re.fullmatch('[0-9a-f]{64}', entry['original_sha256']),
            'existing_new_source_provenance_changed')


def decode_one(store, checked, spec, member, budget):
    import av
    from PIL import Image
    identity = Path(member['path']).stem
    require(re.fullmatch(r'data/[A-Za-z0-9_-]{11}\.mp4', member['path']), 'unexpected_member_path')
    path = MEDIA / 'entries' / (identity + '.json')
    if path.exists():
        entry = json.loads(path.read_bytes()); validate_source_binding(entry, spec, member, checked)
        verify_entry(entry); sync_entry(store, entry); return entry
    started = time.monotonic()
    source, original_sha = extract_member(store, checked, spec, member)
    with source:
        selected, count, indices, declared, passes = decode_selected(source)
    budget.source_admit()
    output = MEDIA / 'frames' / identity; output.mkdir(exist_ok=True)
    frames = []
    for slot, index in enumerate(indices):
        image = selected[index]; encoded = io.BytesIO(); image.save(encoded, format='PNG', compress_level=6); raw = encoded.getvalue()
        require(len(raw) <= 32_000_000, 'encoded_png_memory_limit')
        with Image.open(io.BytesIO(raw)) as loaded:
            loaded.load(); require(loaded.mode == 'RGB' and loaded.size == image.size and loaded.tobytes() == image.tobytes(), 'png_roundtrip_changed')
        frame_path = output / f'{slot:02d}.png'; budget.write(frame_path, raw)
        frames.append({'relative_path': frame_path.relative_to(MEDIA).as_posix(), 'size_bytes': len(raw),
                       'sha256': digest(raw), 'rgb_sha256': digest(image.tobytes()), 'width': image.width,
                       'height': image.height, 'source_frame_index': index})
    entry = {'source_video_id': 'youtube:' + identity, 'original_sha256': original_sha,
             'original_size_bytes': member['size_bytes'], 'decoded_frame_count': count, 'selected_indices': indices,
             'image_paths': [f['relative_path'] for f in frames], 'frames': frames, 'decode_complete': True,
             'container_declared_frame_count': declared, 'full_decode_passes': passes,
             'second_pass_frame_count': count if passes == 2 else None, 'archive_path': spec['path'],
             'archive_sha256': spec['sha256'], 'member_path': member['path'], 'repo_id': 'lmms-lab/Video-MME',
             'revision': REVISION, 'cache_implementation_sha256': sha(__file__),
             'decode_implementation': 'PyAV full sequential decode from OSS ZIP member BytesIO; exact uniform8; thread_count=1',
             'av_version': av.__version__, 'decode_seconds': time.monotonic() - started,
             'oss_archive_etag': checked['etag'], 'oss_archive_receipt_sha256': checked['receipt_sha256'],
             'original_video_written_to_disk': False}
    verify_entry(entry); put_file(path, encode(entry)); sync_entry(store, entry)
    print(json.dumps({'event': 'source_bucket_cached', 'source_video_id': entry['source_video_id'], 'decoded_frame_count': count,
                      'png_bytes': sum(f['size_bytes'] for f in frames), 'total_png_bytes': budget.used,
                      'seconds': round(entry['decode_seconds'], 2)}), flush=True)
    return entry


def finalize(store, directory, sources, cfg, audit):
    require(sum(tree_bytes(root) for root in (P0, P1)) + 100_000_000 < LIMIT, 'final_root_capacity_failed')
    require(tree_bytes(MEDIA / 'frames') <= cfg['frame_cap_bytes'], 'final_png_capacity_failed')
    entries = [json.loads(p.read_bytes()) for p in sorted((MEDIA / 'entries').glob('*.json'))]
    require(len(entries) == 899 and {r['source_video_id'] for r in entries} == sources, 'incomplete_source_set')
    for entry in entries: verify_entry(entry)
    for spec in directory:
        rows = [r for r in entries if r['archive_path'] == spec['path']]
        path = MEDIA / 'archives' / (spec['path'] + '.complete.json')
        if not path.exists():
            value = {'archive_path': spec['path'], 'archive_sha256': spec['sha256'], 'source_count': len(rows),
                     'source_video_ids': sorted(r['source_video_id'] for r in rows),
                     'source_receipt_sha256': {r['source_video_id'].removeprefix('youtube:'): sha(MEDIA / 'entries' / (r['source_video_id'].removeprefix('youtube:') + '.json')) for r in rows},
                     'original_archive_retained_in_oss': True, 'temporary_archive_written': False}
            put_file(path, encode(value))
        store.put_verified(META_PREFIX + 'archives/' + path.name, path.read_bytes())
    receipt = {'schema_version': 1, 'kind': 'videomme_uniform8_frame_cache', 'complete': True, 'num_frames': 8,
               'sampling': 'deepprobe_vlm.data.uniform_indices', 'failures': [], 'repo_id': 'lmms-lab/Video-MME',
               'revision': REVISION, 'source_registry_sha256': REGISTRY_SHA, 'entries': entries,
               'media_archive_plan_sha256': DIRECTORY_SHA, 'archive_directory_sha256': DIRECTORY_SHA,
               'cache_implementation_sha256': sha(__file__),
               'source_cache_implementation_sha256s': sorted({r['cache_implementation_sha256'] for r in entries}),
               'originals_retained': True, 'originals_location': 'OSS archive cache',
               'provenance': 'Audited immutable OSS archive ETag/SHA receipt plus pinned ZIP CRC and complete member SHA/decode',
               'cache_bytes': sum(f['size_bytes'] for r in entries for f in r['frames']),
               'limitations': ['Exact eight uniform RGB frames; no audio/subtitles. MVBench readiness is separate.']}
    receipt_path = MEDIA / 'FRAME_CACHE_RECEIPT.json'; put_file(receipt_path, encode(receipt))
    complete_path = MEDIA / 'FRAME_CACHE_COMPLETE.json'
    put_file(complete_path, encode({'schema_version': 1, 'receipt_sha256': sha(receipt_path), 'source_count': 899, 'frame_count': 7192, 'complete': True}))
    for path in (receipt_path, complete_path, MEDIA / 'videomme_archive_directory.json'):
        store.put_verified(META_PREFIX + path.name, path.read_bytes())
    frames, receipts = [], []
    for entry in entries:
        identity = entry['source_video_id'].removeprefix('youtube:'); path = MEDIA / 'entries' / (identity + '.json')
        receipts.append({'source_video_id': entry['source_video_id'], 'bucket_key': META_PREFIX + 'entries/' + path.name,
                         'size_bytes': path.stat().st_size, 'sha256': sha(path)})
        for f in entry['frames']:
            frames.append({'relative_path': f['relative_path'], 'bucket_key': PNG_PREFIX + f['relative_path'].removeprefix('frames/'),
                           'size_bytes': f['size_bytes'], 'sha256': f['sha256'], 'source_video_id': entry['source_video_id']})
    require(len(frames) == 7192 and set(store.keys(PNG_PREFIX)) == {r['bucket_key'] for r in frames}, 'exact_png_key_set_failed')
    require(set(store.keys(META_PREFIX + 'entries/')) == {r['bucket_key'] for r in receipts}, 'exact_source_key_set_failed')
    # Final full readback covers all reused and newly published frame/source bytes.
    def verify(row):
        raw = store.get(row['bucket_key']); require(raw is not None and len(raw) == row['size_bytes'] and digest(raw) == row['sha256'], 'final_full_oss_readback_failed')
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool: list(pool.map(verify, frames + receipts))
    result = {'schema_version': 1, 'kind': 'videomme_bucket_media_complete_v1', 'status': 'VIDEOMME_BUCKET_MEDIA_COMPLETE',
              'complete': True, 'bucket': BUCKET, 'revision': REVISION, 'source_count': 899, 'frame_count': 7192,
              'media_root': str(MEDIA), 'frame_cache_receipt': {'path': str(receipt_path), 'sha256': sha(receipt_path)},
              'frames': frames, 'source_receipts': receipts, 'frame_prefix': PNG_PREFIX, 'source_receipt_prefix': META_PREFIX + 'entries/',
              'exact_frame_key_set_verified': True, 'exact_source_receipt_key_set_verified': True, 'full_oss_get_sha256_verified': True,
              'cache_bytes': receipt['cache_bytes'], 'source_registry_sha256': REGISTRY_SHA, 'archive_directory_sha256': DIRECTORY_SHA,
              'accepted_archive_catalog_sha256': audit['catalog_sha256'], 'worker_sha256': sha(__file__),
              'all_benchmark_assets_ready': False, 'formal_evaluation_ready': False, 'mvbench_evaluation_ready': False,
              'videomme_media_ready': True, 'ntu_media_included': False, 'audio': False, 'subtitles': False}
    path = MEDIA / 'VIDEOMME_BUCKET_MEDIA_COMPLETE.json'; raw = encode(result)
    store.put_verified(META_PREFIX + path.name, raw); put_file(path, raw)
    print(json.dumps({'event': 'VIDEOMME_BUCKET_MEDIA_COMPLETE', 'source_count': 899, 'frame_count': 7192,
                      'receipt_path': str(path), 'receipt_sha256': digest(raw), 'cache_bytes': receipt['cache_bytes'],
                      'formal_evaluation_ready': False}), flush=True)


def execute(cfg, directory, sources, existing, png, metadata, audit):
    os.nice(10)
    lock = (MEDIA / '.acquisition.lock').open('rb'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    current = sum(tree_bytes(root) for root in (P0, P1)); budget = FrameBudget(cfg['frame_cap_bytes'])
    projected = current + max(0, cfg['frame_cap_bytes'] - budget.used) + 100_000_000 + 2_000_000_000
    require(projected < LIMIT, 'combined_root_capacity_failed')
    print(json.dumps({'event': 'capacity_admitted', 'combined_root_bytes': current, 'existing_png_bytes': budget.used,
                      'total_png_cap': budget.cap, 'projected_bytes': projected, 'workers': cfg['workers']}), flush=True)
    store = OSSStore(cfg['runtime_library_path'])
    catalog = store.get(BASE + 'hf-archives/ARCHIVE_CATALOG_COMPLETE.json')
    require(catalog is not None and digest(catalog) == audit['catalog_sha256'], 'accepted_archive_catalog_changed')
    pinned_entries = {Path(r['source_path']).name: r for r in metadata['entries'] if r['category'] == 'frame_source_receipt'}
    for source in sorted(existing):
        path = MEDIA / 'entries' / (source.removeprefix('youtube:') + '.json')
        require(path.name in pinned_entries and sha(path) == pinned_entries[path.name]['sha256'], 'existing547_receipt_changed')
        verify_entry(json.loads(path.read_bytes()))
    print(json.dumps({'event': 'existing547_local_sha_rgb_verified', 'sources': 547, 'frames': 4376}), flush=True)
    checked = {r['asset_id'].split(':', 1)[1] + '.zip': r for r in audit['checked_archives'] if r['asset_id'].startswith('videomme:')}
    work = []
    for spec in directory:
        validate_archive(store, checked[spec['path']], spec)
        path = MEDIA / 'archives' / (spec['path'] + '.verified.json')
        if not path.exists():
            put_file(path, encode({'archive_path': spec['path'], 'archive_sha256': spec['sha256'], 'size_bytes': spec['size_bytes'],
                       'repo_id': 'lmms-lab/Video-MME', 'revision': REVISION, 'verified': True,
                       'verification': 'independently_accepted_OSS_archive_receipt_and_ETag',
                       'oss_receipt_sha256': checked[spec['path']]['receipt_sha256']}))
        store.put_verified(META_PREFIX + 'archives/' + path.name, path.read_bytes())
        for member in spec['members']:
            if 'youtube:' + Path(member['path']).stem in sources - existing: work.append((checked[spec['path']], spec, member))
    require(len(work) == 352, 'remaining_member_mapping_changed')
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        futures = [pool.submit(decode_one, store, checked, spec, member, budget) for checked, spec, member in work]
        try:
            for future in concurrent.futures.as_completed(futures): future.result()
        except BaseException:
            for future in futures: future.cancel()
            raise
    finalize(store, directory, sources, cfg, audit)
    lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--config-sha256', required=True); p.add_argument('--execute', action='store_true'); args = p.parse_args()
    values = load_inputs(args.config, args.config_sha256)
    if not args.execute:
        print(json.dumps({'status': 'PREPARED_NO_MUTATIONS', 'sources': 899, 'existing': 547, 'remaining': 352})); return
    execute(*values)


if __name__ == '__main__':
    try: main()
    except Exception as error:
        code = str(error) if isinstance(error, SafeError) else 'operation_' + type(error).__name__
        print(json.dumps({'event': 'VIDEOMME_OSS_DECODE_FAILED', 'error_code': code, 'formal_evaluation_ready': False}), flush=True)
        sys.exit(2)
