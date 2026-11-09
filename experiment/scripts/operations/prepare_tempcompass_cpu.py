#!/usr/bin/env python3
"""Prepare the frozen TempCompass MC set from an already accepted OSS ZIP.

One CPU worker, complete sequential decode, native RGB uniform8 PNGs. The ZIP
and one MP4 member remain in memory. No source download, GPU work, deletion,
old-file replacement, automatic retry, or formal GPU admission is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import resource
import stat
import sys
import zipfile

P0 = Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM')
P1 = Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1')
OUTPUT = P1 / 'benchmark-media-tempcompass-20261021-a1'
PREFIX = 'jinhaopeng/Paper/DeepProbe-VLM/benchmark-cache/20261021-tempcompass-uniform8-v1/'
BUCKET = 'ss-devbox-stream'
REVISION = 'c8a67d88a3bc6fd4b2f9ae2f9e112668fbe05722'
SOURCE_REVISION = 'e1b463166400633e6061962d890a9ae85db29f70'
ARCHIVE_KEY = ('jinhaopeng/Paper/DeepProbe-VLM/benchmark-cache/20261017-tempcompass-v1/'
               'hf-archives/lmms-eval/TempCompass/' + REVISION + '/tempcompass_videos.zip')
ARCHIVE_SHA = 'fb7c8a2b742a3d10ae1bd1a6183fb602edb346a4c7fd91c631dba46b753e0f0b'
ARCHIVE_BYTES = 418421516
MAX_MEMBER = 5331577
FRAME_CAP = 4_000_000_000
CONTROL_CAP = 100_000_000
MARGIN = 2_000_000_000
ROOT_LIMIT = 99_000_000_000
RSS_LIMIT = 1_500_000_000
CONTROL_OBJECT_CAP = 31_999_999
CONTENT_PAIR_CAP = 10_000
CONTENT_MATCH_CAP = 50_000
DECODE_MEMORY_INTERVAL = 32
SEARCH_MEMORY_INTERVAL = 256
MAX_PIXELS = 3840 * 2160
MAX_FRAME_COUNT = 1_000_000
BLOCK = 8 * 1024 * 1024
TRAIN_GATE_SHA = '1c08b9e18d7e9f2bcb9664ae81225603c94585ffabfcebe5b7b9bd01fc9ca66c'
PINS = {
    'rows': '803d1b71e85092b9d7118b5679794a2cc1c2203de170d2d8cdcd06b270eb3d96',
    'source_map': '1537ee59ba74b4add2c30e586d5ff72b0ea2913882db798fdea4c69117405738',
    'meta': '72144a7cb5aba97d12c3148ec734f7c8f27a04f6b3a7396ef41003a0a7178419',
    'archive_audit': '33b23a24d01e071a47938ffe975a13e529f630656afa6df363fbedc0a0482c3e',
    'archive_terminal': '4b649527063e084d3117499787ddc4f522304e3a263750b032700b7b6448b05f',
    'git_binding': 'bc5d203c91738d7a9c6d5992c2735a6b05b4c80d23186d7968b3b9134b29e97f',
    'training_receipt': 'a6551ff839f295e39cfddd89b78cc995d61a75d5dd99b6fc4e39ca143dd2053c',
    'training_fingerprints': '85248d73d8d48c5e66352694d76f0229bb8b482cae387cce5de6d2d7b816a4b8',
    'oss_helper': '444187e3155686f8f82344ccc1e798d595b2258472f6afee6b9f3c8bea087c34',
    'content_helper': '4658b04be44f7ffa47a6ca3d55ed9a9c1e013d7a8e7822f16bb87ae848496d43',
    'sampling_source': '43e90fc8dd4e453fe59ae20be1ddf9aae48a28ab9c7e33b832fa0ed4ab2a253e',
    'manifest_source': '52bd9c6a423230d323f606624ff1d81d0af704e982678346a90f0d0fbd9a9e6e',
    'package_init': 'e310e6f3001b127cd6b14c9fc9a1482e6d1d450870fcfc83ac2b4b7affa0b92d',
    'official_scorer': 'ccffa4f717a3e2a29435b2bbe440109bf2def589cea73fde9fd44fd262fb379a',
    'official_score_summary': '514f728e6a4e273db7db2c3feab316c129ebcc08901e62352d5af71efb4e964c',
    'score_adapter': 'f35b543e76400236110964fbe6bb4c2f70d485fb501f7751b5e6ce7544d3f290',
}
DIM_COUNTS = {'action': 338, 'direction': 335, 'speed': 317, 'order': 302, 'attribute_change': 288}
OWN_OUTPUT = False


class PreparationError(ValueError):
    pass


def require(ok, code):
    if not ok:
        raise PreparationError(code)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def encode(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + '\n').encode()


def bounded_encode(value, maximum):
    """Bound control bytes while encoding; never build an oversized full string."""
    require(type(maximum) is int and maximum >= 1, 'control_capacity_failed')
    encoder = json.JSONEncoder(sort_keys=True, ensure_ascii=False, indent=2)
    with io.BytesIO() as output:
        for index, text in enumerate(encoder.iterencode(value)):
            if index % SEARCH_MEMORY_INTERVAL == 0:
                memory_guard()
            # Even a single large string chunk is converted in bounded slices.
            for start in range(0, len(text), 8192):
                raw = text[start:start + 8192].encode()
                require(output.tell() + len(raw) + 1 <= maximum, 'control_capacity_failed')
                output.write(raw)
        output.write(b'\n')
        memory_guard()
        return output.getvalue()


def sha(path):
    h = hashlib.sha256()
    with safe(path).open('rb') as stream:
        for block in iter(lambda: stream.read(BLOCK), b''):
            h.update(block)
    return h.hexdigest()


def safe(path):
    p = Path(path)
    require(p.is_absolute() and '..' not in p.parts and not any(x.is_symlink() for x in (p, *p.parents)), 'unsafe_path')
    return p


def uniform_indices(total):
    require(type(total) is int and 0 < total <= MAX_FRAME_COUNT, 'invalid_frame_count')
    return [round(i * (total - 1) / 7) for i in range(8)]


def source_components(entries):
    """Every clip sharing any underlying source belongs to one connected group."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    seen = set()
    for row in entries:
        require(row['video_id'] not in seen, 'duplicate_clip_id')
        seen.add(row['video_id'])
        ids = row['underlying_source_video_ids']
        require(isinstance(ids, list) and 1 <= len(ids) <= 2 and len(set(ids)) == len(ids), 'invalid_underlying_sources')
        require(all(re.fullmatch(r'shutterstock:[0-9]+', s) for s in ids), 'unresolved_underlying_source')
        for source in ids:
            parent[find(source)] = find(ids[0])
    groups = {}
    for source in sorted(parent):
        groups.setdefault(find(source), []).append(source)
    return {row['video_id']: {
        'source_video_id': 'tempcompass-source-component:' + digest(canonical(groups[find(row['underlying_source_video_ids'][0])])),
        'source_component_ids': groups[find(row['underlying_source_video_ids'][0])],
    } for row in entries}


def build_records(rows, source_map, meta):
    entries = source_map['entries']
    by_clip = {x['video_id']: x for x in entries}
    components = source_components(entries)
    records, seen, counts = [], set(), Counter()
    for raw in rows:
        clip, dim, index = raw['video_id'], raw['dim'], raw['index_within_dimension']
        require(clip in by_clip and dim in DIM_COUNTS and type(index) is int and index >= 0, 'invalid_qa_identity')
        identity = (clip, dim, index)
        require(identity not in seen, 'duplicate_qa_identity')
        seen.add(identity)
        lines = raw['question'].split('\n')
        require(3 <= len(lines) <= 5 and bool(lines[0].strip()), 'invalid_question_layout')
        choices = lines[1:]
        # Frozen upstream includes " D. text", "C.text" and trailing spaces.
        # Retain original strings and exact prefix spans; never change labels.
        parsed = [re.fullmatch(r'([ \t]*[A-D]\. ?)(.*)', x) for x in choices]
        require(all(m is not None and m.group(1).lstrip()[0] == chr(65+i) and m.group(2).strip()
                    for i, m in enumerate(parsed)), 'choice_order_changed')
        require(raw['answer'] in choices, 'gold_answer_not_exact_choice')
        require(dim in by_clip[clip]['dimensions'], 'source_dimension_changed')
        fine = 'order' if dim == 'order' else meta[clip]['eval_dim'][dim]['type']
        item = {
            'sample_id': f'tempcompass:{clip}:{dim}:{index}',
            **components[clip], 'underlying_source_video_ids': by_clip[clip]['underlying_source_video_ids'],
            'processed_media_identity': by_clip[clip]['processed_media_identity'],
            'video_id': clip, 'dataset': 'tempcompass', 'split': 'test',
            'task_type': dim, 'fine_grained_task_type': fine,
            'question': lines[0], 'choices': [m.group(2) for m in parsed], 'answer': raw['answer'][0],
            'original_choices': choices, 'original_choice_prefixes': [m.group(1) for m in parsed],
            'original_question': raw['question'], 'original_answer': raw['answer'],
            'index_within_dimension': index, 'annotation_row_sha256': digest(canonical(raw)),
            'image_paths': [f'frames/{clip}/{slot:02d}.png' for slot in range(8)],
        }
        require(item['question'] + '\n' + '\n'.join(prefix + text for prefix, text in zip(item['original_choice_prefixes'], item['choices'])) == raw['question'], 'question_roundtrip_failed')
        records.append(item)
        counts[clip] += 1
    require(set(counts) == set(by_clip), 'question_clip_set_changed')
    require(all(counts[c] == by_clip[c]['qa_count'] for c in counts), 'per_clip_qa_count_changed')
    for clip in counts:
        for dim in by_clip[clip]['dimensions']:
            indices = [r['index_within_dimension'] for r in rows if r['video_id'] == clip and r['dim'] == dim]
            require(indices == list(range(len(indices))) and indices, 'dimension_question_order_changed')
    return records


def validate_bundle(inputs):
    rows, mapping, audit = inputs['rows'], inputs['source_map'], inputs['archive_audit']
    require(mapping['github_commit'] == SOURCE_REVISION and mapping['qa_count'] == len(rows) == 1580, 'frozen_question_count_changed')
    require(Counter(r['dim'] for r in rows) == DIM_COUNTS, 'dimension_counts_changed')
    require(len(mapping['entries']) == mapping['clip_count'] == 410 and mapping['underlying_source_count'] == 207, 'frozen_source_count_changed')
    sources = {s for e in mapping['entries'] for s in e['underlying_source_video_ids']}
    require(len(sources) == 207, 'source_set_changed')
    require(audit['status'] == 'ARCHIVE_AND_MC_MEMBERS_VERIFIED' and audit['full_oss_get_sha256_passed'] is True
            and audit['every_zip_member_crc32_and_sha256_passed'] is True and audit['all_410_selected_ids_present'] is True
            and audit['extra_video_ids'] == audit['nonvideo_members'] == [], 'archive_audit_not_accepted')
    require(audit['archive']['key'] == ARCHIVE_KEY and audit['archive']['sha256'] == ARCHIVE_SHA
            and audit['archive']['size_bytes'] == ARCHIVE_BYTES and len(audit['members']) == 410, 'archive_binding_changed')
    by_clip = {x['video_id']: x for x in mapping['entries']}
    require(len({x['name'] for x in audit['members']}) == 410
            and {x['video_id'] for x in audit['members']} == set(by_clip), 'archive_member_set_changed')
    for member in audit['members']:
        require(member['name'] == 'videos/' + by_clip[member['video_id']]['expected_member_basename']
                and member['name'] == 'videos/' + member['video_id'] + '.mp4'
                and member['selected_for_mc'] is True and 0 < member['bytes'] <= MAX_MEMBER, 'member_binding_changed')
    terminal = inputs['archive_terminal']
    require(terminal['exit_code'] == 0 and terminal['status'] == 'TEMPCOMPASS_ARCHIVE_CACHED_VERIFIED'
            and terminal['archive_sha256'] == ARCHIVE_SHA and terminal['archive_bytes'] == ARCHIVE_BYTES
            and audit['terminal_sha256'] == PINS['archive_terminal'], 'archive_terminal_changed')
    binding = inputs['git_binding']
    require(binding['revision'] == REVISION and binding['archive']['sha256'] == ARCHIVE_SHA
            and binding['archive']['bytes'] == ARCHIVE_BYTES and binding['git_fsck_exit_code'] == 0, 'git_binding_changed')
    return build_records(rows, mapping, inputs['meta'])


def checked_zip_infos(bundle, audited):
    infos = bundle.infolist()
    require(len({i.filename for i in infos}) == len(infos), 'duplicate_zip_member')
    files = {}
    for info in infos:
        path = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        require(not path.is_absolute() and '..' not in path.parts and '\\' not in info.filename
                and not info.flag_bits & 1 and not stat.S_ISLNK(mode), 'unsafe_zip_member')
        if info.is_dir():
            require(info.filename == 'videos/', 'unexpected_zip_directory')
            continue
        require(stat.S_IFMT(mode) in (0, stat.S_IFREG), 'nonregular_zip_member')
        files[info.filename] = info
    require(set(files) == {m['name'] for m in audited}, 'zip_member_set_changed')
    for member in audited:
        i = files[member['name']]
        require((i.file_size, i.compress_size, i.CRC, i.header_offset, i.compress_type)
                == tuple(member[k] for k in ('bytes', 'compressed_bytes', 'crc32', 'header_offset', 'compression')), 'zip_member_metadata_changed')
    return files


def read_member(bundle, info, member):
    require(0 < info.file_size <= MAX_MEMBER, 'member_memory_cap')
    with bundle.open(info) as stream:
        raw = stream.read(info.file_size + 1)
    require(len(raw) == member['bytes'] and digest(raw) == member['sha256'], 'member_sha_changed')
    return io.BytesIO(raw)


def decode_uniform8(source, av_module=None):
    if av_module is None:
        import av as av_module
    count, dimensions, selected, indices = 0, None, {}, None
    for pass_index in range(2):
        memory_guard()
        source.seek(0)
        with av_module.open(source, options={'err_detect': 'explode'}) as container:
            require(len(container.streams.video) == 1, 'ambiguous_video_streams')
            stream = container.streams.video[0]
            stream.thread_count = 1
            require(0 < stream.width * stream.height <= MAX_PIXELS, 'declared_pixel_cap')
            observed = 0
            for i, frame in enumerate(container.decode(video=0)):
                if i % DECODE_MEMORY_INTERVAL == 0:
                    memory_guard()
                observed += 1
                require(observed <= MAX_FRAME_COUNT and 0 < frame.width * frame.height <= MAX_PIXELS, 'decoded_frame_cap')
                current = (frame.width, frame.height)
                if dimensions is None:
                    dimensions = current
                require(current == dimensions, 'variable_frame_dimensions')
                if pass_index == 1 and i in wanted:
                    selected[i] = frame.to_image().convert('RGB')
                    memory_guard()
            if pass_index == 0:
                count = observed
                indices = uniform_indices(count)
                wanted = set(indices)
            else:
                require(observed == count and set(selected) == wanted, 'complete_decode_count_mismatch')
            memory_guard()
    return selected, count, indices, dimensions


def bounded_candidate_pairs(training_rows, benchmark_rows, policy):
    """The pinned exact-recall search with fail-closed result bounds.

    Crossing either result cap raises rather than returning a truncated list.
    Frames/fingerprints already saved remain available for a reviewed next step.
    """
    require(policy['phash_max_hamming'] == 6 and policy['dhash_max_hamming'] == 10,
            'content_search_policy_changed')
    flat, buckets = [], defaultdict(list)
    for video in training_rows:
        for slot, frame in enumerate(video['frames']):
            if not frame['informative']:
                continue
            if len(flat) % SEARCH_MEMORY_INTERVAL == 0:
                memory_guard()
            index = len(flat)
            phash, dhash = int(frame['phash'], 16), int(frame['dhash'], 16)
            flat.append((video, slot, frame, phash, dhash))
            for block in range(4):
                buckets[block, (phash >> (16 * block)) & 65535].append(index)
    result, matches = {}, 0
    for benchmark in benchmark_rows:
        for slot, frame in enumerate(benchmark['frames']):
            memory_guard()
            if not frame['informative']:
                continue
            phash, dhash = int(frame['phash'], 16), int(frame['dhash'], 16)
            candidates = set()
            for block in range(4):
                value = (phash >> (16 * block)) & 65535
                for mask in (0, *(1 << bit for bit in range(16))):
                    candidates.update(buckets.get((block, value ^ mask), ()))
            for checked, index in enumerate(sorted(candidates)):
                if checked % SEARCH_MEMORY_INTERVAL == 0:
                    memory_guard()
                train, train_slot, train_frame, train_phash, train_dhash = flat[index]
                pd = (phash ^ train_phash).bit_count()
                dd = (dhash ^ train_dhash).bit_count()
                if pd > policy['phash_max_hamming'] or dd > policy['dhash_max_hamming']:
                    continue
                key = train['relative_path'], benchmark['source_video_id']
                require(matches < CONTENT_MATCH_CAP, 'content_frame_match_cap_exceeded')
                if key not in result:
                    require(len(result) < CONTENT_PAIR_CAP, 'content_pair_cap_exceeded')
                    result[key] = {'training_path': key[0], 'training_source_video_id': train['source_video_id'],
                        'benchmark_source_video_id': key[1], 'classification': 'suspected_overlap_requires_review',
                        'frame_matches': []}
                result[key]['frame_matches'].append({'training_slot': train_slot, 'benchmark_slot': slot,
                    'phash_hamming': pd, 'dhash_hamming': dd,
                    'same_decoded_rgb': train_frame['rgb_sha256'] == frame['rgb_sha256']})
                matches += 1
    memory_guard()
    return [result[key] for key in sorted(result)]


def load_module(pin, name):
    require(sha(pin['path']) == pin['sha256'], 'module_pin_changed')
    spec = importlib.util.spec_from_file_location(name, pin['path'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_inputs(cfg):
    require(set(cfg['inputs']) == set(PINS), 'input_pin_set_changed')
    result = {}
    for name, expected in PINS.items():
        pin = cfg['inputs'][name]
        require(set(pin) == {'path', 'sha256'} and pin['sha256'] == expected and sha(pin['path']) == expected, 'input_pin_changed:' + name)
        if name == 'training_fingerprints':
            result[name] = [json.loads(s) for s in safe(pin['path']).read_text().splitlines()]
        elif name not in ('oss_helper', 'content_helper', 'sampling_source', 'manifest_source', 'package_init', 'official_scorer', 'official_score_summary', 'score_adapter'):
            result[name] = json.loads(safe(pin['path']).read_text())
    return result


def validate_training(cfg, inputs, scanner):
    gate = scanner.load_gate(P1 / 'formal-source-final-20261009-v1', TRAIN_GATE_SHA)
    rows, receipt = inputs['training_fingerprints'], inputs['training_receipt']
    require(receipt['passed'] is True and receipt['errors'] == [] and receipt['file_count'] == len(rows) == 3979
            and receipt['fingerprints_sha256'] == PINS['training_fingerprints'] and receipt['policy'] == scanner.POLICY
            and receipt['original_formal_ready_sha256'] == TRAIN_GATE_SHA and receipt['binding'] == gate['binding'], 'training_fingerprint_binding_changed')
    keys = ('relative_path', 'source_video_id', 'size_bytes', 'sha256')
    inventory = [{k: row[k] for k in keys} for row in rows]
    require(inventory == gate['training_inventory']['entries']
            and digest(canonical(inventory)) == receipt['training_entries_sha256'], 'training_inventory_changed')
    media = P0 / 'media/llava-p0'
    for row in rows:
        path = safe(media / row['relative_path'])
        require(path.is_relative_to(media) and path.is_file() and path.stat().st_size == row['size_bytes']
                and sha(path) == row['sha256'], 'training_media_changed')
        require(row['decode_complete'] is True and len(row['frames']) == 8
                and row['selected_indices'] == uniform_indices(row['decoded_frame_count']), 'training_decode_binding_changed')
    return gate


def memory_guard():
    # A periodically observed peak-RSS budget, not an OS-enforced memory limit.
    require(sys.platform == 'linux', 'worker_requires_remote_linux')
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= RSS_LIMIT, 'rss_cap_exceeded')


class Artifacts:
    def __init__(self, store, helper):
        self.store, self.helper = store, helper
        self.png_bytes, self.control_bytes, self.objects = 0, 0, {}

    def capacity(self):
        total = sum(self.helper.tree_bytes(root) for root in (P0, P1))
        projected = total + FRAME_CAP - self.png_bytes + CONTROL_CAP - self.control_bytes + MARGIN
        require(projected < ROOT_LIMIT, 'dual_root_capacity_failed')
        memory_guard()
        return {'current_bytes': total, 'projected_bytes': projected, 'limit_bytes': ROOT_LIMIT}

    def put(self, relative, raw, frame=False):
        path = safe(OUTPUT / relative)
        require(path.is_relative_to(OUTPUT) and not path.exists(), 'output_collision')
        if frame:
            require(self.png_bytes + len(raw) <= FRAME_CAP and len(raw) < 32_000_000, 'png_capacity_failed')
        else:
            require(self.control_bytes + len(raw) <= CONTROL_CAP and len(raw) <= CONTROL_OBJECT_CAP, 'control_capacity_failed')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        self.store.put_verified(PREFIX + relative, raw)
        self.objects[PREFIX + relative] = {'bytes': len(raw), 'sha256': digest(raw)}
        if frame:
            self.png_bytes += len(raw)
        else:
            self.control_bytes += len(raw)
        return self.objects[PREFIX + relative]

    def json(self, relative, value):
        return self.put(relative, bounded_encode(value, min(CONTROL_OBJECT_CAP, CONTROL_CAP - self.control_bytes)))

    def verify_exact(self):
        require(set(self.store.keys(PREFIX)) == set(self.objects), 'output_oss_key_set_changed')
        for key, expected in self.objects.items():
            head = self.store.head(key)
            require(head.content_length == expected['bytes']
                    and head.headers.get('x-oss-meta-sha256') == expected['sha256'], 'output_oss_head_changed')


def fetch_archive(store, audit):
    archive = audit['archive']
    head = store.head(ARCHIVE_KEY)
    require(head.content_length == ARCHIVE_BYTES and head.etag.strip('"') == archive['etag'].strip('"'), 'archive_head_changed')
    raw = io.BytesIO()
    h, size = hashlib.sha256(), 0
    try:
        with store.bucket.get_object(ARCHIVE_KEY, headers={'If-Match': head.etag}) as response:
            for block in iter(lambda: response.read(BLOCK), b''):
                size += len(block)
                require(size <= ARCHIVE_BYTES, 'archive_memory_cap')
                raw.write(block)
                h.update(block)
                memory_guard()
        require(size == ARCHIVE_BYTES and h.hexdigest() == ARCHIVE_SHA, 'archive_full_sha_changed')
        raw.seek(0)
        return raw
    except BaseException:
        raw.close()
        raise


def prepare(cfg, cfg_sha):
    global OWN_OUTPUT
    require(cfg['schema_version'] == 1 and cfg['output_dir'] == str(OUTPUT) and cfg['oss_prefix'] == PREFIX
            and cfg['bucket'] == BUCKET and cfg['workers'] == 1
            and cfg['limits'] == {'frame_bytes': FRAME_CAP, 'control_bytes': CONTROL_CAP, 'margin_bytes': MARGIN,
                                  'dual_root_bytes': ROOT_LIMIT, 'rss_bytes': RSS_LIMIT, 'max_frame_pixels': MAX_PIXELS,
                                  'control_object_bytes': CONTROL_OBJECT_CAP, 'content_pairs': CONTENT_PAIR_CAP,
                                  'content_frame_matches': CONTENT_MATCH_CAP}, 'runtime_config_changed')
    require(cfg['worker_sha256'] == sha(__file__), 'worker_sha_changed')
    inputs = read_inputs(cfg)
    records = validate_bundle(inputs)
    code_root = Path(cfg['inputs']['content_helper']['path']).parents[2]
    for key, relative in [('sampling_source', 'data.py'), ('manifest_source', 'manifest.py'), ('package_init', '__init__.py')]:
        require(Path(cfg['inputs'][key]['path']) == code_root / 'src/deepprobe_vlm' / relative, 'scanner_dependency_path_changed')
    helper = load_module(cfg['inputs']['oss_helper'], 'tempcompass_oss_helper')
    scanner = load_module(cfg['inputs']['content_helper'], 'tempcompass_content_helper')
    require(scanner.sha256_file(scanner.ROOT / 'src/deepprobe_vlm/data.py') == PINS['sampling_source']
            and scanner.sha256_file(scanner.ROOT / 'src/deepprobe_vlm/manifest.py') == PINS['manifest_source']
            and scanner.sha256_file(scanner.ROOT / 'src/deepprobe_vlm/__init__.py') == PINS['package_init'], 'scanner_dependency_changed')
    require(cfg['runtime_library_path'] == str(P1 / 'benchmark-videomme-oss-decode-20261013/python-libs'), 'runtime_library_path_changed')
    store = helper.OSSStore(cfg['runtime_library_path'])
    environment = scanner.versions()
    reference_environment = inputs['training_receipt']['environment']
    require(all(environment[k] == reference_environment[k] for k in ('numpy', 'pillow', 'pyav', 'libav')),
            'fingerprint_runtime_environment_changed')
    require(not list(store.keys(PREFIX)), 'new_oss_prefix_not_empty')
    require(not safe(OUTPUT).exists(), 'new_output_root_exists')
    artifacts = Artifacts(store, helper)
    capacity = artifacts.capacity()
    validate_training(cfg, inputs, scanner)
    OUTPUT.mkdir()
    OWN_OUTPUT = True
    artifacts.json('STARTED.json', {'started_utc': datetime.now(timezone.utc).isoformat(), 'config_sha256': cfg_sha,
                                  'worker_sha256': sha(__file__), 'capacity': capacity, 'environment': environment,
                                  'automatic_retry': False})
    artifacts.json('FROZEN_CONFIG.json', cfg)
    training_rows = inputs['training_fingerprints']
    train_sources = {x['source_video_id'] for x in training_rows}
    test_sources = {s for row in inputs['source_map']['entries'] for s in row['underlying_source_video_ids']}
    intersection = sorted(train_sources & test_sources)
    artifacts.json('SOURCE_ISOLATION.json', {'passed': not intersection, 'training_source_count': len(train_sources),
        'underlying_benchmark_source_count': len(test_sources), 'processed_clip_count': 410,
        'connected_source_components': len({r['source_video_id'] for r in records}), 'identity_intersection': intersection,
        'source_map_sha256': PINS['source_map'], 'training_fingerprints_sha256': PINS['training_fingerprints'],
        'training_media_full_sha_revalidated': True,
        'limitation': 'Exact declared source IDs only. Cross-platform reposts require the separate content audit.'})
    require(not intersection, 'declared_training_source_overlap')
    by_clip = {e['video_id']: e for e in inputs['source_map']['entries']}
    entries, fingerprints = [], []
    with fetch_archive(store, inputs['archive_audit']) as archive, zipfile.ZipFile(archive) as bundle:
        infos = checked_zip_infos(bundle, inputs['archive_audit']['members'])
        for i, member in enumerate(sorted(inputs['archive_audit']['members'], key=lambda m: m['video_id'])):
            if i % 10 == 0:
                artifacts.capacity()
            memory_guard()
            clip = member['video_id']
            with read_member(bundle, infos[member['name']], member) as source:
                images, count, indices, dimensions = decode_uniform8(source)
            frame_rows, hashes = [], []
            for slot, index in enumerate(indices):
                image = images[index]
                with io.BytesIO() as buffer:
                    image.save(buffer, format='PNG')
                    raw = buffer.getvalue()
                relative = f'frames/{clip}/{slot:02d}.png'
                proof = artifacts.put(relative, raw, frame=True)
                frame_rows.append({'relative_path': relative, 'source_frame_index': index, 'size_bytes': proof['bytes'],
                    'sha256': proof['sha256'], 'rgb_sha256': digest(image.tobytes()), 'width': dimensions[0], 'height': dimensions[1]})
                hashes.append(scanner.fingerprint_image(image))
                memory_guard()
            entry = {'video_id': clip, 'processed_media_identity': 'tempcompass:' + clip,
                'underlying_source_video_ids': by_clip[clip]['underlying_source_video_ids'], 'archive_key': ARCHIVE_KEY,
                'archive_sha256': ARCHIVE_SHA, 'archive_etag': inputs['archive_audit']['archive']['etag'],
                'member_path': member['name'], 'original_size_bytes': member['bytes'], 'original_sha256': member['sha256'],
                'decode_complete': True, 'full_decode_passes': 2, 'decoded_frame_count': count,
                'selected_indices': indices, 'frames': frame_rows, 'image_paths': [f['relative_path'] for f in frame_rows]}
            artifacts.json(f'entries/{clip}.json', entry)
            entries.append(entry)
            fingerprints.append({'source_video_id': 'tempcompass:' + clip, 'underlying_source_video_ids': by_clip[clip]['underlying_source_video_ids'], 'frames': hashes})
            for image in images.values():
                image.close()
            del images
            print(json.dumps({'event': 'clip_complete', 'completed': i+1, 'total': 410, 'png_bytes': artifacts.png_bytes}), flush=True)
    artifacts.capacity()
    cache = {'schema_version': 1, 'kind': 'tempcompass_uniform8_frame_cache', 'complete': True,
        'revision': REVISION, 'source_revision': SOURCE_REVISION, 'source_map_sha256': PINS['source_map'],
        'archive_member_audit_sha256': PINS['archive_audit'], 'source_count': 410, 'underlying_source_count': 207,
        'frame_count': 3280, 'num_frames': 8, 'sampling': 'deepprobe_vlm.data.uniform_indices', 'frames_resized': False,
        'png_bytes': artifacts.png_bytes, 'all_png_oss_get_sha256_verified': True, 'entries': entries, 'failures': []}
    cache_proof = artifacts.json('FRAME_CACHE_RECEIPT.json', cache)
    artifacts.json('benchmark_fingerprints.json', fingerprints)
    train_hashes = {}
    for row in training_rows:
        train_hashes.setdefault(row['sha256'], []).append(row['relative_path'])
    exact = [{'training_path': path, 'video_id': m['video_id'], 'sha256': m['sha256']}
             for m in inputs['archive_audit']['members'] for path in train_hashes.get(m['sha256'], [])]
    candidates = bounded_candidate_pairs(training_rows, fingerprints, scanner.POLICY)
    uninformative = ([{'training_path': x['relative_path']} for x in training_rows if not any(f['informative'] for f in x['frames'])]
                     + [{'video_id': x['source_video_id']} for x in fingerprints if not any(f['informative'] for f in x['frames'])])
    content = {'schema_version': 1, 'kind': 'tempcompass_training_content_audit', 'scan_complete': True,
        'passed': not exact and not candidates and not uninformative, 'training_file_count': 3979,
        'benchmark_clip_count': 410, 'underlying_source_count': 207, 'frame_cache_sha256': cache_proof['sha256'],
        'training_receipt_sha256': PINS['training_receipt'], 'training_fingerprints_sha256': PINS['training_fingerprints'],
        'source_map_sha256': PINS['source_map'], 'policy': scanner.POLICY,
        'search_completed_without_truncation': True,
        'resource_limits': {'max_pairs': CONTENT_PAIR_CAP, 'max_frame_matches': CONTENT_MATCH_CAP},
        'exact_duplicate_pairs': exact, 'perceptual_duplicate_pairs': candidates, 'uninformative_sources': uninformative,
        'errors': [], 'unsampled_frames_separation_claimed': False}
    artifacts.json('CONTENT_AUDIT.json', content)
    from deepprobe_vlm.manifest import validate_records
    validate_records(records, OUTPUT, check_media=True)
    manifest_raw = b''.join(canonical(row) + b'\n' for row in records)
    artifacts.put('manifest.jsonl', manifest_raw)
    artifacts.json('MCQ_PROTOCOL.json', {'protocol_id': 'tempcompass-mc-1580-uniform8-noaudio-v1',
        'row_count': 1580, 'clip_count': 410, 'underlying_source_count': 207, 'dimension_counts': DIM_COUNTS,
        'annotation_sha256': PINS['rows'], 'meta_sha256': PINS['meta'], 'source_map_sha256': PINS['source_map'],
        'manifest_sha256': digest(manifest_raw), 'num_frames': 8, 'pool_grid': 4, 'audio': False, 'subtitles': False,
        'question_and_option_text_preserved': True, 'original_label_spacing_preserved_in_provenance': True,
        'canonical_prompt_option_labels': 'A. / B. / C. / D.; source label whitespace alone is normalized',
        'answer_in_model_prompt': False, 'greedy_max_new_tokens': 16, 'official_scorer_mode': 'disable_llm',
        'official_scorer_sha256': PINS['official_scorer'], 'official_summary_sha256': PINS['official_score_summary'],
        'score_adapter_sha256': PINS['score_adapter'],
        'primary_overall_metric': 'QA micro accuracy; official avg weights all 1580 questions equally',
        'additional_metrics': ['per-dimension accuracy and counts', 'per-fine-grained-dimension accuracy and counts',
                               'dimension macro accuracy separately labeled', 'rule-match success rate'],
        'paired_statistics_cluster_key': 'source_video_id (connected components of all underlying source dependencies)',
        'formal_evaluation_ready': False, 'further_required': ['independent CPU data acceptance', 'model and token-budget binding', 'GPU admission']})
    artifacts.verify_exact()
    head = store.head(ARCHIVE_KEY)
    require(head.etag.strip('"') == inputs['archive_audit']['archive']['etag'].strip('"') and head.content_length == ARCHIVE_BYTES, 'archive_changed_during_decode')
    final = {'status': 'CPU_DATA_PREPARED' if content['passed'] else 'CONTENT_REVIEW_REQUIRED',
        'finished_utc': datetime.now(timezone.utc).isoformat(), 'tempcompass_frame_cache_ready': True,
        'cpu_data_ready': content['passed'], 'formal_evaluation_ready': False, 'gpu_submitted': False,
        'config_sha256': cfg_sha, 'worker_sha256': sha(__file__), 'png_count': 3280, 'clip_count': 410,
        'qa_count': 1580, 'png_bytes': artifacts.png_bytes, 'control_bytes_before_terminal': artifacts.control_bytes,
        'content_candidate_pairs': len(candidates), 'exact_duplicate_pairs': len(exact),
        'archive_full_oss_get_sha256_revalidated': True, 'new_local_zip_or_mp4_written': False,
        'objects_before_terminal': artifacts.objects, 'capacity': artifacts.capacity()}
    artifacts.json('TERMINAL.json', final)
    artifacts.verify_exact()
    print(json.dumps({k: final[k] for k in ('status', 'clip_count', 'qa_count', 'png_count', 'cpu_data_ready')}), flush=True)
    return 0 if content['passed'] else 2


def main(argv=None):
    global OWN_OUTPUT
    OWN_OUTPUT = False
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--config-sha256', required=True)
    args = parser.parse_args(argv)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'cpu_environment_required')
    require(sha(args.config) == args.config_sha256, 'config_sha_changed')
    cfg = json.loads(safe(args.config).read_text())
    with safe(P1 / '.tempcompass-preparation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return prepare(cfg, args.config_sha256)
        except Exception as exc:
            # Bounded error class/code only; SDK exceptions may contain credentials.
            failure = {'status': 'FAILED', 'error_type': type(exc).__name__,
                       'code': str(exc) if isinstance(exc, PreparationError) else 'external_operation_failed',
                       'finished_utc': datetime.now(timezone.utc).isoformat(), 'automatic_retry': False}
            if OWN_OUTPUT and OUTPUT.is_dir() and not (OUTPUT / 'FAILED.json').exists():
                with (OUTPUT / 'FAILED.json').open('xb') as f:
                    f.write(encode(failure))
            print(json.dumps(failure), flush=True)
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
