#!/usr/bin/env python3
"""Separate TempCompass data acceptance; never rewrites preparation evidence.

Remote auditing is read-only and may execute from a SHA-bound in-memory bundle.
The creation command writes a NEW LOCAL acceptance directory after that audit.
This data-only gate never grants GPU admission or uploads/deletes anything.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timezone

PROTOCOL = 'tempcompass-mc-1580-uniform8-noaudio-v1'
REMOTE_ROOT = Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/benchmark-media-tempcompass-20261021-a1')
RELEASE = Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/tempcompass-cpu-release-20261021-v2')
PREFIX = 'jinhaopeng/Paper/DeepProbe-VLM/benchmark-cache/20261021-tempcompass-uniform8-v1/'
WORKER_SHA = '408aee7bd42f04efae763a5ab2686bf4b3e72ab9d3abd40ef66ce70bbcfd3873'
CONFIG_SHA = 'cb30c179df974178483badf4b75cc0a9debdcb4dc35043d1458e32ed4c3f53f3'
PINS = {
    'terminal':'27ea8c47968dc74a3f2ee5e01ec421f4aaa35a0da3a3bdf83256be79a1487d73',
    'manifest':'7c8ed89858d0bb3d1fecd50fbe64952b8af0f7fdac9260264fb747373b9cf7b3',
    'cache':'5f3a4a2546dfc7d22c13541613e0f5eaaa043a4ea28c06fd79af62fdd86230ee',
    'audit':'7c397be5225d0aa9fb627f017e3be7e3134d208b0402a9888e057988d2cec761',
    'fingerprints':'83c9eacc10208fd2c72b0d6dbf33658167bdd0bf229d888492810b412314687d',
    'protocol':'e05e48d2c4dbb53bf793b493c7e7f6aa4eedb72565fd0f33cfcceb1c6e22bc2a',
    'root_review':'01869b7216f754000ddad8f3ef612b0029ed39f5ef7de4f0233ebeb62cf08c5f',
    'independent_review':'c9bd8cfd7215d790bd714693140bb2a951723b5bbec7a8a67e442fef228589a2',
    'capture':'cd5e06a051422e77a55c0696497c55c6618d507f0f224df65df73a11d502e5ea',
    'rows':'803d1b71e85092b9d7118b5679794a2cc1c2203de170d2d8cdcd06b270eb3d96',
    'source_map':'1537ee59ba74b4add2c30e586d5ff72b0ea2913882db798fdea4c69117405738',
    'meta':'72144a7cb5aba97d12c3148ec734f7c8f27a04f6b3a7396ef41003a0a7178419',
}
COUNTS = {'qa':1580, 'clips':410, 'underlying_sources':207, 'source_components':188, 'frames':3280}
DIMENSIONS = {'action':338, 'direction':335, 'speed':317, 'order':302, 'attribute_change':288}


def require(ok, code):
    if not ok:
        raise ValueError(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def safe(path):
    path = Path(path).absolute()
    require('..' not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)), 'unsafe_path')
    return path


def sha(path):
    result = hashlib.sha256()
    with safe(path).open('rb') as stream:
        for raw in iter(lambda:stream.read(8 * 1024 * 1024), b''):
            result.update(raw)
    return result.hexdigest()


def code_sha():
    return globals().get('IN_MEMORY_SOURCE_SHA256') or sha(__file__)


def json_once(path, data):
    path = safe(path)
    with path.open('xb') as stream:
        stream.write((json.dumps(data,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
        stream.flush(); os.fsync(stream.fileno())


class Bundle:
    def __init__(self, config, root=None, blobs=None):
        require(config.get('schema_version') == 1 and config.get('protocol') == PROTOCOL and
                config.get('media_root') == str(REMOTE_ROOT) and config.get('oss_prefix') == PREFIX,
                'unsupported_data_config')
        require(set(config['inputs']) == set(PINS), 'input_name_set_changed')
        self.config, self.root, self.blobs = config, Path(root) if root else None, blobs
        for name, expected in PINS.items():
            require(config['inputs'][name]['sha256'] == expected, 'input_pin_changed:' + name)

    def raw(self, item):
        name = item['path']; relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe_bundle_path')
        raw = self.blobs[name] if self.blobs is not None else safe(self.root / relative).read_bytes()
        require(len(raw) == item['bytes'] and digest(raw) == item['sha256'], 'artifact_changed:' + name)
        return raw

    def data(self, name):
        raw = self.raw(self.config['inputs'][name])
        return [json.loads(s) for s in raw.splitlines()] if name == 'manifest' else json.loads(raw)

    def binding(self):
        return digest(canonical(self.config))


def validate_manifest(records, source_map, original_rows, meta):
    """Validate every QA and media identity without using statistical groups as media IDs."""
    require(len(records) == 1580 and len({r['sample_id'] for r in records}) == 1580, 'qa_count_or_identity')
    sources = {r['video_id']:r for r in source_map['entries']}
    require(len(sources) == 410, 'source_map_count')
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x: parent[x] = find(parent[x])
        return parent[x]
    for row in sources.values():
        ids = row['underlying_source_video_ids']
        require(1 <= len(ids) <= 2 and len(set(ids)) == len(ids), 'source_dependency_changed')
        for item in ids: parent[find(item)] = find(ids[0])
    groups = {}
    for item in sorted(parent): groups.setdefault(find(item), []).append(item)
    originals = {(r['video_id'],r['dim'],r['index_within_dimension']):r for r in original_rows}
    require(len(originals) == len(original_rows) == 1580, 'original_qa_identity')
    seen, counts = set(), {}
    for row in records:
        clip, dim, index = row['video_id'], row['task_type'], row['index_within_dimension']
        key = clip, dim, index
        require(key in originals and key not in seen and clip in sources, 'qa_original_mapping')
        seen.add(key); counts[dim] = counts.get(dim, 0) + 1
        source, original = sources[clip], originals[key]
        original_choices = original['question'].split('\n')[1:]
        parsed = [re.fullmatch(r'([ \t]*[A-D]\. ?)(.*)', text) for text in original_choices]
        require(all(m is not None and m.group(1).lstrip()[0] == chr(65+i) for i,m in enumerate(parsed)), 'original_choice_labels')
        component = groups[find(source['underlying_source_video_ids'][0])]
        expected = {'sample_id':f'tempcompass:{clip}:{dim}:{index}', 'dataset':'tempcompass','split':'test',
            'processed_media_identity':'tempcompass:' + clip, 'source_component_ids':component,
            'source_video_id':'tempcompass-source-component:' + digest(canonical(component)),
            'underlying_source_video_ids':source['underlying_source_video_ids'],
            'question':original['question'].split('\n')[0], 'original_question':original['question'],
            'original_choices':original_choices, 'original_choice_prefixes':[m.group(1) for m in parsed],
            'choices':[m.group(2) for m in parsed], 'answer':chr(65+original_choices.index(original['answer'])),
            'original_answer':original['answer'], 'annotation_row_sha256':digest(canonical(original)),
            'fine_grained_task_type':'order' if dim == 'order' else meta[clip]['eval_dim'][dim]['type'],
            'image_paths':[f'frames/{clip}/{i:02d}.png' for i in range(8)]}
        require(all(row.get(k) == value for k,value in expected.items()), 'manifest_record_changed:' + row['sample_id'])
        require(not any(row.get(k) for k in ('video_path','audio_path','subtitle_path','subtitles')), 'extra_model_input')
    require(counts == DIMENSIONS and len({r['video_id'] for r in records}) == 410 and
            len(parent) == 207 and len(groups) == 188, 'frozen_counts_changed')
    return dict(COUNTS)


def validate_reviews(bundle, audit, cache, fingerprints):
    root, independent, capture = (bundle.data(n) for n in ('root_review','independent_review','capture'))
    require(audit['passed'] is False and audit['scan_complete'] is True and
            audit['search_completed_without_truncation'] is True and audit['errors'] == [] and
            audit['exact_duplicate_pairs'] == [] and audit['uninformative_sources'] == [] and
            len(audit['perceptual_duplicate_pairs']) == 1, 'incomplete_or_changed_original_scan')
    pair = audit['perceptual_duplicate_pairs'][0]
    require(len(pair['frame_matches']) == 1 and root['candidate'] == capture['candidate'] == pair, 'candidate_changed')
    require(root['status'] == 'candidate_cleared_as_perceptual_false_positive' and
            root['original_audit_sha256'] == independent['audit_sha256'] == capture['audit_sha256'] == PINS['audit'] and
            root['capture_receipt_sha256'] == independent['capture_receipt_sha256'] == PINS['capture'], 'review_binding_changed')
    require(root['reviewer'] != independent['reviewer'] and independent['passed'] is True and
            independent['independent_of_root_visual_judgment'] is True and
            independent['root_visual_review_read_before_this_receipt'] is False and
            independent['all_matched_frames_and_context_visually_inspected'] is True and
            independent['original_audit_preserved'] is True and independent['qa_frames_and_thresholds_unchanged'] is True,
            'independent_visual_review_incomplete')
    require(root['original_scan_modified'] is False and root['thresholds_changed'] is False and
            root['new_exclusions'] == [] and root['qa_gold_or_model_predictions_inspected'] is False and
            root['all_16_pngs_rehashed'] is True, 'review_altered_scope')
    require(independent['counts'] == {'candidate_pairs':1,'context_frames':16,'different_graphic_collision_pairs':1,
            'matched_frame_pairs':1,'substantive_overlap_in_inspected_samples':0,'unresolved_pairs':0}, 'visual_counts_changed')
    decision, match = independent['pairs'][0], pair['frame_matches'][0]
    require(len(independent['pairs']) == 1 and decision['decision'] == 'flagged_sampled_pair_is_visually_distinct' and
            decision['classification'] == 'different_graphic_collision' and
            all(decision[k] == pair[k] for k in ('training_source_video_id','benchmark_source_video_id')) and
            all(decision[k] == match[k] for k in ('training_slot','benchmark_slot')) and
            decision['candidate_distances_reproduced'] == {k:match[k] for k in ('phash_hamming','dhash_hamming','same_decoded_rgb')},
            'visual_candidate_decision_changed')
    clip = pair['benchmark_source_video_id'].removeprefix('tempcompass:')
    entry = next(e for e in cache['entries'] if e['video_id'] == clip)
    fp = next(e for e in fingerprints if e['source_video_id'] == pair['benchmark_source_video_id'])
    require(capture['benchmark_entry'] == entry and capture['training']['relative_path'] == pair['training_path'] and
            capture['training']['source_video_id'] == pair['training_source_video_id'] and
            capture['sampling_unchanged'] is True and capture['all_16_sampled_rgb_hashes_verified'] is True and
            capture['remote_files_written'] is False and capture['gpu_used'] is False and capture['oss_operations'] is False,
            'capture_binding_changed')
    require(len(capture['files']) == 16 and len(bundle.config['visual_files']) == 18, 'visual_file_count')
    expected_evidence = {x['relative_path']:x for x in independent['evidence_files']}
    require(set(bundle.config['visual_files']) == set(expected_evidence), 'visual_file_set')
    for name, item in bundle.config['visual_files'].items():
        raw = bundle.raw(item); ref = expected_evidence[name]
        require(digest(raw) == ref['sha256'] and len(raw) == ref['bytes'], 'visual_file_changed')
    for item in capture['files']:
        kind, filename = item['path'].split('/')
        slot = int(Path(filename).stem)
        require(kind in ('training','benchmark') and 0 <= slot < 8, 'capture_slot_changed')
        expected = capture['training']['frames'][slot] if kind == 'training' else fp['frames'][slot]
        ref = bundle.config['visual_files']['original-fixed-frames/' + item['path']]
        require(item['fingerprint'] == expected and item['sha256'] == ref['sha256'] and item['bytes'] == ref['bytes'], 'capture_fingerprint_changed')
    return {'passed':True,'original_audit_passed':False,'original_candidates_preserved':True,
            'original_audit_sha256':PINS['audit'],'root_review_sha256':PINS['root_review'],
            'independent_review_sha256':PINS['independent_review'],'capture_sha256':PINS['capture'],
            'candidate_count':1,'matched_frame_count':1,'scope':'reviewed_uniform8_samples_only'}


def validate_documents(bundle):
    records, source_map, originals, meta = (bundle.data(n) for n in ('manifest','source_map','rows','meta'))
    counts = validate_manifest(records, source_map, originals, meta)
    cache, audit, fingerprints, terminal, protocol = (bundle.data(n) for n in ('cache','audit','fingerprints','terminal','protocol'))
    require(terminal['status'] == 'CONTENT_REVIEW_REQUIRED' and terminal['cpu_data_ready'] is False and
            terminal['formal_evaluation_ready'] is False and terminal['worker_sha256'] == WORKER_SHA and
            terminal['config_sha256'] == CONFIG_SHA, 'original_terminal_changed')
    require(cache['complete'] is True and cache['source_count'] == 410 and cache['frame_count'] == 3280 and
            cache['underlying_source_count'] == 207 and cache['frames_resized'] is False and cache['failures'] == [], 'cache_incomplete')
    require(protocol['protocol_id'] == PROTOCOL and protocol['manifest_sha256'] == PINS['manifest'] and
            protocol['num_frames'] == 8 and protocol['pool_grid'] == 4 and protocol['greedy_max_new_tokens'] == 16 and
            protocol['audio'] is False and protocol['subtitles'] is False and protocol['answer_in_model_prompt'] is False,
            'protocol_changed')
    entries = {e['video_id']:e for e in cache['entries']}
    require(len(entries) == len(cache['entries']) == 410 and set(entries) == {r['video_id'] for r in records}, 'cache_clip_set')
    paths = set()
    for entry in entries.values():
        indices = [round(i*(entry['decoded_frame_count']-1)/7) for i in range(8)]
        require(entry['decode_complete'] is True and entry['full_decode_passes'] == 2 and entry['decoded_frame_count'] > 0 and
                entry['selected_indices'] == indices and len(entry['frames']) == 8 and
                entry['image_paths'] == [f['relative_path'] for f in entry['frames']], 'decode_contract_changed')
        for slot, frame in enumerate(entry['frames']):
            expected = f"frames/{entry['video_id']}/{slot:02d}.png"
            require(frame['relative_path'] == expected and expected not in paths and
                    frame['source_frame_index'] == indices[slot] and frame['width']*frame['height'] <= 3840*2160,
                    'frame_identity_changed')
            paths.add(expected)
    for row in records:
        require(row['image_paths'] == entries[row['video_id']]['image_paths'] and
                row['underlying_source_video_ids'] == entries[row['video_id']]['underlying_source_video_ids'], 'media_mapping_changed')
    review = validate_reviews(bundle,audit,cache,fingerprints)
    return {'counts':counts,'review':review,'records':records,'cache':cache,'audit':audit,'fingerprints':fingerprints}


def remote_audit(bundle):
    """Full CPU revalidation: every PNG/fingerprint, original training SHA, overlap and exact OSS set."""
    require(sys.platform == 'linux' and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'linux_cpu_only')
    state = validate_documents(bundle)
    worker_path = RELEASE / 'experiment/scripts/operations/prepare_tempcompass_cpu.py'
    require(sha(worker_path) == WORKER_SHA and sha(RELEASE/'CPU_CONFIG.json') == CONFIG_SHA, 'frozen_runtime_changed')
    spec = importlib.util.spec_from_file_location('tempcompass_frozen_cpu_runtime',worker_path)
    worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    cfg = json.loads((RELEASE/'CPU_CONFIG.json').read_text()); inputs = worker.read_inputs(cfg)
    scanner = worker.load_module(cfg['inputs']['content_helper'],'tempcompass_gate_scanner')
    helper = worker.load_module(cfg['inputs']['oss_helper'],'tempcompass_gate_oss')
    environment, original_environment = scanner.versions(), inputs['training_receipt']['environment']
    # The accepted training scan predates the source-bound cache reader. Only
    # that complete scanner file SHA differs; decoder/fingerprint runtimes and
    # sampling source must still match exactly. Both scanner identities are pinned.
    require(environment['scanner_sha256'] == '4658b04be44f7ffa47a6ca3d55ed9a9c1e013d7a8e7822f16bb87ae848496d43' and
            original_environment['scanner_sha256'] == '09ef001245db4605bf3a676401b5258c78d03f7b877c57b39aeb1c64f8333805' and
            {k:v for k,v in environment.items() if k!='scanner_sha256'} ==
            {k:v for k,v in original_environment.items() if k!='scanner_sha256'}, 'fingerprint_environment_changed')
    worker.validate_training(cfg,inputs,scanner)  # hashes ALL 3979 original train/dev media; no training re-decode
    training = inputs['training_fingerprints']
    capture = bundle.data('capture')
    original_training = next(r for r in training if r['relative_path'] == capture['training']['relative_path'])
    require(all(original_training.get(k)==v for k,v in capture['training'].items()), 'captured_training_identity_changed')
    from PIL import Image
    fresh_fingerprints = []
    for entry in state['cache']['entries']:
        frames = []
        require(sha(REMOTE_ROOT/'entries'/f"{entry['video_id']}.json") == digest((json.dumps(entry,sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode()), 'entry_file_changed')
        for frame in entry['frames']:
            path = safe(REMOTE_ROOT/frame['relative_path'])
            require(path.stat().st_size == frame['size_bytes'] and sha(path) == frame['sha256'], 'current_png_changed')
            with Image.open(path) as image:
                image.load()
                require(image.mode == 'RGB' and image.size == (frame['width'],frame['height']) and
                        digest(image.tobytes()) == frame['rgb_sha256'], 'decoded_rgb_changed')
                frames.append(scanner.fingerprint_image(image))
            worker.memory_guard()
        fresh_fingerprints.append({'source_video_id':'tempcompass:'+entry['video_id'],
            'underlying_source_video_ids':entry['underlying_source_video_ids'],'frames':frames})
    require(fresh_fingerprints == state['fingerprints'], 'benchmark_fingerprints_changed')
    for item in capture['files']:
        with Image.open(io.BytesIO(bundle.raw(bundle.config['visual_files']['original-fixed-frames/'+item['path']]))) as image:
            image.load(); require(scanner.fingerprint_image(image) == item['fingerprint'], 'visual_rgb_fingerprint_changed')
    candidates = worker.bounded_candidate_pairs(training,fresh_fingerprints,scanner.POLICY)
    require(candidates == state['audit']['perceptual_duplicate_pairs'] and state['audit']['policy'] == scanner.POLICY,
            'fresh_overlap_candidates_changed')
    original_hashes = {r['sha256'] for r in training}
    require(not any(e['original_sha256'] in original_hashes for e in state['cache']['entries']), 'exact_training_overlap')
    require(not any(not any(f['informative'] for f in r['frames']) for r in [*training,*fresh_fingerprints]), 'uninformative_source')
    underlying = {s for r in state['records'] for s in r['underlying_source_video_ids']}
    require(not underlying & {r['source_video_id'] for r in training}, 'declared_source_overlap')
    terminal = bundle.data('terminal'); expected = dict(terminal['objects_before_terminal'])
    expected[PREFIX+'TERMINAL.json'] = {'bytes':len(bundle.raw(bundle.config['inputs']['terminal'])),'sha256':PINS['terminal']}
    actual = {PREFIX+str(p.relative_to(REMOTE_ROOT)):p for p in REMOTE_ROOT.rglob('*') if p.is_file()}
    require(set(actual) == set(expected) and len(actual) == 3699,'local_output_set_changed')
    store = helper.OSSStore(cfg['runtime_library_path'])
    require(set(store.keys(PREFIX)) == set(expected),'exact_oss_set_changed')
    for key, ref in expected.items():
        path = actual[key]; require(path.stat().st_size == ref['bytes'] and sha(path) == ref['sha256'],'output_sha_changed')
        head = store.head(key)
        require(head.content_length == ref['bytes'] and head.headers.get('x-oss-meta-sha256') == ref['sha256'],'oss_head_changed')
    head = store.head(worker.ARCHIVE_KEY)
    require(head.content_length == worker.ARCHIVE_BYTES and head.etag.strip('"') == inputs['archive_audit']['archive']['etag'].strip('"'),'input_archive_head_changed')
    return {'schema_version':1,'status':'TEMP_COMPASS_REMOTE_AUDIT_PASSED','finished_utc':datetime.now(timezone.utc).isoformat(),
            'validation_code_sha256':code_sha(),'config_binding_sha256':bundle.binding(),'counts':COUNTS,
            'training_media_sha256_count':3979,'png_sha256_rgb_fingerprint_count':3280,'visual_png_fingerprint_count':16,
            'full_candidate_search_replayed':True,'candidate_pairs':candidates,'exact_training_overlap':[],
            'local_and_oss_objects':3699,'local_and_oss_bytes':sum(e['bytes'] for e in expected.values()),
            'exact_oss_prefix':PREFIX,'bucket':'ss-devbox-stream','training_fingerprints_sha256':worker.PINS['training_fingerprints'],
            'training_sources':sorted({r['source_video_id'] for r in training}),
            'fingerprint_environment':environment,'training_fingerprint_environment':original_environment,
            'all_original_candidates_resolved_by_two_reviews':True,'original_audit_passed':False,
            'review':state['review'],'remote_files_written':0,'oss_writes':0,'gpu_used':False,
            'media_inventory':state['cache'],'formal_evaluation_ready':False,
            'scope':'reviewed uniform8 samples only; no claim about unsampled frames or pretraining contamination'}


def validate_remote_proof(proof,bundle):
    state = validate_documents(bundle)
    expected = {'schema_version':1,'status':'TEMP_COMPASS_REMOTE_AUDIT_PASSED','validation_code_sha256':code_sha(),
        'config_binding_sha256':bundle.binding(),'counts':COUNTS,'training_media_sha256_count':3979,
        'png_sha256_rgb_fingerprint_count':3280,'visual_png_fingerprint_count':16,'full_candidate_search_replayed':True,
        'candidate_pairs':state['audit']['perceptual_duplicate_pairs'],'exact_training_overlap':[],
        'local_and_oss_objects':3699,'local_and_oss_bytes':793097752,'exact_oss_prefix':PREFIX,'bucket':'ss-devbox-stream',
        'all_original_candidates_resolved_by_two_reviews':True,'original_audit_passed':False,'review':state['review'],
        'remote_files_written':0,'oss_writes':0,'gpu_used':False,'media_inventory':state['cache'],'formal_evaluation_ready':False}
    require(all(proof.get(k) == v for k,v in expected.items()),'remote_validation_binding_failed')
    require(datetime.fromisoformat(proof['finished_utc']).tzinfo is not None,'missing_audit_timestamp')
    return state


def create_gate(config_path,config_sha,remote_proof,remote_proof_sha,output_dir):
    require(sha(config_path) == config_sha and sha(remote_proof) == remote_proof_sha,'creation_input_changed')
    bundle = Bundle(json.loads(Path(config_path).read_text()),Path(config_path).parent)
    proof = json.loads(Path(remote_proof).read_text()); validate_remote_proof(proof,bundle)
    output = safe(output_dir)
    require(output.parent == safe(config_path).parent and not output.exists(),'fresh_gate_directory_required')
    output.mkdir()
    json_once(output/'REMOTE_AUDIT.json',proof)
    ready = {'schema_version':1,'kind':'tempcompass_reviewed_data_acceptance','status':'TEMP_COMPASS_DATA_ACCEPTED',
        'protocol':PROTOCOL,'counts':COUNTS,'evaluation_scope':'tempcompass_only','cpu_data_ready':True,
        'formal_evaluation_ready':False,'config_sha256':config_sha,'validation_code_sha256':code_sha(),
        'remote_audit_sha256':sha(output/'REMOTE_AUDIT.json'),'media_root':str(REMOTE_ROOT),
        'manifest_sha256':PINS['manifest'],'original_terminal_sha256':PINS['terminal'],'original_audit_passed':False,
        'review':proof['review'],'gpu_admission_granted':False,
        'remaining':['checkpoint/tokenizer/model-source bindings','all1580 prompt token-budget check','fresh one-job scheduling and GPU admission']}
    json_once(output/'TEMP_COMPASS_DATA_READY.json',ready)
    return ready


def verify_accepted_data(ready_path,expected_sha256,media_root=None):
    ready_path = safe(ready_path); require(sha(ready_path) == expected_sha256,'accepted_gate_sha_changed')
    ready = json.loads(ready_path.read_text()); package = ready_path.parent.parent
    config_path = package/'DATA_CONFIG.json'; require(sha(config_path) == ready['config_sha256'],'accepted_config_changed')
    bundle = Bundle(json.loads(config_path.read_text()),package)
    proof_path = ready_path.parent/'REMOTE_AUDIT.json'; require(sha(proof_path) == ready['remote_audit_sha256'],'accepted_audit_changed')
    proof = json.loads(proof_path.read_text()); state = validate_remote_proof(proof,bundle)
    require(ready['kind'] == 'tempcompass_reviewed_data_acceptance' and ready['status'] == 'TEMP_COMPASS_DATA_ACCEPTED' and
            ready['cpu_data_ready'] is True and ready['formal_evaluation_ready'] is False and ready['counts'] == COUNTS and
            ready['protocol'] == PROTOCOL and ready['validation_code_sha256'] == code_sha() and ready['review'] == proof['review'] and
            ready['manifest_sha256'] == PINS['manifest'] and ready['media_root'] == str(REMOTE_ROOT),'accepted_ready_changed')
    if media_root is not None:
        require(safe(media_root) == REMOTE_ROOT,'evaluation_media_root_changed')
        for entry in state['cache']['entries']:
            for frame in entry['frames']:
                path = safe(Path(media_root)/frame['relative_path'])
                require(path.stat().st_size == frame['size_bytes'] and sha(path) == frame['sha256'],'accepted_png_changed')
    return {**ready,'manifest_path':str(package/bundle.config['inputs']['manifest']['path']),
            'records':state['records'],'training_sources':proof['training_sources'],'ready_sha256':expected_sha256}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True); p.add_argument('--config-sha256',required=True)
    p.add_argument('--remote-proof',type=Path,required=True); p.add_argument('--remote-proof-sha256',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a = p.parse_args(argv)
    result = create_gate(a.config,a.config_sha256,a.remote_proof,a.remote_proof_sha256,a.output_dir)
    print(json.dumps(result,indent=2,sort_keys=True)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
