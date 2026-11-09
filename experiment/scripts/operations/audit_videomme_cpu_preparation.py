#!/usr/bin/env python3
"""Independently rehash the completed Video-MME CPU gate; never submit work.

This audit reads immutable inputs and writes only a new audit directory. It
does not import the training/evaluation runner, load a model or invoke a shell.
Content duplicate scores are checked through the frozen CPU audit's complete
receipt and artifact chain; PNG bytes/RGB and both final models are rehashed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat

BASE = Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM')
P1 = Path(str(BASE) + '-p1')
MEDIA = P1 / 'benchmark-media-20261012/videomme'
RUN_ID = 'dp-benchmark-videomme-r1-r2-s17-a1'
PROTOCOL = 'videomme-899-no-subtitles-no-audio-v1'
ORIGINAL_SHA = '1c08b9e18d7e9f2bcb9664ae81225603c94585ffabfcebe5b7b9bd01fc9ca66c'
RESTORE_SHA = 'fda1b11e76b9fb0fbbd3910b912a67082e91c08eadba9890f283cee7c9efa6fa'
SCOPE_SHA = 'bbbd5235bd00d0f40391401b4168933abd9f32e8ed705f5a3ddc875b3589636f'
SOURCE_LIST_SHA = '9056c23f5ce4bd1131fa96ee5742c2e0bc36d7852ac0a618c48609ab50dd686f'
ANNOTATION_SHA = '7fffab8ed38ecc2f9f0eca4c44d8a11636f0ee96116ede83580c8b9ae0faf986'
REGISTRY_SHA = 'a143d6a364221738b727d3944527039060315e04373599c5829ab3ea60eb0226'
REVISION = 'ead1408f75b618502df9a1d8e0950166bf0a2a0b'
FRAME_PREFIX = 'jinhaopeng/Paper/DeepProbe-VLM/benchmark-cache/20261013-v1/frame-cache/videomme/' + REVISION + '/uniform8/'
ENTRY_PREFIX = 'jinhaopeng/Paper/DeepProbe-VLM/benchmark-cache/20261013-v1/metadata/videomme-frame-cache/entries/'


def require(value, message):
    if not value:
        raise ValueError(message)


def safe(path):
    p = Path(path)
    require(p.is_absolute() and '..' not in p.parts and
            not any(x.is_symlink() for x in (p, *p.parents)), 'Unsafe artifact path')
    return p


def relative(value):
    p = Path(value)
    require(value and not p.is_absolute() and '..' not in p.parts and str(p) == value,
            'Unsafe relative artifact path')
    return p


def sha(path):
    h = hashlib.sha256()
    with safe(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def load(path):
    return json.loads(safe(path).read_text())


def pin(path):
    p = safe(path)
    require(p.is_file(), 'Missing regular artifact')
    return {'path': str(p), 'sha256': sha(p)}


def verify(item):
    require(isinstance(item, dict) and isinstance(item.get('sha256'), str) and
            re.fullmatch('[a-f0-9]{64}', item['sha256']), 'Invalid artifact digest')
    p = safe(item['path'])
    require(p.is_file() and sha(p) == item['sha256'], 'Artifact bytes changed: ' + str(p))
    return p


def save(path, value):
    with safe(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def index(rows, field, expected_count):
    require(isinstance(rows, list) and len(rows) == expected_count, 'Inventory count differs')
    out = {}
    for row in rows:
        require(isinstance(row, dict) and field in row and row[field] not in out, 'Invalid/duplicate inventory identity')
        out[row[field]] = row
    return out


def check_scope(plan):
    require(plan.get('schema_version') == 1 and plan.get('run_id') == RUN_ID and
            plan.get('protocol') == PROTOCOL and plan.get('evaluation_scope') == 'videomme_only' and
            plan.get('all_benchmark_assets_ready') is False and plan.get('mvbench_evaluation_ready') is False and
            plan.get('benchmark_score_based_model_selection') is False, 'Unsupported evaluation scope')
    require(plan.get('run_root') == str(BASE / 'evaluations' / RUN_ID) and
            plan.get('media_root') == str(MEDIA) and plan.get('storage_roots') == [str(BASE), str(P1)] and
            plan.get('storage_limit_bytes') == 99_000_000_000 and
            plan.get('working_reserve_bytes') == 1_000_000_000, 'Run path or capacity contract differs')


def preserved_underlying_gate(scoped, underlying):
    require(isinstance(underlying, dict) and underlying and
            all(key in scoped and scoped[key] == value for key, value in underlying.items()),
            'Scoped gate rewrites the underlying formal gate')


def bound_content_validation(validation, content_path, content):
    require(validation.get('provided') is True and validation.get('passed') is True and
            validation.get('errors') == [] and validation.get('path') == str(content_path) and
            validation.get('sha256') == sha(content_path) and
            validation.get('required_binding') == content['binding'],
            'Formal gate does not bind this complete content audit')


def stage_outputs(root):
    evidence = {}
    scripts = {'content_audit':'audit_benchmark_content.py', 'formal_gate':'prepare_formal_source_gate.py',
               'manifest':'prepare_videomme_evaluation.py', 'cpu_preflight':'run_benchmark_eval.py'}
    for stage in scripts:
        exit_path = root / (stage + '.exit.json'); result = load(exit_path)
        require(type(result.get('returncode')) is int and result['returncode'] == 0, 'CPU stage did not pass')
        for stream in ('stdout', 'stderr'):
            verify({'path': str(root / (stage + '.' + stream)), 'sha256': result[stream + '_sha256']})
        argv = load(root / (stage + '.argv.json'))
        require(argv.get('CUDA_VISIBLE_DEVICES') == '' and isinstance(argv.get('argv'), list), 'CPU stage environment unbound')
        expected_script = P1 / 'benchmark-evaluation-code-20261013-v2/experiment/scripts/operations' / scripts[stage]
        require(len(argv['argv']) >= 2 and argv['argv'][:2] == [str(BASE/'env/bin/python'),str(expected_script)], 'CPU executable changed')
        require(stage != 'cpu_preflight' or '--preflight-only' in argv['argv'], 'GPU runner used outside CPU-only mode')
        require(not any(x in argv['argv'] for x in ('--execute', 'submit')), 'Unexpected side effect in CPU stage')
        evidence[stage] = pin(exit_path)
    return evidence


def verify_checkpoint(item, loops, preflight, sources):
    root = safe(item['path']); rows = index(item['files'], 'path', 20)
    require(item.get('file_count') == 20 and item.get('run_id') == f'dp-p1-ouro-r{loops}-f8-g4-s17-sft-a1', 'Wrong accepted final')
    actual = {}
    def error(exc): raise exc
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=error):
        for name in dirs + files: safe(Path(directory) / name)
        for name in files:
            p = Path(directory) / name
            require(stat.S_ISREG(p.stat().st_mode), 'Nonregular checkpoint file')
            actual[p.relative_to(root).as_posix()] = p
    require(set(actual) == set(rows), 'Checkpoint physical file set differs')
    for name, row in rows.items():
        p = safe(root / relative(name))
        require(p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], 'Checkpoint full SHA mismatch')
    config = load(root / 'deepprobe_config.json')
    require((config['num_loops'], config['num_frames'], config['pool_grid']) == (loops, 8, 4) and config['max_text_tokens'] == 512, 'Model protocol differs')
    provenance = load(root / 'training_provenance.json')
    require(not sources.intersection(provenance.get('training_sources', [])) and
            not sources.intersection(provenance.get('tuning_sources', [])), 'Held-out source leakage in model provenance')
    require(preflight.get('checkpoint') == str(root) and preflight.get('run_id') == item['run_id'] and
            preflight.get('checkpoint_files_sha256') == canonical(item['files']) and
            preflight.get('checkpoint_config_sha256') == sha(root / 'deepprobe_config.json') and
            preflight.get('all_checkpoint_files_sha256_verified') is True and preflight.get('config') == config,
            'CPU preflight checkpoint binding differs')
    token = preflight['token_budget']
    require(token.get('num_examples') == 2697 and token.get('include_answer') is False and
            token.get('truncation') is False and token.get('num_over_budget') == 0 and
            0 < token['max_prompt_text_tokens'] <= token['max_text_tokens'] == 512, 'Inference token protocol differs')
    return {'file_count': 20, 'bytes': sum(r['bytes'] for r in rows.values()), 'config_sha256': sha(root / 'deepprobe_config.json')}


def capacity():
    roots = (BASE, P1); canonical_roots = tuple(p.resolve(strict=True) for p in roots)
    totals, targets, regular = {}, set(), set()
    def error(exc): raise exc
    for root in roots:
        require(safe(root).is_dir(), 'Missing staging root'); total = 0
        for directory, dirs, files in os.walk(root, followlinks=False, onerror=error):
            for name in list(dirs):
                p = Path(directory) / name
                if p.is_symlink():
                    target = p.resolve(strict=True)
                    require(target.is_dir() and any(target.is_relative_to(r) for r in canonical_roots), 'External directory link')
                    dirs.remove(name)
            for name in files:
                p = Path(directory) / name
                if p.is_symlink():
                    target = p.resolve(strict=True); require(target.is_file(), 'Nonfile link')
                    if any(target.is_relative_to(r) for r in canonical_roots): targets.add(target)
                    else: total += target.stat().st_size
                else:
                    require(stat.S_ISREG(p.stat().st_mode), 'Nonregular staging file')
                    regular.add(p.resolve(strict=True)); total += p.stat().st_size
        totals[str(root)] = total
    require(targets <= regular, 'Internal file link target not charged')
    projected = sum(totals.values()) + 1_000_000_000
    fs = os.statvfs(BASE); free = fs.f_bavail * fs.f_frsize
    require(projected < 99_000_000_000 and free >= 2_000_000_000, 'Fresh evaluation capacity failed')
    return {'root_bytes': totals, 'projected_bytes': projected, 'free_bytes': free, 'reserve_bytes': 1_000_000_000}


def audit(root, ready_sha):
    from PIL import Image
    checks = {}
    ready_path = verify({'path': str(root / 'READY_FOR_PARENT_SUBMISSION.json'), 'sha256': ready_sha})
    ready = load(ready_path)
    require(ready.get('status') == 'VIDEOMME_ONLY_CPU_PREPARATION_COMPLETE' and ready.get('gpu_submitted') is False,
            'CPU preparation not complete')
    plan_path = verify(ready['plan']); plan = load(plan_path); check_scope(plan)
    require(plan_path == root / 'EVALUATION_PLAN.json', 'Plan outside CPU preparation')
    checks['fixed_scope_and_completed_cpu_preparation'] = True
    started = load(root / 'CPU_PREPARATION_STARTED.json'); cfg_path = verify(started['config']); cfg = load(cfg_path)
    verify(started['script']); verify(cfg['pipeline_script']); verify(cfg['submission_helper'])
    fixed = cfg['evaluation_source_files']
    require(len(fixed) == 25 and hashlib.sha256((json.dumps(fixed, sort_keys=True, separators=(',', ':')) + '\n').encode()).hexdigest() == SOURCE_LIST_SHA, 'Frozen 25-file source list changed')
    supplied = index(plan['source_files'], 'path', 30)
    required = [*fixed, started['script'], pin(cfg_path), cfg['authorization'], cfg['submission_helper'], pin(MEDIA / 'VIDEOMME_BUCKET_MEDIA_COMPLETE.json')]
    require(len(required) == len(supplied) and {r['path']:r for r in required} == supplied, 'Plan source/control input set differs')
    for item in required: verify(item)
    scope_path = verify(cfg['authorization']); require(sha(scope_path) == SCOPE_SHA, 'Scope authorization changed')
    scope = load(scope_path)
    require(scope['evaluation_scope'] == 'videomme_only' and scope['models'] == ['r1_sft_seed17', 'r2_sft_seed17'] and
            scope['benchmark_score_based_model_selection'] is False, 'Model selection scope differs')
    original = P1 / 'formal-source-final-20261009-v1/FORMAL_READY.json'
    verify({'path': str(original), 'sha256': ORIGINAL_SHA})
    for item in cfg['training_fingerprint_files']: verify(item)
    checks['frozen_source_scope_config_and_training_inputs_rehashed'] = True
    stages = stage_outputs(root)
    scoped_path = verify(plan['formal_ready']); require(pin(scoped_path) == ready['scope_ready'], 'Scoped ready differs')
    scoped = load(scoped_path); underlying = verify(scoped['underlying_formal_ready'])
    require(underlying == root / 'formal-gate/FORMAL_READY.json', 'Unexpected underlying gate path')
    preserved_underlying_gate(scoped, load(underlying))
    require(scoped['formal_evaluation_ready'] is True and scoped['videomme_evaluation_ready'] is True and
            scoped['benchmark_media_gate'] == 'passed' and scoped['content_duplicate_gate'] == 'passed' and
            scoped['evaluation_scope'] == 'videomme_only' and scoped['mvbench_evaluation_ready'] is False and
            scoped['all_benchmark_assets_ready'] is False and scoped['ntu_media_included'] is False, 'Scoped gate failed')
    require(all(value['status'] == 'passed' for value in scoped['evaluation_readiness_gates'].values()), 'An evaluation subgate did not pass')
    for name, digest in scoped['artifact_sha256'].items(): verify({'path': str(scoped_path.parent / relative(name)), 'sha256': digest})
    content_path = root / 'content-audit/CONTENT_AUDIT.json'; content = load(content_path)
    bound_content_validation(load(root / 'formal-gate/content_audit_validation.json'), content_path, content)
    require(content['passed'] is True and content['scan_complete'] is True and content['scope'] == 'uniform8_sampled_frames' and
            content['training_file_count'] == 3979 and content['benchmark_source_count'] == 899 and
            content['original_formal_ready_sha256'] == ORIGINAL_SHA and
            all(content[k] == [] for k in ('errors','exact_duplicate_pairs','perceptual_duplicate_pairs','uninformative_sources')), 'Content audit incomplete or blocked')
    for item in content['artifacts'].values():
        path = Path(item['path']); path = path if path.is_absolute() else content_path.parent / relative(item['path'])
        verify({'path': str(path), 'sha256': item['sha256']})
    checks['four_cpu_stages_and_full_content_gate_passed'] = True
    bucket_path = MEDIA / 'VIDEOMME_BUCKET_MEDIA_COMPLETE.json'; bucket = load(bucket_path)
    require(bucket['kind'] == 'videomme_bucket_media_complete_v1' and bucket['complete'] is True and bucket['source_count'] == 899 and bucket['frame_count'] == 7192 and
            bucket['bucket'] == 'ss-devbox-stream' and bucket['revision'] == REVISION and
            bucket['full_oss_get_sha256_verified'] is True and bucket['exact_frame_key_set_verified'] is True and
            bucket['exact_source_receipt_key_set_verified'] is True and bucket['all_benchmark_assets_ready'] is False,
            'Bucket media completion unverified')
    cache_path = verify(bucket['frame_cache_receipt']); cache = load(cache_path)
    sentinel = load(MEDIA/'FRAME_CACHE_COMPLETE.json')
    require(sentinel['complete'] is True and sentinel['source_count'] == 899 and sentinel['frame_count'] == 7192 and
            sentinel['receipt_sha256'] == sha(cache_path), 'Atomic frame-cache completion changed')
    inventory = load(scoped_path.parent / 'benchmark_media_inventory.json')
    require(inventory['complete'] is True and inventory['scope'] == 'uniform8_sampled_frames' and inventory['root'] == str(MEDIA) and
            inventory['receipt_sha256'] == sha(cache_path), 'Formal media inventory differs')
    entries = index(cache['entries'], 'source_video_id', 899); frame_rows = index(bucket['frames'], 'relative_path', 7192)
    source_rows = index(bucket['source_receipts'], 'source_video_id', 899)
    require(set(source_rows) == set(entries), 'Bucket/source sets differ')
    observed_frames = set()
    for source, entry in entries.items():
        identity = source.removeprefix('youtube:'); require(re.fullmatch('[A-Za-z0-9_-]{11}', identity), 'Invalid source identity')
        entry_path = MEDIA / 'entries' / (identity + '.json'); require(load(entry_path) == entry, 'Source receipt mismatch')
        sr = source_rows[source]
        require(sr['bucket_key'] == ENTRY_PREFIX + identity + '.json' and sr['sha256'] == sha(entry_path) and sr['size_bytes'] == entry_path.stat().st_size, 'Bucket source binding mismatch')
        count = entry['decoded_frame_count']; require(type(count) is int and count > 0, 'Invalid complete decode count')
        indices = [round(i * (count - 1) / 7) for i in range(8)]
        require(entry['decode_complete'] is True and entry['selected_indices'] == indices and len(entry['frames']) == 8 and
                entry['image_paths'] == [r['relative_path'] for r in entry['frames']], 'Uniform8 decode evidence differs')
        for slot, frame in enumerate(entry['frames']):
            name = f'frames/{identity}/{slot:02d}.png'; require(frame['relative_path'] == name and name not in observed_frames and frame['source_frame_index'] == indices[slot], 'Frame order differs')
            observed_frames.add(name); p = safe(MEDIA / name); b = frame_rows[name]
            require(b['bucket_key'] == FRAME_PREFIX + f'{identity}/{slot:02d}.png' and b['sha256'] == frame['sha256'] == sha(p) and
                    b['size_bytes'] == frame['size_bytes'] == p.stat().st_size, 'Full PNG or bucket SHA mismatch')
            with Image.open(p) as image:
                image.load(); require(image.mode == 'RGB' and image.format == 'PNG' and image.size == (frame['width'],frame['height']) and
                                      hashlib.sha256(image.tobytes()).hexdigest() == frame['rgb_sha256'], 'PNG RGB changed')
    require(observed_frames == set(frame_rows) == {r['relative_path'] for r in inventory['entries']}, 'Exact frame inventory mismatch')
    checks['all_899_source_and_7192_png_bytes_and_rgb_reverified'] = True
    manifest_path = verify(plan['manifest']); records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    require(len(records) == 2697 and len({r['sample_id'] for r in records}) == 2697 and {r['source_video_id'] for r in records} == set(entries), 'QA coverage changed')
    binding_path = verify(plan['manifest_binding']); binding = load(binding_path)
    require(binding['status'] == 'EVALUATION_MANIFEST_BOUND' and binding['protocol'] == PROTOCOL and binding['excluded_question_ids'] == ['490-1','490-2','490-3'] and binding['manifest'] == plan['manifest'] and binding['selected_questions'] == 2697 and binding['selected_sources'] == 899 and binding['annotation']['sha256'] == ANNOTATION_SHA and binding['source_registry']['sha256'] == REGISTRY_SHA, 'Fixed manifest binding changed')
    verify(binding['annotation']); verify(binding['source_registry'])
    qa_path = verify({'path': str(binding_path.parent / 'QA_BINDINGS.json'), 'sha256': binding['qa_bindings_sha256']})
    verify({'path': str(binding_path.parent / 'EXCLUSIONS.json'), 'sha256': binding['exclusions_sha256']})
    qa = load(qa_path); require(len(qa) == len(records), 'Missing QA bindings')
    for row, proof in zip(records, qa):
        require(proof['sample_id'] == row['sample_id'] and proof['source_video_id'] == row['source_video_id'] and
                proof['annotation_sha256'] == ANNOTATION_SHA and proof['output_record_sha256'] == canonical(row) and
                row['image_paths'] == entries[row['source_video_id']]['image_paths'], 'Per-QA source/frame/annotation binding differs')
    checks['all_2697_qa_and_frame_order_rebound'] = True
    cpu_path = verify(ready['cpu_preflight']); cpu = load(cpu_path)
    require(cpu_path == root / 'cpu_preflight.stdout' and cpu['status'] == 'CPU_PREFLIGHT_PASSED' and cpu['plan_sha256'] == sha(plan_path) and
            cpu['protocol'] == PROTOCOL and cpu['num_gpus'] == 8 and cpu['sequence'] == ['r1','r2'] and cpu['greedy_max_new_tokens'] == 16 and
            cpu['subtitles'] is False and cpu['audio'] is False, 'CPU inference protocol differs')
    restore_path = verify(plan['checkpoint_restore_receipt']); require(sha(restore_path) == RESTORE_SHA, 'Accepted checkpoint restore changed')
    restore = load(restore_path); models = {}
    for name, loops in (('r1',1),('r2',2)):
        models[name] = verify_checkpoint(restore['checkpoints'][name], loops, cpu['checkpoints'][name], set(entries))
    checks['two_full_sft_finals_and_no_training_source_overlap_reverified'] = True
    current_capacity = capacity(); checks['fresh_two_root_capacity_passed'] = True
    for item in required: verify(item)
    verify({'path':str(original),'sha256':ORIGINAL_SHA}); verify({'path':str(ready_path),'sha256':ready_sha})
    checks['controls_unchanged_after_audit'] = True
    return {'schema_version':1,'kind':'videomme_cpu_preparation_independent_gate_v1','status':'passed',
            'observed_utc':datetime.now(timezone.utc).isoformat(),'evaluation_scope':'videomme_only',
            'all_benchmark_assets_ready':False,'mvbench_evaluation_ready':False,'gpu_submitted':False,
            'plan':pin(plan_path),'cpu_ready':pin(ready_path),'checks':checks,'script':pin(Path(__file__).resolve()),
            'models':models,'capacity':current_capacity,'stage_exit_receipts':stages,
            'content_audit':pin(content_path),'bucket_receipt':pin(bucket_path),
            'limitations':['Bucket GET evidence is bound to the accepted media receipt; this audit rehashes all local source/PNG/RGB bytes and both model finals without another OSS GET.',
                           'Content overlap conclusions retain the frozen uniform8 sampled-frame limitation; scores are not recomputed by this evidence auditor.',
                           'No GPU job or scheduler admission occurs here; the pinned one-submit helper must perform fresh admission after the R4 dependency succeeds.']}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preparation-root',type=Path,required=True)
    p.add_argument('--ready-sha256',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args = p.parse_args(argv)
    root,out = safe(args.preparation_root),safe(args.output_dir)
    require(root.is_dir() and root.is_relative_to(P1) and out.is_relative_to(P1) and not out.exists(), 'Use existing preparation and new audit paths under P1')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Explicit CPU-only audit environment required')
    out.mkdir(parents=True,exist_ok=False)
    try:
        result = audit(root,args.ready_sha256)
        path = out/'VIDEOMME_CPU_PREPARATION_INDEPENDENT_GATE.json'; save(path,result)
        print(json.dumps({'status':'passed','receipt':pin(path),'checks':len(result['checks'])}),flush=True)
        return 0
    except Exception as exc:
        save(out/'INDEPENDENT_GATE_FAILED.json',{'status':'failed','error_type':type(exc).__name__,'error':str(exc),'gpu_submitted':False,'observed_utc':datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
