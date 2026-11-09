#!/usr/bin/env python3
"""Prepare one Video-MME-only evaluation using already cached bucket media.

CPU only. No polling, scheduler calls, inference, automatic submission or input
cleanup. This entry point reuses the accepted frozen evaluation package, then
produces a plan and a parent-owned submission command after every CPU gate passes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

BASE = Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM')
P1 = Path(str(BASE) + '-p1')
EXPERIMENT = P1 / 'benchmark-evaluation-code-20261013-v2/experiment'
MEDIA_ROOT = P1 / 'benchmark-media-20261012/videomme'
RUN_ID = 'dp-benchmark-videomme-r1-r2-s17-a1'
PROTOCOL = 'videomme-899-no-subtitles-no-audio-v1'
REVISION = 'ead1408f75b618502df9a1d8e0950166bf0a2a0b'
BUCKET = 'ss-devbox-stream'
CACHE_PREFIX = 'jinhaopeng/Paper/DeepProbe-VLM/benchmark-cache/20261013-v1/'
FRAME_PREFIX = CACHE_PREFIX + 'frame-cache/videomme/' + REVISION + '/uniform8/'
SOURCE_PREFIX = CACHE_PREFIX + 'metadata/videomme-frame-cache/entries/'
ORIGINAL_READY_SHA = '1c08b9e18d7e9f2bcb9664ae81225603c94585ffabfcebe5b7b9bd01fc9ca66c'
RESTORE_SHA = 'fda1b11e76b9fb0fbbd3910b912a67082e91c08eadba9890f283cee7c9efa6fa'
REGISTRY_SHA = 'a143d6a364221738b727d3944527039060315e04373599c5829ab3ea60eb0226'
ANNOTATION_SHA = '7fffab8ed38ecc2f9f0eca4c44d8a11636f0ee96116ede83580c8b9ae0faf986'
FROZEN_SOURCE_LIST_SHA = '9056c23f5ce4bd1131fa96ee5742c2e0bc36d7852ac0a618c48609ab50dd686f'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'Absolute non-traversing path required')
    require(not any(p.is_symlink() for p in [path, *path.parents]), 'Symlink artifact path forbidden')
    return path


def load(path):
    value = json.loads(safe_path(path).read_text())
    require(isinstance(value, dict), 'JSON object required')
    return value


def pin(path):
    path = safe_path(path)
    return {'path': str(path), 'sha256': sha(path)}


def verify(item):
    path = safe_path(item['path'])
    require(path.is_file() and sha(path) == item['sha256'], 'Pinned artifact changed: ' + str(path))
    return path


def save(path, value):
    with safe_path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def validate_scope(authorization):
    require(authorization.get('schema_version') == 1 and authorization.get('evaluation_scope') == 'videomme_only',
            'Explicit Video-MME-only scope required')
    require(authorization.get('video_mme_evaluation_authorized') is True and
            authorization.get('mvbench_evaluation_authorized') is False and
            authorization.get('benchmark_score_based_model_selection') is False,
            'Scope must authorize only the fixed Video-MME evaluation')
    require(authorization.get('models') == ['r1_sft_seed17', 'r2_sft_seed17'] and
            authorization.get('model_sequence') == ['r1', 'r2'] and
            authorization.get('protocol') == PROTOCOL,
            'Predeclared model order/protocol changed')


def index_rows(rows, key, count, label):
    require(isinstance(rows, list) and len(rows) == count, label + ' count mismatch')
    result = {}
    for row in rows:
        require(isinstance(row, dict) and key in row and row[key] not in result, label + ' duplicate/invalid key')
        result[row[key]] = row
    return result


def verify_bucket_media(receipt_path, frame_cache_path, completion_path, registry_path, media_root):
    """Bind the complete bucket proof to all source receipts and ordered frames.

    The following content-audit stage independently hashes/decodes all local PNGs.
    This step verifies their bucket identities against the same source receipts.
    """
    receipt, cache, complete = load(receipt_path), load(frame_cache_path), load(completion_path)
    require(receipt.get('kind') == 'videomme_bucket_media_complete_v1' and receipt.get('complete') is True and
            receipt.get('status') == 'VIDEOMME_BUCKET_MEDIA_COMPLETE' and receipt.get('videomme_media_ready') is True,
            'Video-MME bucket media completion missing')
    require(receipt.get('bucket') == BUCKET and receipt.get('revision') == REVISION,
            'Bucket/source revision mismatch')
    require(receipt.get('source_count') == 899 and receipt.get('frame_count') == 7192 and
            receipt.get('full_oss_get_sha256_verified') is True and
            receipt.get('exact_frame_key_set_verified') is True and
            receipt.get('exact_source_receipt_key_set_verified') is True,
            'Complete bucket key sets and full readback verification required')
    require(receipt.get('all_benchmark_assets_ready') is False and
            receipt.get('formal_evaluation_ready') is False and receipt.get('mvbench_evaluation_ready') is False and
            receipt.get('ntu_media_included') is False,
            'Video-MME completion must not promote MVBench/NTU readiness')
    require(receipt.get('frame_cache_receipt') == pin(frame_cache_path), 'Bucket proof binds a different frame receipt')
    require(cache.get('kind') == 'videomme_uniform8_frame_cache' and cache.get('complete') is True and
            cache.get('num_frames') == 8 and cache.get('revision') == REVISION and cache.get('failures') == [],
            'Complete frozen uniform8 frame cache required')
    require(complete.get('complete') is True and complete.get('source_count') == 899 and
            complete.get('frame_count') == 7192 and complete.get('receipt_sha256') == sha(frame_cache_path),
            'Frame-cache atomic completion sentinel differs')
    registry = load(verify({'path': str(registry_path), 'sha256': REGISTRY_SHA}))
    accepted = [row for row in registry['entries'] if row['dataset'] == 'videomme']
    selected = {row['source_video_id'] for row in accepted}
    require(len(accepted) == 2697 and len(selected) == 899, 'Original source subset differs')
    entries = index_rows(cache['entries'], 'source_video_id', 899, 'frame source')
    sources = index_rows(receipt['source_receipts'], 'source_video_id', 899, 'bucket source receipt')
    frames = index_rows(receipt['frames'], 'relative_path', 7192, 'bucket PNG')
    require(set(entries) == set(sources) == selected, 'Bucket/source set differs from predeclared899')
    expected_frames, keys, frame_bytes, source_bytes = set(), set(), 0, 0
    for source, entry in entries.items():
        identity = source.removeprefix('youtube:')
        require(source == 'youtube:' + identity and '/' not in identity and '..' not in identity,
                'Unsafe source identity')
        source_path = safe_path(media_root / 'entries' / (identity + '.json'))
        require(load(source_path) == entry, 'Source receipt differs from complete frame cache')
        expected_source = {'source_video_id': source, 'bucket_key': SOURCE_PREFIX + identity + '.json',
                           'size_bytes': source_path.stat().st_size, 'sha256': sha(source_path)}
        require(all(sources[source].get(k) == v for k, v in expected_source.items()), 'Bucket source receipt binding mismatch')
        source_bytes += expected_source['size_bytes']; keys.add(expected_source['bucket_key'])
        ordered = entry.get('frames', [])
        require(entry.get('decode_complete') is True and len(ordered) == 8 and
                entry.get('image_paths') == [row['relative_path'] for row in ordered], 'Ordered8 frame slots required')
        for slot, row in enumerate(ordered):
            relative = f'frames/{identity}/{slot:02d}.png'
            require(row.get('relative_path') == relative and relative not in expected_frames, 'Frame slot path/order changed')
            expected_frames.add(relative)
            expected = {'relative_path': relative, 'bucket_key': FRAME_PREFIX + f'{identity}/{slot:02d}.png',
                        'size_bytes': row['size_bytes'], 'sha256': row['sha256']}
            require(all(frames[relative].get(k) == v for k, v in expected.items()), 'Bucket PNG/source SHA binding mismatch')
            frame = safe_path(media_root / relative)
            require(frame.is_file() and frame.stat().st_size == row['size_bytes'], 'Current staged frame missing or resized')
            keys.add(expected['bucket_key']); frame_bytes += row['size_bytes']
    require(expected_frames == set(frames) and len(keys) == 8091, 'Exact bucket frame/source receipt set mismatch')
    return {'bucket_receipt': pin(receipt_path), 'frame_cache_receipt': pin(frame_cache_path),
            'frame_cache_complete': pin(completion_path), 'source_count': 899, 'frame_count': 7192,
            'frame_bytes': frame_bytes, 'source_receipt_bytes': source_bytes,
            'exact_frame_and_source_keys': 8091, 'full_oss_get_sha256_attested_by_bucket_receipt': True,
            'local_frame_sha_and_RGB_validation': 'required_next_content_audit_stage',
            'evaluation_scope': 'videomme_only', 'all_benchmark_assets_ready': False}


def cpu_environment():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES='',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    env['PATH'] = '/home/jinhaopeng/.local/bin:' + env.get('PATH', '')
    return env


def run_cpu_stage(root, name, argv):
    require(name in {'content_audit', 'formal_gate', 'manifest', 'cpu_preflight'}, 'Unapproved CPU stage')
    save(root / (name + '.argv.json'), {'argv': argv, 'CUDA_VISIBLE_DEVICES': ''})
    with (root / (name + '.stdout')).open('xb') as out, (root / (name + '.stderr')).open('xb') as err:
        result = subprocess.run(argv, stdout=out, stderr=err, env=cpu_environment(), cwd='/home/jinhaopeng')
    save(root / (name + '.exit.json'), {'returncode': result.returncode,
         'stdout_sha256': sha(root / (name + '.stdout')), 'stderr_sha256': sha(root / (name + '.stderr'))})
    require(result.returncode == 0, name + ' failed; no later stage or submission performed')


def prepare(config, config_path, root):
    require(config.get('kind') == 'videomme_bucket_cpu_preparation_v1' and
            config.get('run_id') == RUN_ID and config.get('experiment_root') == str(EXPERIMENT) and
            config.get('media_root') == str(MEDIA_ROOT), 'Frozen Video-MME preparation identity changed')
    authorization_path = verify(config['authorization']); validate_scope(load(authorization_path))
    submission_helper = verify(config['submission_helper'])
    supplied = config['evaluation_source_files']
    encoded = (json.dumps(supplied, sort_keys=True, separators=(',', ':')) + '\n').encode()
    require(hashlib.sha256(encoded).hexdigest() == FROZEN_SOURCE_LIST_SHA,
            'Accepted frozen evaluation package file list changed')
    for item in supplied: verify(item)
    require(not (BASE / 'evaluations' / RUN_ID).exists(), 'Evaluation run already exists; inspect exact job instead')
    original = P1 / 'formal-source-final-20261009-v1/FORMAL_READY.json'
    verify({'path': str(original), 'sha256': ORIGINAL_READY_SHA})
    restore = P1 / 'benchmark-eval-assets-20261012/checkpoints/CHECKPOINT_RESTORE_VERIFIED.json'
    verify({'path': str(restore), 'sha256': RESTORE_SHA})
    for item in config['training_fingerprint_files']: verify(item)
    registry = original.parent / 'benchmark_source_registry.json'
    bucket_path = MEDIA_ROOT / 'VIDEOMME_BUCKET_MEDIA_COMPLETE.json'
    frame_path = MEDIA_ROOT / 'FRAME_CACHE_RECEIPT.json'
    bucket_identity = verify_bucket_media(bucket_path, frame_path, MEDIA_ROOT / 'FRAME_CACHE_COMPLETE.json', registry, MEDIA_ROOT)
    save(root / 'BUCKET_MEDIA_INPUTS_VERIFIED.json', bucket_identity)
    python, ops = str(BASE / 'env/bin/python'), EXPERIMENT / 'scripts/operations'
    audit_root, formal_root, manifest_root = root / 'content-audit', root / 'formal-gate', root / 'manifest'
    run_cpu_stage(root, 'content_audit', [python, str(ops / 'audit_benchmark_content.py'), 'audit',
        '--formal-gate-dir', str(original.parent), '--formal-ready-sha256', ORIGINAL_READY_SHA,
        '--training-fingerprints', str(P1 / 'benchmark-content-training-20261012-attempt1'),
        '--benchmark-frame-cache-receipt', str(frame_path), '--benchmark-media-root', str(MEDIA_ROOT),
        '--output-dir', str(audit_root)])
    vm_assets = BASE / 'assets/videomme' / REVISION
    mv_assets = BASE / 'assets/mvbench/230a2d4fac8900333c61754641c7a13e069ac9c6'
    run_cpu_stage(root, 'formal_gate', [python, str(ops / 'prepare_formal_source_gate.py'),
        '--videomme-root', str(vm_assets), '--videomme-plan', str(vm_assets / 'ASSET_COMMITTED.json'),
        '--mvbench-root', str(mv_assets), '--mvbench-plan', str(mv_assets / 'ASSET_COMMITTED.json'),
        '--training-data-dir', str(P1 / 'formal-training-data-20261009-v1'),
        '--training-media-root', str(BASE / 'media/llava-p0'), '--videomme-media-root', str(MEDIA_ROOT),
        '--videomme-frame-cache-receipt', str(frame_path), '--content-audit', str(audit_root / 'CONTENT_AUDIT.json'),
        '--output-dir', str(formal_root)])
    scoped = load(formal_root / 'FORMAL_READY.json')
    require(scoped.get('formal_evaluation_ready') is True and scoped.get('benchmark_media_gate') == 'passed' and
            scoped.get('content_duplicate_gate') == 'passed', 'Formal Video-MME media/content gate did not pass')
    scoped.update(evaluation_scope='videomme_only', videomme_evaluation_ready=True,
                  mvbench_evaluation_ready=False, all_benchmark_assets_ready=False, ntu_media_included=False,
                  scope_authorization=pin(authorization_path), bucket_media_binding=bucket_identity,
                  underlying_formal_ready=pin(formal_root / 'FORMAL_READY.json'))
    scoped_ready = formal_root / 'VIDEOMME_ONLY_READY.json'; save(scoped_ready, scoped)
    annotation = vm_assets / 'videomme/test-00000-of-00001.parquet'
    verify({'path': str(annotation), 'sha256': ANNOTATION_SHA})
    run_cpu_stage(root, 'manifest', [python, str(ops / 'prepare_videomme_evaluation.py'),
        '--annotation', str(annotation), '--source-registry', str(registry),
        '--frame-cache-receipt', str(frame_path), '--frame-cache-sha256', sha(frame_path),
        '--media-root', str(MEDIA_ROOT), '--output-dir', str(manifest_root)])
    # Additional pinned receipts are checked again by the unchanged runner on
    # the GPU node. Existing full source and checkpoint checks remain mandatory.
    source_files = [*supplied, pin(Path(__file__).resolve()), pin(config_path), pin(authorization_path),
                    pin(bucket_path), pin(submission_helper)]
    require(len({row['path'] for row in source_files}) == len(source_files), 'Duplicate frozen plan input')
    plan = {'schema_version': 1, 'run_id': RUN_ID, 'run_root': str(BASE / 'evaluations' / RUN_ID),
            'protocol': PROTOCOL, 'evaluation_scope': 'videomme_only',
            'scope_authorization': pin(authorization_path), 'bucket_media_binding': bucket_identity,
            'manifest': pin(manifest_root / 'videomme.jsonl'), 'media_root': str(MEDIA_ROOT),
            'manifest_binding': pin(manifest_root / 'MANIFEST_BINDING.json'),
            'formal_ready': pin(scoped_ready), 'checkpoint_restore_receipt': pin(restore),
            'source_files': source_files, 'storage_roots': [str(BASE), str(P1)],
            'storage_limit_bytes': 99_000_000_000, 'working_reserve_bytes': 1_000_000_000,
            'all_benchmark_assets_ready': False, 'mvbench_evaluation_ready': False,
            'benchmark_score_based_model_selection': False}
    plan_path = root / 'EVALUATION_PLAN.json'; save(plan_path, plan)
    run_cpu_stage(root, 'cpu_preflight', [python, str(ops / 'run_benchmark_eval.py'),
        '--plan', str(plan_path), '--plan-sha256', sha(plan_path), '--preflight-only'])
    preflight = load(root / 'cpu_preflight.stdout')
    require(preflight.get('status') == 'CPU_PREFLIGHT_PASSED' and preflight.get('plan_sha256') == sha(plan_path) and
            preflight.get('num_gpus') == 8 and preflight.get('sequence') == ['r1', 'r2'] and
            preflight.get('subtitles') is False and preflight.get('audio') is False,
            'Full R1/R2 CPU preflight did not pass')
    for item in source_files: verify(item)
    verify({'path': str(original), 'sha256': ORIGINAL_READY_SHA})
    parent_command = [python, str(submission_helper), '--plan', str(plan_path),
                      '--plan-sha256', sha(plan_path), '--output-dir', str(root / 'submission'), '--execute']
    save(root / 'READY_FOR_PARENT_SUBMISSION.json', {'status': 'VIDEOMME_ONLY_CPU_PREPARATION_COMPLETE',
         'observed_utc': datetime.now(timezone.utc).isoformat(), 'plan': pin(plan_path),
         'cpu_preflight': pin(root / 'cpu_preflight.stdout'), 'scope_ready': pin(scoped_ready),
         'parent_submission_argv': parent_command, 'gpu_submitted': False,
         'remaining_admission': 'parent-owned fresh exact OSS/job/Kubernetes/stream-low-one-node-eight-GPU-restartNever dry-run and one-submit intent',
         'evaluation_scope': 'videomme_only', 'all_benchmark_assets_ready': False,
         'mvbench_evaluation_ready': False, 'ntu_media_included': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    config_path = verify({'path': str(args.config), 'sha256': args.config_sha256})
    config = load(config_path)
    require(verify(config['pipeline_script']).resolve() == Path(__file__).resolve(),
            'Configuration must pin this exact preparation script')
    root = safe_path(args.output_dir)
    require(root.is_relative_to(P1) and not root.exists(), 'New preparation output path under P1 required')
    lock_path = P1 / 'videomme-bucket-cpu-preparation.lock'
    with safe_path(lock_path).open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir(parents=True, exist_ok=False)
        save(root / 'CPU_PREPARATION_STARTED.json', {'config': pin(config_path),
             'script': pin(Path(__file__).resolve()), 'gpu_submitted': False,
             'observed_utc': datetime.now(timezone.utc).isoformat()})
        try:
            prepare(config, config_path, root)
        except Exception as error:
            save(root / 'CPU_PREPARATION_FAILED.json', {'error_type': type(error).__name__, 'error': str(error),
                 'gpu_submitted': False, 'automatic_retry': False,
                 'observed_utc': datetime.now(timezone.utc).isoformat()})
            raise


if __name__ == '__main__':
    main()
