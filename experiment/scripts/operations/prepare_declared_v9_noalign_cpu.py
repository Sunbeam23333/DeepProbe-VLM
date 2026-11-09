#!/usr/bin/env python3
"""Prepare the declared no-alignment inputs on CPU; no scheduler or OSS calls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True
BASE = Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM')
P1 = Path(str(BASE) + '-p1')
RELEASE = P1 / 'releases/deepprobe-20261021-declared-v9'
EXPERIMENT = RELEASE / 'experiment'
OUTPUT = P1 / 'formal-preflight-v9-noalign-r4-s17-20261021-prepare-1'
DATA = P1 / 'formal-training-data-20261009-v1'
FORMAL_READY = P1 / 'formal-source-final-20261009-v1/FORMAL_READY.json'
MEDIA = BASE / 'media/llava-p0'
DECODE = P1 / 'formal-preflight-20261009-v1/media-decode-r1-f8.json'
PYTHON = BASE / 'env/bin/python'
HUB = BASE / 'hf/hub'
CONFIG_RELATIVE = 'configs/train/declared_v9/ouro_r4_sft_s17_no_alignment.json'
FIXED_FORMAL_SHA = '1c08b9e18d7e9f2bcb9664ae81225603c94585ffabfcebe5b7b9bd01fc9ca66c'
FIXED_DECODE_SHA = 'c946d14c6e8dad6277358b0f1e4143b91939ffd767f8eab67d70f80eec1ec6f2'
OTHER_MEDIA_RESERVE = 6_000_000_000
OTHER_CPU_RESERVE = 1_000_000_000


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), 'JSON object required')
    return value


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def verify_release(manifest_sha):
    manifest_path = RELEASE / 'RELEASE_MANIFEST.json'
    require(sha(manifest_path) == manifest_sha, 'Release manifest SHA changed')
    manifest = read(manifest_path)
    require(manifest.get('status') == 'complete_source_inventory' and manifest.get('variant') == 'declared-v9', 'Wrong release kind')
    rows = manifest.get('files')
    require(isinstance(rows, list) and len(rows) == manifest['file_count'], 'Release count mismatch')
    names = set()
    for row in rows:
        relative = Path(row['path'])
        require(not relative.is_absolute() and '..' not in relative.parts and row['path'] not in names, 'Unsafe release path')
        names.add(row['path'])
        path = RELEASE / relative
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Release byte mismatch: ' + str(relative))
    observed = set()
    for path in (RELEASE, *RELEASE.rglob('*')):
        mode = path.lstat().st_mode
        require(not stat.S_ISLNK(mode) and not mode & 0o222, 'Release must be read only and have no symlinks')
        if stat.S_ISREG(mode):
            observed.add(path.relative_to(RELEASE).as_posix())
        else:
            require(stat.S_ISDIR(mode), 'Nonregular release entry')
    require(observed == names | {'RELEASE_MANIFEST.json'}, 'Release has unexpected files')
    require(sum(row['bytes'] for row in rows) == manifest['total_bytes'], 'Release byte total mismatch')
    require(sha(RELEASE / 'P1_SOURCE_MANIFEST.json') == manifest['p1_source_manifest_sha256'], 'Runtime source manifest changed')
    return {'manifest_sha256': manifest_sha, 'files_verified': len(rows), 'total_bytes': manifest['total_bytes'],
            'source_manifest_sha256': manifest['p1_source_manifest_sha256']}


def load_runner():
    path = EXPERIMENT / 'scripts/operations/run_p1_stage_v9.py'
    spec = importlib.util.spec_from_file_location('frozen_v9_noalign_cpu_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_configs(module):
    bindings = read(RELEASE / 'CONFIG_BINDINGS.json')
    rows = bindings['configs']
    require(len(rows) == 17 and len({r['config'] for r in rows}) == 17, 'Exact 17 configurations required')
    result = []
    for row in rows:
        for key in ('config', 'baseline'):
            value = row[key]
            require(isinstance(value, str) and value and not Path(value).is_absolute()
                    and '..' not in Path(value).parts, 'Unsafe config/baseline binding path')
        config, baseline = EXPERIMENT / row['config'], RELEASE / row['baseline']
        require(config.resolve().is_relative_to(EXPERIMENT.resolve()) and baseline.resolve().is_relative_to(RELEASE.resolve())
                and config.is_file() and baseline.is_file() and not config.is_symlink() and not baseline.is_symlink(),
                'Config/baseline binding must stay inside reviewed release')
        require(sha(config) == row['sha256'] and sha(baseline) == row['baseline_sha256'], 'Config/baseline bytes changed')
        raw, previous = read(config), read(baseline)
        restored = json.loads(json.dumps(raw))
        restored.pop('experiment_protocol')
        restored['training']['seed'] = 17
        require(restored == previous, 'Configuration differs beyond frozen seed/declaration')
        recipe = module.validate_recipe(raw, row['stage'])
        require(raw['experiment_protocol']['method'] == row['method'] and raw['training']['seed'] == row['seed'], 'Config identity differs')
        result.append({'config': row['config'], 'sha256': row['sha256'], 'recipe': recipe,
                       'cpu_inputs_prepared': False, 'ready_for_gpu_submission': False,
                       'pending': 'This CPU execution prepares only the R4 seed17 no-alignment cell; every other cell retains its own future admission.'})
    selected = [row for row in rows if row['config'] == CONFIG_RELATIVE]
    require(len(selected) == 1 and selected[0]['method'] == 'no_alignment' and selected[0]['seed'] == 17 and selected[0]['stage'] == 'sft', 'Wrong selected no-alignment cell')
    config = EXPERIMENT / CONFIG_RELATIVE
    mode = module.declared.initialization_mode(read(config), no_alignment=True)
    require(mode == 'pretrained_backbones_random_connector', 'No-alignment initializer changed')
    return result


def verify_decode(formal):
    require(sha(DECODE) == FIXED_DECODE_SHA, 'Full CPU decode evidence changed')
    document = read(DECODE)
    clips = document.get('clips')
    require(document.get('status') == 'all_selected_media_decoded_and_processed' and document.get('device') == 'cpu' and document.get('gpu_validated') is False
            and document.get('unique_clips') == 3979 and isinstance(clips, list) and len(clips) == 3979, 'Full decode receipt incomplete')
    require(all(c.get('decoded_frames') == 8 and c.get('pixel_shape') == [8, 3, 384, 384] and c.get('finite') is True for c in clips), 'Failed decode row')
    require(len({c.get('video_path') for c in clips}) == len({c.get('source_video_id') for c in clips}) == 3979, 'Duplicate decode sources/paths')
    require(document.get('manifest_sha256') == {n: formal['data_sha256'][n] for n in ('train.jsonl', 'validation.jsonl')}, 'Decode does not bind current data')
    inventory = read(DATA / 'training_media_inventory.json')
    require({(c['video_path'], c['source_video_id']) for c in clips} ==
            {(c['relative_path'], c['source_video_id']) for c in inventory['entries']},
            'Decode source/path mapping differs from current media inventory')
    return {'sha256': FIXED_DECODE_SHA, 'clips': 3979, 'all_eight_frames_finite': True,
            'boundary': 'Prior complete CPU decode receipt revalidated against current fully hashed source/media; no redundant video re-decode.'}


def verify_tokens(report, runtime_sha, formal, runtime):
    require(report.get('passed') is True and report.get('samples') == 96298 and report.get('overflow_count') == 0
            and report.get('overflow_samples') == [] and report.get('config_sha256') == runtime_sha, 'Full token budget failed')
    require(report.get('mode') == 'training' and report.get('include_answer') is True and report.get('max_text_tokens') == 512 and report.get('num_visual_tokens') == 128,
            'Token protocol changed')
    require(report.get('manifest_sha256') == {str(DATA / n): formal['data_sha256'][n] for n in ('train.jsonl', 'validation.jsonl')}, 'Token data identity differs')
    require(report.get('language_model') == runtime['model']['language_model'] and report.get('language_revision') == runtime['model']['language_revision']
            and report.get('tokenizer_source') == runtime['model']['language_model'] and 'checkpoint_tokenizer_sha256' not in report, 'Fresh tokenizer identity differs')
    require(report.get('prompt_source_sha256') == sha(EXPERIMENT / 'src/deepprobe_vlm/data.py') and report.get('media_decoded') is False and report.get('model_weights_loaded') is False,
            'Token implementation/scope changed')


def capacity(module, runtime):
    roots = [BASE, P1]
    require(BASE.stat().st_dev == P1.stat().st_dev, 'Staging roots must share filesystem')
    counts = module.declared.staging_bytes(roots)
    used = sum(counts.values())
    plan = module.estimate_stage_storage(runtime, runtime['training']['stage'])
    require(runtime['training']['stage'] == 'sft' and plan['save_total_limit'] == 1 and plan['required_additional_bytes'] == 47_884_490_928,
            'No-alignment must reserve the complete SFT storage peak')
    charged = used + OTHER_MEDIA_RESERVE + OTHER_CPU_RESERVE
    free = shutil.disk_usage(BASE).free
    snapshot = {'utc': now(), 'root_bytes': counts, 'combined_accounted_bytes': used,
                'other_media_full_cap_bytes': OTHER_MEDIA_RESERVE, 'other_cpu_reserve_bytes': OTHER_CPU_RESERVE,
                'plan': plan, 'projected_bytes': charged + plan['required_additional_bytes'], 'filesystem_free_bytes': free}
    write(OUTPUT / 'CAPACITY_SNAPSHOT.json', snapshot)
    snapshot['admission'] = module.require_storage_admission(charged, free, plan)
    require(free >= plan['required_additional_bytes'] + OTHER_MEDIA_RESERVE + OTHER_CPU_RESERVE + 1_000_000_000, 'Shared free-space guard failed')
    return snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-manifest-sha256', required=True)
    args = parser.parse_args(argv)
    require(Path(__file__).resolve() == EXPERIMENT / 'scripts/operations/prepare_declared_v9_noalign_cpu.py', 'Execute only the new reviewed frozen release')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'CPU-only environment required')
    release = verify_release(args.release_manifest_sha256)
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), 'Never overwrite or rerun a CPU attempt directory')
    OUTPUT.mkdir()
    (OUTPUT / 'tmp').mkdir(); (OUTPUT / 'cache').mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', HF_HOME=str(BASE / 'hf'), HF_HUB_CACHE=str(HUB),
               TMPDIR=str(OUTPUT / 'tmp'), XDG_CACHE_HOME=str(OUTPUT / 'cache'), HF_MODULES_CACHE=str(OUTPUT / 'cache/modules'),
               PYTHONPATH=str(EXPERIMENT / 'src'), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    progress = {'schema_version': 1, 'started_utc': now(), 'mode': 'prepare_only', 'release': release,
                'ready_for_gpu_submission': False, 'gpu_job_submitted': False, 'scheduler_gate_checked': False,
                'scheduler_dry_run_executed': False, 'oss_checked': False, 'oss_mutated': False}
    write(OUTPUT / 'CPU_PREPARATION_INTENT.json', progress)
    try:
        print('START exact source and all declared configurations', flush=True)
        module = load_runner()
        progress['source'] = module.verify_source_manifest(EXPERIMENT, RELEASE / 'P1_SOURCE_MANIFEST.json')
        progress['configuration_checks'] = verify_configs(module)
        require(sha(FORMAL_READY) == FIXED_FORMAL_SHA, 'Formal-ready pin changed')
        print('START full current formal training/media hashes', flush=True)
        formal = module.verify_formal_inputs(DATA, FORMAL_READY, MEDIA)
        require(formal['training_media']['file_count'] == 3979 and formal['training_media']['total_bytes'] == 5_168_858_321, 'Formal media count/bytes changed')
        counts = read(DATA / 'READY.json')['counts']
        require(counts['selected_qa'] == 96298 and counts['train'] == {'qa': 91826, 'source_videos': 3794}
                and counts['validation'] == {'qa': 4472, 'source_videos': 185}, 'Formal QA/source counts changed')
        progress['formal'] = formal
        progress['decode'] = verify_decode(formal)
        print('START fixed pretrained backbone byte inventories', flush=True)
        runtime_path = OUTPUT / 'config.runtime.json'
        recipe, offline = module.prepare_offline_config(EXPERIMENT / CONFIG_RELATIVE, runtime_path, HUB, 'sft')
        expected_models = read(RELEASE / 'PINNED_OFFLINE_MODEL_MAP.json')
        require(offline == expected_models['offline'], 'Pretrained snapshot files differ from accepted original model bytes')
        runtime = read(runtime_path)
        require(module.declared.initialization_mode(runtime, no_alignment=True) == 'pretrained_backbones_random_connector', 'Runtime no-alignment identity differs')
        progress.update(requested_config_sha256=sha(EXPERIMENT / CONFIG_RELATIVE), runtime_config_sha256=sha(runtime_path), recipe=recipe,
                        offline=offline, initialization={'kind': 'pretrained_backbones_random_connector', 'seed': 17, 'method': 'no_alignment',
                        'alignment_final': None, 'resume_checkpoint': None, 'actual_connector_state_sha256': None,
                        'connector_state_boundary': 'The actual seeded connector is constructed and hashed by native training; CPU preparation never loads model weights.'})
        print('START complete 96298-QA fresh-tokenizer budget', flush=True)
        command = [str(PYTHON), '-B', str(EXPERIMENT / 'scripts/data/check_token_budget.py'), str(DATA / 'train.jsonl'), str(DATA / 'validation.jsonl'),
                   '--config', str(runtime_path), '--mode', 'training', '--local-files-only', '--report', str(OUTPUT / 'TOKEN_BUDGET.json')]
        call = subprocess.run(command, env=env, capture_output=True, text=True, timeout=1200, check=False)
        (OUTPUT / 'token_budget.stdout').write_text(call.stdout); (OUTPUT / 'token_budget.stderr').write_text(call.stderr)
        write(OUTPUT / 'TOKEN_COMMAND.json', {'argv': command, 'returncode': call.returncode, 'stdout_sha256': sha(OUTPUT / 'token_budget.stdout'), 'stderr_sha256': sha(OUTPUT / 'token_budget.stderr')})
        require(call.returncode == 0, 'Full token checker failed')
        tokens = read(OUTPUT / 'TOKEN_BUDGET.json')
        verify_tokens(tokens, sha(runtime_path), formal, runtime)
        progress['training_token_budget'] = tokens
        progress['token_report_sha256'] = sha(OUTPUT / 'TOKEN_BUDGET.json')
        require(module.verify_formal_inputs(DATA, FORMAL_READY, MEDIA) == formal, 'Formal media/data changed during preparation')
        require(module.verify_source_manifest(EXPERIMENT, RELEASE / 'P1_SOURCE_MANIFEST.json') == progress['source'], 'Source changed during preparation')
        write(OUTPUT / 'CPU_INPUTS_VERIFIED_PENDING_CAPACITY.json', progress)
        print('START fresh dual-root SFT capacity', flush=True)
        progress['capacity'] = capacity(module, runtime)
        progress.update(status='CPU_PREPARED_PENDING_INDEPENDENT_AND_SCHEDULER_GATE', completed_utc=now(), cpu_preparation_passed=True,
                        total_training_budget_matches_two_stage=False, expected_optimizer_steps=718,
                        next_step='Independent current CPU acceptance and later full scheduler/OSS/dry-run admission owned by root; never submit from this checker.')
        write(OUTPUT / 'CPU_PREPARED_PENDING_SCHEDULER_GATE.json', progress)
        print(json.dumps({'status': progress['status'], 'receipt_sha256': sha(OUTPUT / 'CPU_PREPARED_PENDING_SCHEDULER_GATE.json'), 'ready_for_gpu_submission': False}), flush=True)
        return 0
    except Exception as exc:
        progress.update(status='CPU_PREPARATION_FAILED_GATED', failed_utc=now(), error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        write(OUTPUT / 'CPU_PREPARATION_FAILED.json', progress)
        print(json.dumps({'status': progress['status'], 'error': progress['error'], 'receipt_sha256': sha(OUTPUT / 'CPU_PREPARATION_FAILED.json')}), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
