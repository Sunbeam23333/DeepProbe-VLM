#!/usr/bin/env python3
"""CPU-only prepare of the fresh R4 seed29 alignment cell in the exact v9 release."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


h = load('declared_v9_alignment_shared_cpu', Path(__file__).with_name('prepare_declared_v9_noalign_cpu.py'))
OUTPUT = h.P1 / 'formal-preflight-v9-r4-s29-alignment-20261021-prepare-1'
CONFIG_RELATIVE = 'configs/train/declared_v9/ouro_r4_alignment_s29.json'


def selected_config(module):
    config = h.EXPERIMENT / CONFIG_RELATIVE
    raw = h.read(config)
    row = [item for item in h.read(h.RELEASE / 'CONFIG_BINDINGS.json')['configs'] if item['config'] == CONFIG_RELATIVE]
    h.require(len(row) == 1 and row[0]['stage'] == 'alignment' and row[0]['method'] == 'two_stage'
              and row[0]['seed'] == 29 and h.sha(config) == row[0]['sha256'], 'Wrong fresh R4 seed29 alignment configuration')
    recipe = module.validate_recipe(raw, 'alignment')
    h.require(raw['model']['num_loops'] == 4 and raw['model']['language_model'] == 'ByteDance/Ouro-1.4B'
              and raw['training']['seed'] == 29 and raw['training']['stage'] == 'alignment'
              and raw['training']['save_total_limit'] == 1 and raw['experiment_protocol']['method'] == 'two_stage',
              'R4 seed29 alignment recipe differs')
    h.require(module.declared.initialization_mode(raw) == 'pretrained_backbones_random_connector', 'Alignment requires fresh pretrained initialization')
    return raw, recipe


def capacity(module, runtime):
    h.require(h.BASE.stat().st_dev == h.P1.stat().st_dev, 'Staging roots must share filesystem')
    counts = module.declared.staging_bytes([h.BASE, h.P1])
    used = sum(counts.values())
    plan = module.estimate_stage_storage(runtime, runtime['training']['stage'])
    h.require(runtime['training']['stage'] == 'alignment' and runtime['training']['save_total_limit'] == 1
              and plan['save_total_limit'] == 1 and plan['required_additional_bytes'] == 18_070_438_560,
              'Fresh R4 alignment must reserve its actual limit1 storage peak')
    free = shutil.disk_usage(h.BASE).free
    charged = used + h.OTHER_MEDIA_RESERVE + h.OTHER_CPU_RESERVE
    snapshot = {'utc': h.now(), 'root_bytes': counts, 'combined_accounted_bytes': used,
                'other_media_full_cap_bytes': h.OTHER_MEDIA_RESERVE, 'other_cpu_reserve_bytes': h.OTHER_CPU_RESERVE,
                'plan': plan, 'projected_bytes': charged + plan['required_additional_bytes'], 'filesystem_free_bytes': free}
    h.write(OUTPUT / 'CAPACITY_SNAPSHOT.json', snapshot)
    snapshot['admission'] = module.require_storage_admission(charged, free, plan)
    h.require(free >= plan['required_additional_bytes'] + h.OTHER_MEDIA_RESERVE + h.OTHER_CPU_RESERVE + 1_000_000_000,
              'Shared free-space guard failed')
    return snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-manifest-sha256', required=True)
    args = parser.parse_args(argv)
    h.require(Path(__file__).resolve() == h.EXPERIMENT / 'scripts/operations/prepare_declared_v9_alignment_cpu.py',
              'Execute only the new reviewed frozen release')
    h.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'CPU-only environment required')
    release = h.verify_release(args.release_manifest_sha256)
    h.require(not OUTPUT.exists() and not OUTPUT.is_symlink(), 'Never overwrite or rerun a CPU attempt directory')
    OUTPUT.mkdir()
    (OUTPUT / 'tmp').mkdir()
    (OUTPUT / 'cache').mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', HF_HOME=str(h.BASE / 'hf'), HF_HUB_CACHE=str(h.HUB),
               TMPDIR=str(OUTPUT / 'tmp'), XDG_CACHE_HOME=str(OUTPUT / 'cache'), HF_MODULES_CACHE=str(OUTPUT / 'cache/modules'),
               PYTHONPATH=str(h.EXPERIMENT / 'src'), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    progress = {'schema_version': 1, 'started_utc': h.now(), 'mode': 'prepare_only', 'release': release,
                'selected_cell': 'ouro_r4_alignment_s29', 'ready_for_gpu_submission': False, 'gpu_job_submitted': False,
                'scheduler_gate_checked': False, 'scheduler_dry_run_executed': False, 'oss_checked': False, 'oss_mutated': False}
    h.write(OUTPUT / 'CPU_PREPARATION_INTENT.json', progress)
    try:
        module = h.load_runner()
        progress['source'] = module.verify_source_manifest(h.EXPERIMENT, h.RELEASE / 'P1_SOURCE_MANIFEST.json')
        progress['configuration_checks'] = h.verify_configs(module)
        for item in progress['configuration_checks']:
            item['pending'] = 'This CPU execution prepares only the R4 seed29 alignment cell; every other cell retains its own future admission.'
        selected_config(module)
        print('START lightweight current capacity before full hashes', flush=True)
        inspector = load('declared_v9_alignment_capacity', Path(__file__).with_name('inspect_declared_v9_capacity.py'))
        preview = inspector.capacity_inspection(module)
        h.write(OUTPUT / 'CAPACITY_PRECHECK.json', preview)
        candidates = [row for row in preview['cells'] if row['config'] == CONFIG_RELATIVE]
        h.require(len(candidates) == 1 and candidates[0]['capacity_admitted'] is True,
                  'Current lightweight capacity is gated; do not start full data/model/token verification')
        progress['capacity_precheck'] = preview
        h.require(h.sha(h.FORMAL_READY) == h.FIXED_FORMAL_SHA, 'Formal-ready pin changed')
        print('START complete current formal training/media hashes', flush=True)
        formal = module.verify_formal_inputs(h.DATA, h.FORMAL_READY, h.MEDIA)
        h.require(formal['training_media']['file_count'] == 3979 and formal['training_media']['total_bytes'] == 5_168_858_321,
                  'Formal media count/bytes changed')
        counts = h.read(h.DATA / 'READY.json')['counts']
        h.require(counts['selected_qa'] == 96298 and counts['train'] == {'qa': 91826, 'source_videos': 3794}
                  and counts['validation'] == {'qa': 4472, 'source_videos': 185}, 'Formal QA/source counts changed')
        progress['formal'] = formal
        progress['decode'] = h.verify_decode(formal)
        print('START fixed pretrained backbone byte inventories', flush=True)
        runtime_path = OUTPUT / 'config.runtime.json'
        recipe, offline = module.prepare_offline_config(h.EXPERIMENT / CONFIG_RELATIVE, runtime_path, h.HUB, 'alignment')
        expected = h.read(h.RELEASE / 'PINNED_OFFLINE_MODEL_MAP.json')
        h.require(offline == expected['offline'], 'Pretrained snapshot files differ from accepted original model bytes')
        runtime = h.read(runtime_path)
        h.require(module.declared.initialization_mode(runtime) == 'pretrained_backbones_random_connector'
                  and runtime['training']['seed'] == 29 and runtime['training']['stage'] == 'alignment'
                  and runtime['experiment_protocol']['method'] == 'two_stage', 'Runtime fresh alignment identity differs')
        progress.update(requested_config_sha256=h.sha(h.EXPERIMENT / CONFIG_RELATIVE), runtime_config_sha256=h.sha(runtime_path), recipe=recipe,
                        offline=offline, initialization={'kind': 'pretrained_backbones_random_connector', 'seed': 29, 'method': 'two_stage',
                        'alignment_final': None, 'resume_checkpoint': None, 'actual_connector_state_sha256': None,
                        'connector_state_boundary': 'Native training must construct and hash the seeded connector; CPU preparation never loads model weights.'})
        print('START complete 96298-QA fresh-tokenizer budget', flush=True)
        command = [str(h.PYTHON), '-B', str(h.EXPERIMENT / 'scripts/data/check_token_budget.py'), str(h.DATA / 'train.jsonl'), str(h.DATA / 'validation.jsonl'),
                   '--config', str(runtime_path), '--mode', 'training', '--local-files-only', '--report', str(OUTPUT / 'TOKEN_BUDGET.json')]
        call = subprocess.run(command, env=env, capture_output=True, text=True, timeout=1200, check=False)
        (OUTPUT / 'token_budget.stdout').write_text(call.stdout)
        (OUTPUT / 'token_budget.stderr').write_text(call.stderr)
        h.write(OUTPUT / 'TOKEN_COMMAND.json', {'argv': command, 'returncode': call.returncode,
                'stdout_sha256': h.sha(OUTPUT / 'token_budget.stdout'), 'stderr_sha256': h.sha(OUTPUT / 'token_budget.stderr')})
        h.require(call.returncode == 0, 'Full token checker failed')
        tokens = h.read(OUTPUT / 'TOKEN_BUDGET.json')
        h.verify_tokens(tokens, h.sha(runtime_path), formal, runtime)
        progress['training_token_budget'] = tokens
        progress['token_report_sha256'] = h.sha(OUTPUT / 'TOKEN_BUDGET.json')
        h.require(module.verify_formal_inputs(h.DATA, h.FORMAL_READY, h.MEDIA) == formal, 'Formal media/data changed during preparation')
        h.require(module.verify_source_manifest(h.EXPERIMENT, h.RELEASE / 'P1_SOURCE_MANIFEST.json') == progress['source'], 'Source changed during preparation')
        h.write(OUTPUT / 'CPU_INPUTS_VERIFIED_PENDING_CAPACITY.json', progress)
        print('START fresh final dual-root alignment capacity', flush=True)
        progress['capacity'] = capacity(module, runtime)
        progress.update(status='CPU_PREPARED_PENDING_INDEPENDENT_AND_SCHEDULER_GATE', completed_utc=h.now(),
                        cpu_preparation_passed=True, expected_optimizer_steps=718,
                        next_step='Independent current CPU acceptance and root-owned scheduler/OSS/dry-run admission required; checker never submits.')
        h.write(OUTPUT / 'CPU_PREPARED_PENDING_SCHEDULER_GATE.json', progress)
        print(json.dumps({'status': progress['status'], 'receipt_sha256': h.sha(OUTPUT / 'CPU_PREPARED_PENDING_SCHEDULER_GATE.json'),
                          'ready_for_gpu_submission': False}), flush=True)
        return 0
    except Exception as exc:
        progress.update(status='CPU_PREPARATION_FAILED_GATED', failed_utc=h.now(), error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        h.write(OUTPUT / 'CPU_PREPARATION_FAILED.json', progress)
        print(json.dumps({'status': progress['status'], 'error': progress['error'],
                          'receipt_sha256': h.sha(OUTPUT / 'CPU_PREPARATION_FAILED.json')}), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
