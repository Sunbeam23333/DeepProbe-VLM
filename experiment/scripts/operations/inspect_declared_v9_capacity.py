#!/usr/bin/env python3
"""Read-only capacity inspection before any full v9 model/media/data hashing.

Only the reviewed release, small model configs, safetensors headers, and file
metadata are read. A capacity pass is not full CPU or GPU admission.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys

sys.dont_write_bytecode = True
BASE = Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM')
P1 = Path(str(BASE) + '-p1')
RELEASE = P1 / 'releases/deepprobe-20261021-declared-v9'
EXPERIMENT = RELEASE / 'experiment'
HUB = BASE / 'hf/hub'
OTHER_MEDIA_RESERVE = 6_000_000_000
OTHER_CPU_RESERVE = 1_000_000_000
MODEL_SAVED_BYTES = 7_480_748_488
PROJECTOR_SAVED_BYTES = 27_235_396
CELLS = (
    ('configs/train/declared_v9/ouro_r4_sft_s17_no_alignment.json', 'sft', 17, 'no_alignment', 47_884_490_928),
    ('configs/train/declared_v9/ouro_r4_alignment_s29.json', 'alignment', 29, 'two_stage', 18_070_438_560),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), 'JSON object required: ' + str(path))
    return value


def boundary():
    return {'schema_version': 1, 'capacity_only': True, 'ready_for_gpu_submission': False,
            'full_cpu_preparation_passed': False, 'gpu_job_submitted': False,
            'scheduler_gate_checked': False, 'oss_checked': False, 'oss_mutated': False,
            'model_payload_hashed': False, 'model_weights_loaded': False,
            'training_data_or_media_hashed': False, 'token_budget_checked': False,
            'files_written': False,
            'limitation': 'Model configuration bytes, weight headers and file metadata only; model payload identity, full formal inputs, tokens, scheduler, OSS and GPU admission remain unverified.'}


def require_cpu_environment():
    expected = {'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
                'PYTHONNOUSERSITE': '1', 'HF_HUB_OFFLINE': '1',
                'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
                'HF_HOME': str(BASE / 'hf'), 'HF_HUB_CACHE': str(HUB)}
    require(all(os.environ.get(key) == value for key, value in expected.items()),
            'Fixed CPU-only offline environment required')


def verify_snapshot_metadata(row, identifier, revision):
    """Check identity and stat every mapped file, hashing only small config JSON."""
    expected = HUB / ('models--' + identifier.replace('/', '--')) / 'snapshots' / revision
    require(isinstance(row, dict) and row.get('configured_id') == identifier
            and row.get('revision') == revision and row.get('resolved_path') == str(expected),
            'Pinned offline ID/revision/path changed')
    require(expected.is_dir() and not expected.is_symlink() and expected.resolve() == expected,
            'Missing or aliased pinned snapshot directory')
    files = row.get('files')
    require(isinstance(files, dict) and 'config.json' in files
            and any(name.endswith('.safetensors') for name in files), 'Incomplete pinned offline file map')
    for relative, item in files.items():
        require(isinstance(relative, str) and relative and not Path(relative).is_absolute()
                and '..' not in Path(relative).parts and isinstance(item, dict)
                and type(item.get('bytes')) is int and item['bytes'] > 0
                and isinstance(item.get('sha256'), str) and re.fullmatch(r'[0-9a-f]{64}', item['sha256']),
                'Unsafe pinned offline file entry')
    require(type(row.get('total_bytes')) is int and row['total_bytes'] == sum(item['bytes'] for item in files.values()),
            'Pinned offline file byte total differs')
    observed = set()
    for path in expected.rglob('*'):
        if path.is_dir():
            require(not path.is_symlink(), 'Snapshot directory symlink forbidden')
            continue
        target = path.resolve(strict=True)
        require(any(target.is_relative_to(root) for root in (BASE, P1))
                and stat.S_ISREG(target.stat().st_mode), 'Snapshot file escapes accounted roots or is nonregular')
        relative = path.relative_to(expected).as_posix()
        require(relative in files and target.stat().st_size == files[relative]['bytes'],
                'Current snapshot file set or size differs: ' + relative)
        observed.add(relative)
    require(observed == set(files), 'Current snapshot is missing pinned files')
    config = expected / 'config.json'
    require(config.stat().st_size <= 1_000_000, 'Unexpectedly large model configuration')
    payload = config.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == files['config.json']['sha256'],
            'Current model configuration bytes changed')
    require(isinstance(json.loads(payload), dict), 'Model configuration must be an object')
    return {'path': str(expected), 'id': identifier, 'revision': revision,
            'files_statted': len(files), 'mapped_bytes': row['total_bytes'],
            'config_sha256': files['config.json']['sha256'], 'model_payload_hashed': False}


def memory_runtime(module, config, stage, seed, method, pinned_map):
    raw = read(EXPERIMENT / config)
    require(raw.get('training', {}).get('stage') == stage and raw['training'].get('seed') == seed
            and raw.get('experiment_protocol', {}).get('method') == method
            and raw.get('model', {}).get('num_loops') == 4, 'Selected R4 cell identity changed')
    module.validate_recipe(raw, stage)
    require(module.declared.initialization_mode(raw, no_alignment=method == 'no_alignment')
            == 'pretrained_backbones_random_connector', 'Both inspected cells must start fresh')
    offline = pinned_map.get('offline')
    require(isinstance(offline, dict) and set(offline) == {'language', 'vision'}, 'Expected exactly two pinned backbones')
    runtime = json.loads(json.dumps(raw))
    evidence = {}
    specs = (('language', 'language_model', 'language_revision', 'ByteDance/Ouro-1.4B',
              module.declared.LANGUAGE_REVISIONS['ByteDance/Ouro-1.4B']),
             ('vision', 'vision_model', 'vision_revision', module.declared.VISION_ID, module.declared.VISION_REVISION))
    for label, model_key, revision_key, identifier, revision in specs:
        require(raw['model'][model_key] == identifier and raw['model'][revision_key] == revision,
                'Selected config backbone differs')
        evidence[label] = verify_snapshot_metadata(offline[label], identifier, revision)
        runtime['model'][model_key] = offline[label]['resolved_path']
    return runtime, evidence


def capacity_inspection(module, pinned_map=None):
    """Return both cell verdicts; caller must authenticate the frozen release first.

    This function performs no write, subprocess, network or full payload hash.
    Shared filesystem/accounting errors raise; per-cell failures stay gated so a
    blocked SFT capacity does not prevent inspection of the alignment cell.
    """
    require_cpu_environment()
    require(all(root.is_dir() and not root.is_symlink() and root.resolve() == root for root in (BASE, P1)),
            'Fixed staging roots must be regular canonical directories')
    require(BASE.stat().st_dev == P1.stat().st_dev, 'Staging roots must share filesystem')
    pinned_map = read(RELEASE / 'PINNED_OFFLINE_MODEL_MAP.json') if pinned_map is None else pinned_map
    require(isinstance(pinned_map, dict), 'Pinned offline map must be an object')
    root_bytes = module.declared.staging_bytes([BASE, P1])
    require(set(root_bytes) == {str(BASE), str(P1)} and all(type(v) is int and v >= 0 for v in root_bytes.values()),
            'Invalid dual-root byte inventory')
    used = sum(root_bytes.values())
    free = shutil.disk_usage(BASE).free
    require(type(free) is int and free >= 0, 'Invalid filesystem free bytes')
    result = {**boundary(), 'status': 'CAPACITY_ONLY_CHECKED_NOT_GPU_ADMITTED',
              'observed_utc': datetime.now(timezone.utc).isoformat(), 'root_bytes': root_bytes,
              'combined_accounted_bytes': used, 'other_media_full_cap_bytes': OTHER_MEDIA_RESERVE,
              'other_cpu_reserve_bytes': OTHER_CPU_RESERVE, 'filesystem_free_bytes': free, 'cells': []}
    charged = used + OTHER_MEDIA_RESERVE + OTHER_CPU_RESERVE
    for config, stage, seed, method, expected_reserve in CELLS:
        cell = {'config': config, 'stage': stage, 'seed': seed, 'method': method,
                'capacity_admitted': False, 'ready_for_gpu_submission': False, 'plan': None,
                'expected_required_additional_bytes': expected_reserve}
        try:
            runtime, snapshots = memory_runtime(module, config, stage, seed, method, pinned_map)
            plan = module.estimate_stage_storage(runtime, stage)
            cell['plan'] = {key: value for key, value in plan.items() if key != 'components'}
            shapes = plan['components']['projector']
            require(plan['stage'] == stage and plan['save_total_limit'] == 1
                    and type(plan['required_additional_bytes']) is int and plan['required_additional_bytes'] == expected_reserve
                    and plan['model_saved_bytes'] == MODEL_SAVED_BYTES and plan['projector_saved_bytes'] == PROJECTOR_SAVED_BYTES
                    and shapes['language_hidden_size'] == 2048 and shapes['vision_hidden_size'] == 1152,
                    'Current model header/shape-derived reserve differs from the pinned R4 plan')
            cell.update(snapshot_metadata=snapshots, projector_shape=shapes,
                        projected_bytes=charged + expected_reserve,
                        required_filesystem_free_bytes=expected_reserve + OTHER_MEDIA_RESERVE + OTHER_CPU_RESERVE + 1_000_000_000)
            admission = module.require_storage_admission(charged, free, plan)
            require(free >= cell['required_filesystem_free_bytes'], 'Shared free-space guard failed')
            cell.update(capacity_admitted=True, status='CAPACITY_ONLY_PASSED',
                        admission={key: value for key, value in admission.items() if key != 'plan'})
        except Exception as exc:
            cell.update(status='CAPACITY_ONLY_GATED', error=f'{type(exc).__name__}: {exc}')
        result['cells'].append(cell)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--release-manifest-sha256', required=True)
    args = parser.parse_args(argv)
    try:
        require(re.fullmatch(r'[0-9a-f]{64}', args.release_manifest_sha256), 'Exact release manifest SHA required')
        require(Path(__file__).resolve() == EXPERIMENT / 'scripts/operations/inspect_declared_v9_capacity.py',
                'Execute only the new reviewed frozen release')
        require_cpu_environment()
        path = EXPERIMENT / 'scripts/operations/prepare_declared_v9_noalign_cpu.py'
        spec = importlib.util.spec_from_file_location('frozen_v9_capacity_helper', path)
        require(spec is not None and spec.loader is not None, 'Frozen no-alignment CPU helper unavailable')
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        release = helper.verify_release(args.release_manifest_sha256)
        module = helper.load_runner()
        configs = helper.verify_configs(module)
        result = capacity_inspection(module)
        result.update(release=release, configuration_checks_passed=len(configs))
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if all(cell['capacity_admitted'] for cell in result['cells']) else 2
    except Exception as exc:
        print(json.dumps({**boundary(), 'status': 'CAPACITY_ONLY_FAILED_GATED',
                          'error': f'{type(exc).__name__}: {exc}', 'cells': []}, sort_keys=True, allow_nan=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
