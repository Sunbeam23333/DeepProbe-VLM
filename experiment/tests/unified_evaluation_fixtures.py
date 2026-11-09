"""Tiny synthetic matched-SFT graphs for CPU tests; never model or benchmark evidence."""
import json
from pathlib import Path
import sys

OPS = Path(__file__).resolve().parents[1] / 'scripts/operations'
sys.path.insert(0, str(OPS))
import model_registry as m


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    return {'path': str(path), 'sha256': m.sha(path), 'bytes': path.stat().st_size}


def config(family, loops, alignment=False):
    ouro = family == 'ouro'
    return {'language_backend': 'ouro' if ouro else 'causal_lm', 'num_loops': loops,
            'num_frames': 8, 'pool_grid': 4, 'max_text_tokens': 512,
            'attn_implementation': 'sdpa', 'freeze_vision': alignment,
            'vision_model': 'google/siglip-so400m-patch14-384',
            'vision_revision': '9fdffc58afc957d1a03a25b10dba0329ab15c2a3',
            'language_model': 'ByteDance/Ouro-1.4B' if ouro else 'HuggingFaceTB/SmolLM2-1.7B-Instruct',
            'language_revision': '574fa66cb8bf5abdc979642d01cf2b79b16bfab1' if ouro
                                 else '31b70e2e869a7173562077fd711b654946d38674'}


def fixture(root, *, family='ouro', loops=4, seed=17, contract_seed=None,
            alignment_loops=None, acceptance_overrides=None, stage_name='sft'):
    name = 'r4' if family == 'ouro' and seed == 17 and loops == 4 else f'model_{family}_{loops}_{seed}'
    b = {'model_id': name, 'family': family, 'language_backend': 'ouro' if family == 'ouro' else 'causal_lm',
         'num_loops': loops, 'seed': seed, 'stage': 'sft', 'training_method': 'two_stage'}
    family_run = f'ouro-r{loops}' if family == 'ouro' else family
    run_id = f'dp-p1-{family_run}-f8-g4-s{seed}-sft-a1'
    cfg = config(family, loops)
    runtime = {'model': cfg, 'training': {'stage': 'sft', 'seed': seed}}
    g = {'runtime_config': write(root / 'inputs/config.runtime.json', runtime)}
    source_pins = {k: str(i) * 64 for i, k in enumerate(
        ('train_manifest_sha256', 'eval_manifest_sha256', 'formal_ready_sha256', 'frozen_source_manifest_sha256'), 1)}
    def provenance(stage, actual_seed):
        return {'training_sources': ['train:001'], 'tuning_sources': ['dev:001'],
                'training_contract': {'stage': stage, 'formal_training': True,
                    'training': {'seed': str(actual_seed)}, 'global_batch_size': 128, 'world_size': 8,
                    'config_sha256': g['runtime_config']['sha256'], **source_pins}}
    final = root / 'final'
    write(final / 'deepprobe_config.json', cfg)
    write(final / 'training_provenance.json', provenance(stage_name, seed if contract_seed is None else contract_seed))
    for name_ in ['projector.safetensors', 'language_model/config.json', 'language_model/model.safetensors',
                  'vision_model/config.json', 'vision_model/model.safetensors', 'tokenizer/tokenizer.json',
                  'tokenizer/tokenizer_config.json', 'image_processor/preprocessor_config.json', 'training_args.bin']:
        write(final / name_, {'synthetic_fixture_only': name_})
    files = [{'path': str(p.relative_to(final)), 'bytes': p.stat().st_size, 'sha256': m.sha(p)}
             for p in sorted(final.rglob('*')) if p.is_file()]
    g['alignment_config'] = write(root / 'alignment/deepprobe_config.json', config(family, loops if alignment_loops is None else alignment_loops, True))
    g['alignment_provenance'] = write(root / 'alignment/training_provenance.json', provenance('alignment', seed))
    init = {'path': f'/home/jinhaopeng/.cache/Paper/DeepProbe-VLM/runs/dp-p1-{family_run}-f8-g4-s{seed}-alignment-a1/training/final',
            'rfg': {k: cfg[k] for k in ('num_loops', 'num_frames', 'pool_grid')},
            'files': {n: {k: g[f][k] for k in ('bytes', 'sha256')} for f, n in
                      [('alignment_config', 'deepprobe_config.json'), ('alignment_provenance', 'training_provenance.json')]}}
    g['alignment_initializer'] = write(root / 'inputs/alignment_initializer.json', init)
    formal = {'status': 'formal_training_complete', 'stage': 'sft', 'optimizer_step': 718, 'world_size': 8, **source_pins}
    g['formal_train_complete'] = write(root / 'training/FORMAL_TRAIN_COMPLETE.json', formal)
    stage = {'status': 'p1_formal_stage_complete', 'run_id': run_id, 'stage': 'sft', 'seed': seed,
             'rfg': init['rfg'], 'formal_training_result': True, 'runtime_config_sha256': g['runtime_config']['sha256'],
             'alignment_initializer': init, 'resume_initializer': None,
             'training_result': {'final_files': m.file_map(files), 'optimizer_steps': 718,
                                 'formal_completion_sha256': g['formal_train_complete']['sha256']}}
    g['stage_complete'] = write(root / 'P1_STAGE_COMPLETE.json', stage)
    g['exit_status'] = write(root / 'EXIT_STATUS.json', {'run_id': run_id, 'stage': 'sft', 'exit_code': 0, 'stages': [{'returncode': 0}]})
    archived = [{'path': 'training/final/' + f['path'], 'bytes': f['bytes'], 'sha256': f['sha256']} for f in files]
    for field, path in [('runtime_config', 'inputs/config.runtime.json'), ('alignment_initializer', 'inputs/alignment_initializer.json'),
                        ('stage_complete', 'P1_STAGE_COMPLETE.json'), ('formal_train_complete', 'training/FORMAL_TRAIN_COMPLETE.json'),
                        ('exit_status', 'EXIT_STATUS.json')]:
        archived.append({'path': path, 'bytes': g[field]['bytes'], 'sha256': g[field]['sha256']})
    g['archive_manifest'] = write(root / 'ARCHIVE_MANIFEST.json', {'run_id': run_id, 'files': archived})
    facts = {'schema_version': 1, 'kind': 'MATCHED_SFT_FINAL_INDEPENDENT_ACCEPTANCE_V1',
              'status': 'matched_sft_final_and_archive_accepted', 'independent_acceptance': True,
              'acceptance_errors': [], 'all_final_files_sha256_verified': True,
              'model_identity': {k: b[k] for k in ('family', 'language_backend', 'num_loops', 'seed', 'stage', 'training_method')},
              'exact_job': 'synthetic-fixture-not-a-real-job',
              'run_id': run_id, **{f + '_sha256': g[f]['sha256'] for f in
              ('archive_manifest', 'stage_complete', 'formal_train_complete', 'exit_status')},
              'job': {'scheduler_state': 'Succeeded', 'pod_exit_codes': [0], 'pool': 'stream', 'priority': 'low',
                      'gpu_count': 8, 'replicas': 1, 'restart_policy': 'Never', 'pod_phase': 'Succeeded',
                      'kubernetes_succeeded_condition': True}, 'exact_oss_paths_and_lengths_match': True}
    for k, v in (acceptance_overrides or {}).items():
        (facts['job'] if k in facts['job'] else facts)[k] = v
    accept = write(root / 'NATIVE_ACCEPTANCE.json', facts)
    b.update(training_acceptance=accept, training_graph=g,
             checkpoint={'path': str(final), 'run_id': run_id, 'file_count': len(files), 'files': files,
                         'archive_manifest_sha256': g['archive_manifest']['sha256']})
    return b


def catalog():
    bindings = {}
    for family, depths in [('ouro', (1, 2, 4)), ('smollm2', (1,))]:
        for seed in (17, 29, 43):
            for loops in depths:
                name = f'r{loops}' if family == 'ouro' and seed == 17 else f'{family}_r{loops}_s{seed}'
                if family == 'smollm2':
                    name = f'smollm2_s{seed}'
                bindings[name] = {'model_id': name, 'family': family,
                    'language_backend': 'ouro' if family == 'ouro' else 'causal_lm',
                    'num_loops': loops, 'seed': seed, 'stage': 'sft', 'training_method': 'two_stage',
                    'checkpoint': None, 'training_acceptance': None, 'training_graph': None}
    return {'model_sequence': list(bindings), 'model_bindings': bindings,
            'benchmark_score_based_model_selection': False}
