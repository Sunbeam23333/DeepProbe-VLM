"""CPU-only identities for the declared, independently accepted two-stage SFTs.

A name is an output label, never a source of backend/seed/recurrence settings.
Native independent-audit layouts use fixed, supported kind/status parsers;
expected values come from actual pinned artifacts, never caller-selected claims.
This module neither creates acceptance nor submits.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import stat


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def safe_path(value):
    require(isinstance(value, str) and value, 'missing_absolute_path')
    p = Path(value)
    require(p.is_absolute() and '..' not in p.parts, 'unsafe_absolute_path')
    require(not any(q.is_symlink() for q in (p, *p.parents)), 'symlink_in_artifact_path')
    return p


def relative(value):
    require(isinstance(value, str) and value and '\\' not in value, 'unsafe_relative_path')
    p = Path(value)
    require(not p.is_absolute() and '..' not in p.parts and str(p) == value and value != '.',
            'unsafe_relative_path')
    return value


def artifact(ref):
    require(isinstance(ref, dict) and isinstance(ref.get('sha256'), str) and
            re.fullmatch('[0-9a-f]{64}', ref['sha256']), 'missing_pinned_artifact')
    p = safe_path(ref.get('path'))
    require(p.is_file() and stat.S_ISREG(p.stat().st_mode), 'missing_regular_artifact')
    raw = p.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == ref['sha256'] and
            ('bytes' not in ref or len(raw) == ref['bytes']), 'artifact_bytes_or_sha_changed')
    return json.loads(raw)


def file_map(rows):
    require(isinstance(rows, list) and rows, 'empty_file_graph')
    result = {}
    for r in rows:
        require(isinstance(r, dict) and set(r) == {'path', 'bytes', 'sha256'}, 'invalid_file_graph_row')
        name = relative(r['path'])
        require(name not in result and type(r['bytes']) is int and r['bytes'] >= 0 and
                isinstance(r['sha256'], str) and re.fullmatch('[0-9a-f]{64}', r['sha256']),
                'duplicate_or_invalid_file_graph')
        result[name] = {'bytes': r['bytes'], 'sha256': r['sha256']}
    return result


def model_sequence(plan, *, require_bound=True):
    names, bindings = plan.get('model_sequence'), plan.get('model_bindings')
    require(isinstance(names, list) and names and len(names) <= 12 and
            all(isinstance(n, str) and re.fullmatch('[a-z][a-z0-9_]{0,63}', n) for n in names) and
            len(set(names)) == len(names), 'invalid_model_sequence')
    require(isinstance(bindings, dict) and set(bindings) == set(names), 'model_set_differs_from_sequence')
    require(plan.get('benchmark_score_based_model_selection') is False, 'heldout_model_selection_forbidden')
    seen = set()
    for name in names:
        b = bindings[name]
        require(isinstance(b, dict) and b.get('model_id') == name and b.get('stage') == 'sft' and
                b.get('training_method') == 'two_stage', 'matched_sft_identity_required')
        seed, loops, family, backend = (b.get(k) for k in ('seed', 'num_loops', 'family', 'language_backend'))
        require(type(seed) is int and seed in (17, 29, 43) and type(loops) is int, 'invalid_seed_or_loops')
        require((family == 'ouro' and backend == 'ouro' and loops in (1, 2, 4)) or
                (family == 'smollm2' and backend == 'causal_lm' and loops == 1), 'invalid_family_backend_loops')
        # Preserve the historical output aliases only for their original models.
        if name in ('r1', 'r2', 'r4'):
            require(family == 'ouro' and seed == 17 and loops == {'r1': 1, 'r2': 2, 'r4': 4}[name],
                    'historical_model_alias_changed')
        identity = (family, seed, loops)
        require(identity not in seen, 'duplicate_declared_model')
        seen.add(identity)
        if require_bound:
            require(all(isinstance(b.get(k), dict) for k in
                        ('checkpoint', 'training_acceptance', 'training_graph')), 'MODEL_BINDINGS_GATED:' + name)
    if require_bound:
        for key in ('path', 'run_id'):
            values = [bindings[n]['checkpoint'].get(key) for n in names]
            require(all(isinstance(v, str) and v for v in values) and len(set(values)) == len(values),
                    'checkpoint_reused_between_models')
        pins = [bindings[n]['training_acceptance'].get('sha256') for n in names]
        require(all(isinstance(v, str) for v in pins) and len(set(pins)) == len(pins),
                'training_acceptance_reused_between_models')
    return names


def validate_model_config(config, binding, *, alignment=False):
    expected = {'language_backend': binding['language_backend'], 'num_loops': binding['num_loops'],
                'num_frames': 8, 'pool_grid': 4, 'max_text_tokens': 512, 'attn_implementation': 'sdpa',
                'freeze_vision': alignment, 'vision_revision': '9fdffc58afc957d1a03a25b10dba0329ab15c2a3',
                'language_revision': ('574fa66cb8bf5abdc979642d01cf2b79b16bfab1' if binding['family'] == 'ouro'
                                      else '31b70e2e869a7173562077fd711b654946d38674')}
    require(all(config.get(k) == v and type(config.get(k)) is type(v) for k, v in expected.items()),
            'trained_model_config_differs_from_binding')
    for key, identifier in [('language_model', 'ByteDance/Ouro-1.4B' if binding['family'] == 'ouro'
                             else 'HuggingFaceTB/SmolLM2-1.7B-Instruct'),
                            ('vision_model', 'google/siglip-so400m-patch14-384')]:
        value = config.get(key)
        require(isinstance(value, str), 'missing_backbone_identity')
        revision = config[key.replace('_model', '_revision')]
        suffix = '/models--' + identifier.replace('/', '--') + '/snapshots/' + revision
        require(value == identifier or (value.startswith('/home/jinhaopeng/') and value.endswith(suffix)
                                        and '..' not in Path(value).parts), 'backbone_source_changed')


def training_contract(provenance, binding, *, alignment=False):
    c = provenance.get('training_contract', {})
    require(c.get('stage') == ('alignment' if alignment else 'sft') and c.get('formal_training') is True and
            c.get('training', {}).get('seed') == str(binding['seed']) and
            c.get('world_size') == 8 and c.get('global_batch_size') == 128, 'training_stage_seed_or_budget_changed')
    declared = c.get('declared_protocol')
    if declared is not None:
        require(declared.get('method') == 'two_stage' and declared.get('seed') == binding['seed'] and
                declared.get('heldout_selection') is False and declared.get('selection') == 'final_step',
                'training_method_or_selection_changed')
    for field in ('training_sources', 'tuning_sources'):
        require(isinstance(provenance.get(field), list) and
                all(isinstance(s, str) and s for s in provenance[field]), 'invalid_training_source_provenance')
    return c


def verify_native_acceptance(audit, binding, graph):
    """Known original R1/R2 layouts plus an explicit future independent-audit schema."""
    run_id = binding['checkpoint']['run_id']
    require(audit.get('schema_version') == 1 and audit.get('run_id') == run_id,
            'native_acceptance_schema_or_run_changed')
    require(audit.get('passed') is not False and audit.get('independent_acceptance') is not False and
            not audit.get('acceptance_errors') and not audit.get('errors'), 'native_acceptance_negative_verdict')
    kind, status = audit.get('kind'), audit.get('status')
    job = audit.get('job', {})
    require(job.get('scheduler_state') == 'Succeeded' and job.get('pod_exit_codes') == [0] and
            type(job['pod_exit_codes'][0]) is int and job.get('pod_phase') == 'Succeeded' and
            isinstance(audit.get('exact_job'), str) and audit['exact_job'], 'native_terminal_not_accepted')
    refs = {'archive_manifest_sha256': 'archive_manifest', 'stage_complete_sha256': 'stage_complete',
            'formal_train_complete_sha256': 'formal_train_complete', 'exit_status_sha256': 'exit_status'}
    if kind == 'P1_OURO_R1_S17_SFT_A1_SECOND_INDEPENDENT_SUCCESS_AUDIT':
        require(status == 'sft_stage_and_archive_verified' and run_id == 'dp-p1-ouro-r1-f8-g4-s17-sft-a1',
                'native_acceptance_kind_or_status_changed')
        require(job.get('kubernetes_condition') == 'Succeeded=True', 'native_kubernetes_not_succeeded')
    elif kind == 'P1_R2_S17_SFT_A1_INDEPENDENT_COMPLETION_ARCHIVE_CHECK':
        require(status == 'completion_and_exact_archive_listing_key_sha_verified' and
                run_id == 'dp-p1-ouro-r2-f8-g4-s17-sft-a1' and audit.get('formal_training_result') is True,
                'native_acceptance_kind_or_status_changed')
        require(job.get('kubernetes_succeeded_condition') is True and
                job.get('stream_low_one_node_eight_gpu_restart_never') is True and
                audit.get('exact_oss_object_paths_lengths_match') is True, 'native_r2_archive_or_job_not_accepted')
        members = {'archive_manifest': 'ARCHIVE_MANIFEST.json', 'stage_complete': 'P1_STAGE_COMPLETE.json',
                   'formal_train_complete': 'training/FORMAL_TRAIN_COMPLETE.json', 'exit_status': 'EXIT_STATUS.json'}
        require(audit.get('archive_manifest_sha256') == graph['archive_manifest']['sha256'] and
                audit.get('p1_stage_complete_sha256') == graph['stage_complete']['sha256'], 'native_acceptance_graph_changed')
        for key, member in members.items():
            row = audit.get('readbacks', {}).get(member, {})
            require(row.get('passed') is True and row.get('sha256') == graph[key]['sha256'],
                    'native_acceptance_readback_changed')
        return
    elif kind == 'MATCHED_SFT_FINAL_INDEPENDENT_ACCEPTANCE_V1':
        require(status == 'matched_sft_final_and_archive_accepted' and audit.get('independent_acceptance') is True and
                audit.get('acceptance_errors') == [] and audit.get('all_final_files_sha256_verified') is True,
                'native_acceptance_kind_or_status_changed')
        require(audit.get('model_identity') == {k: binding[k] for k in
                    ('family', 'language_backend', 'num_loops', 'seed', 'stage', 'training_method')},
                'native_accepted_model_identity_changed')
        require(job.get('kubernetes_succeeded_condition') is True, 'native_kubernetes_not_succeeded')
    else:
        raise ValueError('unsupported_native_acceptance_kind')
    for key, value in {'pool': 'stream', 'priority': 'low', 'gpu_count': 8, 'replicas': 1, 'restart_policy': 'Never'}.items():
        require(type(job.get(key)) is type(value) and job[key] == value, 'native_job_contract_changed')
    require(audit.get('exact_oss_paths_and_lengths_match') is True, 'native_exact_archive_not_accepted')
    for claim, key in refs.items():
        require(audit.get(claim) == graph[key]['sha256'], 'native_acceptance_graph_changed')


def tree_snapshot(root):
    result = {}
    def walk_error(error):
        raise error
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        for leaf in dirs + files:
            require(not (Path(directory) / leaf).is_symlink(), 'symlink_in_final_graph')
        for leaf in files:
            p = Path(directory) / leaf
            s = p.stat()
            require(stat.S_ISREG(s.st_mode), 'nonregular_final_file')
            result[p.relative_to(root).as_posix()] = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    return result


def verify_registered_checkpoint(binding):
    """Read each complete final graph and bind it back to native accepted SFT artifacts."""
    name = binding['model_id']
    model_sequence({'model_sequence': [name], 'model_bindings': {name: binding},
                    'benchmark_score_based_model_selection': False})
    item, graph, accepted = (binding[k] for k in ('checkpoint', 'training_graph', 'training_acceptance'))
    run_id = item['run_id']
    family = f"ouro-r{binding['num_loops']}" if binding['family'] == 'ouro' else 'smollm2'
    require(re.fullmatch(f"dp-p1-{family}-f8-g4-s{binding['seed']}-sft-a[1-9][0-9]*", run_id),
            'checkpoint_run_not_matching_sft')
    root = safe_path(item['path'])
    require(root.name == 'final' and root.is_dir(), 'complete_final_directory_required')
    expected = file_map(item['files'])
    require(type(item.get('file_count')) is int and item['file_count'] == len(expected), 'final_file_count_changed')
    before = tree_snapshot(root)
    actual = {name: {'bytes': metadata[2], 'sha256': sha(root / name)} for name, metadata in before.items()}
    require(actual == expected, 'final_graph_bytes_or_sha_changed')
    def final_json(name):
        return artifact({'path': str(root / name), **actual[name]})
    required = {'deepprobe_config.json', 'training_provenance.json', 'projector.safetensors',
                'language_model/config.json', 'vision_model/config.json', 'tokenizer/tokenizer.json',
                'tokenizer/tokenizer_config.json', 'image_processor/preprocessor_config.json'}
    require(required <= set(actual), 'incomplete_final_loader_graph')
    for folder in ('language_model', 'vision_model'):
        index = root / folder / 'model.safetensors.index.json'
        if index.is_file():
            weights = final_json(folder + '/model.safetensors.index.json').get('weight_map', {})
            require(isinstance(weights, dict) and weights and all(
                folder + '/' + relative(s) in actual for s in weights.values()), 'missing_indexed_model_shard')
        else:
            require(folder + '/model.safetensors' in actual, 'missing_model_weights')
    config = final_json('deepprobe_config.json')
    validate_model_config(config, binding)
    contract = training_contract(final_json('training_provenance.json'), binding)
    artifacts = {k: artifact(graph[k]) for k in ('archive_manifest', 'stage_complete', 'formal_train_complete',
                 'exit_status', 'runtime_config', 'alignment_initializer', 'alignment_config', 'alignment_provenance')}
    archive, stage, formal, exit_status = (artifacts[k] for k in
                                         ('archive_manifest', 'stage_complete', 'formal_train_complete', 'exit_status'))
    require(archive.get('run_id') == run_id and item.get('archive_manifest_sha256') == graph['archive_manifest']['sha256'],
            'archive_run_or_pin_changed')
    archived = file_map(archive['files'])
    archived_final = {k[len('training/final/'):]: v for k, v in archived.items() if k.startswith('training/final/')}
    require(archived_final == actual, 'final_graph_not_equal_to_accepted_archive')
    for key, member in [('stage_complete', 'P1_STAGE_COMPLETE.json'), ('formal_train_complete', 'training/FORMAL_TRAIN_COMPLETE.json'),
                        ('exit_status', 'EXIT_STATUS.json'), ('runtime_config', 'inputs/config.runtime.json'),
                        ('alignment_initializer', 'inputs/alignment_initializer.json')]:
        require(member in archived and archived[member]['sha256'] == graph[key]['sha256'], 'training_artifact_not_in_archive')
    require(stage.get('status') == 'p1_formal_stage_complete' and stage.get('run_id') == run_id and
            stage.get('stage') == 'sft' and stage.get('seed') == binding['seed'] and
            stage.get('rfg') == {k: config[k] for k in ('num_loops', 'num_frames', 'pool_grid')} and
            stage.get('formal_training_result') is True and stage.get('runtime_config_sha256') == graph['runtime_config']['sha256'],
            'matched_sft_completion_required')
    result = stage['training_result']
    require(result.get('final_files') == actual and result.get('formal_completion_sha256') == graph['formal_train_complete']['sha256'] and
            result.get('optimizer_steps') == 718, 'final_not_from_completed_matched_sft')
    require(formal.get('status') == 'formal_training_complete' and formal.get('stage') == 'sft' and
            formal.get('optimizer_step') == 718 and formal.get('world_size') == 8, 'formal_completion_not_matched_sft')
    require(exit_status.get('run_id') == run_id and exit_status.get('stage') == 'sft' and
            type(exit_status.get('exit_code')) is int and exit_status['exit_code'] == 0 and
            isinstance(exit_status.get('stages'), list) and exit_status['stages'] and
            all(type(s.get('returncode')) is int and s['returncode'] == 0 for s in exit_status['stages']),
            'training_exit_not_success')
    runtime = artifacts['runtime_config']
    require(runtime.get('model') == config and runtime.get('training', {}).get('stage') == 'sft' and
            runtime['training'].get('seed') == binding['seed'] and contract.get('config_sha256') == graph['runtime_config']['sha256'],
            'runtime_config_or_seed_not_bound_to_final')
    for field in ('train_manifest_sha256', 'eval_manifest_sha256', 'formal_ready_sha256', 'frozen_source_manifest_sha256'):
        require(contract.get(field) == formal.get(field) and isinstance(contract.get(field), str) and
                re.fullmatch('[0-9a-f]{64}', contract[field]), 'formal_source_contract_changed')
    initializer = artifacts['alignment_initializer']
    require(stage.get('alignment_initializer') == initializer and stage.get('resume_initializer') is None and
            initializer.get('rfg') == stage['rfg'], 'matched_alignment_initializer_required')
    require(re.fullmatch(f"/home/jinhaopeng/.cache/Paper/DeepProbe-VLM/runs/dp-p1-{family}-f8-g4-s{binding['seed']}-alignment-a[1-9][0-9]*/training/final",
                         initializer.get('path', '')), 'alignment_run_not_matching_seed_and_depth')
    init_files = initializer.get('files', {})
    require(isinstance(init_files, dict), 'initializer_file_graph_missing')
    for field, member in [('alignment_config', 'deepprobe_config.json'), ('alignment_provenance', 'training_provenance.json')]:
        require(init_files.get(member, {}).get('sha256') == graph[field]['sha256'], 'alignment_metadata_not_in_initializer_graph')
    validate_model_config(artifacts['alignment_config'], binding, alignment=True)
    training_contract(artifacts['alignment_provenance'], binding, alignment=True)
    require(set(accepted) <= {'path', 'bytes', 'sha256'}, 'native_acceptance_claim_overrides_forbidden')
    verify_native_acceptance(artifact(accepted), binding, graph)
    require(before == tree_snapshot(root), 'final_graph_changed_during_verification')
    return {'checkpoint': str(root), 'run_id': run_id, 'archive_manifest_sha256': graph['archive_manifest']['sha256'],
            'checkpoint_files_sha256': canonical_sha(item['files']),
            'checkpoint_config_sha256': actual['deepprobe_config.json']['sha256'],
            'num_files': len(actual), 'total_bytes': sum(v['bytes'] for v in actual.values()),
            'all_checkpoint_files_sha256_verified': True, 'config': config,
            'model_binding_sha256': canonical_sha(binding), 'training_seed': binding['seed'],
            'training_acceptance_sha256': accepted['sha256'],
            'training_graph_sha256': canonical_sha(graph), 'matched_sft_training_source_verified': True}
