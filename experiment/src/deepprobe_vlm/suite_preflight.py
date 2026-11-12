"""CPU admission for a complete suite, without scheduler or GPU operations.

The default path reads local bytes only. ``--prepare-media`` may hydrate bound
bucket PNGs on Linux; historical model payloads are restored and SHA-checked by
the serial actions just before use, never all materialized during admission.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat

from . import checkpoint_retention as retention
from . import suite_runner as runner
from .suite_actions import inventory, load_operation, validate_benchmark, verify_inventory, save
from .suite_bindings import validate_environment, reserve_for
from .experiment_suite import validate_plan

require = runner.require


def _source_gate(environment, training, binding=None):
    from .formal_integrity import verify_frozen_source_manifest, source_inventory
    root = runner.safe(environment['experiment_root'])
    proof = verify_frozen_source_manifest(root, Path(training['frozen_source_manifest']['path']),
                                          training['frozen_source_manifest']['sha256'])
    if binding is not None:
        require(binding['source_files'] == environment['source_files'], 'Binding/environment source pins differ')
        declared = {p['path']: p['sha256'] for p in binding['source_files']}
        require(len(declared) == len(binding['source_files']), 'Duplicated executable source pin')
        observed = {str(root / name): sha for name, sha in source_inventory(root).items()}
        require(all(declared.get(name) == sha for name, sha in observed.items()),
                'Executable binding omits or changes a file in the frozen source tree')
    return proof


def _formal_gate(training):
    from .train import verify_formal_ready
    return verify_formal_ready(training['formal_ready']['path'], training['train_manifest']['sha256'],
                               training['eval_manifest']['sha256'], training_data_dir=training['data_dir'],
                               media_root=training['media_root'])


def _training_gate(training, declared):
    from .manifest import load_manifest, validate_disjoint
    from .formal_integrity import manifest_media, verify_attested_training_media
    require(training['train_manifest']['sha256'] == declared['train_manifest_sha256']
            and training['eval_manifest']['sha256'] == declared['validation_manifest_sha256'],
            'Training input hashes differ from the scientific suite')
    train = load_manifest(training['train_manifest']['path'], media_root=training['media_root'])
    dev = load_manifest(training['eval_manifest']['path'], media_root=training['media_root'])
    require(len(train) == declared['train_examples'] and len(dev) == declared['validation_examples']
            and len(train) + len(dev) == declared['total_qa'], 'Formal QA split differs')
    validate_disjoint(train, dev)
    require(all(x['split'] == 'train' for x in train) and all(x['split'] == 'validation' for x in dev),
            'Wrong training split labels')
    sources = {r['source_video_id'] for r in [*train, *dev]}
    require(len(sources) == declared['source_clips'], 'Formal training source-clip coverage differs')
    ready = runner.read(training['formal_ready']['path'])
    media = verify_attested_training_media(Path(training['data_dir']) / 'training_media_inventory.json',
        runner.safe(training['media_root']), ready['immutable_input_sha256']['training_media_inventory'],
        manifest_media([*train, *dev]))
    return {'train_qa': len(train), 'validation_qa': len(dev), 'source_clips': len(sources),
            'no_train_validation_source_overlap': True, 'all_current_media_sha256_verified': True,
            'media_verification': media}


def _bound_environment(binding, environment, environment_sha256):
    require(binding['suite_id'] == environment['suite_id'] and binding['storage'] == environment['storage']
            and binding['source_files'] == environment['source_files'], 'Binding belongs to another machine environment')
    expected = dict(environment['runtime_environment'])
    expected['PYTHONPATH'] = str(Path(environment['experiment_root']) / 'src')
    require(binding['environment'] == expected, 'Bound runtime environment differs')
    for command in [binding['startup'], *binding['tasks'].values()]:
        argv = command['argv']
        require(argv[0] == environment['python'] and '-m' in argv
                and argv[argv.index('-m') + 1] == 'deepprobe_vlm.suite_actions',
                'Preflight requires the concrete reviewed suite action adapter')
        require('--environment-sha256' in argv and argv[argv.index('--environment-sha256') + 1] == environment_sha256,
                'Command pins another environment')
    for key, value in environment['tools'].items():
        path = runner.executable_path(value) if key in {'kubectl', 'ossutil'} else runner.safe(value)
        require(path.is_file() and os.access(path, os.X_OK if key in {'kubectl', 'ossutil'} else os.R_OK),
                'Required installed tool/config is unavailable: ' + key)
    python = runner.executable_path(environment['python'])
    require(python.is_file() and os.access(python, os.X_OK), 'Suite Python is unavailable')
    return {'concrete_actions_bound': True, 'installed_tools_present': True}


def _backbone_gate(family, assets, environment, plan):
    from .declared_training import tree_identity
    models = [m for m in plan['models'] if m['family'] == family]
    require(models, 'Unused backbone family')
    task = next(t for t in plan['tasks'] if t['kind'] == 'train' and t['model_id'] == models[0]['id'])
    recipe = runner.read(Path(environment['experiment_root']) / task['config']['path'])['model']
    runtime = environment['runtime_environment']
    hub = runtime.get('HF_HUB_CACHE')
    if hub is None and runtime.get('HF_HOME'):
        hub = str(Path(runtime['HF_HOME']) / 'hub')
    require(hub is not None, 'Offline backbone cache needs explicit HF_HUB_CACHE or HF_HOME')
    hub = runner.safe(hub)
    require(not runtime.get('TRANSFORMERS_CACHE') or runner.safe(runtime['TRANSFORMERS_CACHE']) == hub,
            'Transformers and Hub cache locations differ')
    pins = assets.get('files')
    require(isinstance(pins, list) and pins and len({p['path'] for p in pins}) == len(pins),
            'Backbone byte inventory is empty or duplicated')
    for pin in pins:
        require(set(pin) == {'path', 'bytes', 'sha256'} and Path(pin['path']).is_absolute()
                and type(pin['bytes']) is int and pin['bytes'] >= 0 and retention.valid_sha(pin['sha256']),
                'Invalid backbone file pin')
    actual, roots = {}, {}
    for label in ('language', 'vision'):
        identifier, revision = recipe[label + '_model'], recipe[label + '_revision']
        root = runner.safe(assets[label + '_root'])
        expected = hub / ('models--' + identifier.replace('/', '--')) / 'snapshots' / revision
        require(root == expected, 'Audited backbone differs from the actual offline Hub revision: ' + label)
        identity = tree_identity(root)
        names = set(identity['files'])
        if label == 'language':
            require(bool(names & {'tokenizer.json', 'tokenizer.model'}) or {'vocab.json', 'merges.txt'} <= names,
                    'Language snapshot lacks complete tokenizer payload')
        else:
            require('preprocessor_config.json' in names, 'Vision snapshot lacks preprocessor configuration')
        for name in names:
            if name.endswith('.safetensors.index.json'):
                # Hub snapshot files may alias already-checked personal blobs.
                mapping = json.loads((root / name).read_text()).get('weight_map')
                require(isinstance(mapping, dict) and mapping and set(mapping.values()) <= names,
                        'Model weight index references a missing shard')
        for name, value in identity['files'].items():
            actual[str(root / name)] = value
        roots[label] = {'root': str(root), 'files': len(names), 'inventory_sha256': identity['files_sha256']}
    require({p['path']: {'bytes': p['bytes'], 'sha256': p['sha256']} for p in pins} == actual,
            'Backbone pins do not cover the complete current snapshots')
    return {'family': family, 'snapshots': roots, 'all_file_sha256_verified': True}


def _checkpoint_rows(files, *, resume):
    require(isinstance(files, list) and files and len({p['path'] for p in files}) == len(files),
            'Empty or duplicated historical checkpoint inventory')
    for row in files:
        retention.relative(row['path'])
        require(set(row) == {'path', 'bytes', 'sha256'} and type(row['bytes']) is int
                and row['bytes'] > 0 and retention.valid_sha(row['sha256']), 'Invalid historical checkpoint pin')
    names = {row['path'] for row in files}
    required = {'deepprobe_config.json', 'training_provenance.json', 'projector.safetensors',
                'language_model/config.json', 'vision_model/config.json',
                'tokenizer/tokenizer_config.json', 'image_processor/preprocessor_config.json'}
    require(required <= names and all(any(n.startswith(label + '/') and n.endswith('.safetensors')
                for n in names) for label in ('language_model', 'vision_model')), 'Historical model inventory is incomplete')
    if resume:
        require({'trainer_state.json', 'optimizer.pt', 'scheduler.pt',
                 *(f'rng_state_{rank}.pth' for rank in range(8))} <= names, 'Historical recovery state is incomplete')
        require({n for n in names if n.startswith('rng_state') and n.endswith('.pth')}
                == {f'rng_state_{rank}.pth' for rank in range(8)}, 'Historical RNG layout differs from eight ranks')
    return {row['path']: row for row in files}


def _history_gate(task, environment):
    require(task['id'] in environment['history'], 'Historical candidate is unbound; never restart from zero')
    h = environment['history'][task['id']]
    require(h.get('mode') in {'accepted', 'resume'}, 'Unsupported historical reuse mode')
    for name in ('acceptance', 'config', 'frozen_source_manifest'):
        runner.verify_pin(h[name])
    audit = runner.read(h['acceptance']['path'])
    expected = {'status': 'SUITE_HISTORY_ACCEPTED', 'task_id': task['id'], 'mode': h['mode'],
                'config_sha256': h['config']['sha256'], 'frozen_source_manifest_sha256': h['frozen_source_manifest']['sha256'],
                'checkpoint_inventory_sha256': runner.digest(h['checkpoint']['files'])}
    require(all(audit.get(k) == v for k, v in expected.items()) and audit.get('evidence'),
            'History audit does not bind exact recovery/accepted inputs')
    for item in audit['evidence']:
        runner.verify_pin(item)
    require(runner.read(h['config']['path']) == runner.read(Path(environment['experiment_root']) / task['config']['path']),
            'Historical recipe changed')
    from .formal_integrity import verify_frozen_source_manifest
    source = verify_frozen_source_manifest(runner.safe(h['source_root']), Path(h['frozen_source_manifest']['path']),
                                           h['frozen_source_manifest']['sha256'])
    resume = h['mode'] == 'resume'
    rows = _checkpoint_rows(h['checkpoint']['files'], resume=resume)
    root = runner.safe(h['checkpoint']['path'])
    remote = h['checkpoint'].get('oss_prefix')
    if not remote:
        verify_inventory(root, h['checkpoint']['files'])
    else:
        require(isinstance(remote, str) and remote.startswith('oss://ss-devbox-stream/')
                and '/runs/' in remote and remote.endswith('/') and '..' not in remote.split('/'),
                'Historical archive must use an exact stream run prefix')
        require(audit.get('checkpoint_oss_prefix') == remote, 'Archive prefix mismatch')
        # The historical independent audit must attest a real completed full
        # GET, and its proof must itself be a pinned retained evidence file.
        archive_pin = h.get('archive_acceptance')
        require(archive_pin in audit['evidence'], 'Historical full-GET archive acceptance is not pinned by the audit')
        runner.verify_pin(archive_pin)
        archive = runner.read(archive_pin['path'])
        require(archive.get('all_objects_full_get_sha256_verified') is True
                and archive.get('checkpoint_oss_prefix') == remote
                and archive.get('checkpoint_inventory_sha256') == runner.digest(h['checkpoint']['files']),
                'Independent archive acceptance lacks this checkpoint full-GET binding')
    # Metadata may be retained locally or supplied separately as SHA-bound
    # readbacks. Do not hydrate weight/optimizer payloads during CPU admission.
    metadata_pins = h.get('checkpoint_metadata', {})
    metadata = {}
    for name in ('deepprobe_config.json', 'training_provenance.json', *(['trainer_state.json'] if resume else [])):
        row = rows[name]
        candidate = metadata_pins.get(name, {**row, 'path': str(root / name)})
        require(candidate['sha256'] == row['sha256'] and candidate.get('bytes') == row['bytes'],
                'Historical metadata is not bound to the archived file inventory')
        runner.verify_pin(candidate)
        metadata[name] = runner.read(candidate['path'])
    config = runner.read(h['config']['path'])['model']
    require(all(metadata['deepprobe_config.json'].get(k) == config.get(k) for k in
                ('num_frames', 'pool_grid', 'num_loops', 'language_revision', 'vision_revision')),
            'Historical checkpoint model shape/revision differs')
    training = environment['assets']['training']
    contract = metadata['training_provenance.json'].get('training_contract', {})
    expected_contract = {'stage': task['stage'], 'config_sha256': h['config']['sha256'],
        'frozen_source_manifest_sha256': h['frozen_source_manifest']['sha256'],
        'train_manifest_sha256': training['train_manifest']['sha256'],
        'eval_manifest_sha256': training['eval_manifest']['sha256'],
        'formal_ready_sha256': training['formal_ready']['sha256'], 'formal_training': True,
        'world_size': 8, 'global_batch_size': 128}
    require(all(type(contract.get(k)) is type(v) and contract.get(k) == v for k, v in expected_contract.items()),
            'Historical training provenance differs from source/data/config/DDP contract')
    if resume:
        step = task['historical_hint']['checkpoint_step']
        state = metadata['trainer_state.json']
        require(type(state.get('global_step')) is int and state['global_step'] == step
                and type(state.get('max_steps')) is int and state['max_steps'] == task['expected_optimizer_steps'],
                'Historical Trainer state differs from the exact recovery budget')
        require(audit.get('optimizer_step') == step and audit.get('exact_resume_accepted') is True,
                'Recovery audit incomplete')
        if not remote:
            from .resume_checkpoint import verify_resume_checkpoint
            verify_resume_checkpoint(root, world_size=8, expected_max_steps=task['expected_optimizer_steps'])
    else:
        require(audit.get('optimizer_step') == task['expected_optimizer_steps']
                and audit.get('formal_training_complete') is True, 'Accepted training incomplete')
    return {'mode': h['mode'], 'checkpoint_bytes': sum(r['bytes'] for r in rows.values()),
            'historical_source': source, 'all_current_checkpoint_payloads_sha256_verified': not bool(remote),
            'fresh_remote_payload_verification_deferred_to_restore': bool(remote)}


def _cached_media(data):
    """Reconstruct hydration's exact output using only current local evidence."""
    from . import suite_media as media
    rows, binding, items, frames, _, keys = media._bound_cache_inventory(data['cache'])
    root = runner.safe(data['media_root'])
    needed = sum(item['bytes'] for item in items)
    require(type(data['max_bytes']) is int and 0 < needed <= data['max_bytes'], 'Frame cache exceeds its capacity cap')
    owner = {'schema_version': 1, 'kind': 'suite_benchmark_media',
             'cache_binding_sha256': retention.digest(data['cache']), 'media_root': str(root),
             'max_bytes': data['max_bytes'], 'png_bytes': needed}
    require(root.is_dir() and runner.read(root / '.suite-media-owner.json') == owner,
            'Node media is absent or belongs to another cache; run CPU --prepare-media first')
    files = [{k: item[k] for k in ('path', 'bytes', 'sha256')} for item in items]
    expected_names = {row['path'] for row in files} | {'.suite-media-owner.json'}
    actual_names = set()
    for path in root.rglob('*'):
        runner.safe(path)
        require(stat.S_ISREG(path.stat().st_mode) or path.is_dir(), 'Unexpected media filesystem entry')
        if path.is_file():
            actual_names.add(path.relative_to(root).as_posix())
    require(actual_names == expected_names, 'Local frame cache contains missing or unowned files')
    metadata = runner.safe(root.parent / (root.name + '.acceptance'))
    ready_path = metadata / 'SUITE_BENCHMARK_READY.json'
    ready = runner.read(ready_path)
    require(ready.get('cache_binding_sha256') == retention.digest(data['cache'])
            and ready.get('png_bytes') == needed and ready.get('source_count') == len(binding['sources'])
            and ready.get('all_png_full_oss_get_and_rgb_verified') is True
            and ready.get('evidence') == [data['cache'][key] for key in keys], 'Media acceptance cache/source binding differs')
    result = {'manifest': data['cache']['manifest'], 'media_root': str(root), 'media_files': files,
              'acceptance': retention.file_pin(ready_path)}
    require(runner.read(metadata / 'ENVIRONMENT_FRAGMENT.json') == result, 'Hydrated environment fragment differs')
    return result


def _benchmark_gate(data, benchmark, frames, environment, *, prepare_media=False):
    if 'cache' in data:
        if prepare_media:
            from .suite_media import hydrate_frame_cache, OSSStore, PROJECT_PREFIX
            runner.capacity({'storage': environment['storage']}, data['max_bytes'])
            hydrate_frame_cache(data['cache'], media_root=data['media_root'], max_bytes=data['max_bytes'],
                                store=OSSStore(environment['tools']['oss_config'], allowed_prefix=PROJECT_PREFIX + 'suite-media-readonly/'),
                                resume=True)
        data = _cached_media(data)
    records = validate_benchmark(data, {'benchmark': benchmark, 'frames': frames})
    return {'qa': len(records), 'frames': frames, 'acceptance': data['acceptance'],
            'network_or_hydration_requested': prepare_media}


def _development_gate(environment):
    from . import benchmark_latency as latency
    from .manifest import load_manifest
    dev = environment['assets']['development']
    for name in ('manifest', 'source_metadata'):
        runner.verify_pin(dev[name])
    metadata = runner.read(dev['source_metadata']['path'])
    require(metadata.get('scope') == 'development' and metadata.get('quality_score_based_selection') is False
            and metadata.get('manifest_sha256') == dev['manifest']['sha256']
            and isinstance(metadata.get('source_description'), str) and metadata['source_description'].strip(),
            'Unbound latency development selection')
    records = load_manifest(dev['manifest']['path'], media_root=dev['media_root'])
    selected = latency.select_records(records)
    files = [retention.file_pin(runner.safe(dev['media_root']) / name) for name in latency.media_names(selected)]
    return {'sample_count': len(selected), 'heldout_used': False, 'current_selected_media': files}


def _initial_storage_gate(plan, binding, environment, estimates):
    # The first accepted initializer remains resident while its first actual
    # training consumer runs. Never certify those separate costs independently.
    restored = 0
    first_training_reserve = None
    for task in plan['tasks']:
        if task['kind'] != 'train':
            continue
        history = environment['history'].get(task['id'], {})
        if history.get('checkpoint'):
            checkpoint = 'final' if history['mode'] == 'accepted' else 'checkpoint-' + str(task['historical_hint']['checkpoint_step'])
            target = runner.safe(environment['run_root']) / (environment['suite_id'] + '--' + task['id']) / 'training' / checkpoint
            for row in history['checkpoint']['files']:
                retention.relative(row['path'])
                destination = runner.safe(target / row['path'])
                if destination.exists():
                    retention.check_pin({**row, 'path': str(destination)})
                else:
                    restored += row['bytes']
        if history.get('mode') != 'accepted':
            first_training_reserve = max(estimates[task['id']]['required_additional_bytes'],
                binding['tasks'][task['id']]['reserve_bytes'], reserve_for(task, environment))
            break
    require(first_training_reserve is not None, 'No actual training task was costed')
    earliest = max(binding['startup']['reserve_bytes'], reserve_for(plan['tasks'][0], environment),
                   restored + first_training_reserve)
    result = runner.capacity(binding, earliest)
    return {**result, 'initial_restored_checkpoint_bytes': restored,
            'first_training_reserve_bytes': first_training_reserve, 'future_cleanup_credited': False}


def inspect(plan, binding, environment, *, plan_sha256, binding_sha256, environment_sha256,
            state_root, prepare_media=False):
    checks, failures, outputs, estimates = [], [], [], {}
    def check(name, callback):
        try:
            value = callback()
            checks.append({'check': name, 'status': 'passed', 'result': value})
            return value
        except Exception as exc:
            # SDK exceptions can contain signed request data; retain their
            # class, while ordinary local contract failures retain the reason.
            detail = str(exc) if isinstance(exc, (ValueError, FileNotFoundError, KeyError)) else type(exc).__name__
            failures.append({'check': name, 'error_type': type(exc).__name__, 'error': detail})
            return None
    def result():
        return {'schema_version': 1, 'status': 'SUITE_CPU_GATED' if failures else 'SUITE_CPU_ADMITTED',
            'generated_utc': runner.utc(), 'plan_sha256': plan_sha256, 'binding_sha256': binding_sha256,
            'environment_sha256': environment_sha256,
            'source_identity_sha256': runner.source_identity(binding) if isinstance(binding, dict) and 'source_files' in binding else None,
            'failures': failures, 'checks': checks, 'outputs': outputs, 'training_peak_estimates': estimates,
            'gpu_used': False, 'media_hydration_requested': prepare_media, 'future_capacity_rechecked_per_task': True,
            'remote_checkpoint_fresh_payload_check_deferred_to_restore': True,
            'scheduler_admission_still_required': True}
    check('scientific_plan', lambda: bool(validate_plan(plan, config_root=environment['experiment_root'])))
    check('environment', lambda: bool(validate_environment(environment, plan)))
    check('binding', lambda: bool(runner.validate_binding(plan, binding, plan_sha256=plan_sha256)))
    if failures:
        return result()
    def state_location():
        state = runner.safe(state_root)
        roots = [runner.safe(root) for root in environment['storage']['roots']]
        require(any(state != root and state.is_relative_to(root) for root in roots),
                'Suite state must live inside an accounted storage root')
        return {'state_root': str(state), 'created': False}
    check('accounted_state_location', state_location)
    check('bound_environment_and_installed_tools', lambda: _bound_environment(binding, environment, environment_sha256))
    check('all_executable_sources', lambda: runner.verify_sources(binding))
    training = environment['assets']['training']
    for key in ('train_manifest', 'eval_manifest', 'formal_ready', 'frozen_source_manifest'):
        pin = check('training/' + key, lambda k=key: runner.verify_pin(training[k]))
        if pin:
            outputs.append(pin)
    check('formal_ready_contents', lambda: _formal_gate(training))
    check('training_data_and_live_media', lambda: _training_gate(training, plan['training_data']))
    source_verified = check('frozen_training_source', lambda: _source_gate(environment, training, binding))
    families = sorted({m['family'] for m in plan['models']})
    for family in families:
        check('backbones/' + family, lambda f=family: _backbone_gate(f, environment['assets']['backbones'][f], environment, plan))
    def load_estimator():
        require(source_verified is not None, 'Cannot import a storage estimator from an unverified source tree')
        return load_operation(environment['experiment_root'], 'run_p1_stage_v9')
    estimator = check('storage_estimator', load_estimator)
    # A Python module is not an evidence artifact or JSON-serializable result.
    if estimator is not None:
        checks[-1]['result'] = {'module': 'run_p1_stage_v9', 'loaded': True}
    for task in plan['tasks']:
        if task['kind'] != 'train':
            continue
        def estimate(t=task):
            require(estimator is not None, 'Storage estimator unavailable')
            model = next(m for m in plan['models'] if m['id'] == t['model_id'])
            config = runner.read(Path(environment['experiment_root']) / t['config']['path'])
            assets = environment['assets']['backbones'][model['family']]
            config['model']['language_model'], config['model']['vision_model'] = assets['language_root'], assets['vision_root']
            value = estimator.estimate_stage_storage(config, t['stage'])
            require(type(value.get('required_additional_bytes')) is int and value['required_additional_bytes'] > 0,
                    'Missing positive measured training peak estimate')
            estimates[t['id']] = value
            if environment['history'].get(t['id'], {}).get('mode') != 'accepted':
                require(reserve_for(t, environment) >= value['required_additional_bytes']
                        and binding['tasks'][t['id']]['reserve_bytes'] >= value['required_additional_bytes'],
                        'Bound training reserve understates the measured checkpoint peak')
            return value
        check('peak_capacity/' + task['id'], estimate)
        if task['historical_hint']['status'] in {'resume_candidate', 'reuse_candidate'}:
            check('historical_acceptance/' + task['id'], lambda t=task: _history_gate(t, environment))
    hydration_permitted = prepare_media and not failures
    for benchmark in ('videomme899', 'tempcompass_mc'):
        for frames in (4, 8, 16):
            def benchmark_gate(b=benchmark, f=frames):
                require(not prepare_media or hydration_permitted, 'Upstream input failures prevent media hydration')
                return _benchmark_gate(environment['assets']['benchmarks'][b][str(f)], b, f, environment,
                                       prepare_media=prepare_media)
            proof = check('benchmark/' + benchmark + '/' + str(frames), benchmark_gate)
            if proof:
                outputs.append(proof['acceptance'])
    check('development_latency_inputs', lambda: _development_gate(environment))
    check('initial_storage', lambda: _initial_storage_gate(plan, binding, environment, estimates))
    return result()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('plan', 'binding', 'environment'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--state-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-media', action='store_true')
    args = parser.parse_args(argv)
    for name in ('plan', 'binding', 'environment'):
        runner.verify_pin({'path': str(getattr(args, name)), 'sha256': getattr(args, name + '_sha256')})
    value = inspect(runner.read(args.plan), runner.read(args.binding), runner.read(args.environment),
                    plan_sha256=args.plan_sha256, binding_sha256=args.binding_sha256,
                    environment_sha256=args.environment_sha256, state_root=args.state_root, prepare_media=args.prepare_media)
    save(args.output, value)
    print(json.dumps({'status': value['status'], 'failures': value['failures']}, indent=2))
    return 0 if value['status'] == 'SUITE_CPU_ADMITTED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
