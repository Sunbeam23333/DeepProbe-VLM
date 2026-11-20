"""Propose and authenticate explicit same-source private savepoint recovery.

This module reads only small pinned metadata and source files. It neither
hydrates checkpoints nor submits workers. A proposal is never an acceptance:
an independently supplied history-import review remains mandatory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re

from . import suite_bindings as bindings, suite_runner as runner


METADATA_NAMES = frozenset({'deepprobe_config.json', 'training_provenance.json', 'trainer_state.json'})
_UNREVIEWED = {'path': '/proposal-only/NO_HISTORY_ACCEPTANCE.json', 'bytes': 1, 'sha256': '0' * 64}


def _equal_fields(value, expected, message):
    runner.require(isinstance(value, dict) and all(type(value.get(k)) is type(v) and value[k] == v
                                                  for k, v in expected.items()), message)


def _pin(value, *, verify):
    bindings._pin_shape(value)
    if verify:
        runner.verify_pin(value)
    return value


def _same_bytes(first, second):
    return all(first[k] == second[k] for k in ('sha256', 'bytes'))


def _entry_shape(task, entry, environment, *, reviewed):
    runner.require(task.get('kind') == 'train' and entry.get('mode') == 'resume',
                   'Private savepoint recovery requires an explicit training resume')
    runner.require(environment.get('execution_profile', {}).get('schema_version') == 3,
                   'Private savepoint recovery requires scratch schema 3')
    bindings.validate_history_shape(task, entry if reviewed else {**entry, 'acceptance': _UNREVIEWED})
    runner.require('tokenizer/tokenizer.json' in {row['path'] for row in entry['checkpoint']['files']},
                   'Private resume inventory lacks the exported fast-tokenizer vocabulary')
    runner.require('archive_acceptance' in entry and set(entry.get('checkpoint_metadata', {})) == METADATA_NAMES,
                   'Private resume requires an export acceptance and all three retained metadata pins')
    current = environment['assets']['training']['frozen_source_manifest']
    _pin(current, verify=False)
    runner.require(_same_bytes(current, entry['frozen_source_manifest']),
                   'Private resume cannot change the frozen source manifest')
    runner.require(entry['config']['sha256'] == task['config']['sha256'],
                   'Private resume configuration must have the exact declared byte SHA')
    runner.safe(environment['experiment_root'])
    return current


def _owner_and_prefix(task, owner, prefix, step, generation, environment):
    runner.require(isinstance(owner, dict) and set(owner) == {
        'schema_version', 'suite_id', 'run_id', 'runs_root', 'run_root', 'source_sha256'}
        and type(owner['schema_version']) is int and owner['schema_version'] == 1,
        'Private savepoint original run ownership is incomplete')
    runner.require(all(isinstance(owner[k], str) and re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,127}', owner[k])
                       for k in ('suite_id', 'run_id'))
                   and owner['run_id'] == owner['suite_id'] + '--' + task['id']
                   and isinstance(owner['source_sha256'], str) and runner.SHA.fullmatch(owner['source_sha256']),
                   'Private savepoint original suite/run/task identity differs')
    root, runs = runner.safe(owner['run_root']), runner.safe(owner['runs_root'])
    runner.require(root == runs / owner['run_id'] and str(root) == owner['run_root']
                   and str(runs) == owner['runs_root'], 'Private savepoint original run path differs')
    # The old Pod may be gone. Ownership is authenticated by the reviewed
    # manifest, never by adopting/recreating its former local directory.
    runner.require(isinstance(prefix, str) and re.fullmatch(
        r'oss://ss-devbox-stream/(?:[A-Za-z0-9._-]+/)+', prefix)
        and not {'.', '..'}.intersection(prefix[6:].split('/')),
        'Private savepoint needs a canonical exact stream prefix')
    suffix = '/savepoints/runs/' + owner['run_id'] + '/checkpoint-' + str(step) + '/' + generation + '/'
    runner.require(prefix.endswith(suffix) and prefix.count('/savepoints/runs/') == 1,
                   'Private savepoint prefix differs from its original run/step/inventory')
    base = prefix[:-len(suffix)]
    runner.require(base.startswith('oss://ss-devbox-stream/') and not base.endswith('/')
                   and '/runs/' not in base, 'Private savepoint archive base is invalid')
    allowed = environment['storage']['oss_prefixes']
    from .suite_storage import prefix as validate_prefix
    runner.require(isinstance(allowed, list) and allowed, 'Private resume requires approved OSS project prefixes')
    for value in allowed:
        validate_prefix(value)
        runner.require(re.fullmatch(r'oss://ss-devbox-stream/(?:[A-Za-z0-9._-]+/)+', value)
                       and not {'.', '..'}.intersection(value[6:].split('/')), 'Noncanonical allowed OSS prefix')
    runner.require(any(prefix.startswith(value) for value in allowed),
                   'Private savepoint archive escapes the approved project prefixes')
    return root / 'training' / ('checkpoint-' + str(step))


def _verify_source_and_metadata(task, entry, environment, current):
    from .formal_integrity import verify_frozen_source_manifest
    current_root, original_root = runner.safe(environment['experiment_root']), runner.safe(entry['source_root'])
    for item in (current, entry['frozen_source_manifest'], entry['config']):
        _pin(item, verify=True)
    current_source = verify_frozen_source_manifest(current_root, Path(current['path']), current['sha256'])
    original_source = verify_frozen_source_manifest(original_root, Path(entry['frozen_source_manifest']['path']),
                                                    entry['frozen_source_manifest']['sha256'])
    runner.require(current_source == original_source, 'Private resume executable source trees differ')
    files = runner.read(current['path'])['files']
    expected = {str(current_root / name): digest for name, digest in files.items()}
    actual = {}
    for item in environment['source_files']:
        _pin(item, verify=True)
        runner.require(item['path'] not in actual, 'Duplicate current source pin')
        actual[item['path']] = item['sha256']
    runner.require(actual == expected, 'Current source pins do not cover the exact frozen executable tree')
    declared_config = runner.safe(current_root / task['config']['path'])
    runner.require(declared_config.is_relative_to(current_root)
                   and runner.sha(declared_config) == task['config']['sha256'], 'Declared resume config changed')
    metadata = {name: runner.read(_pin(value, verify=True)['path'])
                for name, value in entry['checkpoint_metadata'].items()}
    configuration = runner.read(entry['config']['path'])['model']
    _equal_fields(metadata['deepprobe_config.json'], {key: configuration.get(key) for key in
        ('num_frames', 'pool_grid', 'num_loops', 'language_revision', 'vision_revision')},
        'Private savepoint model shape or backbone revision differs')
    training = environment['assets']['training']
    for name in ('train_manifest', 'eval_manifest', 'formal_ready'):
        _pin(training[name], verify=True)
    expected_contract = {'stage': task['stage'], 'config_sha256': entry['config']['sha256'],
        'frozen_source_manifest_sha256': current['sha256'],
        'train_manifest_sha256': training['train_manifest']['sha256'],
        'eval_manifest_sha256': training['eval_manifest']['sha256'],
        'formal_ready_sha256': training['formal_ready']['sha256'], 'formal_training': True,
        'world_size': 8, 'global_batch_size': 128}
    _equal_fields(metadata['training_provenance.json'].get('training_contract'), expected_contract,
                  'Private savepoint training source/data/config/DDP contract differs')
    _equal_fields(metadata['trainer_state.json'], {'global_step': entry['checkpoint_step'],
                  'max_steps': task['expected_optimizer_steps']}, 'Private savepoint Trainer step or schedule differs')
    return current_source


def _verify_export(task, entry, environment):
    export_pin = _pin(entry['archive_acceptance'], verify=True)
    exported = runner.read(export_pin['path'])
    files, step = entry['checkpoint']['files'], entry['checkpoint_step']
    generation, prefix = runner.digest(files), entry['checkpoint']['oss_prefix']
    _equal_fields(exported, {'schema_version': 1, 'status': 'SUITE_SAVEPOINT_EXPORTED',
        'archive_prefix': prefix, 'checkpoint_oss_prefix': prefix, 'optimizer_step': step,
        'checkpoint_inventory_sha256': generation, 'all_objects_full_get_sha256_verified': True,
        'ready_published_last': True}, 'Private savepoint export acceptance differs')
    runner.require(isinstance(exported.get('stable_listing_sha256'), str)
                   and runner.SHA.fullmatch(exported['stable_listing_sha256']), 'Savepoint stable listing proof is missing')
    manifest_pin, ready_pin = (_pin(exported[name], verify=True) for name in ('manifest', 'ready'))
    manifest, ready = runner.read(manifest_pin['path']), runner.read(ready_pin['path'])
    _equal_fields(manifest, {'schema_version': 1, 'status': 'SUITE_NATIVE_CHECKPOINT_MANIFEST',
        'task_id': task['id'], 'task_sha256': runner.digest(task), 'optimizer_step': step,
        'expected_optimizer_steps': task['expected_optimizer_steps'], 'world_size': 8,
        'checkpoint_inventory_sha256': generation, 'archive_prefix': prefix, 'files': files},
        'Private savepoint manifest differs from the exact task/checkpoint')
    _pin(manifest['frozen_source_manifest'], verify=False)
    runner.require(_same_bytes(manifest['frozen_source_manifest'], entry['frozen_source_manifest']),
                   'Private savepoint manifest source differs')
    original_checkpoint = _owner_and_prefix(task, manifest['owner'], prefix, step, generation, environment)
    runner.require(str(original_checkpoint) == entry['checkpoint']['path'], 'Private savepoint original checkpoint path differs')
    _equal_fields(ready, {'schema_version': 1, 'status': 'SUITE_NATIVE_CHECKPOINT_OSS_READY',
        'manifest_sha256': manifest_pin['sha256'], 'manifest_bytes': manifest_pin['bytes'],
        'manifest_object': prefix + 'MANIFEST.json', 'checkpoint_inventory_sha256': generation,
        'optimizer_step': step, 'world_size': 8, 'payload_objects': len(files),
        'payload_bytes': sum(row['bytes'] for row in files), 'full_get_sha256_verified': True,
        'control_backend': 'gloo', 'control_timeout_seconds': 21600, 'automatic_resume_authorized': False},
        'Private savepoint READY does not bind its complete manifest')
    runner.require(exported.get('checkpoint_metadata') == entry['checkpoint_metadata'],
                   'Private resume metadata differs from the exported shared pins')
    for item in entry['checkpoint_metadata'].values():
        runner.require(item['bytes'] <= 4 * 1024 * 1024, 'Private savepoint metadata exceeds the CPU-only bound')
        _pin(item, verify=True)
    return {'export_acceptance': deepcopy(export_pin), 'manifest': deepcopy(manifest_pin),
            'ready': deepcopy(ready_pin), 'stable_listing_sha256': exported['stable_listing_sha256']}


def _facts(task, entry):
    return {'source_root': entry['source_root'], 'config_sha256': entry['config']['sha256'],
        'frozen_source_manifest_sha256': entry['frozen_source_manifest']['sha256'],
        'checkpoint_inventory_sha256': runner.digest(entry['checkpoint']['files']),
        'checkpoint_oss_prefix': entry['checkpoint']['oss_prefix'], 'optimizer_step': entry['checkpoint_step'],
        'archive_verified': True, 'resume_metadata_verified': True, 'optimizer_scheduler_rng_deserialized': False}


def _review(task, entry, review_pin):
    value = runner.read(_pin(review_pin, verify=True)['path'])
    _equal_fields(value, {'status': 'SUITE_HISTORY_IMPORT_REVIEW_ACCEPTED', 'approved': True,
        'production_evidence': True, 'task_id': task['id'], 'task_sha256': runner.digest(task),
        'entry_sha256': runner.digest(entry)}, 'Private resume independent review does not bind this exact entry')
    _equal_fields(value.get('verified_facts'), _facts(task, entry), 'Private resume review facts differ')
    runner.require(entry['archive_acceptance'] in value.get('evidence', []),
                   'Private resume review omits its original export evidence')
    for item in value['evidence']:
        _pin(item, verify=True)
    return review_pin


def _external_review(task, history):
    acceptance = runner.read(history['acceptance']['path'])
    runner.require(acceptance.get('import_scope') == 'existing_independent_evidence_only',
                   'Private resume must use a separately reviewed history import')
    _equal_fields(acceptance, _facts(task, history), 'Private resume history facts differ or overstate runtime verification')
    entry = {key: value for key, value in history.items() if key != 'acceptance'}
    # import_history_entry puts the independently supplied review first. Other
    # evidence may be arbitrary pinned logs or source files, not JSON reviews.
    return _review(task, entry, acceptance['evidence'][0])


def verify_private_resume(task, history, environment, verify_files=True):
    """Verify reviewed metadata only; actual payload restore and CPU/GPU gates remain required."""
    if task.get('id') == 'train__ouro_r4_s17__sft' and environment.get('legacy_private_resume'):
        from .suite_legacy_private_resume import verify_legacy_private_resume
        return verify_legacy_private_resume(task, history, environment, verify_files=verify_files)
    runner.require(type(verify_files) is bool, 'Private resume verification mode must be boolean')
    current = _entry_shape(task, history, environment, reviewed=True)
    if not verify_files:
        return {'status': 'PRIVATE_RESUME_SHAPE_VALID_NOT_VERIFIED', 'task_id': task['id'], 'eligible': False}
    bindings.verify_history_binding({'tasks': [task]}, task, history, verify_files=True)
    exported = _verify_export(task, history, environment)
    source = _verify_source_and_metadata(task, history, environment, current)
    review = _external_review(task, history)
    return {'status': 'PRIVATE_RESUME_ELIGIBLE', 'task_id': task['id'], 'eligible': True,
            'optimizer_step': history['checkpoint_step'], 'source': source, 'independent_review': review,
            **exported, 'fresh_remote_payload_verification_deferred_to_restore': True,
            'optimizer_scheduler_rng_deserialized': False, 'runtime_resume_verified': False, 'scheduler_admitted': False}


def prepare_resume_candidate(plan, environment, task_id, export_pin, output, *, review_pin=None):
    """Write a proposal, or import an already supplied independent review; never invent one."""
    from .suite_actions import save
    tasks = {task['id']: task for task in plan['tasks']}
    runner.require(task_id in tasks and tasks[task_id]['kind'] == 'train', 'Unknown private resume training task')
    task = tasks[task_id]
    exported = runner.read(_pin(export_pin, verify=True)['path'])
    manifest = runner.read(_pin(exported['manifest'], verify=True)['path'])
    source = runner.safe(environment['experiment_root'])
    config_path = runner.safe(source / task['config']['path'])
    runner.require(config_path.is_relative_to(source) and config_path != source,
                   'Resume candidate config escapes its frozen source tree')
    config = runner.artifact(config_path)
    step = manifest['optimizer_step']
    entry = {'mode': 'resume', 'config': config,
        'frozen_source_manifest': deepcopy(environment['assets']['training']['frozen_source_manifest']),
        'source_root': str(source), 'checkpoint_step': step,
        'checkpoint': {'path': str(runner.safe(manifest['owner']['run_root']) / 'training' / ('checkpoint-' + str(step))),
                       'files': deepcopy(manifest['files']), 'oss_prefix': manifest['archive_prefix']},
        'archive_acceptance': deepcopy(export_pin), 'checkpoint_metadata': deepcopy(exported['checkpoint_metadata'])}
    current = _entry_shape(task, entry, environment, reviewed=False)
    proof = _verify_export(task, entry, environment)
    _verify_source_and_metadata(task, entry, environment, current)
    directory = runner.safe(output)
    entry_pin = save(directory / 'RESUME_ENTRY.json', entry)
    if review_pin is None:
        return save(directory / 'RESUME_PROPOSAL.json', {'schema_version': 1,
            'status': 'SUITE_PRIVATE_RESUME_PROPOSED', 'task_id': task_id, 'task_sha256': runner.digest(task),
            'entry': entry_pin, 'entry_sha256': runner.digest(entry), 'export_evidence': proof,
            'independent_review_required': True, 'accepted': False, 'gpu_submitted': False,
            'remaining_optimizer_updates': task['expected_optimizer_steps'] - step,
            'runtime_resume_verified': False})
    # This is the only path that can publish a history acceptance. The existing
    # importer authenticates the caller-supplied independent review first.
    _review(task, entry, review_pin)
    history = bindings.import_history_entry(plan, task_id, entry, review_pin, directory / 'HISTORY_ACCEPTANCE.json')
    verified = verify_private_resume(task, history, environment)
    return save(directory / 'RESUME_HISTORY.json', {'schema_version': 1,
        'status': 'SUITE_PRIVATE_RESUME_HISTORY_IMPORTED', 'history': history, 'verification': verified,
        'gpu_submitted': False, 'runtime_resume_verified': False})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('plan', 'environment', 'export'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review', type=Path)
    parser.add_argument('--review-sha256')
    args = parser.parse_args(argv)
    runner.require(bool(args.review) == bool(args.review_sha256), 'Review path and SHA must be supplied together')
    for name in ('plan', 'environment', 'export', *(['review'] if args.review else [])):
        runner.verify_pin({'path': str(getattr(args, name)), 'sha256': getattr(args, name + '_sha256')})
    result = prepare_resume_candidate(runner.read(args.plan), runner.read(args.environment), args.task,
        runner.artifact(args.export), args.output, review_pin=runner.artifact(args.review) if args.review else None)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
