"""Read-only, independently reviewed bridge for one legacy R4 SFT checkpoint.

The old scientific source continues training unchanged. The current source may
schedule it and export a complete native checkpoint after successful exit; it
does not provide an intermediate checkpoint hook to the old training process.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
from pathlib import Path

from . import suite_bindings as bindings, suite_runner as runner


TASK_ID = 'train__ouro_r4_s17__sft'
LEGACY_SOURCE_ROOT = '/home/jinhaopeng/.cache/Paper/DeepProbe-VLM/runs/dp-p1-ouro-r4-f8-g4-s17-sft-a1/source'
LEGACY_FROZEN_SHA = 'ccc99b4780849055ea1df69ef69dfc645f3bdc66740bd0366f36958a07ac0cd2'
LEGACY_CONFIG_SHA = '0ec088d7e769a6043792aa36ad97a8c718289a14370c530ddccaf11c87a2a606'
LEGACY_TRAIN_SHA = 'cf7877633d210f7cef0651ee39278223496db1289c7ef146a3f98879abc1808a'
LEGACY_INVENTORY_SHA = '5450d7c5854a1829e96f74920af99712b94b1c42f3eec9172fb599c1b2a97dd4'
LEGACY_OSS_PREFIX = ('oss://ss-devbox-stream/jinhaopeng/Paper/DeepProbe-VLM/runs/'
                     'dp-p1-ouro-r4-f8-g4-s17-sft-a1/failure-evidence/original_run/training/checkpoint-600/')
METADATA_NAMES = frozenset({'deepprobe_config.json', 'training_provenance.json', 'trainer_state.json'})
DATA_NAMES = ('train_manifest', 'eval_manifest', 'formal_ready')
REVIEW_STATUS = 'SUITE_LEGACY_PRIVATE_RESUME_REVIEW_ACCEPTED'


def _equal(value, expected, message):
    runner.require(isinstance(value, dict) and all(type(value.get(key)) is type(wanted)
                   and value[key] == wanted for key, wanted in expected.items()), message)


def _pin(pin, *, verify=False):
    bindings._pin_shape(pin)
    if verify:
        runner.verify_pin(pin)
    return pin


def _shape(task, history, environment):
    _equal(task, {'id': TASK_ID, 'kind': 'train', 'stage': 'sft', 'expected_optimizer_steps': 718},
           'Legacy private bridge is limited to the original R4 SFT task ending at step 718')
    schema = environment.get('execution_profile', {}).get('schema_version')
    runner.require(type(schema) is int and schema == 3,
                   'Legacy private bridge requires the reviewed private scratch profile')
    bindings.validate_history_shape(task, history)
    _equal(history, {'mode': 'resume', 'checkpoint_step': 600, 'source_root': LEGACY_SOURCE_ROOT},
           'Legacy bridge cannot change the original source or checkpoint 600')
    runner.require(history['frozen_source_manifest']['sha256'] == LEGACY_FROZEN_SHA
                   and history['config']['sha256'] == LEGACY_CONFIG_SHA,
                   'Legacy scientific source or original runtime config bytes changed')
    rows = history['checkpoint']['files']
    runner.require(len(rows) == 31 and runner.digest(rows) == LEGACY_INVENTORY_SHA
                   and history['checkpoint']['oss_prefix'] == LEGACY_OSS_PREFIX,
                   'Legacy bridge requires the exact 31-object archived checkpoint 600')
    runner.require(set(history.get('checkpoint_metadata', {})) == METADATA_NAMES
                   and 'archive_acceptance' in history, 'Legacy bridge requires all retained metadata and full-GET proof')
    _pin(history['archive_acceptance'])
    inventory = {row['path']: row for row in rows}
    for name, pin in history['checkpoint_metadata'].items():
        _equal(pin, {key: inventory[name][key] for key in ('bytes', 'sha256')},
               'Legacy retained metadata differs from its fixed checkpoint inventory')
        runner.require(pin['bytes'] <= 4 * 1024 * 1024, 'Legacy checkpoint metadata exceeds the CPU bound')
    training = environment['assets']['training']
    for name in (*DATA_NAMES, 'frozen_source_manifest'):
        _pin(training[name])
    outer_root = runner.safe(environment['experiment_root'])
    runner.require(str(outer_root) == environment['experiment_root']
                   and outer_root != runner.safe(LEGACY_SOURCE_ROOT)
                   and training['frozen_source_manifest']['sha256'] != LEGACY_FROZEN_SHA,
                   'Outer scheduling source must remain distinct from the legacy scientific source')
    runner.require(isinstance(environment.get('source_files'), list) and environment['source_files'],
                   'Outer scheduling source inventory is missing')
    for pin in environment['source_files']:
        _pin(pin)
    bound_history = environment.get('history', {}).get(TASK_ID)
    runner.require(isinstance(bound_history, dict) and runner.canonical(bound_history) == runner.canonical(history),
                   'Legacy bridge cannot replace the environment historical entry')
    return training


def expected_review_facts(task, history, environment):
    """Describe the exact claims a separate reviewer must verify; never approve."""
    training = _shape(task, history, environment)
    return {
        'bridge': 'r4-s17-sft-cp600-to718-unchanged-science',
        'legacy_source_root': history['source_root'],
        'legacy_frozen_source_manifest': deepcopy(history['frozen_source_manifest']),
        'legacy_config': deepcopy(history['config']),
        'declared_config': deepcopy(task['config']),
        'training_data': {name: deepcopy(training[name]) for name in DATA_NAMES},
        'checkpoint_metadata': deepcopy(history['checkpoint_metadata']),
        'checkpoint_inventory_sha256': runner.digest(history['checkpoint']['files']),
        'checkpoint_oss_prefix': history['checkpoint']['oss_prefix'], 'checkpoint_payload_objects': 31,
        'checkpoint_full_get_evidence': deepcopy(history['archive_acceptance']),
        'outer_source_root': environment['experiment_root'],
        # prepare_workflow retains identical manifest bytes at its own control
        # path; only the outer manifest permits this physical-path relocation.
        'outer_frozen_source_manifest': {key: training['frozen_source_manifest'][key]
                                          for key in ('bytes', 'sha256')},
        'outer_source_files_sha256': runner.source_identity(environment),
        'optimizer_step': 600, 'expected_optimizer_steps': 718, 'world_size': 8, 'global_batch_size': 128,
        'legacy_checkpoint_export_hook_present': False, 'legacy_scientific_source_modified': False,
        'legacy_final_native_checkpoint_supported': True,
        'outer_source_role': 'scheduling_and_post_success_export_only',
        'interrupted_recovery_checkpoint_step': 600,
        'post_success_export': {'required': True, 'optimizer_step': 718, 'world_size': 8,
                                'execution': 'single_process', 'format': 'full_native',
                                'requires_training_exit_code': 0},
    }


def _verify_review(task, history, environment, facts):
    pin = _pin(environment.get('legacy_private_resume'), verify=True)
    review = runner.read(pin['path'])
    _equal(review, {'schema_version': 1, 'status': REVIEW_STATUS, 'approved': True,
                   'production_evidence': True, 'task_id': TASK_ID, 'task_sha256': runner.digest(task),
                   'history_sha256': runner.digest(history)},
           'Independent legacy bridge review does not bind the exact task and complete history')
    runner.require(isinstance(review.get('verified_facts'), dict)
                   and runner.canonical(review['verified_facts']) == runner.canonical(facts),
                   'Independent legacy bridge facts changed or overstate hooks')
    evidence = review.get('evidence')
    runner.require(isinstance(evidence, list) and evidence, 'Legacy bridge review lacks independent pinned evidence')
    required = [history['acceptance'], history['archive_acceptance'], history['config'],
                history['frozen_source_manifest'],
                *history['checkpoint_metadata'].values(),
                *(environment['assets']['training'][name] for name in DATA_NAMES)]
    runner.require(all(item in evidence for item in required),
                   'Legacy bridge review omits original source/data/history or full-GET evidence')
    paths = set()
    for item in evidence:
        _pin(item, verify=True)
        runner.require(item['path'] not in paths, 'Duplicate legacy bridge review evidence path')
        paths.add(item['path'])
    outer = environment['assets']['training']['frozen_source_manifest']
    runner.require(any(all(item[key] == outer[key] for key in ('bytes', 'sha256')) for item in evidence),
                   'Legacy bridge review lacks verified identical outer manifest bytes')
    return deepcopy(pin)


def _verify_training_entry(old_root, old_files):
    train_name = 'src/deepprobe_vlm/train.py'
    runner.require(old_files.get(train_name) == LEGACY_TRAIN_SHA,
                   'Legacy final-native checkpoint training entry differs from the reviewed source')
    train_bytes = (old_root / train_name).read_bytes()
    runner.require(hashlib.sha256(train_bytes).hexdigest() == LEGACY_TRAIN_SHA,
                   'Legacy training entry bytes differ from the reviewed source')
    train_text = train_bytes.decode('utf-8')
    train_ast = ast.parse(train_text)
    trainers = [node for node in train_ast.body
                if isinstance(node, ast.ClassDef) and node.name == 'DeepProbeTrainer']
    mains = [node for node in train_ast.body
             if isinstance(node, ast.FunctionDef) and node.name == 'main']
    runner.require(len(trainers) == len(mains) == 1,
                   'Legacy training entry requires its original Trainer class and main')
    savers = [node for node in trainers[0].body
              if isinstance(node, ast.FunctionDef) and node.name == 'save_final_resume_checkpoint']
    runner.require(len(savers) == 1 and len(savers[0].args.args) == 1
                   and savers[0].args.args[0].arg == 'self',
                   'Legacy Trainer class lacks its final native checkpoint method')

    def body_nodes(function):
        pending = list(function.body)
        while pending:
            node = pending.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            yield node
            pending.extend(ast.iter_child_nodes(node))

    def calls(function, receiver, method):
        return any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and isinstance(node.func.value, ast.Name) and node.func.value.id == receiver
                   and node.func.attr == method for node in body_nodes(function))

    runner.require(calls(mains[0], 'trainer', 'save_final_resume_checkpoint'),
                   'Legacy train main does not call its final native checkpoint saver')
    runner.require(calls(savers[0], 'self', '_save_checkpoint'),
                   'Legacy final saver does not call the native Trainer checkpoint method')
    runner.require('src/deepprobe_vlm/suite_checkpoint_export.py' not in old_files
                   and 'suite_checkpoint_export' not in train_text,
                   'Legacy source unexpectedly claims an intermediate checkpoint export hook')


def _verify_sources_and_metadata(task, history, environment, training):
    from .formal_integrity import verify_frozen_source_manifest
    from .suite_actions import verify_historical_recipe

    old_root, outer_root = runner.safe(history['source_root']), runner.safe(environment['experiment_root'])
    old_pin, outer_pin = history['frozen_source_manifest'], training['frozen_source_manifest']
    for pin in (old_pin, outer_pin, history['config'], *(training[name] for name in DATA_NAMES)):
        _pin(pin, verify=True)
    old_source = verify_frozen_source_manifest(old_root, Path(old_pin['path']), old_pin['sha256'])
    outer_source = verify_frozen_source_manifest(outer_root, Path(outer_pin['path']), outer_pin['sha256'])
    outer_files = runner.read(outer_pin['path'])['files']
    expected = {str(outer_root / name): digest for name, digest in outer_files.items()}
    actual = {}
    for pin in environment['source_files']:
        _pin(pin, verify=True)
        runner.require(pin['path'] not in actual, 'Duplicate outer source file pin')
        actual[pin['path']] = pin['sha256']
    runner.require(actual == expected, 'Outer source pins do not cover the complete frozen source tree')
    _verify_training_entry(old_root, runner.read(old_pin['path'])['files'])
    config_path = runner.safe(outer_root / task['config']['path'])
    runner.require(config_path.is_relative_to(outer_root) and runner.sha(config_path) == task['config']['sha256'],
                   'Declared outer recipe bytes changed')
    historical_recipe, declared_recipe = runner.read(history['config']['path']), runner.read(config_path)
    verify_historical_recipe(historical_recipe, declared_recipe)
    model = historical_recipe['model']
    metadata = {name: runner.read(_pin(pin, verify=True)['path'])
                for name, pin in history['checkpoint_metadata'].items()}
    _equal(metadata['deepprobe_config.json'], {key: model.get(key) for key in
        ('num_frames', 'pool_grid', 'num_loops', 'language_revision', 'vision_revision')},
        'Legacy checkpoint model shape or revision differs')
    _equal(metadata['training_provenance.json'].get('training_contract'), {
        'stage': 'sft', 'config_sha256': history['config']['sha256'],
        'frozen_source_manifest_sha256': LEGACY_FROZEN_SHA,
        'train_manifest_sha256': training['train_manifest']['sha256'],
        'eval_manifest_sha256': training['eval_manifest']['sha256'],
        'formal_ready_sha256': training['formal_ready']['sha256'],
        'formal_training': True, 'world_size': 8, 'global_batch_size': 128,
    }, 'Legacy checkpoint scientific source/config/data/world-size contract differs')
    _equal(metadata['trainer_state.json'], {'global_step': 600, 'max_steps': 718},
           'Legacy native Trainer state differs from step 600 to 718')
    return old_source, outer_source


def verify_legacy_private_resume(task, history, environment, verify_files=True):
    """Check pinned metadata and both source trees; do not hydrate or launch."""
    runner.require(type(verify_files) is bool, 'Legacy resume verification mode must be boolean')
    facts = expected_review_facts(task, history, environment)
    _pin(environment.get('legacy_private_resume'))
    if not verify_files:
        return {'status': 'LEGACY_PRIVATE_RESUME_SHAPE_VALID_NOT_VERIFIED', 'task_id': TASK_ID, 'eligible': False}
    review = _verify_review(task, history, environment, facts)
    # This lower-level verifier has no environment/private dispatch, so it
    # cannot recurse through _history_gate -> verify_private_resume -> here.
    bindings.verify_history_binding({'tasks': [task]}, task, history, verify_files=True)
    acceptance = runner.read(history['acceptance']['path'])
    _equal(acceptance, {'resume_metadata_verified': True, 'optimizer_scheduler_rng_deserialized': False},
           'Legacy history overstates deserialization or lacks reviewed resume metadata')
    archive = runner.read(_pin(history['archive_acceptance'], verify=True)['path'])
    _equal(archive, {'all_objects_full_get_sha256_verified': True,
                     'checkpoint_oss_prefix': LEGACY_OSS_PREFIX,
                     'checkpoint_inventory_sha256': LEGACY_INVENTORY_SHA},
           'Existing checkpoint 600 full-GET audit differs from this complete native inventory')
    old_source, outer_source = _verify_sources_and_metadata(task, history, environment, environment['assets']['training'])
    return {'status': 'LEGACY_PRIVATE_RESUME_ELIGIBLE', 'task_id': TASK_ID, 'eligible': True,
            'optimizer_step': 600, 'expected_optimizer_steps': 718, 'world_size': 8,
            'source': old_source, 'outer_source': outer_source, 'independent_review': review,
            'history_sha256': runner.digest(history), 'export_acceptance': deepcopy(history['archive_acceptance']),
            'legacy_checkpoint_export_hook_present': False, 'interrupted_recovery_checkpoint_step': 600,
            'legacy_final_native_checkpoint_supported': True,
            'post_success_export': deepcopy(facts['post_success_export']),
            'fresh_remote_payload_verification_deferred_to_restore': True,
            'optimizer_scheduler_rng_deserialized': False, 'runtime_resume_verified': False, 'scheduler_admitted': False}
