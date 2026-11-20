"""Synthetic, CPU-only checks of the narrowly pinned legacy resume bridge."""
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil

import pytest

from deepprobe_vlm import formal_integrity, suite_legacy_private_resume as legacy
from deepprobe_vlm import suite_runner as runner


LEGACY_TRAIN_ENTRY = (Path(__file__).parent / 'fixtures' / 'legacy_r4_train_entry.py').read_bytes()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else runner.canonical(value) + b'\n')
    return runner.artifact(path)


def review(f, **overrides):
    facts = legacy.expected_review_facts(f['task'], f['history'], f['env'])
    h, data = f['history'], f['env']['assets']['training']
    required = [h['acceptance'], h['archive_acceptance'], h['config'], h['frozen_source_manifest'],
                data['frozen_source_manifest'], *h['checkpoint_metadata'].values(),
                *(data[name] for name in legacy.DATA_NAMES)]
    # This is a synthetic trust-boundary test fixture, never a production review.
    value = {'schema_version': 1, 'status': legacy.REVIEW_STATUS, 'approved': True,
             'production_evidence': True, 'task_id': legacy.TASK_ID,
             'task_sha256': runner.digest(f['task']), 'history_sha256': runner.digest(h),
             'verified_facts': facts, 'evidence': required, **overrides}
    f['review'] = value
    f['env']['legacy_private_resume'] = write(f['root'] / 'BRIDGE_REVIEW.json', value)


def refresh_history(f):
    h = f['history']
    for row in h['checkpoint']['files']:
        if row['path'] in h['checkpoint_metadata']:
            row.update({key: h['checkpoint_metadata'][row['path']][key] for key in ('bytes', 'sha256')})
    inventory_sha = runner.digest(h['checkpoint']['files'])
    f['monkeypatch'].setattr(legacy, 'LEGACY_INVENTORY_SHA', inventory_sha)
    archive = {'all_objects_full_get_sha256_verified': True, 'checkpoint_oss_prefix': legacy.LEGACY_OSS_PREFIX,
               'checkpoint_inventory_sha256': inventory_sha}
    h['archive_acceptance'] = write(f['root'] / 'ARCHIVE_ACCEPTANCE.json', archive)
    acceptance = {
        'schema_version': 1, 'status': 'SUITE_HISTORY_ACCEPTED', 'production_evidence': True,
        'task_id': legacy.TASK_ID, 'task_sha256': runner.digest(f['task']), 'mode': 'resume',
        'source_root': h['source_root'], 'config_sha256': h['config']['sha256'],
        'frozen_source_manifest_sha256': h['frozen_source_manifest']['sha256'],
        'checkpoint_inventory_sha256': inventory_sha, 'checkpoint_oss_prefix': legacy.LEGACY_OSS_PREFIX,
        'optimizer_step': 600, 'archive_verified': True, 'resume_metadata_verified': True,
        'optimizer_scheduler_rng_deserialized': False, 'evidence': [h['archive_acceptance']],
    }
    h['acceptance'] = write(f['root'] / 'HISTORY_ACCEPTANCE.json', acceptance)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = tmp_path.resolve(); old = root / 'old-source'; outer = root / 'outer-source'
    for name in formal_integrity.SOURCE_DIRS:
        (old / name).mkdir(parents=True)
    for name in (*formal_integrity.REQUIRED_P1_SOURCE, *formal_integrity.SOURCE_FILES):
        write(old / name, b'# synthetic old scientific source, no checkpoint hook\n')
    write(old / 'src/deepprobe_vlm/train.py', LEGACY_TRAIN_ENTRY)
    model = {'num_frames': 8, 'pool_grid': 4, 'num_loops': 4,
             'language_revision': 'a' * 40, 'vision_revision': 'b' * 40}
    recipe = {'model': model, 'training': {'seed': 17, 'stage': 'sft', 'max_steps': 718}}
    write(old / 'configs/recipe.json', recipe)
    config = write(root / 'inputs' / 'config.runtime.json', recipe)
    old_manifest = write(root / 'OLD_SOURCE.json', {'schema_version': 1,
        'purpose': 'DEEPPROBE_FROZEN_P1_SOURCE', 'files': formal_integrity.source_inventory(old)})
    shutil.copytree(old, outer)
    write(outer / 'src/deepprobe_vlm/suite_checkpoint_export.py', b'# synthetic outer-only exporter\n')
    outer_manifest = write(root / 'OUTER_SOURCE.json', {'schema_version': 1,
        'purpose': 'DEEPPROBE_FROZEN_P1_SOURCE', 'files': formal_integrity.source_inventory(outer)})
    data = {name: write(root / (name + '.json'), {'synthetic': name}) for name in legacy.DATA_NAMES}
    task = {'id': legacy.TASK_ID, 'kind': 'train', 'stage': 'sft', 'expected_optimizer_steps': 718,
            'config': {'path': 'configs/recipe.json', 'sha256': runner.sha(outer / 'configs/recipe.json')}}
    contract = {'stage': 'sft', 'config_sha256': config['sha256'],
                'frozen_source_manifest_sha256': old_manifest['sha256'],
                **{name + '_sha256': data[name]['sha256'] for name in legacy.DATA_NAMES},
                'formal_training': True, 'world_size': 8, 'global_batch_size': 128}
    metadata = {name: write(root / 'metadata' / name, value) for name, value in {
        'deepprobe_config.json': model, 'training_provenance.json': {'training_contract': contract},
        'trainer_state.json': {'global_step': 600, 'max_steps': 718}}.items()}
    names = {'projector.safetensors', 'language_model/config.json', 'language_model/model.safetensors',
             'vision_model/config.json', 'vision_model/model.safetensors', 'tokenizer/tokenizer_config.json',
             'tokenizer/tokenizer.json', 'image_processor/preprocessor_config.json', 'training_args.bin',
             'optimizer.pt', 'scheduler.pt', *(f'rng_state_{i}.pth' for i in range(8)), *metadata}
    names.update('synthetic-extra-' + str(i) for i in range(31 - len(names)))
    payload = b'synthetic native checkpoint payload, never deserialized'
    rows = [{'path': name, 'bytes': metadata[name]['bytes'] if name in metadata else len(payload),
             'sha256': metadata[name]['sha256'] if name in metadata else hashlib.sha256(payload).hexdigest()}
            for name in sorted(names)]
    history = {'mode': 'resume', 'config': config, 'frozen_source_manifest': old_manifest,
               'source_root': str(old), 'checkpoint_step': 600, 'checkpoint_metadata': metadata,
               'checkpoint': {'path': str(root / 'absent-private-checkpoint-600'),
                              'files': rows, 'oss_prefix': legacy.LEGACY_OSS_PREFIX}}
    env = {'execution_profile': {'schema_version': 3}, 'experiment_root': str(outer),
           'source_files': [runner.artifact(outer / name) for name in sorted(formal_integrity.source_inventory(outer))],
           'assets': {'training': {**data, 'frozen_source_manifest': outer_manifest}},
           'history': {legacy.TASK_ID: history}}
    monkeypatch.setattr(legacy, 'LEGACY_SOURCE_ROOT', str(old))
    monkeypatch.setattr(legacy, 'LEGACY_FROZEN_SHA', old_manifest['sha256'])
    monkeypatch.setattr(legacy, 'LEGACY_CONFIG_SHA', config['sha256'])
    monkeypatch.setattr(legacy, 'LEGACY_TRAIN_SHA', runner.sha(old / 'src/deepprobe_vlm/train.py'))
    f = {'root': root, 'old': old, 'outer': outer, 'task': task, 'history': history,
         'env': env, 'monkeypatch': monkeypatch}
    refresh_history(f); review(f)
    return f


def verify(f, **kwargs):
    return legacy.verify_legacy_private_resume(f['task'], f['history'], f['env'], **kwargs)


def test_production_bridge_constants_are_exact():
    assert legacy.TASK_ID == 'train__ouro_r4_s17__sft'
    assert legacy.LEGACY_FROZEN_SHA == 'ccc99b4780849055ea1df69ef69dfc645f3bdc66740bd0366f36958a07ac0cd2'
    assert legacy.LEGACY_CONFIG_SHA == '0ec088d7e769a6043792aa36ad97a8c718289a14370c530ddccaf11c87a2a606'
    assert legacy.LEGACY_TRAIN_SHA == 'cf7877633d210f7cef0651ee39278223496db1289c7ef146a3f98879abc1808a'
    assert legacy.LEGACY_INVENTORY_SHA == '5450d7c5854a1829e96f74920af99712b94b1c42f3eec9172fb599c1b2a97dd4'
    assert legacy.LEGACY_SOURCE_ROOT.endswith('/dp-p1-ouro-r4-f8-g4-s17-sft-a1/source')
    assert legacy.LEGACY_OSS_PREFIX.endswith('/dp-p1-ouro-r4-f8-g4-s17-sft-a1/failure-evidence/original_run/training/checkpoint-600/')


def test_metadata_only_bridge_keeps_old_history_unchanged_and_does_not_claim_hook(fixture):
    f = fixture; before = deepcopy(f['history'])
    result = verify(f)
    assert result['eligible'] is True and result['optimizer_step'] == 600
    assert result['source']['manifest_sha256'] != result['outer_source']['manifest_sha256']
    assert result['legacy_checkpoint_export_hook_present'] is False
    assert result['legacy_final_native_checkpoint_supported'] is True
    assert result['interrupted_recovery_checkpoint_step'] == 600
    assert result['post_success_export'] == {'required': True, 'optimizer_step': 718, 'world_size': 8,
                                             'execution': 'single_process', 'format': 'full_native',
                                             'requires_training_exit_code': 0}
    assert result['runtime_resume_verified'] is False and result['scheduler_admitted'] is False
    assert f['history'] == before
    assert not Path(f['history']['checkpoint']['path']).exists()


def test_shape_only_is_not_eligible_and_does_not_read_pin_payloads(fixture):
    f = fixture
    Path(f['env']['legacy_private_resume']['path']).unlink()
    result = verify(f, verify_files=False)
    assert result['eligible'] is False
    with pytest.raises(FileNotFoundError):
        verify(f)


@pytest.mark.parametrize('change', ['task', 'stage', 'target-step', 'resume-step', 'mode', 'source-root',
                                  'source-manifest', 'config', 'inventory', 'objects', 'oss-prefix', 'profile',
                                  'metadata-missing', 'history-not-bound', 'review-missing'])
def test_rejects_any_other_bridge_identity(fixture, change):
    f = fixture; h, env, task = f['history'], f['env'], f['task']
    if change == 'task': task['id'] = 'train__ouro_r2_s17__sft'
    elif change == 'stage': task['stage'] = 'alignment'
    elif change == 'target-step': task['expected_optimizer_steps'] = 719
    elif change == 'resume-step': h['checkpoint_step'] = 650
    elif change == 'mode': h['mode'] = 'accepted'
    elif change == 'source-root': h['source_root'] = str(f['outer'])
    elif change == 'source-manifest': h['frozen_source_manifest']['sha256'] = '0' * 64
    elif change == 'config': h['config']['sha256'] = '0' * 64
    elif change == 'inventory': h['checkpoint']['files'][0]['sha256'] = '0' * 64
    elif change == 'objects': h['checkpoint']['files'].pop()
    elif change == 'oss-prefix': h['checkpoint']['oss_prefix'] += 'copy/'
    elif change == 'profile': env['execution_profile']['schema_version'] = 2
    elif change == 'metadata-missing': h['checkpoint_metadata'].pop('trainer_state.json')
    elif change == 'history-not-bound': env['history'] = {}
    elif change == 'review-missing': env.pop('legacy_private_resume')
    with pytest.raises((ValueError, KeyError)):
        verify(f)


@pytest.mark.parametrize('field,value', [('approved', False), ('approved', 1), ('production_evidence', False),
    ('status', 'PROPOSED'), ('task_id', 'train__other__sft'), ('task_sha256', '0' * 64),
    ('history_sha256', '0' * 64), ('schema_version', True)])
def test_independent_production_review_required(fixture, field, value):
    review(fixture, **{field: value})
    with pytest.raises(ValueError):
        verify(fixture)


@pytest.mark.parametrize('change', ['claims-hook', 'hook-number-zero', 'world-float', 'new-interrupted-step', 'export-before-success',
                                  'export-not-native', 'different-outer', 'data-pin', 'missing-fullget',
                                  'duplicate-evidence', 'history-repinned'])
def test_review_binds_sources_data_honest_export_boundary_and_complete_history(fixture, change):
    f = fixture; value = deepcopy(f['review']); facts = value['verified_facts']
    if change == 'claims-hook': facts['legacy_checkpoint_export_hook_present'] = True
    elif change == 'hook-number-zero': facts['legacy_checkpoint_export_hook_present'] = 0
    elif change == 'world-float': facts['world_size'] = 8.0
    elif change == 'new-interrupted-step': facts['interrupted_recovery_checkpoint_step'] = 650
    elif change == 'export-before-success': facts['post_success_export']['requires_training_exit_code'] = 1
    elif change == 'export-not-native': facts['post_success_export']['format'] = 'weights_only'
    elif change == 'different-outer': facts['outer_source_files_sha256'] = '0' * 64
    elif change == 'data-pin': facts['training_data']['formal_ready']['sha256'] = '0' * 64
    elif change == 'missing-fullget': value['evidence'].remove(f['history']['archive_acceptance'])
    elif change == 'duplicate-evidence': value['evidence'].append(value['evidence'][0])
    elif change == 'history-repinned': f['history']['checkpoint']['path'] += '-changed'
    f['env']['legacy_private_resume'] = write(f['root'] / 'BRIDGE_REVIEW.json', value)
    with pytest.raises(ValueError):
        verify(f)


@pytest.mark.parametrize('change', ['old-tree', 'outer-tree', 'data-bytes', 'metadata-bytes', 'missing-source-pin',
                                  'duplicate-source-pin', 'source-pin-outside', 'review-bytes'])
def test_live_artifact_bytes_and_complete_source_trees_are_rechecked(fixture, change):
    f = fixture
    if change == 'old-tree': (f['old'] / 'src/deepprobe_vlm/train.py').write_text('# changed old source\n')
    elif change == 'outer-tree': (f['outer'] / 'src/deepprobe_vlm/train.py').write_text('# changed outer source\n')
    elif change == 'data-bytes': Path(f['env']['assets']['training']['train_manifest']['path']).write_text('{}')
    elif change == 'metadata-bytes': Path(f['history']['checkpoint_metadata']['trainer_state.json']['path']).write_text('{}')
    elif change == 'missing-source-pin':
        f['env']['source_files'].pop(); review(f)
    elif change == 'duplicate-source-pin':
        f['env']['source_files'].append(f['env']['source_files'][0]); review(f)
    elif change == 'source-pin-outside':
        f['env']['source_files'][0] = write(f['root'] / 'not-outer.py', b'# unrelated\n'); review(f)
    elif change == 'review-bytes': Path(f['env']['legacy_private_resume']['path']).write_text('{}')
    with pytest.raises(ValueError):
        verify(f)


@pytest.mark.parametrize('name,field,value', [
    ('training_provenance.json', 'world_size', 4), ('training_provenance.json', 'world_size', True),
    ('training_provenance.json', 'global_batch_size', 64),
    ('training_provenance.json', 'frozen_source_manifest_sha256', '0' * 64),
    ('training_provenance.json', 'train_manifest_sha256', '0' * 64),
    ('training_provenance.json', 'formal_training', False),
    ('trainer_state.json', 'global_step', 650), ('trainer_state.json', 'max_steps', 719),
    ('deepprobe_config.json', 'num_loops', 2),
])
def test_even_repinned_reviews_cannot_bypass_original_metadata_contract(fixture, name, field, value):
    f = fixture; pin = f['history']['checkpoint_metadata'][name]; data = runner.read(pin['path'])
    (data['training_contract'] if name == 'training_provenance.json' else data)[field] = value
    f['history']['checkpoint_metadata'][name] = write(Path(pin['path']), data)
    refresh_history(f); review(f)
    with pytest.raises(ValueError):
        verify(f)


def test_reaudited_archive_still_needs_full_get_true(fixture):
    f = fixture; h = f['history']; archive = runner.read(h['archive_acceptance']['path'])
    archive['all_objects_full_get_sha256_verified'] = False
    h['archive_acceptance'] = write(Path(h['archive_acceptance']['path']), archive)
    acceptance = runner.read(h['acceptance']['path']); acceptance['evidence'] = [h['archive_acceptance']]
    h['acceptance'] = write(Path(h['acceptance']['path']), acceptance); review(f)
    with pytest.raises(ValueError, match='full-GET'):
        verify(f)


def test_legacy_training_entry_accepts_audited_class_method_structure(tmp_path, monkeypatch):
    root = tmp_path.resolve(); name = 'src/deepprobe_vlm/train.py'
    pin = write(root / name, LEGACY_TRAIN_ENTRY)
    monkeypatch.setattr(legacy, 'LEGACY_TRAIN_SHA', pin['sha256'])
    legacy._verify_training_entry(root, {name: pin['sha256']})


@pytest.mark.parametrize('change', ['missing-saver', 'missing-main', 'main-does-not-call',
                                  'hook-import', 'hook-module', 'wrong-train-sha', 'changed-live-bytes',
                                  'wrong-class', 'duplicate-class', 'duplicate-main', 'wrong-self',
                                  'missing-native-save', 'wrong-main-receiver', 'wrong-native-receiver',
                                  'nested-main-call', 'nested-native-call', 'top-level-helper'])
def test_legacy_training_entry_requires_final_native_saver_and_no_intermediate_hook(tmp_path, monkeypatch, change):
    root = tmp_path.resolve(); name = 'src/deepprobe_vlm/train.py'
    source = LEGACY_TRAIN_ENTRY.decode()
    if change == 'missing-saver': source = source.replace('def save_final_resume_checkpoint(self):', 'def other(self):')
    elif change == 'missing-main': source = source.replace('def main(argv=None):', 'def other(argv=None):')
    elif change == 'main-does-not-call': source = source.replace('trainer.save_final_resume_checkpoint()', 'pass')
    elif change == 'hook-import': source += '\nfrom . import suite_checkpoint_export\n'
    elif change == 'wrong-class': source = source.replace('class DeepProbeTrainer(Trainer):', 'class OtherTrainer(Trainer):')
    elif change == 'duplicate-class': source += '\nclass DeepProbeTrainer(Trainer):\n    pass\n'
    elif change == 'duplicate-main': source += '\ndef main():\n    pass\n'
    elif change == 'wrong-self': source = source.replace('def save_final_resume_checkpoint(self):', 'def save_final_resume_checkpoint(other):')
    elif change == 'missing-native-save': source = source.replace('self._save_checkpoint(self.model, trial=None)', 'pass')
    elif change == 'wrong-main-receiver': source = source.replace('trainer.save_final_resume_checkpoint()', 'other.save_final_resume_checkpoint()')
    elif change == 'wrong-native-receiver': source = source.replace('self._save_checkpoint(', 'other._save_checkpoint(')
    elif change == 'nested-main-call': source = source.replace('trainer.save_final_resume_checkpoint()', 'def unused():\n            trainer.save_final_resume_checkpoint()')
    elif change == 'nested-native-call': source = source.replace('self._save_checkpoint(self.model, trial=None)', 'def unused():\n            self._save_checkpoint(self.model, trial=None)')
    elif change == 'top-level-helper': source = 'def save_final_resume_checkpoint(trainer):\n    pass\n\ndef main():\n    save_final_resume_checkpoint(None)\n'
    pin = write(root / name, source.encode())
    monkeypatch.setattr(legacy, 'LEGACY_TRAIN_SHA', pin['sha256'])
    files = {name: pin['sha256']}
    if change == 'hook-module': files['src/deepprobe_vlm/suite_checkpoint_export.py'] = '0' * 64
    elif change == 'wrong-train-sha': files[name] = '0' * 64
    elif change == 'changed-live-bytes': (root / name).write_text(source + '\n# changed after manifest verification\n')
    with pytest.raises(ValueError):
        legacy._verify_training_entry(root, files)


@pytest.mark.parametrize('key,value', [('bytes', 1), ('sha256', '0' * 64)])
def test_retained_metadata_pin_must_equal_native_inventory(fixture, key, value):
    fixture['history']['checkpoint_metadata']['trainer_state.json'][key] = value
    with pytest.raises(ValueError, match='metadata'):
        verify(fixture)


def test_real_identical_outer_manifest_path_rebinding_preserves_original_review(fixture):
    f = fixture; original_review = deepcopy(f['env']['legacy_private_resume'])
    original_outer = deepcopy(f['env']['assets']['training']['frozen_source_manifest'])
    rebound = write(f['root'] / 'workflow' / 'P1_SOURCE_MANIFEST.json', Path(original_outer['path']).read_bytes())
    f['env']['assets']['training']['frozen_source_manifest'] = rebound
    assert rebound['path'] != original_outer['path']
    assert verify(f)['eligible'] is True
    assert f['env']['legacy_private_resume'] == original_review
    assert original_outer in runner.read(original_review['path'])['evidence']


@pytest.mark.parametrize('which', ['outer-changed-bytes', 'legacy-manifest', 'legacy-config', 'training-data'])
def test_rebinding_exception_is_only_for_identical_outer_manifest_bytes(fixture, which):
    f = fixture; training = f['env']['assets']['training']
    if which == 'outer-changed-bytes':
        pin = training['frozen_source_manifest']
        training['frozen_source_manifest'] = write(f['root'] / 'rebound.json', Path(pin['path']).read_bytes() + b' ')
    elif which in {'legacy-manifest', 'legacy-config'}:
        name = 'frozen_source_manifest' if which == 'legacy-manifest' else 'config'
        pin = f['history'][name]
        f['history'][name] = write(f['root'] / 'rebound.json', Path(pin['path']).read_bytes())
    else:
        pin = training['formal_ready']
        training['formal_ready'] = write(f['root'] / 'rebound.json', Path(pin['path']).read_bytes())
    with pytest.raises(ValueError):
        verify(f)


def test_prepare_workflow_rebinds_manifest_then_passes_real_legacy_guard(fixture, monkeypatch):
    from deepprobe_vlm import suite_bindings, suite_preflight, suite_run_root, suite_workflow

    f = fixture; environment = deepcopy(f['env']); original = deepcopy(environment['assets']['training']['frozen_source_manifest'])
    environment['storage'] = {'schema_version': 3}
    environment['assets']['development'] = {'source_metadata': {'synthetic_fixture': True}}
    plan = {'tasks': [f['task']]}
    candidate = {'schema_version': 1, 'environment': environment, 'state_root': str(f['root'] / 'new-state')}
    # This integration keeps real source_inventory/save/rebinding and the real
    # guard; unrelated storage/asset/scheduler admission is isolated on CPU.
    monkeypatch.setattr(suite_workflow, 'workflow_report', lambda *args: {'synthetic_fixture': True})
    monkeypatch.setattr(suite_run_root, 'validate_run_root_location', lambda env: None)
    monkeypatch.setattr(suite_run_root, 'prepare_run_root', lambda env: {'synthetic_fixture': True})
    monkeypatch.setattr(suite_bindings, 'remaining_summary', lambda *args: {'blocked_history_tasks': []})
    checked = []

    def build_binding(plan, env, **kwargs):
        checked.append(legacy.verify_legacy_private_resume(f['task'], env['history'][legacy.TASK_ID], env))
        return {'synthetic_binding': True}

    def inspect(plan, binding, env, **kwargs):
        checked.append(legacy.verify_legacy_private_resume(f['task'], env['history'][legacy.TASK_ID], env))
        return {'status': 'SUITE_CPU_ADMITTED', 'synthetic_unrelated_gates': True}

    monkeypatch.setattr(suite_bindings, 'build_binding', build_binding)
    monkeypatch.setattr(suite_preflight, 'inspect', inspect)
    result = suite_workflow.prepare_workflow(plan, candidate, f['root'] / 'control' / 'workflow',
                                            prepare_assets=True, cpu_admission=True)
    emitted = runner.read(result['environment']['path'])
    current = emitted['assets']['training']['frozen_source_manifest']
    assert current['path'] != original['path']
    assert all(current[key] == original[key] for key in ('bytes', 'sha256'))
    assert emitted['legacy_private_resume'] == f['env']['legacy_private_resume']
    assert emitted['history'] == f['env']['history']
    assert result['cpu_admitted'] is True and result['gpu_submitted'] is False
    assert len(checked) == 2 and all(row['eligible'] for row in checked)
