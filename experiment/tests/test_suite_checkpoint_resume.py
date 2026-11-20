"""Synthetic CPU metadata recovery; no real review, model, bucket or allocation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from deepprobe_vlm import formal_integrity, suite_bindings, suite_checkpoint_resume as resume
from deepprobe_vlm import suite_runner as runner


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else runner.canonical(value) + b'\n')
    return runner.artifact(path)


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path.resolve(); source = root / 'current-source'; original = root / 'original-source'
    for name in formal_integrity.SOURCE_DIRS:
        (source / name).mkdir(parents=True)
    for name in (*formal_integrity.REQUIRED_P1_SOURCE, *formal_integrity.SOURCE_FILES,
                 'src/deepprobe_vlm/suite_checkpoint_export.py'):
        write(source / name, b'# synthetic frozen source\n')
    model = {'num_frames': 8, 'pool_grid': 4, 'num_loops': 4,
             'language_revision': 'a' * 40, 'vision_revision': 'b' * 40}
    config = write(source / 'configs/recipe.json', {'model': model, 'training': {'max_steps': 718}})
    task = {'id': 'train__synthetic__sft', 'kind': 'train', 'stage': 'sft',
            'config': {'path': 'configs/recipe.json', 'sha256': config['sha256']},
            'expected_optimizer_steps': 718}
    files = formal_integrity.source_inventory(source)
    frozen = write(root / 'shared' / 'SOURCE.json', {'schema_version': 1,
        'purpose': 'DEEPPROBE_FROZEN_P1_SOURCE', 'files': files})
    shutil.copytree(source, original)
    input_pin = write(root / 'shared' / 'DATA.json', {'synthetic': 'input-identities'})
    env = {'execution_profile': {'schema_version': 3}, 'experiment_root': str(source),
        'source_files': [runner.artifact(source / name) for name in sorted(files)],
        'archive_root': 'oss://ss-devbox-stream/fixture/new-suite',
        'storage': {'oss_prefixes': ['oss://ss-devbox-stream/fixture/']},
        'assets': {'training': {'frozen_source_manifest': frozen, 'train_manifest': input_pin,
                               'eval_manifest': input_pin, 'formal_ready': input_pin}}}
    contract = {'stage': 'sft', 'config_sha256': config['sha256'],
        'frozen_source_manifest_sha256': frozen['sha256'], 'train_manifest_sha256': input_pin['sha256'],
        'eval_manifest_sha256': input_pin['sha256'], 'formal_ready_sha256': input_pin['sha256'],
        'formal_training': True, 'world_size': 8, 'global_batch_size': 128}
    payload = {name: b'synthetic-native-payload' for name in (
        'projector.safetensors', 'language_model/config.json', 'language_model/model.safetensors',
        'vision_model/config.json', 'vision_model/model.safetensors', 'tokenizer/tokenizer_config.json',
        'tokenizer/tokenizer.json', 'image_processor/preprocessor_config.json', 'training_args.bin',
        'optimizer.pt', 'scheduler.pt', *(f'rng_state_{i}.pth' for i in range(8)))}
    metadata = {}
    for name, value in {'deepprobe_config.json': model, 'training_provenance.json': {'training_contract': contract},
                        'trainer_state.json': {'global_step': 350, 'max_steps': 718}}.items():
        metadata[name] = write(root / 'shared' / 'metadata' / name, value)
        payload[name] = Path(metadata[name]['path']).read_bytes()
    rows = [{'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
            for name, raw in sorted(payload.items())]
    generation = runner.digest(rows)
    run_id = 'previous-suite--' + task['id']; runs = root / 'lost-pod' / 'runs' / 'worker-01'
    owner = {'schema_version': 1, 'suite_id': 'previous-suite', 'run_id': run_id,
             'runs_root': str(runs), 'run_root': str(runs / run_id), 'source_sha256': 'c' * 64}
    prefix = 'oss://ss-devbox-stream/fixture/old-suite/savepoints/runs/' + run_id + '/checkpoint-350/' + generation + '/'
    manifest = write(root / 'shared' / 'MANIFEST.json', {'schema_version': 1,
        'status': 'SUITE_NATIVE_CHECKPOINT_MANIFEST', 'owner': owner, 'task_id': task['id'],
        'task_sha256': runner.digest(task), 'optimizer_step': 350, 'expected_optimizer_steps': 718,
        'world_size': 8, 'frozen_source_manifest': frozen, 'checkpoint_inventory_sha256': generation,
        'archive_prefix': prefix, 'files': rows})
    ready = write(root / 'shared' / 'READY.json', {'schema_version': 1, 'status': 'SUITE_NATIVE_CHECKPOINT_OSS_READY',
        'manifest_sha256': manifest['sha256'], 'manifest_bytes': manifest['bytes'],
        'manifest_object': prefix + 'MANIFEST.json', 'checkpoint_inventory_sha256': generation,
        'optimizer_step': 350, 'world_size': 8, 'payload_objects': len(rows),
        'payload_bytes': sum(row['bytes'] for row in rows), 'full_get_sha256_verified': True,
        'control_backend': 'gloo', 'control_timeout_seconds': 21600, 'automatic_resume_authorized': False})
    exported = write(root / 'shared' / 'EXPORT_ACCEPTED.json', {'schema_version': 1,
        'status': 'SUITE_SAVEPOINT_EXPORTED', 'manifest': manifest, 'ready': ready, 'archive_prefix': prefix,
        'checkpoint_oss_prefix': prefix, 'all_objects_full_get_sha256_verified': True,
        'ready_published_last': True, 'stable_listing_sha256': 'd' * 64,
        'checkpoint_metadata': metadata, 'checkpoint_inventory_sha256': generation, 'optimizer_step': 350})
    entry = {'mode': 'resume', 'config': config, 'frozen_source_manifest': frozen,
        'source_root': str(original), 'checkpoint_step': 350,
        'checkpoint': {'path': str(runs / run_id / 'training/checkpoint-350'), 'files': rows, 'oss_prefix': prefix},
        'archive_acceptance': exported, 'checkpoint_metadata': metadata}
    return {'root': root, 'source': source, 'original': original, 'task': task, 'plan': {'tasks': [task]},
            'env': env, 'entry': entry, 'export': exported}


def review(f, entry, **changes):
    # Explicit synthetic trust-boundary fixture. Production evidence must come
    # from a separate external review; the candidate generator cannot make it.
    facts = {'source_root': entry['source_root'], 'config_sha256': entry['config']['sha256'],
        'frozen_source_manifest_sha256': entry['frozen_source_manifest']['sha256'],
        'checkpoint_inventory_sha256': runner.digest(entry['checkpoint']['files']),
        'checkpoint_oss_prefix': entry['checkpoint']['oss_prefix'], 'optimizer_step': entry['checkpoint_step'],
        'archive_verified': True, 'resume_metadata_verified': True, 'optimizer_scheduler_rng_deserialized': False}
    value = {'status': 'SUITE_HISTORY_IMPORT_REVIEW_ACCEPTED', 'approved': True, 'production_evidence': True,
             'task_id': f['task']['id'], 'task_sha256': runner.digest(f['task']), 'entry_sha256': runner.digest(entry),
             'evidence': [entry['archive_acceptance']], 'verified_facts': facts, **changes}
    return write(f['root'] / 'external-reviews' / (runner.digest(value) + '.json'), value)


def imported(f, entry=None):
    entry = deepcopy(entry or f['entry'])
    target = f['root'] / 'external-imports' / (runner.digest(entry) + '.json')
    return suite_bindings.import_history_entry(f['plan'], f['task']['id'], entry, review(f, entry), target)


def test_reviewed_resume_accepts_identical_relocated_source_and_never_hydrates_old_pod(fixture):
    f = fixture; history = imported(f)
    result = resume.verify_private_resume(f['task'], history, f['env'])
    assert result['eligible'] and result['optimizer_step'] == 350
    assert result['fresh_remote_payload_verification_deferred_to_restore']
    assert not result['runtime_resume_verified'] and not result['scheduler_admitted']
    assert not (f['root'] / 'lost-pod').exists()
    assert history['source_root'] != f['env']['experiment_root']


def test_shape_only_never_claims_resume_eligibility_or_reads_source(fixture):
    f = fixture; history = imported(f)
    shutil.rmtree(f['original'])
    assert resume.verify_private_resume(f['task'], history, f['env'], verify_files=False)['eligible'] is False
    with pytest.raises(FileNotFoundError):
        resume.verify_private_resume(f['task'], history, f['env'])


def test_proposal_cannot_self_authorize_and_external_review_is_exact(fixture):
    f = fixture; output = f['root'] / 'proposal'
    proposal = resume.prepare_resume_candidate(f['plan'], f['env'], f['task']['id'], f['export'], output)
    value = runner.read(proposal['path']); entry = runner.read(value['entry']['path'])
    assert not value['accepted'] and value['remaining_optimizer_updates'] == 368
    assert 'acceptance' not in entry and not (output / 'HISTORY_ACCEPTANCE.json').exists()
    with pytest.raises(ValueError, match='independent review'):
        resume.prepare_resume_candidate(f['plan'], f['env'], f['task']['id'], f['export'], output, review_pin=proposal)
    assert not (output / 'HISTORY_ACCEPTANCE.json').exists()
    result = resume.prepare_resume_candidate(f['plan'], f['env'], f['task']['id'], f['export'], output,
                                             review_pin=review(f, entry))
    accepted = runner.read(result['path'])
    assert accepted['verification']['eligible'] and not accepted['gpu_submitted']
    assert not (f['root'] / 'lost-pod').exists()


@pytest.mark.parametrize('field', ['approved', 'production_evidence', 'entry_sha256', 'runtime_claim'])
def test_bad_independent_review_does_not_publish_history_acceptance(fixture, field):
    f = fixture; output = f['root'] / 'proposal'
    proposal = runner.read(resume.prepare_resume_candidate(f['plan'], f['env'], f['task']['id'], f['export'], output)['path'])
    entry = runner.read(proposal['entry']['path']); pin = review(f, entry)
    value = runner.read(pin['path'])
    if field == 'runtime_claim': value['verified_facts']['optimizer_scheduler_rng_deserialized'] = True
    else: value[field] = '0' * 64 if field == 'entry_sha256' else False
    pin = write(f['root'] / 'bad-review.json', value)
    with pytest.raises(ValueError, match='review'):
        resume.prepare_resume_candidate(f['plan'], f['env'], f['task']['id'], f['export'], output, review_pin=pin)
    assert not (output / 'HISTORY_ACCEPTANCE.json').exists()


@pytest.mark.parametrize('change', ['old_source_sha', 'current_source_bytes', 'historical_source_bytes',
                                  'missing_source_pin', 'declared_config_sha', 'outside_archive', 'wrong_local_checkpoint'])
def test_source_config_and_archive_scope_cannot_be_rebound(fixture, change):
    f = fixture; history = imported(f)
    if change == 'old_source_sha':
        f['env']['assets']['training']['frozen_source_manifest'] = {**history['frozen_source_manifest'], 'sha256': 'e' * 64}
    elif change == 'current_source_bytes':
        (f['source'] / 'src/deepprobe_vlm/train.py').write_text('changed current source')
    elif change == 'historical_source_bytes':
        (f['original'] / 'src/deepprobe_vlm/train.py').write_text('changed original source')
    elif change == 'missing_source_pin': f['env']['source_files'].pop()
    elif change == 'declared_config_sha': f['task']['config']['sha256'] = 'e' * 64
    elif change == 'outside_archive': f['env']['storage']['oss_prefixes'] = ['oss://ss-devbox-stream/fixture-other/']
    else:
        entry = deepcopy(f['entry']); entry['checkpoint']['path'] = str(f['root'] / 'arbitrary-checkpoint-350')
        history = imported(f, entry)
    with pytest.raises(ValueError):
        resume.verify_private_resume(f['task'], history, f['env'])


@pytest.mark.parametrize('document,key,value', [
    ('export', 'ready_published_last', False), ('export', 'all_objects_full_get_sha256_verified', False),
    ('export', 'stable_listing_sha256', ''), ('manifest', 'world_size', 4), ('manifest', 'task_sha256', 'f' * 64),
    ('manifest', 'optimizer_step', 349), ('ready', 'manifest_sha256', 'f' * 64),
    ('ready', 'payload_objects', 1), ('ready', 'payload_bytes', 1), ('ready', 'full_get_sha256_verified', False),
    ('ready', 'automatic_resume_authorized', True)])
def test_export_manifest_ready_tampering_rejected_even_if_external_history_is_repinned(fixture, document, key, value):
    f = fixture; entry = deepcopy(f['entry']); exported = runner.read(entry['archive_acceptance']['path'])
    if document == 'export': exported[key] = value
    else:
        target = runner.read(exported[document]['path']); target[key] = value
        exported[document] = write(f['root'] / ('modified-' + document + '.json'), target)
        if document == 'manifest':
            ready = runner.read(exported['ready']['path'])
            ready.update(manifest_sha256=exported['manifest']['sha256'], manifest_bytes=exported['manifest']['bytes'])
            exported['ready'] = write(f['root'] / 'modified-ready.json', ready)
    entry['archive_acceptance'] = write(f['root'] / 'modified-export.json', exported)
    history = imported(f, entry)
    with pytest.raises(ValueError):
        resume.verify_private_resume(f['task'], history, f['env'])


def test_missing_rank_rng_vocabulary_or_retained_metadata_is_never_eligible(fixture):
    f = fixture
    for name in ('rng_state_7.pth', 'tokenizer/tokenizer.json', 'trainer_state.json'):
        history = deepcopy(imported(f))
        if name != 'trainer_state.json':
            history['checkpoint']['files'] = [r for r in history['checkpoint']['files'] if r['path'] != name]
        else: history['checkpoint_metadata'].pop(name)
        with pytest.raises(ValueError):
            resume.verify_private_resume(f['task'], history, f['env'], verify_files=False)


def test_direct_self_signed_acceptance_is_not_a_private_resume_review(fixture):
    f = fixture; history = imported(f)
    receipt = runner.read(history['acceptance']['path'])
    receipt.pop('import_scope'); receipt['evidence'] = [history['archive_acceptance']]
    history['acceptance'] = write(f['root'] / 'self-signed.json', receipt)
    with pytest.raises(ValueError, match='separately reviewed'):
        resume.verify_private_resume(f['task'], history, f['env'])


def test_cli_defaults_to_pinned_proposal_only(fixture, capsys):
    f = fixture
    plan = write(f['root'] / 'PLAN.json', f['plan']); env = write(f['root'] / 'ENV.json', f['env'])
    args = ['--task', f['task']['id'], '--output', str(f['root'] / 'cli')]
    for name, pin in [('plan', plan), ('environment', env), ('export', f['export'])]:
        args += ['--' + name, pin['path'], '--' + name + '-sha256', pin['sha256']]
    assert resume.main(args) == 0
    result = runner.read(json.loads(capsys.readouterr().out)['path'])
    assert result['independent_review_required'] and not result['accepted']
    assert not (f['root'] / 'cli' / 'HISTORY_ACCEPTANCE.json').exists()
