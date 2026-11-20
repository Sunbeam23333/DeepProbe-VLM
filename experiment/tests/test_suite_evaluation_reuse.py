"""Synthetic complete history receipts must skip inference and retain provenance."""
import json

import pytest

from deepprobe_vlm import experiment_suite, suite_actions as actions


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return actions.pin(path)


def fixture(tmp_path, monkeypatch, benchmark='videomme899'):
    plan = experiment_suite.default_plan()
    task = next(t for t in plan['tasks'] if t['kind'] == 'evaluate'
                and t['model_id'] == 'ouro_r1_s17' and t['benchmark'] == benchmark)
    qa = 2697 if benchmark == 'videomme899' else 1580
    rows = [{'sample_id': str(i), 'correct': i % 2,
             'question_identity_sha256': actions.digest({'synthetic_question': i})} for i in range(qa)]
    result = write(tmp_path / 'original/EVALUATION.json', {
        'status': 'SUITE_EVALUATION_COMPLETE', 'suite_protocol': task['protocol'],
        'task_sha256': actions.digest(task), 'qa': qa, 'failed_units': 0, 'rows': rows})
    parent = {'checkpoint_files': [{'path': 'model.safetensors', 'bytes': 1, 'sha256': 'a' * 64}],
              'training_complete': write(tmp_path / 'parent/COMPLETE.json', {'synthetic': True})}
    evidence = write(tmp_path / 'review.json', {'synthetic_test_evidence_only': True})
    acceptance = write(tmp_path / 'HISTORY_ACCEPTED.json', {
        'schema_version': 1, 'status': 'SUITE_HISTORY_ACCEPTED', 'production_evidence': True,
        'task_id': task['id'], 'task_sha256': actions.digest(task), 'mode': 'accepted',
        'evidence': [evidence], 'evaluation_sha256': result['sha256'],
        'checkpoint_inventory_sha256': actions.digest(parent['checkpoint_files']),
        'protocol': task['protocol'], 'qa': qa, 'failed_units': 0})
    history = {'mode': 'accepted', 'acceptance': acceptance, 'evaluation': result,
               'checkpoint_inventory_sha256': actions.digest(parent['checkpoint_files'])}
    env = {'experiment_root': str(tmp_path), 'run_root': str(tmp_path / 'runs'),
           'source_files': [], 'history': {task['id']: history}}
    worker = actions.Actions(plan, env, tmp_path / 'state')
    monkeypatch.setattr(worker, 'completed', lambda tid: parent)
    def forbidden(*args, **kwargs):
        raise AssertionError('Accepted evaluation must not load benchmark media or execute inference')
    monkeypatch.setattr(worker, 'benchmark_asset', forbidden)
    monkeypatch.setattr(worker, 'command', forbidden)
    return worker, task, history, parent


@pytest.mark.parametrize('benchmark', ['videomme899', 'tempcompass_mc'])
def test_reuses_complete_result_and_emits_exact_checkpoint_consumer(tmp_path, monkeypatch, benchmark):
    worker, task, history, parent = fixture(tmp_path, monkeypatch, benchmark)
    original = actions.read(history['evaluation']['path'])
    result = worker.evaluate(task)
    assert result['evaluation'] == history['evaluation']
    assert actions.read(result['evaluation']['path']) == original
    reuse = actions.read(result['outputs'][1]['path'])
    assert reuse['history_acceptance'] == history['acceptance']
    assert reuse['parent_training_complete'] == parent['training_complete']
    assert reuse['inference_executed'] is reuse['new_quality_measurement'] is False
    consumer = actions.read(result['consumer_receipts'][0]['path'])
    assert consumer['completed_units'] == original['qa'] and consumer['failed_units'] == 0
    assert consumer['checkpoint_inventory_sha256'] == actions.digest([
        {**r, 'path': 'training/final/' + r['path']} for r in parent['checkpoint_files']])
    assert worker.evaluate(task) == result


@pytest.mark.parametrize('damage', ['result_bytes', 'review_bytes', 'parent', 'protocol', 'coverage', 'mode'])
def test_invalid_history_never_falls_back_to_rerun(tmp_path, monkeypatch, damage):
    worker, task, history, parent = fixture(tmp_path, monkeypatch)
    if damage == 'result_bytes':
        write(tmp_path / 'original/EVALUATION.json', {'tampered': True})
    elif damage == 'review_bytes':
        write(tmp_path / 'review.json', {'changed_review': True})
    elif damage == 'parent':
        parent['checkpoint_files'][0]['sha256'] = 'b' * 64
    elif damage == 'mode':
        history['mode'] = 'resume'
    else:
        value = actions.read(history['evaluation']['path'])
        if damage == 'protocol': value['suite_protocol'] = 'different-protocol'
        else: value['rows'].pop()
        history['evaluation'] = write(tmp_path / 'original/EVALUATION.json', value)
        accepted = actions.read(history['acceptance']['path'])
        accepted['evaluation_sha256'] = history['evaluation']['sha256']
        history['acceptance'] = write(tmp_path / 'HISTORY_ACCEPTED.json', accepted)
    with pytest.raises(ValueError):
        worker.evaluate(task)
    assert not (worker.state / 'artifacts' / task['id'] / 'HISTORICAL_EVALUATION_REUSE.json').exists()
