"""Synthetic CPU scheduling tests; no production cache or GPU acceptance."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from deepprobe_vlm import suite_actions, suite_fleet, suite_preflight, suite_progressive
from deepprobe_vlm import suite_runner as runner, suite_workflow


@pytest.fixture
def scheduler():
    import test_suite_fleet as fixtures
    fixture = fixtures.FleetTest(methodName='test_stale_claim_never_expires_and_duplicate_worker_cannot_join')
    fixture.setUp()
    try:
        yield fixture
    finally:
        fixture.doCleanups()


def state_for(fixture, tasks):
    fixture.plan['tasks'] = tasks
    state = suite_fleet.FleetState(fixture.plan, fixture.binding, fixture.state,
        plan_sha256=runner.digest(fixture.plan), binding_sha256=runner.digest(fixture.binding))
    state.initialize()
    return state


def test_pending_evaluation_never_claims_reserves_or_blocks_ready_training(scheduler):
    tasks = [{'id': 'consumer', 'kind': 'evaluate', 'dependencies': []},
             {'id': 'first', 'kind': 'train', 'dependencies': []}]
    state = state_for(scheduler, tasks)
    reserved = []
    def reserve(task, owner):
        reserved.append(task['id']); return 'synthetic-token'
    selected = state.claim('worker-01', 'synthetic-job', reserve,
        readiness=lambda task: None if task['kind'] == 'evaluate' else {})
    assert selected['task']['id'] == 'first' and reserved == ['first']
    root = scheduler.state / 'tasks/consumer'
    assert not (root / 'CLAIM.json').exists() and not (root / 'attempts').exists()
    assert not list(scheduler.state.rglob('FAILED.json'))


def test_no_active_tasks_wait_for_external_cache_then_claim_exact_ready_once(scheduler):
    tasks = [{'id': 'consumer', 'kind': 'evaluate', 'dependencies': []}]
    state = state_for(scheduler, tasks)
    reserve = mock.Mock(return_value='synthetic-token')
    result = state.claim('worker-01', 'job', reserve, readiness=lambda task: None)
    assert result['status'] == 'wait' and result['asset_pending_task_ids'] == ['consumer']
    assert result['active_task_ids'] == []
    reserve.assert_not_called()
    ready_file = scheduler.home / 'READY.json'; ready_file.write_text('{"synthetic":true}')
    proof = {'ready_pin': runner.artifact(ready_file), 'contract_sha256': 'a' * 64,
             'cache_binding_sha256': 'b' * 64}
    selected = state.claim('worker-01', 'job', reserve, readiness=lambda task: proof)
    assert selected['claim']['asset_readiness'] == proof
    assert state.claim('worker-02', 'job2', reserve, readiness=lambda task: proof)['status'] == 'wait'
    assert reserve.call_count == 1


def test_present_invalid_readiness_fails_before_attempt_or_storage(scheduler):
    state = state_for(scheduler, [{'id': 'consumer', 'kind': 'evaluate', 'dependencies': []}])
    reserve = mock.Mock(side_effect=AssertionError('No reserve for invalid cache'))
    with pytest.raises(ValueError, match='corrupt READY'):
        state.claim('worker-01', 'job', reserve,
                    readiness=mock.Mock(side_effect=ValueError('corrupt READY')))
    assert not list(scheduler.state.rglob('CLAIM.json'))
    assert not list(scheduler.state.rglob('FAILED.json'))
    reserve.assert_not_called()


def test_claimed_ready_is_pinned_again_before_any_hydration(tmp_path, monkeypatch):
    task = {'id': 'evaluate', 'kind': 'evaluate', 'benchmark': 'videomme899', 'frames': 8}
    state = tmp_path / 'state'; attempt = state / 'tasks/evaluate/attempts/attempt-001'; attempt.mkdir(parents=True)
    ready_file = tmp_path / 'READY.json'; ready_file.write_text('{"accepted":true}')
    old_pin = runner.artifact(ready_file)
    proof = {'ready_pin': old_pin, 'contract_sha256': 'a' * 64, 'cache_binding_sha256': 'b' * 64}
    claim = {'task_id': task['id'], 'task_sha256': runner.digest(task), 'attempt': str(attempt), 'asset_readiness': proof}
    runner.write_once(state / 'tasks/evaluate/CLAIM.json', claim)
    runner.write_once(attempt / 'TASK_INTENT.json', {'task_id': task['id'], 'claim': claim})
    environment = {'experiment_root': str(tmp_path), 'run_root': str(tmp_path / 'runs'),
        'source_files': [], 'benchmark_preparations': {'synthetic': True}}
    actions = suite_actions.Actions({'tasks': [task]}, environment, state)
    ready_file.write_text('{"changed":true}')
    def resolve(env, benchmark, frames, expected_pin=None):
        assert expected_pin == old_pin
        runner.verify_pin(expected_pin)
        raise AssertionError('Changed claim must have failed')
    monkeypatch.setattr(suite_progressive, 'resolve_ready', resolve)
    with pytest.raises(ValueError, match='changed'):
        with actions.benchmark_asset(task):
            pytest.fail('Must not consume changed READY')


@pytest.fixture
def admission(tmp_path, monkeypatch):
    task = {'id': 'train', 'kind': 'train', 'stage': 'alignment', 'model_id': 'tiny',
            'config': {'path': 'recipe.json'}, 'historical_hint': {'status': 'planned'}}
    plan = {'tasks': [task], 'models': [{'id': 'tiny', 'family': 'ouro_r1'}], 'training_data': {}}
    source = tmp_path / 'source.json'; source.write_text('{"synthetic":true}')
    source_pin = runner.artifact(source)
    (tmp_path / 'recipe.json').write_text('{"model":{},"training":{}}')
    declarations = {suite_progressive.key(b, f): {'contract_sha256': 'a' * 64,
        'ready_path': str(tmp_path / (b + '-' + str(f) + '.json'))}
        for b in suite_progressive.BENCHMARKS for f in suite_progressive.FRAMES}
    (tmp_path / 'runs').mkdir()
    env = {'experiment_root': str(tmp_path), 'source_files': [source_pin], 'history': {},
        'run_root': str(tmp_path / 'runs'),
        'storage': {'roots': [str(tmp_path)]}, 'task_reserve_bytes': {'train_alignment': 300},
        'benchmark_preparations': declarations,
        'assets': {'training': {k: source_pin for k in ('train_manifest', 'eval_manifest', 'formal_ready', 'frozen_source_manifest')},
            'backbones': {'ouro_r1': {'language_root': '/lm', 'vision_root': '/vision'}}, 'benchmarks': {}}}
    binding = {'source_files': [source_pin], 'tasks': {'train': {'reserve_bytes': 300}}}
    for name in ('validate_plan', 'validate_environment'):
        monkeypatch.setattr(suite_preflight, name, lambda *args, **kwargs: True)
    monkeypatch.setattr(runner, 'validate_binding', lambda *args, **kwargs: True)
    for name in ('_bound_environment', '_formal_gate', '_training_gate', '_source_gate',
                 '_backbone_gate', '_development_gate', '_initial_storage_gate', '_chain_storage_gate'):
        monkeypatch.setattr(suite_preflight, name, lambda *args, **kwargs: {'synthetic': True})
    monkeypatch.setattr(suite_preflight, 'load_operation', lambda *_: SimpleNamespace(
        estimate_stage_storage=lambda *_: {'required_additional_bytes': 200}))
    monkeypatch.setattr(suite_progressive, 'resolve_ready', lambda *args, **kwargs: None)
    def run():
        return suite_preflight.inspect(plan, binding, env, plan_sha256='a' * 64, binding_sha256='b' * 64,
            environment_sha256='c' * 64, state_root=tmp_path / 'state')
    return env, binding, run


def test_missing_run_root_gates_cpu_without_creating_it_or_hydrating(admission, monkeypatch):
    env, _, run = admission
    root = Path(env['run_root']); root.rmdir()
    with mock.patch.object(suite_preflight, '_bound_environment', side_effect=AssertionError('must stop before model checks')):
        result = run()
    assert result['status'] == 'SUITE_CPU_GATED'
    assert [row['check'] for row in result['failures']] == ['run_root_directory']
    assert 'missing' in result['failures'][0]['error']
    assert not root.exists() and not (root.parent / 'state').exists()


def test_six_pending_caches_get_explicit_deferred_status_but_training_is_checked(admission):
    env, binding, run = admission
    result = run()
    assert result['status'] == 'SUITE_CPU_PROGRESSIVE_ADMITTED'
    assert len(result['deferred_benchmarks']) == 6 and not result['failures']
    assert all(c['status'] == 'deferred' for c in result['checks'] if c['check'].startswith('benchmark/'))
    assert any(c['check'] == 'training_data_and_live_media' and c['status'] == 'passed' for c in result['checks'])
    assert result['training_peak_estimates']['train']['required_additional_bytes'] == 200
    binding['tasks']['train']['reserve_bytes'] = 100
    assert run()['status'] == 'SUITE_CPU_GATED'


def test_training_media_failure_still_gates_progressive_mode(admission, monkeypatch):
    _, _, run = admission
    monkeypatch.setattr(suite_preflight, '_training_gate', mock.Mock(side_effect=ValueError('Current training SHA changed')))
    result = run()
    assert result['status'] == 'SUITE_CPU_GATED'
    assert any(f['check'] == 'training_data_and_live_media' for f in result['failures'])


def test_strict_default_still_blocks_missing_cache(admission):
    env, _, run = admission
    del env['benchmark_preparations']
    result = run()
    assert result['status'] == 'SUITE_CPU_GATED' and result['deferred_benchmarks'] == []
    assert len([f for f in result['failures'] if f['check'].startswith('benchmark/')]) == 6


def test_external_cache_workflow_keeps_full_dag_and_never_replays_history_or_cache_build(tmp_path, monkeypatch):
    from deepprobe_vlm import experiment_suite, suite_bindings
    plan = experiment_suite.default_plan(); original = deepcopy(plan)
    source = tmp_path / 'source.json'; source.write_text('{"synthetic":true}')
    pin = runner.artifact(source)
    requests = [{'benchmark': b, 'frames': f, 'source_manifest': pin, 'oss_prefix': 'synthetic/' + b + str(f),
                 'max_png_bytes': 1000, 'metadata_reserve_bytes': 1000}
                for b in suite_progressive.BENCHMARKS for f in suite_progressive.FRAMES]
    declarations = {suite_progressive.key(r['benchmark'], r['frames']): {
        'contract': {'request': r}, 'contract_sha256': 'a' * 64, 'ready_path': str(tmp_path / ('ready-' + str(i)))}
        for i, r in enumerate(requests)}
    (tmp_path / 'runs').mkdir()
    env = {'experiment_root': str(tmp_path), 'source_files': [], 'history': {'prepared': {'synthetic': True}},
        'run_root': str(tmp_path / 'runs'), 'storage': {'roots': [str(tmp_path)]},
        'assets': {'training': {}, 'development': {'source_metadata': pin}},
        'benchmark_preparations': declarations}
    candidate = {'schema_version': 1, 'environment': env, 'state_root': str(tmp_path / 'state'),
        'benchmark_preparations': requests, 'benchmark_preparation_mode': 'external_progressive'}
    monkeypatch.setattr(suite_progressive, 'validate_declarations', lambda *args: declarations)
    monkeypatch.setattr(suite_bindings, 'remaining_summary', lambda *args: {'blocked_history_tasks': []})
    monkeypatch.setattr(suite_bindings, 'build_binding', lambda *args, **kwargs: {'synthetic': True})
    monkeypatch.setattr(suite_workflow, 'prepare_benchmark', mock.Mock(side_effect=AssertionError('CPU producer owns cache work')))
    monkeypatch.setattr('deepprobe_vlm.suite_recovery.prepare_recovery', mock.Mock(side_effect=AssertionError('No repeated checkpoint GET')))
    monkeypatch.setattr('deepprobe_vlm.formal_integrity.source_inventory', lambda root: {'source.json': pin['sha256']})
    result = suite_workflow.prepare_workflow(plan, candidate, tmp_path / 'workflow', prepare_assets=True)
    assert result['status'] == 'SUITE_WORKFLOW_BOUND_NOT_ADMITTED' and len(result['cache_preparations']) == 6
    assert plan == original and len(plan['tasks']) == 185
    assert runner.read(result['environment']['path'])['history'] == env['history']
    assert not result['gpu_submitted']
