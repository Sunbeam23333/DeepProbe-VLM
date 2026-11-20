"""CPU-only publisher coordination; full PNG validation lives in progressive tests."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from unittest import mock

import pytest

from deepprobe_vlm import suite_cache_publisher as publisher, suite_runner as runner


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    request = {'benchmark': 'videomme899', 'frames': 8, 'synthetic': True}
    source = tmp_path / 'manifest.json'; source.write_text('{"source":true}')
    producer = {'source_root': str(tmp_path / 'source'), 'source_manifest': runner.artifact(source)}
    declaration = {'contract': {'request': request, 'producer': producer},
                   'ready_path': str(tmp_path / 'READY.json'), 'contract_sha256': 'a' * 64}
    environment = {'suite_id': 'synthetic-suite'}
    candidate = tmp_path / 'CANDIDATE.json'
    runner.write_once(candidate, {'benchmark_preparations': [request],
        'environment': {'experiment_root': producer['source_root']}})
    value = {'status': 'CPU_CACHE_ASSET_PREPARED_NOT_FLEET_ADMITTED', 'request': request,
             'request_sha256': runner.digest(request), 'producer_source': {
                 'root': producer['source_root'], 'manifest': producer['source_manifest']},
             'candidate': runner.artifact(candidate), 'asset': {'synthetic': True}}
    path = tmp_path / 'PREPARED_ASSET.json'; runner.write_once(path, value)
    budget = mock.Mock()
    budget.asset_lock.side_effect = lambda owner: nullcontext()
    budget.reserve.return_value = 'owned-token'
    monkeypatch.setattr(publisher, 'budget_for', mock.Mock(return_value=budget))
    monkeypatch.setattr(publisher.progressive, 'resolve_ready', mock.Mock(return_value=None))
    monkeypatch.setattr(publisher.progressive, 'publish_ready', mock.Mock(return_value={'proof': True}))
    return environment, declaration, value, path, budget


def test_pending_descriptor_does_not_create_reservation(prepared):
    env, declaration, _, path, budget = prepared
    path.unlink()
    assert publisher.publish_prepared(env, declaration) is None
    budget.reserve.assert_not_called()
    publisher.progressive.publish_ready.assert_not_called()


def test_ready_reuse_does_not_repeat_remote_publication(prepared):
    env, declaration, _, path, budget = prepared
    path.unlink()
    publisher.progressive.resolve_ready.return_value = {'existing': True}
    assert publisher.publish_prepared(env, declaration) == {'existing': True}
    budget.reserve.assert_not_called()
    publisher.progressive.publish_ready.assert_not_called()


@pytest.mark.parametrize('change', ['request', 'producer', 'candidate'])
def test_wrong_producer_descriptor_fails_before_storage_or_get(prepared, change):
    env, declaration, value, path, budget = prepared
    value = deepcopy(value)
    if change == 'request':
        value['request']['frames'] = 4
    elif change == 'producer':
        value['producer_source']['root'] += '-other'
    else:
        value['candidate']['sha256'] = '0' * 64
    path.write_bytes(runner.canonical(value))
    with pytest.raises(ValueError):
        publisher.publish_prepared(env, declaration)
    budget.reserve.assert_not_called()
    publisher.progressive.publish_ready.assert_not_called()


def test_publication_has_owned_budget_and_releases_after_sync_verification(prepared):
    env, declaration, value, _, budget = prepared
    assert publisher.publish_prepared(env, declaration) == {'proof': True}
    owner = {'suite_id': env['suite_id'], 'operation': 'cache-readiness-publication',
             'contract_sha256': declaration['contract_sha256']}
    budget.reserve.assert_called_once_with(owner, publisher.PUBLICATION_RESERVE_BYTES, 0)
    publisher.progressive.publish_ready.assert_called_once_with(env, declaration, value['asset'])
    budget.release.assert_called_once()
    args, kwargs = budget.release.call_args
    assert args == ('owned-token',)
    assert kwargs['quiescent'](owner) == {'owner': owner, 'active_writers': [], 'active_uploads': []}


def test_failed_full_verification_releases_only_its_own_reservation(prepared):
    env, declaration, _, _, budget = prepared
    publisher.progressive.publish_ready.side_effect = ValueError('PNG RGB SHA differs')
    with pytest.raises(ValueError, match='PNG RGB SHA differs'):
        publisher.publish_prepared(env, declaration)
    assert budget.release.call_args.args == ('owned-token',)


def test_producer_descriptor_race_is_detected_after_publication(prepared):
    env, declaration, _, path, budget = prepared
    def publish(*args):
        path.write_text('{"changed":true}')
        return {'proof': True}
    publisher.progressive.publish_ready.side_effect = publish
    with pytest.raises(ValueError, match='changed'):
        publisher.publish_prepared(env, declaration)
    budget.release.assert_called_once()


def test_six_shapes_cannot_report_complete_with_any_pending(monkeypatch):
    declarations = {str(i): {'i': i} for i in range(6)}
    monkeypatch.setattr(publisher.progressive, 'validate_declarations', lambda *args: declarations)
    publish = mock.Mock(side_effect=lambda env, d: None if d['i'] == 5 else {'asset': {}, 'ready_pin': d})
    monkeypatch.setattr(publisher, 'publish_prepared', publish)
    result = publisher.publish_available({'suite_id': 'synthetic'}, {})
    assert result['status'] == 'SUITE_CACHE_READINESS_PENDING' and result['pending'] == ['5']
    assert len(result['ready']) == 5 and publish.call_count == 6
    assert result['checkpoint_payload_gets'] == 0 and result['gpu_submitted'] is False


def test_failed_producer_marks_only_missing_descriptor_before_stopping(prepared, monkeypatch):
    env, declaration, _, path, budget = prepared
    path.unlink()
    monkeypatch.setattr(publisher.progressive, 'validate_declarations', lambda *args: {'one': declaration})
    monkeypatch.setattr(publisher, 'producer_outcome', lambda *args: {'successful': False, 'evidence': []})
    with pytest.raises(ValueError, match='unsuccessfully'):
        publisher.publish_available(env, {}, producer_start={'synthetic': True})
    failure = path.with_name('READY.FAILED.json')
    value = runner.read(failure)
    assert value['status'] == 'SUITE_BENCHMARK_PREPARATION_FAILED'
    assert value['contract'] == declaration['contract'] and value['automatic_retry'] is False
    assert budget.reserve.call_count == 1 and budget.release.call_count == 1


def test_later_producer_failure_does_not_revoke_ready(prepared, monkeypatch):
    env, declaration, _, path, budget = prepared
    path.with_name('READY.json').write_text('{"synthetic":true}')
    monkeypatch.setattr(publisher.progressive, 'validate_declarations', lambda *args: {'one': declaration})
    monkeypatch.setattr(publisher, 'producer_outcome', lambda *args: {'successful': False, 'evidence': []})
    publisher.progressive.resolve_ready.return_value = {'ready_pin': {'synthetic': True}, 'asset': {}}
    result = publisher.publish_available(env, {}, producer_start={'synthetic': True})
    assert result['status'] == 'SUITE_CACHE_READINESS_COMPLETE'
    assert not path.with_name('READY.FAILED.json').exists()
    budget.reserve.assert_not_called()


def test_terminal_producer_marks_all_missing_shapes_even_when_f4_precedes_f8(prepared, monkeypatch):
    env, declaration, _, path, _ = prepared
    path.unlink()
    first, second = deepcopy(declaration), deepcopy(declaration)
    first['contract']['request']['frames'] = 4
    first['ready_path'] = str(path.parent / 'f4/READY.json')
    first['contract_sha256'] = 'b' * 64
    second['ready_path'] = str(path.parent / 'f8/READY.json')
    monkeypatch.setattr(publisher.progressive, 'validate_declarations', lambda *args: {'f4': first, 'f8': second})
    monkeypatch.setattr(publisher, 'producer_outcome', lambda *args: {'successful': False, 'evidence': []})
    with pytest.raises(ValueError, match='unsuccessfully'):
        publisher.publish_available(env, {}, producer_start={'synthetic': True})
    for declaration in (first, second):
        failed = runner.read(Path(declaration['ready_path']).with_name('READY.FAILED.json'))
        assert failed['contract_sha256'] == declaration['contract_sha256']


def test_full_get_failure_is_durably_propagated_to_waiting_consumer(prepared, monkeypatch):
    env, declaration, _, path, budget = prepared
    monkeypatch.setattr(publisher.progressive, 'validate_declarations', lambda *args: {'one': declaration})
    publisher.progressive.publish_ready.side_effect = ValueError('real PNG SHA mismatch')
    with pytest.raises(ValueError, match='real PNG SHA mismatch'):
        publisher.publish_available(env, {})
    assert runner.read(path.with_name('READY.FAILED.json'))['error'] == 'real PNG SHA mismatch'
    assert budget.reserve.call_count == 2 and budget.release.call_count == 2


def test_sdk_failure_marks_every_unpublished_shape_and_omits_signed_url(prepared, monkeypatch):
    env, declaration, _, path, _ = prepared
    other = deepcopy(declaration)
    other['ready_path'] = str(path.parent / 'other/READY.json')
    other['contract_sha256'] = 'b' * 64
    monkeypatch.setattr(publisher.progressive, 'validate_declarations', lambda *args: {'first': declaration, 'other': other})
    class SyntheticSdkError(Exception):
        pass
    publisher.progressive.publish_ready.side_effect = SyntheticSdkError('https://example.invalid?secret=must-not-persist')
    with pytest.raises(SyntheticSdkError):
        publisher.publish_available(env, {})
    for declaration in (declaration, other):
        failed = runner.read(Path(declaration['ready_path']).with_name('READY.FAILED.json'))
        assert failed['error_type'] == 'SyntheticSdkError' and failed['error'] == 'SyntheticSdkError'
        assert 'must-not-persist' not in runner.canonical(failed).decode()


def test_producer_exit_binds_exact_started_candidate_and_source(prepared):
    _, declaration, value, path, _ = prepared
    attempt = path.parent / 'attempt'; attempt.mkdir()
    driver = path.parent / 'driver.py'; driver.write_text('# synthetic CPU producer\n')
    start = attempt / 'START.json'
    runner.write_once(start, {'status': 'CPU_CACHE_PREPARATION_STARTED', 'output_root': str(attempt),
        'source_manifest': declaration['contract']['producer']['source_manifest'],
        'candidate': value['candidate'], 'driver': runner.artifact(driver)})
    pin = runner.artifact(start)
    assert publisher.producer_outcome(pin, {'one': declaration}) is None
    runner.write_once(attempt / 'EXIT.json', {'status': 'CPU_CACHE_PREPARATION_EXITED',
        'returncode': 1, 'group_cleanup_verified': True})
    result = publisher.producer_outcome(pin, {'one': declaration})
    assert result['successful'] is False and result['evidence'][0] == pin
    declaration['contract']['request']['frames'] = 16
    with pytest.raises(ValueError, match='does not own'):
        publisher.producer_outcome(pin, {'one': declaration})
