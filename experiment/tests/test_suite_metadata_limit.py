"""Synthetic local metadata accounting; no real home ledger or training."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from deepprobe_vlm import suite_runner as runner
from deepprobe_vlm.suite_actions import save
from deepprobe_vlm.suite_actions import Actions
from deepprobe_vlm.suite_durable_metadata import retain_pin
from deepprobe_vlm.suite_metadata_limit import check_metadata, metadata_write, CONTROL_HEADROOM_BYTES


@pytest.fixture
def metadata(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    state, ledger = root / 'state', root / 'ledger'
    task = 'train__synthetic__sft'
    attempt = state / 'tasks' / task / 'attempts' / 'attempt-000001'
    attempt.mkdir(parents=True); ledger.mkdir()
    owner = {'fleet_id': 'synthetic', 'worker_id': 'worker-01', 'task_id': task,
             'claim_id': 'synthetic-claim', 'job_uid': 'synthetic-job'}
    claim = {'owner': owner, 'attempt': str(attempt), 'storage_token': {'home': 'home-token', 'scratch': 'private-token'}}
    runner.write_once(attempt.parents[1] / 'CLAIM.json', claim)
    runner.write_once(attempt / 'TASK_INTENT.json', {'task_id': task, 'kind': 'train'})
    value = {'schema_version': 2, 'reservations': {'home-token': {
        'owner': owner, 'home_bytes': CONTROL_HEADROOM_BYTES + 200_000, 'state': 'ACTIVE'}}}
    runner.write_once(ledger / 'LEDGER.json', value)
    runtime = {'DP_SUITE_FLEET_STATE': str(state), 'DP_SUITE_TASK_ID': task,
               'DP_SUITE_STORAGE_LEDGER': str(ledger), 'DP_SUITE_STORAGE_TOKEN': 'home-token',
               'DP_SUITE_SCRATCH_TOKEN': 'private-token', 'DP_SUITE_WORKER_ID': 'worker-01'}
    for key, value in runtime.items():
        monkeypatch.setenv(key, value)
    return state, task, ledger, runtime


def test_all_task_receipts_and_previous_savepoints_share_one_existing_token(metadata):
    state, task, ledger, runtime = metadata
    first = save(state / 'checkpoint-exports' / task / 'checkpoint-100' / 'META.json', {'synthetic': 'a' * 50_000})
    save(state / 'artifacts' / task / 'TRAIN.json', {'synthetic': 'b' * 50_000})
    proof = check_metadata(state, task, environ=runtime)
    assert proof['observed_bytes'] > 100_000 and proof['lifetime_quota_claimed'] is False
    assert runner.verify_pin(first)
    target = state / 'checkpoint-exports' / task / 'checkpoint-200' / 'META.json'
    with pytest.raises(ValueError, match='exceeds its held token'):
        save(target, {'synthetic': 'c' * 100_000})
    assert not target.exists()
    assert len(runner.read(ledger / 'LEDGER.json')['reservations']) == 1


def test_copy_reserves_pending_publication_and_provenance_and_counts_retained_bytes(metadata, tmp_path):
    state, task, _, runtime = metadata
    private = tmp_path.resolve() / 'private'; private.mkdir()
    source = private / 'TRAIN_COMPLETE.json'; source.write_bytes(b's' * 80_000)
    kept = retain_pin(runner.artifact(source), private, state / 'durable-metadata' / task)
    source.unlink()
    assert runner.verify_pin(kept)
    assert check_metadata(state, task, environ=runtime)['observed_bytes'] >= 80_000
    oversized = private / 'OTHER.json'; oversized.write_bytes(b's' * 60_000)
    with pytest.raises(ValueError, match='exceeds its held token'):
        retain_pin(runner.artifact(oversized), private, state / 'durable-metadata' / task)
    assert len(list((state / 'durable-metadata' / task).glob('*.SOURCE.json'))) == 1


@pytest.mark.parametrize('change', ['released', 'owner', 'token', 'worker', 'state'])
def test_wrong_or_released_token_fails_closed(metadata, change):
    state, task, ledger, runtime = metadata
    runtime = deepcopy(runtime)
    document = runner.read(ledger / 'LEDGER.json')
    row = document['reservations']['home-token']
    if change == 'released':
        row['state'] = 'RELEASED'
    elif change == 'owner':
        row['owner']['job_uid'] = 'another-job'
    elif change == 'token':
        runtime['DP_SUITE_STORAGE_TOKEN'] = 'another-token'
    elif change == 'worker':
        runtime['DP_SUITE_WORKER_ID'] = 'worker-02'
    else:
        runtime['DP_SUITE_FLEET_STATE'] = str(state.parent / 'another-state')
    (ledger / 'LEDGER.json').write_text(json.dumps(document))
    with pytest.raises(ValueError):
        check_metadata(state, task, environ=runtime)


def test_unexpected_postwrite_growth_is_not_accepted(metadata):
    state, task, _, _ = metadata
    path = state / 'artifacts' / task / 'UNEXPECTED.bin'; path.parent.mkdir(parents=True)
    with pytest.raises(ValueError, match='exceeds its held token'):
        with metadata_write(path, 10):
            path.write_bytes(b'x' * 250_000)
    assert path.exists()  # Preserve failed artifacts; never silently delete them.


def test_old_finished_tasks_are_actual_inventory_not_current_task_token(metadata):
    state, task, _, runtime = metadata
    old = state / 'artifacts' / 'train__old__sft' / 'OLD.json'; old.parent.mkdir(parents=True)
    old.write_bytes(b'x' * 500_000)
    proof = check_metadata(state, task, environ=runtime)
    assert proof['observed_bytes'] < 500_000
    assert old.exists()


def test_legacy_save_without_private_token_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv('DP_SUITE_SCRATCH_TOKEN', raising=False)
    result = save(tmp_path.resolve() / 'LEGACY.json', {'legacy': True})
    assert runner.read(result['path']) == {'legacy': True}


def test_native_progress_logs_stay_in_owned_private_run(metadata, tmp_path, monkeypatch):
    state, task, _, _ = metadata
    private = tmp_path.resolve() / 'private-run'; private.mkdir()
    for key in ('TMPDIR', 'CUDA_CACHE_PATH', 'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR',
                'TORCHINDUCTOR_CACHE_DIR'):
        monkeypatch.setenv(key, str(private))
    environment = {'experiment_root': str(tmp_path.resolve()), 'run_root': str(private),
                   'source_files': [], 'runtime_environment': {}}
    actions = Actions({'tasks': [{'id': task, 'kind': 'train'}]}, environment, state)
    actions.private_scratch = True
    actions.owner = lambda task_id: {'run_root': str(private)}
    shared = state / 'artifacts' / task
    actions.command([sys.executable, '-c', 'print("synthetic-native-log")'], shared, 'training')
    stdout = list((private / '.suite-logs').glob('training-*.stdout'))
    assert len(stdout) == 1 and stdout[0].read_text() == 'synthetic-native-log\n'
    assert not list(shared.glob('training-*'))
