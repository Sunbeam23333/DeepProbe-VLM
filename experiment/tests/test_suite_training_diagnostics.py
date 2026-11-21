"""Real failing CPU children and bounded, token-accounted diagnostic snapshots."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

from deepprobe_vlm import suite_actions, suite_runner as runner
from deepprobe_vlm import suite_training_diagnostics as diagnostics
from deepprobe_vlm import suite_training_process as training
from deepprobe_vlm.suite_metadata_limit import check_metadata
from test_suite_metadata_limit import metadata


@pytest.fixture
def action(metadata, tmp_path, monkeypatch):
    state, task, ledger, runtime = metadata
    private = tmp_path.resolve() / 'private-run'; private.mkdir()
    source = tmp_path.resolve() / 'scientific-source'; source.mkdir()
    for key in ('TMPDIR', 'CUDA_CACHE_PATH', 'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR',
                'TORCHINDUCTOR_CACHE_DIR'):
        monkeypatch.setenv(key, str(private))
    environment = {'experiment_root': str(source), 'run_root': str(private),
                   'source_files': [], 'runtime_environment': {}}
    value = suite_actions.Actions({'tasks': [{'id': task, 'kind': 'train'}]}, environment, state)
    value.private_scratch = True
    value.owner = lambda _task: {'run_root': str(private)}
    return value, private, state / 'artifacts' / task, state / 'durable-metadata' / task, ledger, runtime


def report_at(destination):
    paths = list(destination.glob('training-failure-*/DIAGNOSTICS.json'))
    assert len(paths) == 1
    return runner.read(paths[0]), paths[0]


def test_real_exit7_diagnostics_survive_private_loss_without_becoming_success(action):
    value, private, output, destination, ledger, runtime = action
    original_ledger = (ledger / 'LEDGER.json').read_bytes()
    with pytest.raises(ValueError, match='Training command exited unsuccessfully'):
        value.command([sys.executable, '-B', '-c',
            'import sys; print("before failure"); print("actual stderr",file=sys.stderr); sys.exit(7)'], output, 'training')
    report, path = report_at(destination)
    assert report['training_success'] is report['process_quiescence_claimed'] is False
    assert report['original_error_type'] == 'ValueError' and report['automatic_retry'] is False
    for name, row in report['files'].items():
        original = Path(row['source_path']).read_bytes()
        retained = Path(row['retained']['path']).read_bytes()
        assert original == retained
        assert row['complete_source_file'] is True and row['source_stable_during_capture'] is True
        assert row['source_sha256'] == hashlib.sha256(original).hexdigest() == row['retained']['sha256']
        assert row['range_start_inclusive'] == 0 and row['range_end_exclusive'] == len(original)
    assert (path.parent / 'stderr').read_text() == 'actual stderr\n'
    exited = runner.read(path.parent / 'exit.json')
    assert exited['status'] == training.FAILED and exited['returncode'] == 7
    intent_bytes = (path.parent / 'intent.json').read_bytes()
    assert exited['intent']['sha256'] == hashlib.sha256(intent_bytes).hexdigest()
    retained_proof = {key: runner.artifact(path.parent / (key + '.json')) for key in ('intent', 'exit')}
    shutil.rmtree(private)
    with pytest.raises(ValueError, match='not successful and quiescent'):
        training.verify_training_exit(retained_proof, expected_source_root=value.root)
    assert runner.read(path)['files'] == report['files']
    assert (ledger / 'LEDGER.json').read_bytes() == original_ledger
    assert check_metadata(value.state, runtime['DP_SUITE_TASK_ID'], environ=runtime)['observed_bytes'] > report['retained_payload_bytes']


def test_missing_executable_keeps_empty_streams_and_failed_exit(action):
    value, private, output, destination, _, _ = action
    with pytest.raises(FileNotFoundError):
        value.command([str(private / 'not-an-executable')], output, 'training')
    report, path = report_at(destination)
    for name in ('stdout', 'stderr'):
        row = report['files'][name]
        assert row['source_bytes'] == row['retained']['bytes'] == 0
        assert row['source_sha256'] == hashlib.sha256(b'').hexdigest()
        assert (path.parent / name).read_bytes() == b''
    assert runner.read(path.parent / 'exit.json')['returncode'] is None


def test_large_logs_are_fixed_ranges_not_full_source_hashes(action):
    value, private, _, destination, ledger, _ = action
    config = runner.read(ledger / 'LEDGER.json')
    config['reservations']['home-token']['home_bytes'] = 4 * 1024 * 1024
    (ledger / 'LEDGER.json').write_text(json.dumps(config))
    prefix = private / '.suite-logs/training-fixture'; prefix.parent.mkdir()
    original = b'beginning-' + b'X' * (diagnostics.LOG_TAIL_BYTES * 3) + b'-actual-tail'
    for name in ('stdout', 'stderr'):
        prefix.with_name(prefix.name + '.' + name).write_bytes(original)
    pin = diagnostics.retain_failure(prefix, private_run_root=private, destination=destination, error_type='RuntimeError')
    report = runner.read(pin['path'])
    assert report['retained_payload_bytes'] == 2 * diagnostics.LOG_TAIL_BYTES
    assert report['files']['intent.json']['status'] == report['files']['exit.json']['status'] == 'MISSING'
    for name in ('stdout', 'stderr'):
        row = report['files'][name]
        assert row['complete_source_file'] is False and row['truncated'] is True
        assert 'source_sha256' not in row and 'source_bytes' not in row
        assert row['source_stat_before']['size'] == len(original)
        assert row['range_start_inclusive'] == len(original) - diagnostics.LOG_TAIL_BYTES
        assert row['range_end_exclusive'] == len(original)
        tail = Path(row['retained']['path']).read_bytes()
        assert tail == original[-diagnostics.LOG_TAIL_BYTES:]
        assert row['captured_sha256'] == hashlib.sha256(tail).hexdigest()
    before = Path(pin['path']).read_bytes()
    with pytest.raises(ValueError, match='already retained'):
        diagnostics.retain_failure(prefix, private_run_root=private, destination=destination, error_type='RuntimeError')
    assert Path(pin['path']).read_bytes() == before


def test_diagnostic_budget_rejection_never_hides_original_training_error(action):
    value, _, output, destination, ledger, _ = action
    config = runner.read(ledger / 'LEDGER.json')
    config['reservations']['home-token']['home_bytes'] = 1
    (ledger / 'LEDGER.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='Training command exited unsuccessfully') as exc:
        value.command([sys.executable, '-B', '-c', 'raise SystemExit(7)'], output, 'training')
    assert not destination.exists()
    assert any('diagnostics could not be retained' in note for note in getattr(exc.value, '__notes__', [])) or sys.version_info < (3, 11)


def test_private_exit_receipt_failure_keeps_original_error_and_other_diagnostics(action, monkeypatch):
    value, _, output, destination, _, _ = action
    def broken_log(_path):
        raise OSError('synthetic exit metadata failure')
    monkeypatch.setattr(training, '_failure_log', broken_log)
    with pytest.raises(ValueError, match='Training command exited unsuccessfully') as exc:
        value.command([sys.executable, '-B', '-c', 'print("diagnostic retained"); raise SystemExit(7)'], output, 'training')
    report, path = report_at(destination)
    assert report['files']['exit.json']['status'] == 'MISSING'
    assert (path.parent / 'stdout').read_text() == 'diagnostic retained\n'
    assert any('exit evidence could not be retained' in note for note in getattr(exc.value, '__notes__', [])) or sys.version_info < (3, 11)


@pytest.mark.parametrize('error_kind', ['interrupt', 'nonquiescent', 'no_add_note', 'broken_add_note'])
def test_interrupt_or_note_failure_never_changes_original_exception(action, monkeypatch, error_kind):
    value, _, output, destination, _, _ = action
    class OldPythonError(RuntimeError):
        add_note = None
    class BrokenNoteError(RuntimeError):
        def add_note(self, _message):
            raise OSError('note failure')
    error = {'interrupt': KeyboardInterrupt(), 'nonquiescent': RuntimeError('not quiescent'),
             'no_add_note': OldPythonError('original'), 'broken_add_note': BrokenNoteError('original')}[error_kind]
    def fail(*_a, **_kw):
        raise error
    monkeypatch.setattr(training.suite_process, 'run_group', fail)
    if error_kind in {'no_add_note', 'broken_add_note'}:
        monkeypatch.setattr(diagnostics, 'retain_failure', lambda *_a, **_k: (_ for _ in ()).throw(OSError('copy failure')))
        monkeypatch.setattr(training, '_failure_log', lambda *_a: (_ for _ in ()).throw(OSError('receipt failure')))
    with pytest.raises(type(error)) as exc:
        value.command([sys.executable, '-B', '-c', 'pass'], output, 'training')
    assert exc.value is error
    if error_kind in {'interrupt', 'nonquiescent'}:
        _, path = report_at(destination)
        exited = runner.read(path.parent / 'exit.json')
        assert exited['status'] == training.FAILED and exited['group_cleanup_verified'] is False


def test_changing_source_cannot_claim_a_complete_source_hash(tmp_path, monkeypatch):
    source = tmp_path.resolve() / 'stderr'; source.write_bytes(b'initial')
    actual = diagnostics.os.fstat
    count = 0
    def changed(fd):
        nonlocal count
        count += 1
        if count == 2:
            with source.open('ab') as stream:
                stream.write(b'-later')
        return actual(fd)
    monkeypatch.setattr(diagnostics.os, 'fstat', changed)
    row, payload = diagnostics._capture(source, diagnostics.LOG_TAIL_BYTES)
    assert payload == b'initial'
    assert row['source_stable_during_capture'] is row['complete_source_file'] is False
    assert 'source_sha256' not in row
    assert row['range_start_inclusive'] == 0 and row['range_end_exclusive'] == len(payload)


def test_failure_exit_does_not_hash_whole_logs_before_bounded_capture(action, monkeypatch):
    value, _, output, destination, _, _ = action
    def forbidden_whole_hash(_path):
        raise AssertionError('Failure path must not hash an unbounded source log')
    monkeypatch.setattr(training, '_log', forbidden_whole_hash)
    with pytest.raises(ValueError, match='Training command exited unsuccessfully'):
        value.command([sys.executable, '-B', '-c', 'print("actual log"); raise SystemExit(7)'], output, 'training')
    _, path = report_at(destination)
    exited = runner.read(path.parent / 'exit.json')
    assert exited['logs']['stdout']['observed_bytes'] == len(b'actual log\n')
    assert exited['logs']['stdout']['full_source_sha256_recorded'] is False
    assert 'sha256' not in exited['logs']['stdout']


def test_fifo_and_symlink_sources_are_rejected_without_blocking(tmp_path):
    root = tmp_path.resolve()
    fifo = root / 'fifo'; os.mkfifo(fifo)
    with pytest.raises(ValueError, match='regular file'):
        diagnostics._capture(fifo, diagnostics.LOG_TAIL_BYTES)
    ordinary = root / 'ordinary'; ordinary.write_bytes(b'not a log')
    alias = root / 'alias'; alias.symlink_to(ordinary)
    with pytest.raises(ValueError, match='symlink'):
        diagnostics._capture(alias, diagnostics.LOG_TAIL_BYTES)


def test_private_inductor_cache_overrides_shared_environment(action):
    value, private, output, _, _, _ = action
    value.env['runtime_environment']['TORCHINDUCTOR_CACHE_DIR'] = '/unwritable/shared/cache'
    proof = value.command([sys.executable, '-c',
        'import os; from pathlib import Path; '
        'p=Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])/"inductor-write-probe"; '
        'p.write_text("cache-writable")'], output, 'training')
    assert (private / 'inductor-write-probe').read_text() == 'cache-writable'
    assert runner.read(proof['exit']['path'])['returncode'] == 0
