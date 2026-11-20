"""Real CPU process-tree probes plus pinned-proof tamper/relocation checks."""
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

from deepprobe_vlm import suite_runner as runner, suite_training_process as training


def invoke(tmp_path, code):
    source = tmp_path.resolve() / 'source'; source.mkdir(exist_ok=True)
    argv = [sys.executable, '-B', '-c', code]
    return training.run_training(argv, cwd=source, env=dict(os.environ),
        logs_prefix=tmp_path.resolve() / 'private-run/logs/training'), source, argv


def repin(path, value):
    path = Path(path); path.write_text(json.dumps(value)); return runner.artifact(path)


def test_real_joined_child_and_leader_produce_quiescent_exit(tmp_path):
    proof, source, argv = invoke(tmp_path,
        'import subprocess,sys; subprocess.run([sys.executable,"-c","print(123)"],check=True)')
    verified = training.verify_training_exit(proof, expected_source_root=source,
                                             expected_argv_sha256=runner.digest(argv))
    assert verified['status'] == training.ACCEPTED
    exited = runner.read(proof['exit']['path'])
    assert exited['returncode'] == 0 and exited['group_cleanup_verified']
    assert exited['error_type'] is None
    assert Path(exited['logs']['stdout']['path']).read_text().strip() == '123'
    assert 'env' not in runner.read(proof['intent']['path'])


def test_real_leader_exit_with_live_child_is_failed_even_after_cleanup(tmp_path):
    marker = tmp_path.resolve() / 'child.pid'
    code = ('import subprocess,sys,time,pathlib; '
            'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); '
            'pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(.15)')
    source = tmp_path.resolve() / 'source'; source.mkdir()
    prefix = tmp_path.resolve() / 'private-run/logs/training'
    with pytest.raises(RuntimeError, match='owned descendants remained'):
        training.run_training([sys.executable, '-B', '-c', code, str(marker)],
                              cwd=source, env=dict(os.environ), logs_prefix=prefix)
    proof = {key: runner.artifact(prefix.with_name(prefix.name + '.' + key + '.json'))
             for key in ('intent', 'exit')}
    exited = runner.read(proof['exit']['path'])
    # A cleaned-up failure is still a failure, even when the leader returned 0.
    assert exited['status'] == training.FAILED
    assert exited['returncode'] == 0 and exited['group_cleanup_verified']
    assert exited['error_type'] == 'RuntimeError'
    with pytest.raises(ValueError, match='not successful and quiescent'):
        training.verify_training_exit(proof, expected_source_root=source)
    from deepprobe_vlm.suite_process import process_table
    child = process_table().get(int(marker.read_text()))
    assert child is None or child['state'] in {'Z', 'X'}


@pytest.mark.parametrize('code', ['raise RuntimeError("synthetic failure")', 'raise SystemExit(7)'])
def test_real_nonzero_training_never_accepts(tmp_path, code):
    with pytest.raises(ValueError, match='exited unsuccessfully'): invoke(tmp_path, code)
    prefix = tmp_path.resolve() / 'private-run/logs/training'
    proof = {key: runner.artifact(prefix.with_name(prefix.name + '.' + key + '.json'))
             for key in ('intent', 'exit')}
    exited = runner.read(proof['exit']['path'])
    assert exited['status'] == training.FAILED and exited['returncode'] != 0
    assert exited['group_cleanup_verified'] and exited['error_type'] is not None
    with pytest.raises(ValueError): training.verify_training_exit(proof, expected_source_root=tmp_path.resolve() / 'source')


def test_spawn_exception_preserves_failed_exit_receipt(tmp_path):
    root = tmp_path.resolve(); prefix = root / 'private-run/training'
    with pytest.raises(FileNotFoundError):
        training.run_training([str(root / 'missing-executable')], cwd=root,
                              env=dict(os.environ), logs_prefix=prefix)
    exited = runner.read(prefix.with_name(prefix.name + '.exit.json'))
    assert exited['status'] == training.FAILED and exited['returncode'] is None
    assert exited['error_type'] == 'FileNotFoundError'


def test_private_loss_allows_only_byte_preserving_receipt_relocation(tmp_path):
    proof, source, argv = invoke(tmp_path, 'pass')
    shared = tmp_path.resolve() / 'shared'; shared.mkdir()
    retained = {}
    for key, pin in proof.items():
        destination = shared / (key + '.json')
        destination.write_bytes(Path(pin['path']).read_bytes()); retained[key] = runner.artifact(destination)
    original_intent_path = runner.read(retained['exit']['path'])['intent']['path']
    shutil.rmtree(tmp_path / 'private-run')
    assert not Path(original_intent_path).exists()
    # The old mount is irrelevant, including if it has since become an alias.
    (tmp_path / 'private-run').symlink_to(shared, target_is_directory=True)
    assert training.verify_training_exit(retained, expected_source_root=source,
        expected_argv_sha256=runner.digest(argv))['status'] == training.ACCEPTED
    changed = runner.read(retained['intent']['path']); changed['extra'] = 'changed bytes'
    retained['intent'] = repin(retained['intent']['path'], changed)
    with pytest.raises(ValueError, match='exact intent bytes'):
        training.verify_training_exit(retained, expected_source_root=source)


@pytest.mark.parametrize('field,value', [('returncode', 1), ('returncode', False),
    ('group_cleanup_verified', False), ('group_cleanup_verified', 1),
    ('error_type', 'RuntimeError'), ('source_root', '/different'),
    ('argv_sha256', 'f' * 64), ('status', training.FAILED)])
def test_repinned_invalid_exit_fields_are_not_accepted(tmp_path, field, value):
    proof, source, _ = invoke(tmp_path, 'pass')
    exited = runner.read(proof['exit']['path']); exited[field] = value
    proof['exit'] = repin(proof['exit']['path'], exited)
    with pytest.raises(ValueError, match='not successful and quiescent'):
        training.verify_training_exit(proof, expected_source_root=source)


def test_swapped_intent_and_wrong_expected_command_are_rejected(tmp_path):
    proof, source, argv = invoke(tmp_path, 'pass')
    with pytest.raises(ValueError, match='expected argv'):
        training.verify_training_exit(proof, expected_source_root=source, expected_argv_sha256='e' * 64)
    altered = runner.read(proof['intent']['path']); altered['argv'][-1] = 'print(1)'
    altered['argv_sha256'] = runner.digest(altered['argv'])
    proof['intent'] = repin(proof['intent']['path'], altered)
    with pytest.raises(ValueError): training.verify_training_exit(proof, expected_source_root=source)


def test_existing_logs_are_never_overwritten_or_reexecuted(tmp_path):
    proof, source, argv = invoke(tmp_path, 'pass')
    original = {name: Path(pin['path']).read_bytes() for name, pin in proof.items()}
    with pytest.raises(ValueError, match='already exist'):
        training.run_training(argv, cwd=source, env=dict(os.environ),
            logs_prefix=tmp_path.resolve() / 'private-run/logs/training')
    assert original == {name: Path(pin['path']).read_bytes() for name, pin in proof.items()}
