"""Local routing tests; guard trust checks are covered by its dedicated suite."""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from deepprobe_vlm import suite_actions as actions, suite_checkpoint_export as exporter
from deepprobe_vlm import suite_checkpoint_resume as recovery, suite_runner as runner
from deepprobe_vlm import suite_training_process as training_process
import test_suite_actions_history as history_fixtures


@pytest.fixture
def bridge(monkeypatch):
    case = history_fixtures.PlannedTrainingHistoryTest(); case.setUp()
    try:
        case.task = next(t for t in case.plan['tasks'] if t['id'] == 'train__ouro_r4_s17__sft')
        history = case.history('resume')
        previous = case.old_checkpoint
        case.old_checkpoint = previous.with_name('checkpoint-600')
        previous.rename(case.old_checkpoint)
        state = case.old_checkpoint / 'trainer_state.json'
        state.write_text(json.dumps({'global_step': 600, 'max_steps': 718}))
        history['checkpoint_step'] = 600
        history['checkpoint'].update(path=str(case.old_checkpoint), files=actions.inventory(case.old_checkpoint),
            oss_prefix=history['checkpoint']['oss_prefix'].replace('checkpoint-350', 'checkpoint-600'))
        acceptance = runner.read(history['acceptance']['path'])
        acceptance.update(optimizer_step=600, checkpoint_inventory_sha256=runner.digest(history['checkpoint']['files']),
                          checkpoint_oss_prefix=history['checkpoint']['oss_prefix'])
        history['acceptance'] = case.pin('inputs/bridge-resume-acceptance.json', acceptance)
        case.environment['legacy_private_resume'] = case.pin('inputs/independent-bridge-review.json', {'synthetic': True})
        case.environment.update(archive_root='oss://ss-devbox-stream/synthetic-bridge', tools={})
        case.worker.private_scratch = True
        case.worker.consume = Mock(return_value=case.pin('outputs/initializer-consumer.json', {'synthetic': True}))
        proof = {'status': 'LEGACY_PRIVATE_RESUME_ELIGIBLE', 'eligible': True,
                 'legacy_checkpoint_export_hook_present': False, 'interrupted_recovery_checkpoint_step': 600}
        monkeypatch.setattr(recovery, 'verify_private_resume', Mock(return_value=proof))
        original_command = case.finish_synthetic_training
        def complete_with_exit(argv, directory, label, *, source_root):
            original_command(argv, directory, label, source_root=source_root)
            intent = case.pin('outputs/real-intent-shape.json', {
                'schema_version': 1, 'status': 'SUITE_TRAINING_PROCESS_INTENT', 'argv': list(map(str, argv)),
                'argv_sha256': runner.digest(list(map(str, argv))), 'source_root': str(source_root), 'automatic_retry': False})
            exited = case.pin('outputs/real-exit-shape.json', {'schema_version': 1,
                'status': training_process.ACCEPTED, 'intent': intent,
                'argv_sha256': runner.digest(list(map(str, argv))), 'source_root': str(source_root),
                'returncode': 0, 'group_cleanup_verified': True, 'error_type': None, 'automatic_retry': False})
            return {'exit': exited, 'intent': intent}
        case.finish_synthetic_training = complete_with_exit
        case.worker.command = Mock(side_effect=complete_with_exit)
        yield case, history
    finally:
        case.doCleanups()


def test_unchanged_legacy_source_resumes_600_and_exports_only_after_complete_exit(bridge, monkeypatch):
    case, history = bridge
    original = actions.inventory(case.old_checkpoint)
    exported = case.pin('outputs/export-native718.json', {'status': 'SYNTHETIC_EXPORT_ONLY'})
    events = []
    def command(argv, directory, label, *, source_root, **kwargs):
        assert kwargs == {}  # No export environment injected into the old trainer.
        assert source_root == case.old_source
        assert argv[argv.index('--frozen-source-manifest-sha256') + 1] == case.old_frozen['sha256']
        assert Path(argv[argv.index('--resume') + 1]).name == 'checkpoint-600'
        events.append('legacy-native-training')
        return case.finish_synthetic_training(argv, directory, label, source_root=source_root)
    def export(config, checkpoint, step, *, post_training_exit):
        assert events == ['legacy-native-training'] and step == 718
        assert checkpoint.name == 'checkpoint-718'
        assert (checkpoint.parent / 'FORMAL_TRAIN_COMPLETE.json').is_file()
        assert config['frozen_source_manifest'] == case.old_frozen
        assert runner.read(post_training_exit['path'])['native_checkpoint'] == str(checkpoint)
        events.append('post-exit-native-export')
        return exported
    case.worker.command = Mock(side_effect=command)
    monkeypatch.setattr(exporter, 'export_checkpoint', Mock(side_effect=export))
    result = case.worker.train(case.task)
    assert events == ['legacy-native-training', 'post-exit-native-export']
    complete = runner.read(result['training_complete']['path'])
    assert exported in complete['outputs']
    assert actions.inventory(case.old_checkpoint) == original
    contract = runner.read(case.worker.task_dir(case.task['id']) / 'LEGACY_PRIVATE_RESUME.json')
    assert contract['durable_recovery_step_until_exit'] == 600
    assert contract['in_training_export_hook_enabled'] is False


def test_legacy_failure_never_exports_or_accepts_training(bridge, monkeypatch):
    case, _ = bridge
    case.worker.command = Mock(side_effect=ValueError('synthetic training failed before 718'))
    export = Mock(); monkeypatch.setattr(exporter, 'export_checkpoint', export)
    with pytest.raises(ValueError, match='before 718'):
        case.worker.train(case.task)
    export.assert_not_called()
    assert not (case.worker.task_dir(case.task['id']) / 'TRAINING_COMPLETE.json').exists()


def test_postexit_export_failure_prevents_acceptance_and_preserves_native_save(bridge, monkeypatch):
    case, _ = bridge
    monkeypatch.setattr(exporter, 'export_checkpoint', Mock(side_effect=ValueError('synthetic readback failed')))
    with pytest.raises(ValueError, match='readback failed'):
        case.worker.train(case.task)
    assert not (case.worker.task_dir(case.task['id']) / 'TRAINING_COMPLETE.json').exists()
    owner = case.worker.owner(case.task['id'])
    assert (Path(owner['run_root']) / 'training' / 'checkpoint-718').is_dir()


def test_existing_formal_marker_without_retained_quiescence_cannot_export(bridge, monkeypatch):
    case, _ = bridge
    export = Mock(side_effect=ValueError('first synthetic export interruption'))
    monkeypatch.setattr(exporter, 'export_checkpoint', export)
    with pytest.raises(ValueError, match='export interruption'):
        case.worker.train(case.task)
    (case.worker.task_dir(case.task['id']) / 'NATIVE_TRAINING_EXIT.json').unlink()
    case.worker.command = Mock(side_effect=AssertionError('completed marker must not train again'))
    export.reset_mock()
    with pytest.raises(FileNotFoundError):
        case.worker.train(case.task)
    export.assert_not_called()


def test_legacy_bridge_rejects_a_different_owned_resume_step(bridge, monkeypatch):
    case, _ = bridge
    case.worker.resume = True
    root = Path(case.worker.owner(case.task['id'])['run_root']) / 'training' / 'checkpoint-700'
    case.checkpoint(root, 700)
    export = Mock(); monkeypatch.setattr(exporter, 'export_checkpoint', export)
    with pytest.raises(ValueError, match='only the reviewed native checkpoint 600'):
        case.worker.train(case.task)
    case.worker.command.assert_not_called()
    export.assert_not_called()
