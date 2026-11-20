"""CPU routing regressions with synthetic checkpoints, never real training evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

from deepprobe_vlm import experiment_suite
from deepprobe_vlm import suite_actions as actions
from deepprobe_vlm import suite_runner as runner


class PlannedTrainingHistoryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.experiment = Path(__file__).resolve().parents[1]
        self.plan = experiment_suite.default_plan()
        self.task = next(t for t in self.plan['tasks'] if t['kind'] == 'train'
                         and t['stage'] == 'alignment' and t['historical_hint']['status'] == 'planned')
        runs = self.root / 'runs'; runs.mkdir()
        self.old_source = self.root / 'historical-source'; self.old_source.mkdir()
        self.source = self.pin('inputs/source.py', '# synthetic current source\n')
        self.current_frozen = self.pin('inputs/current-frozen.json', {'synthetic': 'current'})
        self.old_frozen = self.pin('inputs/old-frozen.json', {'synthetic': 'historical'})
        input_pin = self.pin('inputs/data.json', {'synthetic': 'training-input'})
        self.environment = {
            'suite_id': 'synthetic-history-routing', 'experiment_root': str(self.experiment),
            'run_root': str(runs), 'python': sys.executable, 'source_files': [self.source],
            'history': {}, 'assets': {'training': {
                'train_manifest': input_pin, 'eval_manifest': input_pin, 'formal_ready': input_pin,
                'frozen_source_manifest': self.current_frozen,
                'data_dir': str(self.root / 'data'), 'media_root': str(self.root / 'media')}},
            'execution_profile': {'schema_version': 2, 'fleet_id': 'synthetic-history-routing',
                'priority': 'low', 'gpu_product': 'NVIDIA-B300',
                'node_selector': {'sslaunch-pool': 'stream', 'aliyun.accelerator/nvidia_name': 'NVIDIA-L20D'},
                'workers': [{'worker_id': 'worker-01', 'job_name': 'jinhaopeng-dpvlm-history-fixture',
                             'namespace': 'default'}]},
        }
        self.worker = actions.Actions(self.plan, self.environment, self.root / 'state')
        self.worker.budget = Mock(return_value={'synthetic': 'capacity'})
        self.worker.training_storage = Mock(return_value={'synthetic': 'measured-peak'})
        self.worker.command = Mock(side_effect=self.finish_synthetic_training)
        self.worker.restore_files = Mock(side_effect=self.restore_fixture)

    def pin(self, relative, value):
        path = self.root / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value))
        return runner.artifact(path)

    def checkpoint(self, root, step):
        names = {'deepprobe_config.json', 'training_provenance.json', 'projector.safetensors',
                 'language_model/config.json', 'vision_model/config.json', 'language_model/model.safetensors',
                 'vision_model/model.safetensors', 'tokenizer/tokenizer_config.json',
                 'image_processor/preprocessor_config.json', 'training_args.bin', 'trainer_state.json',
                 'optimizer.pt', 'scheduler.pt'} | {f'rng_state_{i}.pth' for i in range(8)}
        for name in names:
            path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(json.dumps({'global_step': step, 'max_steps': 718}).encode()
                             if name == 'trainer_state.json' else b'synthetic-checkpoint-only')
        return actions.inventory(root)

    def history(self, mode):
        step = 718 if mode == 'accepted' else 350
        self.old_checkpoint = self.root / 'original-run' / ('final' if mode == 'accepted' else f'checkpoint-{step}')
        files = self.checkpoint(self.old_checkpoint, step)
        config = self.pin('inputs/historical-config.json',
                          runner.read(self.experiment / self.task['config']['path']))
        prefix = 'oss://ss-devbox-stream/synthetic-history/runs/original/training/' + self.old_checkpoint.name + '/'
        value = {'mode': mode, 'checkpoint_step': step, 'source_root': str(self.old_source),
                 'config': config, 'frozen_source_manifest': self.old_frozen,
                 'checkpoint': {'path': str(self.old_checkpoint), 'files': files, 'oss_prefix': prefix}}
        # This is a deliberately synthetic trust-boundary fixture. Real history
        # requires independent original evidence and fresh archive verification.
        receipt = {'schema_version': 1, 'status': 'SUITE_HISTORY_ACCEPTED', 'production_evidence': True,
            'task_id': self.task['id'], 'task_sha256': runner.digest(self.task), 'mode': mode,
            'evidence': [self.source], 'config_sha256': config['sha256'],
            'frozen_source_manifest_sha256': self.old_frozen['sha256'],
            'checkpoint_inventory_sha256': runner.digest(files), 'checkpoint_oss_prefix': prefix,
            'optimizer_step': step, 'source_root': str(self.old_source), 'archive_verified': True,
            'formal_training_complete' if mode == 'accepted' else 'resume_metadata_verified': True}
        value['acceptance'] = self.pin('inputs/history-acceptance.json', receipt)
        self.environment['history'][self.task['id']] = value
        return value

    def restore_fixture(self, prefix, target, files, state_dir):
        self.assertEqual(prefix, self.environment['history'][self.task['id']]['checkpoint']['oss_prefix'])
        self.worker.copy_files(self.old_checkpoint, target, files)

    def finish_synthetic_training(self, argv, directory, label, *, source_root):
        self.assertEqual(label, 'training')
        output = Path(argv[argv.index('--output-dir') + 1]); output.mkdir(parents=True, exist_ok=True)
        self.checkpoint(output / 'final', 718)
        self.checkpoint(output / 'checkpoint-718', 718)
        for name, value in {'trainer_state.json': {'global_step': 718}, 'run_manifest.json': {'synthetic': True},
                            'train_results.json': {'train_loss': 0.0}, 'eval_results.json': {'eval_loss': 0.0}}.items():
            (output / name).write_text(json.dumps(value))
        training = self.environment['assets']['training']
        receipt = {'status': 'formal_training_complete', 'stage': self.task['stage'], 'world_size': 8,
            'optimizer_step': 718, 'train_manifest_sha256': training['train_manifest']['sha256'],
            'eval_manifest_sha256': training['eval_manifest']['sha256'],
            'formal_ready_sha256': training['formal_ready']['sha256'],
            'frozen_source_manifest_sha256': argv[argv.index('--frozen-source-manifest-sha256') + 1],
            'media_before': {'synthetic': True}, 'media_after': {'synthetic': True},
            'trainer_state_sha256': runner.sha(output / 'trainer_state.json'),
            'run_manifest_sha256': runner.sha(output / 'run_manifest.json')}
        (output / 'FORMAL_TRAIN_COMPLETE.json').write_text(json.dumps(receipt))

    def test_planned_accepted_history_never_calls_trainer_or_training_estimator(self):
        history = self.history('accepted')
        self.worker.command.side_effect = AssertionError('accepted history must not train')
        self.worker.training_storage.side_effect = AssertionError('accepted history must not reserve fresh training')
        result = self.worker.train(self.task)
        self.worker.command.assert_not_called()
        self.worker.training_storage.assert_not_called()
        self.worker.restore_files.assert_called_once()
        self.assertEqual(self.task['historical_hint']['status'], 'planned')
        self.assertEqual(actions.inventory(Path(result['checkpoint'])), history['checkpoint']['files'])
        self.assertEqual(actions.inventory(self.old_checkpoint), history['checkpoint']['files'])
        self.assertEqual(runner.read(result['training_complete']['path'])['outputs'], [history['acceptance']])

    def test_planned_resume_preserves_audited_config_source_and_exact_checkpoint(self):
        history = self.history('resume')
        result = self.worker.train(self.task)
        self.worker.command.assert_called_once()
        argv = self.worker.command.call_args.args[0]
        self.assertEqual(argv[argv.index('--config') + 1], history['config']['path'])
        self.assertEqual(self.worker.command.call_args.kwargs['source_root'], self.old_source)
        self.assertEqual(argv[argv.index('--frozen-source-manifest') + 1], self.old_frozen['path'])
        self.assertEqual(argv[argv.index('--frozen-source-manifest-sha256') + 1], self.old_frozen['sha256'])
        expected = Path(result['checkpoint']).parent / 'checkpoint-350'
        self.assertEqual(argv[argv.index('--resume') + 1], str(expected))
        self.assertEqual(actions.inventory(expected), history['checkpoint']['files'])
        self.assertNotIn('--init-checkpoint', argv)
        self.assertEqual(actions.inventory(self.old_checkpoint), history['checkpoint']['files'])

    def test_planned_without_history_still_runs_fresh_training(self):
        self.worker.train(self.task)
        self.worker.command.assert_called_once()
        self.worker.restore_files.assert_not_called()
        argv = self.worker.command.call_args.args[0]
        self.assertEqual(argv[argv.index('--config') + 1], str(self.experiment / self.task['config']['path']))
        self.assertEqual(self.worker.command.call_args.kwargs['source_root'], self.experiment)
        self.assertEqual(argv[argv.index('--frozen-source-manifest') + 1], self.current_frozen['path'])
        self.assertNotIn('--resume', argv)

    def test_explicit_invalid_history_cannot_fall_back_to_fresh_training(self):
        self.environment['history'][self.task['id']] = {}
        with self.assertRaisesRegex(ValueError, 'audited binding'):
            self.worker.train(self.task)
        self.worker.command.assert_not_called()

    def test_original_historical_hint_still_requires_an_explicit_binding(self):
        task = next(t for t in self.plan['tasks'] if t['kind'] == 'train'
                    and t['historical_hint']['status'] == 'reuse_candidate')
        with self.assertRaisesRegex(ValueError, 'audited binding'):
            self.worker.train(task)
        self.worker.command.assert_not_called()

    def test_planned_accepted_sft_still_rejects_wrong_alignment_initializer(self):
        self.task = next(t for t in self.plan['tasks'] if t['kind'] == 'train' and t['stage'] == 'sft'
                         and t['historical_hint']['status'] == 'planned'
                         and t['initializer'].get('checkpoint_task_id'))
        self.history('accepted')
        self.worker.completed = Mock(return_value={'checkpoint_files': [{'path': 'synthetic-parent', 'sha256': 'a' * 64}]})
        with self.assertRaisesRegex(ValueError, 'matching alignment initializer'):
            self.worker.train(self.task)
        self.worker.command.assert_not_called()


if __name__ == '__main__':
    unittest.main()
