"""Offline derivation tests; synthetic snapshots do not certify real assets."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

from deepprobe_vlm import experiment_suite, suite_runner as runner

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/operations/derive_suite_cache_continuation.py'
SPEC = importlib.util.spec_from_file_location('suite_cache_continuation', SCRIPT)
continuation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(continuation)


class CacheContinuationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        task = next(t for t in experiment_suite.default_plan()['tasks']
                    if t['id'] == 'train__smollm2_s29__alignment')
        self.tid = task['id']
        base = self.root / 'old-preparation'
        pin = {'path': str(self.root / 'SYNTHETIC_ONLY.json'), 'bytes': 1, 'sha256': '1' * 64}
        names = {'deepprobe_config.json', 'training_provenance.json', 'projector.safetensors',
                 'language_model/config.json', 'vision_model/config.json', 'language_model/model.safetensors',
                 'vision_model/model.safetensors', 'tokenizer/tokenizer_config.json',
                 'image_processor/preprocessor_config.json', 'trainer_state.json', 'optimizer.pt',
                 'scheduler.pt', 'training_args.bin'} | {f'rng_state_{i}.pth' for i in range(8)}
        checkpoint = {'path': str(self.root / 'old-run/checkpoint-350'),
                      'oss_prefix': 'oss://ss-devbox-stream/project/runs/synthetic/training/checkpoint-350/',
                      'files': [{'path': name, 'bytes': 1, 'sha256': '1' * 64} for name in sorted(names)]}
        request = {'task_id': self.tid, 'output_dir': str(base), 'operation': 'archive_local',
                   'entry': pin, 'review': pin}
        history = {'mode': 'resume', 'checkpoint_step': 350, 'source_root': str(self.root / 'historical-code'),
                   'config': pin, 'frozen_source_manifest': pin, 'checkpoint': checkpoint}
        documents = {'original_request': request,
            'archive_acceptance': {'status': 'SUITE_RECOVERY_ARCHIVE_VERIFIED', 'production_evidence': True,
                'all_objects_full_get_sha256_verified': True,
                'checkpoint_inventory_sha256': runner.digest(checkpoint['files']),
                'checkpoint_oss_prefix': checkpoint['oss_prefix'],
                'objects': [{'uri': checkpoint['oss_prefix'] + row['path'], 'bytes': row['bytes'],
                             'sha256': row['sha256']} for row in checkpoint['files']]},
            'native_metadata': {'status': 'SUITE_NATIVE_METADATA_VERIFIED',
                'checkpoint_inventory_sha256': runner.digest(checkpoint['files']),
                'optimizer_scheduler_rng_deserialized': False, 'runtime_resume_verified': False},
            'history_acceptance': {'scope': 'SYNTHETIC_ONLY'}}
        receipts, captured = {}, {}
        def capture(key, document):
            name = continuation.RECEIPT_NAMES[key]
            raw = runner.canonical(document) + b'\n'
            artifact = {'path': str(base / name), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
            receipts[key] = artifact; captured[name] = {**artifact, 'content': deepcopy(document)}
        for key, document in documents.items():
            capture(key, document)
        history.update(acceptance=receipts['history_acceptance'], archive_acceptance=receipts['archive_acceptance'])
        capture('prepared', {'status': 'SUITE_RECOVERY_PREPARED', 'history': history,
            'checkpoint_step': 350, 'remaining_optimizer_updates': task['expected_optimizer_steps'] - 350,
            'gpu_submitted': False, 'runtime_resume_verified': False})
        receipts.update(history_eligible=True, scientific_recipe_equivalent=True,
            archive_full_get_is_existing_measurement_not_this_review=True, checkpoint_files=len(checkpoint['files']),
            checkpoint_bytes=len(checkpoint['files']), checkpoint_step=350, mode='resume')
        self.candidate = {'schema_version': 1, 'environment': {
            'experiment_root': str(self.root / 'source-v1/experiment'), 'source_files': [], 'history': {},
            'runtime_environment': {}, 'execution_profile': {'priority': 'normal'}},
            'state_root': str(self.root / 'state'), 'recovery_preparations': [request], 'history_preparations': [],
            'history_imports': {'evaluate__synthetic': {'entry': pin, 'review': pin}},
            'benchmark_preparations': [{'output_dir': str(self.root / 'cache-output'),
                                      'media_root': str(self.root / 'cache-media'), 'frames': 4}],
            'input_indexes': {'recovery_preparations': pin, 'history_imports': pin},
            'job_template_document': {'synthetic': True}, 'job_template_source': pin}
        self.candidate_pin = {'path': str(self.root / 'copied-candidate.json'), 'bytes': 123, 'sha256': 'c' * 64}
        remote_pin = {**self.candidate_pin, 'path': str(self.root / 'old-control/CANDIDATE.json')}
        self.verified = {'candidate': remote_pin, 'status': 'COMPLETED_HISTORY_PREPARATIONS_REUSE_VERIFIED',
            'new_checkpoint_payload_gets': 0, 'remote_mutations': False, 'observed_utc': 'SYNTHETIC_ONLY',
            'history': {self.tid: history}, 'receipts': {self.tid: receipts}}
        self.snapshot = {'candidate': {**remote_pin, 'content': deepcopy(self.candidate)},
                         'prepared': {self.tid: captured}}
        self.kwargs = {'experiment_root': self.root / 'source-v2/experiment',
                       'workflow_output': self.root / 'new-cache-control'}

    def derive(self):
        return continuation.derive(self.candidate, self.candidate_pin, self.verified, self.snapshot, **self.kwargs)

    def test_exact_histories_and_eval_cache_inputs_preserved_with_no_launch_template(self):
        original = deepcopy(self.candidate)
        value, receipt = self.derive()
        self.assertEqual(value['environment']['history'], self.verified['history'])
        self.assertEqual(value['history_imports'], original['history_imports'])
        self.assertEqual(value['benchmark_preparations'], original['benchmark_preparations'])
        self.assertEqual(value['environment']['execution_profile']['priority'], 'normal')
        self.assertEqual(value['input_indexes'], {'history_imports': original['input_indexes']['history_imports']})
        self.assertFalse(set(continuation.PREPARATION_FIELDS + continuation.TEMPLATE_FIELDS).intersection(value))
        self.assertFalse(receipt['gpu_submitted']); self.assertFalse(receipt['cache_build_executed'])
        self.assertEqual(self.candidate, original)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_rejects_snapshot_bound_to_different_original_candidate(self):
        self.verified['candidate']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'exact original candidate'): self.derive()

    def test_rejects_incomplete_or_duplicate_preparations(self):
        self.verified['history'].clear()
        with self.assertRaisesRegex(ValueError, 'cover every preparation'): self.derive()

    def test_rejects_changed_history_even_when_its_shape_is_valid(self):
        self.verified['history'][self.tid] = deepcopy(self.verified['history'][self.tid])
        self.verified['history'][self.tid]['source_root'] = str(self.root / 'wrong-source')
        with self.assertRaisesRegex(ValueError, 'Prepared history identity'): self.derive()

    def test_rejects_captured_receipt_content_tampering(self):
        self.snapshot['prepared'][self.tid]['RECOVERY_REQUEST.json']['content']['operation'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'content differs'): self.derive()

    def test_rejects_reused_source_or_control_output(self):
        self.kwargs['experiment_root'] = self.candidate['environment']['experiment_root']
        with self.assertRaisesRegex(ValueError, 'immutable source'): self.derive()
        self.kwargs['experiment_root'] = self.root / 'source-v2/experiment'
        self.kwargs['workflow_output'] = self.root / 'old-control/new-attempt'
        with self.assertRaisesRegex(ValueError, 'workflow output'): self.derive()

    def test_remote_paths_never_depend_on_local_symlinks(self):
        from unittest import mock
        with mock.patch.object(Path, 'is_symlink', side_effect=AssertionError('local path probe')):
            value, _ = self.derive()
            self.assertEqual(len(value['environment']['history']), 1)
        for value in ('relative/path', '/remote/../escape', '//remote/share', '/remote\\file'):
            with self.assertRaises(ValueError): continuation.remote_path(value)

    def test_new_source_and_workflow_cannot_overlap_retained_history(self):
        history = self.verified['history'][self.tid]
        for field in ('experiment_root', 'workflow_output'):
            for protected in (history['source_root'], history['checkpoint']['path']):
                previous = self.kwargs[field]
                self.kwargs[field] = protected
                with self.assertRaisesRegex(ValueError, 'separate new'): self.derive()
                self.kwargs[field] = previous

    def test_changes_only_an_exact_pythonpath_and_rejects_unexpected_value(self):
        old_source = self.candidate['environment']['experiment_root']
        self.candidate['environment']['runtime_environment']['PYTHONPATH'] = old_source + '/src'
        self.snapshot['candidate']['content'] = deepcopy(self.candidate)
        value, _ = self.derive()
        self.assertEqual(value['environment']['runtime_environment']['PYTHONPATH'],
                         str(self.kwargs['experiment_root'] / 'src'))
        self.candidate['environment']['runtime_environment']['PYTHONPATH'] = '/unexpected'
        self.snapshot['candidate']['content'] = deepcopy(self.candidate)
        with self.assertRaisesRegex(ValueError, 'PYTHONPATH'): self.derive()


if __name__ == '__main__':
    unittest.main()
