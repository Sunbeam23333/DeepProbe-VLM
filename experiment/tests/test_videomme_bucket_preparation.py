"""Offline bucket-identity and CPU-only stage routing contracts."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location('bucket_vm_prep', Path(__file__).resolve().parents[1] /
                                             'scripts/operations/prepare_videomme_bucket_evaluation.py')
p = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(p)


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def scope():
    return {'schema_version': 1, 'evaluation_scope': 'videomme_only',
            'video_mme_evaluation_authorized': True, 'mvbench_evaluation_authorized': False,
            'benchmark_score_based_model_selection': False,
            'models': ['r1_sft_seed17', 'r2_sft_seed17'], 'model_sequence': ['r1', 'r2'], 'protocol': p.PROTOCOL}


class BucketIdentityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name).resolve()
        (cls.root / 'entries').mkdir(); (cls.root / 'frames').mkdir()
        cls.frame = cls.root / 'FRAME_CACHE_RECEIPT.json'
        cls.complete = cls.root / 'FRAME_CACHE_COMPLETE.json'
        cls.bucket = cls.root / 'VIDEOMME_BUCKET_MEDIA_COMPLETE.json'
        cls.registry = cls.root / 'registry.json'
        entries, sources, frames, registry = [], [], [], []
        frame_sha = hashlib.sha256(b'x').hexdigest()
        for index in range(899):
            identity = f'video_{index:04d}'; source = 'youtube:' + identity
            (cls.root / 'frames' / identity).mkdir()
            row = {'source_video_id': source, 'decode_complete': True, 'frames': [], 'image_paths': []}
            for slot in range(8):
                relative = f'frames/{identity}/{slot:02d}.png'
                (cls.root / relative).write_bytes(b'x')
                item = {'relative_path': relative, 'size_bytes': 1, 'sha256': frame_sha}
                row['frames'].append(item); row['image_paths'].append(relative)
                frames.append({**item, 'bucket_key': p.FRAME_PREFIX + f'{identity}/{slot:02d}.png'})
            entries.append(row)
            path = cls.root / 'entries' / (identity + '.json'); write(path, row)
            sources.append({'source_video_id': source, 'bucket_key': p.SOURCE_PREFIX + identity + '.json',
                            'size_bytes': path.stat().st_size, 'sha256': p.sha(path)})
            registry.extend([{'dataset': 'videomme', 'source_video_id': source, 'question_id': f'{index}-{q}'}
                             for q in range(3)])
        write(cls.registry, {'entries': registry})
        cls.registry_sha = p.sha(cls.registry)
        write(cls.frame, {'kind': 'videomme_uniform8_frame_cache', 'complete': True,
                         'num_frames': 8, 'revision': p.REVISION, 'failures': [], 'entries': entries})
        write(cls.complete, {'complete': True, 'source_count': 899, 'frame_count': 7192,
                            'receipt_sha256': p.sha(cls.frame)})
        cls.good = {'kind': 'videomme_bucket_media_complete_v1', 'complete': True, 'bucket': p.BUCKET,
                    'status': 'VIDEOMME_BUCKET_MEDIA_COMPLETE', 'videomme_media_ready': True,
                    'revision': p.REVISION, 'source_count': 899, 'frame_count': 7192,
                    'full_oss_get_sha256_verified': True, 'exact_frame_key_set_verified': True,
                    'exact_source_receipt_key_set_verified': True, 'frame_cache_receipt': p.pin(cls.frame),
                    'frames': frames, 'source_receipts': sources,
                    'all_benchmark_assets_ready': False, 'formal_evaluation_ready': False, 'mvbench_evaluation_ready': False,
                    'ntu_media_included': False}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def verify(self, value=None):
        write(self.bucket, self.good if value is None else value)
        with mock.patch.object(p, 'REGISTRY_SHA', self.registry_sha):
            return p.verify_bucket_media(self.bucket, self.frame, self.complete, self.registry, self.root)

    def test_complete899_and7192_bucket_bindings_accept(self):
        value = self.verify()
        self.assertEqual(value['source_count'], 899); self.assertEqual(value['frame_count'], 7192)
        self.assertEqual(value['exact_frame_and_source_keys'], 8091)
        self.assertIs(value['all_benchmark_assets_ready'], False)
        self.assertIn('required_next_content_audit', value['local_frame_sha_and_RGB_validation'])

    def test_missing_or_duplicate_frame(self):
        for duplicate in [False, True]:
            value = copy.deepcopy(self.good)
            if duplicate: value['frames'][-1] = value['frames'][0]
            else: value['frames'].pop()
            with self.assertRaises(ValueError): self.verify(value)

    def test_bucket_png_key_sha_and_size_must_match_source_receipt(self):
        for field, invalid in [('bucket_key', 'other/prefix/00.png'), ('sha256', '0' * 64), ('size_bytes', 2)]:
            value = copy.deepcopy(self.good); value['frames'][0][field] = invalid
            with self.assertRaisesRegex(ValueError, 'Bucket PNG/source SHA'): self.verify(value)

    def test_source_receipt_key_sha_and_size_must_match_local(self):
        for field, invalid in [('bucket_key', 'other/key'), ('sha256', '0' * 64), ('size_bytes', 1)]:
            value = copy.deepcopy(self.good); value['source_receipts'][0][field] = invalid
            with self.assertRaisesRegex(ValueError, 'Bucket source receipt'): self.verify(value)

    def test_alternate_source_is_rejected_even_with899_rows(self):
        value = copy.deepcopy(self.good); value['source_receipts'][0]['source_video_id'] = 'youtube:other'
        with self.assertRaisesRegex(ValueError, 'source set differs'): self.verify(value)

    def test_bucket_verification_and_scope_flags_are_strict(self):
        for field, invalid in [('full_oss_get_sha256_verified', 1), ('exact_frame_key_set_verified', False),
                               ('exact_source_receipt_key_set_verified', False), ('all_benchmark_assets_ready', True),
                               ('mvbench_evaluation_ready', True), ('ntu_media_included', True)]:
            value = copy.deepcopy(self.good); value[field] = invalid
            with self.assertRaises(ValueError): self.verify(value)

    def test_changed_frame_cache_binding_is_rejected(self):
        value = copy.deepcopy(self.good); value['frame_cache_receipt']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'different frame receipt'): self.verify(value)

    def test_wrong_bucket_or_revision_is_rejected(self):
        for field in ['bucket', 'revision']:
            value = copy.deepcopy(self.good); value[field] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'revision mismatch'): self.verify(value)


class ScopeAndStageTest(unittest.TestCase):
    def test_scope_is_vm_only_fixed_model_order_without_score_selection(self):
        p.validate_scope(scope())
        for field, invalid in [('evaluation_scope', 'all'), ('video_mme_evaluation_authorized', False),
                               ('mvbench_evaluation_authorized', True), ('benchmark_score_based_model_selection', True),
                               ('model_sequence', ['r2', 'r1']), ('models', ['best_test_score'])]:
            value = scope(); value[field] = invalid
            with self.assertRaises(ValueError): p.validate_scope(value)

    def test_cpu_environment_has_no_visible_cuda_and_is_offline(self):
        env = p.cpu_environment()
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(env['HF_HUB_OFFLINE'], '1')

    def test_stage_cannot_be_submit_or_training(self):
        with self.assertRaisesRegex(ValueError, 'Unapproved CPU stage'):
            p.run_cpu_stage(Path('/nonexistent'), 'submit', ['sslaunch', 'submit'])

    def test_failed_cpu_stage_stops_and_records_no_later_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with mock.patch.object(p.subprocess, 'run', return_value=type('Result', (), {'returncode': 2})()) as run:
                with self.assertRaisesRegex(ValueError, 'no later stage'):
                    p.run_cpu_stage(root, 'content_audit', ['frozen-python', 'frozen-audit'])
            self.assertEqual(run.call_count, 1)
            self.assertEqual(json.loads((root / 'content_audit.exit.json').read_text())['returncode'], 2)
            self.assertFalse((root / 'READY_FOR_PARENT_SUBMISSION.json').exists())

    def test_prepare_routes_four_cpu_stages_and_only_emits_parent_submission_argv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); output = root / 'out'; output.mkdir()
            auth = root / 'authorization.json'; write(auth, scope())
            config_path = root / 'config.json'; config_path.write_text('{}\n')
            base, p1, experiment, media = (root / name for name in ['base', 'p1', 'experiment', 'media'])
            config = {'kind': 'videomme_bucket_cpu_preparation_v1', 'run_id': p.RUN_ID,
                      'experiment_root': str(experiment), 'media_root': str(media),
                      'authorization': p.pin(auth), 'evaluation_source_files': [], 'training_fingerprint_files': []}
            config['submission_helper'] = {'path': str(root / 'submit_wrapper.py'), 'sha256': 'a' * 64}
            calls = []
            def stage(out, name, argv):
                calls.append((name, argv))
                if name == 'formal_gate':
                    formal = output / 'formal-gate'; formal.mkdir()
                    write(formal / 'FORMAL_READY.json', {'formal_evaluation_ready': True,
                          'benchmark_media_gate': 'passed', 'content_duplicate_gate': 'passed'})
                if name == 'cpu_preflight':
                    write(output / 'cpu_preflight.stdout', {'status': 'CPU_PREFLIGHT_PASSED',
                          'plan_sha256': p.sha(output / 'EVALUATION_PLAN.json'), 'num_gpus': 8,
                          'sequence': ['r1', 'r2'], 'subtitles': False, 'audio': False})
            # This unit covers routing. Media bytes are covered by the separate
            # full899/7192 fixture and the unchanged frozen content-audit tests.
            original_pin = p.pin
            def fake_pin(path):
                path = Path(path)
                return original_pin(path) if path.is_file() else {'path': str(path), 'sha256': 'a' * 64}
            with mock.patch.multiple(p, BASE=base, P1=p1, EXPERIMENT=experiment, MEDIA_ROOT=media,
                                     FROZEN_SOURCE_LIST_SHA=hashlib.sha256(b'[]\n').hexdigest()), \
                 mock.patch.object(p, 'verify', side_effect=lambda item: Path(item['path'])), \
                 mock.patch.object(p, 'verify_bucket_media', return_value={'evaluation_scope': 'videomme_only'}), \
                 mock.patch.object(p, 'run_cpu_stage', side_effect=stage), mock.patch.object(p, 'pin', side_effect=fake_pin), \
                 mock.patch.object(p, 'sha', wraps=p.sha) as hashed:
                # The manifest command needs the source receipt SHA, which is
                # supplied as a fixed fixture without copying any real media.
                media.mkdir(); (media / 'FRAME_CACHE_RECEIPT.json').write_text('{}')
                p.prepare(config, config_path, output)
            self.assertEqual([name for name, _ in calls], ['content_audit', 'formal_gate', 'manifest', 'cpu_preflight'])
            self.assertTrue(all('sslaunch' not in ' '.join(argv) for _, argv in calls))
            self.assertIn('--preflight-only', calls[-1][1])
            receipt = json.loads((output / 'READY_FOR_PARENT_SUBMISSION.json').read_text())
            self.assertIs(receipt['gpu_submitted'], False)
            self.assertIs(receipt['all_benchmark_assets_ready'], False)
            self.assertEqual(receipt['parent_submission_argv'][-1], '--execute')
            scoped = json.loads((output / 'formal-gate/VIDEOMME_ONLY_READY.json').read_text())
            self.assertIs(scoped['videomme_evaluation_ready'], True)
            self.assertIs(scoped['mvbench_evaluation_ready'], False)
            self.assertEqual(scoped['evaluation_scope'], 'videomme_only')


if __name__ == '__main__': unittest.main(verbosity=2)
