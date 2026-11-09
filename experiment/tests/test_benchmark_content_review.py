"""Adversarial tests for separate primary/independent content review admission."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('content_review', ROOT / 'scripts/operations/validate_benchmark_content_review.py')
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


class ContentReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.audit_path = self.root / 'CONTENT_AUDIT.json'
        self.review_path = self.root / 'CANDIDATE_VISUAL_ADJUDICATION.json'
        self.independent_path = self.root / 'INDEPENDENT_REVIEW.json'
        self.capture_path = self.root / 'derivatives/DERIVATIVE_CAPTURE_RECEIPT.json'
        self.train_id, self.bench_id = 'youtube:train000001', 'youtube:bench000001'
        self.fps = [{'rgb_sha256': str(i) * 64, 'phash': '0' * 16, 'dhash': '0' * 16,
                     'informative': True} for i in range(8)]
        train = {'relative_path': 'clip.mp4', 'source_video_id': self.train_id, 'frames': self.fps,
                 'sha256': 'a' * 64, 'size_bytes': 7, 'selected_indices': list(range(8)), 'decoded_frame_count': 8}
        tfp = self.root / 'training_fingerprints.jsonl'; write(tfp, train)
        self.policy = {'limitation': 'uniform8', 'phash_max_hamming': 6, 'dhash_max_hamming': 10}
        self.binding = {'fixture': 1}
        tr = self.root / 'TRAINING_FINGERPRINTS.json'
        write(tr, {'fingerprints_sha256': v.sha(tfp), 'passed': True, 'policy': self.policy,
                   'binding': self.binding, 'original_formal_ready_sha256': 'b' * 64})
        bfp = self.root / 'benchmark_fingerprints.json'; write(bfp, [{'source_video_id': self.bench_id, 'frames': self.fps}])
        self.candidate = {'training_path': 'clip.mp4', 'training_source_video_id': self.train_id,
                          'benchmark_source_video_id': self.bench_id, 'classification': 'suspected_overlap_requires_review',
                          'frame_matches': [{'training_slot': 0, 'benchmark_slot': 3, 'phash_hamming': 0,
                                             'dhash_hamming': 0, 'same_decoded_rgb': False}]}
        self.audit = {'schema_version': 1, 'passed': False, 'scan_complete': True, 'errors': [],
                      'uninformative_sources': [], 'exact_duplicate_pairs': [], 'perceptual_duplicate_pairs': [self.candidate],
                      'binding': self.binding, 'media_binding': {'frame_cache_receipt_sha256': 'f' * 64},
                      'policy': self.policy, 'original_formal_ready_sha256': 'b' * 64,
                      'scope': 'uniform8_sampled_frames', 'method': 'fixed scan', 'training_file_count': 1,
                      'benchmark_source_count': 1, 'environment': {'scanner_sha256': 'e' * 64},
                      'artifacts': {'training_fingerprints_receipt': {'path': str(tr), 'sha256': v.sha(tr)},
                                    'benchmark_fingerprints': {'path': str(bfp), 'sha256': v.sha(bfp)}}}
        write(self.audit_path, self.audit)
        files = []
        for name in sorted(v.pair_evidence_names(0, self.candidate)):
            path = self.capture_path.parent / name
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(('pixels:' + name).encode())
            files.append({'relative_path': name, 'bytes': path.stat().st_size, 'sha256': v.sha(path)})
        self.bench_record = {'source_video_id': self.bench_id, 'original_sha256': 'c' * 64,
                             'original_size_bytes': 8, 'selected_indices': list(range(8)),
                             'frames': [{'relative_path': str(i)+'.png', 'sha256': str(i)*64, 'size_bytes': i+1} for i in range(8)]}
        self.benchmark = {'complete': True, 'expected_files': 8, 'expected_source_videos': 1,
                          'root': str(self.root / 'cache'), 'records': [self.bench_record]}
        self.training = {'complete': True, 'expected_files': 1}
        self.capture = {'kind': 'readonly_sha_bound_candidate_contact_sheets', 'audit_sha256': v.sha(self.audit_path),
                        'policy_unchanged': True, 'original_sample_slots_unchanged': True,
                        'all_152_frame_fingerprints_equal_existing': True, 'remote_files_written': 0, 'gpu_used': False,
                        'scanner_sha256': 'e' * 64, 'frame_cache_receipt_sha256': 'f' * 64,
                        'training_receipt_sha256': v.sha(tr), 'training_fingerprints_sha256': v.sha(tfp),
                        'benchmark_fingerprints_sha256': v.sha(bfp), 'training_videos_redecoded': 1,
                        'benchmark_existing_pngs_read': 8, 'files': files, 'sources': [
                            {'kind': 'train', 'source_video_id': self.train_id, 'fingerprints_equal_existing': True,
                             'fingerprints': copy.deepcopy(self.fps), 'original_sha256': 'a' * 64, 'original_size_bytes': 7,
                             'indices': list(range(8)), 'decoded_frame_count': 8},
                            {'kind': 'benchmark', 'source_video_id': self.bench_id, 'fingerprints_equal_existing': True,
                             'fingerprints': copy.deepcopy(self.fps), 'original_sha256': 'c'*64, 'original_size_bytes': 8,
                             'indices': list(range(8)), 'cached_pngs': [
                                 {'path': str(Path(self.benchmark['root'])/f['relative_path']), 'sha256': f['sha256'], 'size_bytes': f['size_bytes']}
                                 for f in self.bench_record['frames']]}]}
        write(self.capture_path, self.capture)
        meta = self.root / 'SELECTED_METADATA.json'
        write(meta, {'audit_sha256': v.sha(self.audit_path), 'pairs': [self.candidate]})
        self.review = {'schema_version': 1, 'kind': 'separate_visual_candidate_adjudication',
                       'original_audit': {'sha256': v.sha(self.audit_path), 'passed': False},
                       'policy': self.policy, 'binding': self.binding, 'media_binding': self.audit['media_binding'],
                       'original_formal_ready_sha256': 'b'*64,
                       'reviewer': 'primary-reviewer', 'reviewed_utc': '2026-10-17T08:00:00Z',
                       'derivative_capture': {'path': 'derivatives/DERIVATIVE_CAPTURE_RECEIPT.json', 'sha256': v.sha(self.capture_path)},
                       'selected_metadata': {'path': meta.name, 'sha256': v.sha(meta)},
                       'counts': {'candidate_pairs': 1, 'candidate_frame_matches': 1, 'unique_training_sources': 1,
                                  'unique_benchmark_sources': 1, 'source_context_frames_visually_inspected': 16,
                                  'distinct_content_hash_collision_pairs': 1, 'distinct_content_hash_collision_frame_matches': 1,
                                  'shared_publisher_boilerplate_only_pairs': 0, 'shared_publisher_boilerplate_only_frame_matches': 0,
                                  'unresolved_pairs': 0, 'substantive_overlapping_scene_pairs_observed': 0},
                       'pairs': [{'pair_index': 1, **{k: val for k, val in self.candidate.items() if k != 'classification'},
                                  'original_classification': self.candidate['classification'],
                                  'decision': v.RESOLVED, 'resolution_category': 'distinct_content_hash_collision',
                                  'matched_frame_observation': 'Different centered graphics.',
                                  'eight_frame_context_observation': 'Distinct story/action contexts.',
                                  'all_original_candidate_matches_visually_inspected': True,
                                  'training_slots_inspected': list(range(8)), 'benchmark_slots_inspected': list(range(8)),
                                  'evidence': [dict(x, relative_path='derivatives/'+x['relative_path']) for x in files]}]}
        for key in ('candidate_review_complete', 'all_original_candidates_resolved',
                    'all_original_candidate_rows_preserved', 'original_audit_not_overwritten',
                    'independent_review_required_before_use'):
            self.review[key] = True
        for key in ('hash_thresholds_changed', 'qa_or_sources_removed', 'frames_removed_or_resampled',
                    'training_or_benchmark_media_modified', 'gold_answers_or_model_predictions_used',
                    'formal_ready_created', 'gpu_submission_performed'):
            self.review[key] = False
        self.content = types.SimpleNamespace(POLICY=self.policy, media_binding=lambda t, b: self.audit['media_binding'])

    def independent(self, review):
        return {'schema_version': 1, 'kind': 'independent_visual_candidate_review_v1',
                'audit_sha256': v.sha(self.audit_path), 'primary_review_sha256': v.sha(self.review_path),
                'derivative_capture_sha256': v.sha(self.capture_path), 'reviewer': 'second-reviewer',
                'reviewed_utc': '2026-10-17T09:00:00Z', 'independent_of_primary': True, 'passed': True,
                'all_matched_frames_and_context_visually_inspected': True,
                'pairs': [{'pair_index': i, 'primary_pair_sha256': v.canonical_sha(p),
                           'decision': 'agree', 'reason': 'Independent inspection sees distinct graphics and context.'}
                          for i, p in enumerate(review['pairs'], 1)],
                'evidence_files': [dict(x, relative_path='derivatives/'+x['relative_path']) for x in self.capture['files']]}

    def validate(self, review=None, independent=None, omit_independent=False):
        review = self.review if review is None else review
        write(self.review_path, review)
        independent = self.independent(review) if independent is None else independent
        write(self.independent_path, independent)
        return v.validate_content_audit(self.audit_path, self.binding, self.training, self.benchmark,
            content_module=self.content, review_path=self.review_path, review_sha256=v.sha(self.review_path),
            independent_path=None if omit_independent else self.independent_path,
            independent_sha256=None if omit_independent else v.sha(self.independent_path))

    def test_complete_review_preserves_raw_failure_and_candidates(self):
        before = self.audit_path.read_bytes(); result = self.validate()
        self.assertTrue(result['passed'], result['errors'])
        self.assertEqual(result['adjudication']['candidate_count'], 1)
        self.assertFalse(result['adjudication']['original_audit_passed'])
        self.assertEqual(result['adjudication']['perceptual_duplicate_pairs'], [self.candidate])
        self.assertEqual(before, self.audit_path.read_bytes())
        self.assertFalse(json.loads(before)['passed'])
        self.assertIn(str(self.independent_path), [r['path'] for r in result['adjudication']['artifacts']])

    def test_shared_boilerplate_is_disclosed_and_not_called_false_positive(self):
        r = copy.deepcopy(self.review); r['pairs'][0]['resolution_category'] = 'shared_publisher_boilerplate_only'
        for suffix in ('pairs', 'frame_matches'):
            r['counts']['distinct_content_hash_collision_'+suffix] = 0
            r['counts']['shared_publisher_boilerplate_only_'+suffix] = 1
        result = self.validate(r)
        self.assertTrue(result['passed'], result['errors'])
        self.assertEqual(result['adjudication']['decisions'][0]['resolution_category'], 'shared_publisher_boilerplate_only')

    def test_original_candidate_scan_blocks_without_explicit_reviews(self):
        self.assertFalse(v.validate_content_audit(self.audit_path, self.binding)['passed'])
        self.assertFalse(self.validate(omit_independent=True)['passed'])

    def test_true_unresolved_missing_duplicate_or_changed_candidate_blocks(self):
        variants = []
        for decision in ('true_overlap', 'unresolved', 'approved', 'false_positive'):
            r = copy.deepcopy(self.review); r['pairs'][0]['decision'] = decision; variants.append(r)
        r = copy.deepcopy(self.review); r['pairs'] = []; variants.append(r)
        r = copy.deepcopy(self.review); r['pairs'] *= 2; variants.append(r)
        r = copy.deepcopy(self.review); r['pairs'][0]['frame_matches'] = []; variants.append(r)
        r = copy.deepcopy(self.review); r['pairs'][0]['pair_index'] = 0; variants.append(r)
        r = copy.deepcopy(self.review); r['pairs'][0]['resolution_category'] = 'harmless'; variants.append(r)
        for r in variants:
            with self.subTest(r=r): self.assertFalse(self.validate(r)['passed'])

    def test_changed_threshold_binding_media_or_raw_audit_blocks(self):
        for field in ('policy', 'binding', 'media_binding', 'original_audit'):
            r = copy.deepcopy(self.review); r[field] = {}
            with self.subTest(field=field): self.assertFalse(self.validate(r)['passed'])
        self.audit['passed'] = True; write(self.audit_path, self.audit)
        self.assertFalse(self.validate()['passed'])

    def test_missing_changed_or_extra_image_evidence_blocks(self):
        for i in range(3):
            r = copy.deepcopy(self.review); r['pairs'][0]['evidence'].pop(i)
            self.assertFalse(self.validate(r)['passed'])
        r = copy.deepcopy(self.review); r['pairs'][0]['evidence'][0]['sha256'] = '0'*64
        self.assertFalse(self.validate(r)['passed'])
        path = self.capture_path.parent/self.capture['files'][0]['relative_path']; path.write_bytes(b'changed')
        self.assertFalse(self.validate()['passed'])

    def test_named_complete_review_and_no_scope_mutation_required(self):
        for field, value in [('matched_frame_observation', ''), ('all_original_candidate_matches_visually_inspected', False),
                             ('training_slots_inspected', [1, 2, 3, 4, 5, 6, 7])]:
            r = copy.deepcopy(self.review); r['pairs'][0][field] = value
            self.assertFalse(self.validate(r)['passed'])
        for field, value in [('reviewer', ''), ('reviewed_utc', '2026-10-17T08:00:00'),
                             ('frames_removed_or_resampled', True), ('hash_thresholds_changed', True),
                             ('qa_or_sources_removed', True), ('gold_answers_or_model_predictions_used', True)]:
            r = copy.deepcopy(self.review); r[field] = value
            self.assertFalse(self.validate(r)['passed'])

    def test_rehashed_capture_cannot_change_original_frame(self):
        self.capture['sources'][0]['fingerprints'][0] = {'other': 'frame'}
        write(self.capture_path, self.capture)
        self.review['derivative_capture']['sha256'] = v.sha(self.capture_path)
        self.assertFalse(self.validate()['passed'])

    def test_current_media_indices_identity_and_completeness_bound(self):
        for field in ('original_sha256', 'selected_indices'):
            old = self.bench_record[field]; self.bench_record[field] = 'changed'
            self.assertFalse(self.validate()['passed']); self.bench_record[field] = old
        self.benchmark['complete'] = False; self.assertFalse(self.validate()['passed'])

    def test_independent_must_bind_all_pairs_and_images_with_distinct_reviewer(self):
        write(self.review_path, self.review); base = self.independent(self.review)
        for field, value in [('reviewer', self.review['reviewer']), ('passed', False),
                             ('independent_of_primary', False), ('reviewed_utc', '2026-10-16T00:00:00Z'),
                             ('primary_review_sha256', '0'*64), ('pairs', []), ('evidence_files', [])]:
            report = copy.deepcopy(base); report[field] = value
            self.assertFalse(self.validate(independent=report)['passed'])
        for field, value in [('decision', 'disagree'), ('reason', ''), ('primary_pair_sha256', '0'*64)]:
            report = copy.deepcopy(base); report['pairs'][0][field] = value
            self.assertFalse(self.validate(independent=report)['passed'])

    def test_changed_review_file_and_symlink_paths_block(self):
        write(self.review_path, self.review); digest = v.sha(self.review_path)
        self.review_path.write_text('{}')
        with self.assertRaises(ValueError): v.validate_review(self.audit_path, self.audit, self.review_path, digest)
        symlink = self.root/'linked.json'; symlink.symlink_to(self.audit_path)
        with self.assertRaises(ValueError): v.sha(symlink)


if __name__ == '__main__': unittest.main()
