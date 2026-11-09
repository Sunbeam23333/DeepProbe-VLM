"""Synthetic CPU checks for explicit model identity and accepted SFT provenance."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from unified_evaluation_fixtures import catalog, fixture, write, m


class Boundaries(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_twelve_declared_registrations_and_null_gate(self):
        p = catalog()
        self.assertEqual(len(m.model_sequence(p, require_bound=False)), 12)
        with self.assertRaisesRegex(ValueError, 'MODEL_BINDINGS_GATED'):
            m.model_sequence(p)
        for n, b in p['model_bindings'].items():
            self.assertIsNone(b['checkpoint'])
            self.assertIsNone(b['training_acceptance'])
            self.assertIsNone(b['training_graph'])

    def test_r4_matches_actual_sft_and_nonconstant_file_count(self):
        b = fixture(self.root)
        result = m.verify_registered_checkpoint(b)
        self.assertEqual(result['config']['num_loops'], 4)
        self.assertEqual(result['num_files'], 11)
        self.assertTrue(result['matched_sft_training_source_verified'])

    def test_smol_uses_explicit_causal_backend_and_seed29(self):
        b = fixture(self.root, family='smollm2', loops=1, seed=29)
        result = m.verify_registered_checkpoint(b)
        self.assertEqual(result['config']['language_backend'], 'causal_lm')
        self.assertEqual(result['training_seed'], 29)

    def test_wrong_training_seed_and_alignment_depth_rejected(self):
        for i, (kwargs, reason) in enumerate([({'contract_seed': 29}, 'training_stage_seed_or_budget_changed'),
                ({'alignment_loops': 2}, 'trained_model_config_differs'),
                ({'stage_name': 'alignment'}, 'training_stage_seed_or_budget_changed')]):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, reason):
                m.verify_registered_checkpoint(fixture(self.root / str(i), **kwargs))

    def test_wrong_native_acceptance_not_satisfied_by_plan_reuse(self):
        for i, kwargs in enumerate([{'scheduler_state': 'Failed'}, {'priority': 'normal'},
                                    {'pod_exit_codes': [1]}, {'archive_manifest_sha256': '0' * 64}]):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, 'native_'):
                m.verify_registered_checkpoint(fixture(self.root / str(i), acceptance_overrides=kwargs))

    def test_rejected_or_unknown_native_verdict_cannot_supply_success_scalars(self):
        for i, override in enumerate([{'status': 'REJECTED'}, {'kind': 'FAILED_INDEPENDENT_REVIEW'},
                    {'independent_acceptance': False}, {'acceptance_errors': ['archive_payload_not_fully_verified']}]):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, 'native_acceptance'):
                m.verify_registered_checkpoint(fixture(self.root / str(i), acceptance_overrides=override))

    def test_concurrent_new_file_after_walk_is_detected(self):
        b = fixture(self.root); original = m.sha
        def hashing(path):
            digest = original(path)
            if Path(path).name == 'projector.safetensors':
                (Path(b['checkpoint']['path']) / 'concurrent-extra.bin').write_bytes(b'added after walk')
            return digest
        with patch.object(m, 'sha', hashing), self.assertRaisesRegex(ValueError, 'final_graph_changed_during_verification'):
            m.verify_registered_checkpoint(b)

    def test_native_r1_and_r2_acceptance_layouts_with_synthetic_graphs(self):
        for loops in (1, 2):
            b = fixture(self.root / str(loops), loops=loops)
            ref = b['training_acceptance']
            audit = json.loads(Path(ref['path']).read_text())
            graph = b['training_graph']
            if loops == 1:
                audit.update(kind='P1_OURO_R1_S17_SFT_A1_SECOND_INDEPENDENT_SUCCESS_AUDIT',
                             status='sft_stage_and_archive_verified')
                audit['job']['kubernetes_condition'] = 'Succeeded=True'
            else:
                audit.update(kind='P1_R2_S17_SFT_A1_INDEPENDENT_COMPLETION_ARCHIVE_CHECK',
                             status='completion_and_exact_archive_listing_key_sha_verified',
                             formal_training_result=True, exact_oss_object_paths_lengths_match=True,
                             p1_stage_complete_sha256=graph['stage_complete']['sha256'])
                audit['job']['stream_low_one_node_eight_gpu_restart_never'] = True
                members = {'archive_manifest': 'ARCHIVE_MANIFEST.json',
                           'stage_complete': 'P1_STAGE_COMPLETE.json',
                           'formal_train_complete': 'training/FORMAL_TRAIN_COMPLETE.json',
                           'exit_status': 'EXIT_STATUS.json'}
                audit['readbacks'] = {member: {'passed': True, 'sha256': graph[key]['sha256']}
                                     for key, member in members.items()}
            b['training_acceptance'] = write(Path(ref['path']), audit)
            with self.subTest(loops=loops):
                self.assertTrue(m.verify_registered_checkpoint(b)['matched_sft_training_source_verified'])
                audit['status'] = 'rejected'
                b['training_acceptance'] = write(Path(ref['path']), audit)
                with self.assertRaisesRegex(ValueError, 'native_acceptance'):
                    m.verify_registered_checkpoint(b)

    def test_full_file_graph_rejects_missing_extra_mutated_symlink(self):
        for i, mutation in enumerate(('missing', 'extra', 'changed', 'symlink')):
            b = fixture(self.root / str(i)); final = Path(b['checkpoint']['path'])
            target = final / 'projector.safetensors'
            if mutation == 'missing': target.unlink()
            elif mutation == 'extra': (final / 'unlisted.bin').write_bytes(b'extra')
            elif mutation == 'changed': target.write_bytes(b'changed')
            else:
                target.unlink(); target.symlink_to(final / 'training_args.bin')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                m.verify_registered_checkpoint(b)

    def test_names_cannot_override_identity_and_causal_lm_cannot_recur(self):
        p = catalog()
        for field, value in [('num_loops', 4), ('seed', True), ('language_backend', 'ouro')]:
            q = copy.deepcopy(p); q['model_bindings']['smollm2_s29'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                m.model_sequence(q, require_bound=False)
        q = copy.deepcopy(p); q['model_bindings']['r4']['num_loops'] = 2
        with self.assertRaises(ValueError): m.model_sequence(q, require_bound=False)

    def test_duplicate_run_or_acceptance_cannot_cover_another_model(self):
        a = fixture(self.root / 'r4'); b = fixture(self.root / 'smol', family='smollm2', loops=1, seed=29)
        p = {'model_sequence': [a['model_id'], b['model_id']], 'model_bindings': {a['model_id']: a, b['model_id']: b},
             'benchmark_score_based_model_selection': False}
        self.assertEqual(len(m.model_sequence(p)), 2)
        b['training_acceptance']['sha256'] = a['training_acceptance']['sha256']
        with self.assertRaisesRegex(ValueError, 'acceptance_reused'): m.model_sequence(p)


if __name__ == "__main__":
    unittest.main()
