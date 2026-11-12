"""Scientific matrix completeness and checkpoint-lifetime contracts, CPU only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from deepprobe_vlm import experiment_suite as suite


class CompleteExperimentSuiteTest(unittest.TestCase):
    def setUp(self):
        self.plan = suite.default_plan()
        self.by_id = {task['id']: task for task in self.plan['tasks']}

    def test_complete_fixed_matrix_and_declared_gpu_policy(self):
        summary = suite.summarize_plan(self.plan)
        self.assertEqual(summary['scope']['scientific_model_configs'], 18)
        self.assertEqual(summary['task_counts'], {'gate': 1, 'train': 34, 'archive': 52,
            'evaluate': 36, 'latency': 15, 'compare_latency': 12, 'cleanup': 34, 'aggregate': 1})
        self.assertEqual(self.plan['allocation']['priority'], 'low')
        self.assertEqual(self.plan['allocation']['gpus'], 8)
        self.assertEqual(self.plan['allocation']['nodes'], 1)
        self.assertEqual(self.plan['allocation']['max_concurrent_gpu_tasks'], 1)
        self.assertFalse(summary['gpu_submitted'])
        self.assertFalse(summary['training_or_evaluation_completed'])
        suite.validate_plan(self.plan, config_root=suite.EXPERIMENT_ROOT)

    def test_three_core_seeds_are_independent_and_smol29_cannot_replace17(self):
        core = [m for m in self.plan['models'] if m['group'] == 'core']
        self.assertEqual({(m['family'], m['seed']) for m in core},
            {(family, seed) for family in ('ouro_r1', 'ouro_r2', 'ouro_r4', 'smollm2') for seed in (17, 29, 43)})
        s17 = self.by_id['train__smollm2_s17__alignment']
        s29 = self.by_id['train__smollm2_s29__alignment']
        self.assertNotEqual(s17['config']['sha256'], s29['config']['sha256'])
        self.assertEqual(s17['historical_hint']['status'], 'planned')
        self.assertEqual(s29['historical_hint']['checkpoint_step'], 300)

    def test_all_shape_ablations_use_matching_independent_alignment(self):
        for axis in ('f4', 'f16', 'g2', 'g8'):
            model = f'ouro_r4_{axis}_s17'
            sft = self.by_id[suite.task_id('train', model, 'sft')]
            self.assertEqual(sft['initializer']['checkpoint_task_id'], suite.task_id('train', model, 'alignment'))
            self.assertIn(suite.task_id('archive', model, 'alignment'), sft['dependencies'])
            for benchmark in suite.BENCHMARKS:
                evaluation = self.by_id[suite.task_id('evaluate', model, benchmark)]
                self.assertIn('shape-ablation', evaluation['protocol'])
                self.assertFalse(evaluation['resample_existing_frame_cache'])
                self.assertEqual(evaluation['frame_sampling'], 'uniform_indices_from_original_video')

    def test_frozen_vision_shared_initializer_has_two_consumers_and_safe_cleanup(self):
        alignment = 'train__ouro_r4_s17__alignment'
        expected = ['train__ouro_r4_s17__sft', 'train__ouro_r4_frozen_vision_s17__sft']
        self.assertEqual(suite.checkpoint_consumers(self.plan, alignment), expected)
        cleanup = self.by_id['cleanup__ouro_r4_s17__alignment']
        self.assertTrue(set(expected) <= set(cleanup['dependencies']))
        self.assertIn('archive__ouro_r4_s17__results', cleanup['dependencies'])
        self.assertNotIn(expected[1], self.by_id['cleanup__ouro_r4_s17__sft']['dependencies'])

    def test_no_alignment_is_one_stage_with_fresh_connector(self):
        task = self.by_id['train__ouro_r4_no_alignment_s17__sft']
        self.assertNotIn('train__ouro_r4_no_alignment_s17__alignment', self.by_id)
        self.assertEqual(task['initializer'], {'kind': 'pretrained_backbones_random_connector'})
        self.assertEqual(task['entry'], 'deepprobe_vlm.train_declared')

    def test_historical_success_is_only_reuse_candidate_and_resume_stays_gated(self):
        training = [t for t in self.plan['tasks'] if t['kind'] == 'train']
        self.assertEqual(sum(t['historical_hint']['status'] == 'reuse_candidate' for t in training), 5)
        resumes = [t for t in training if t['historical_hint']['status'] == 'resume_candidate']
        self.assertEqual({t['historical_hint']['checkpoint_step'] for t in resumes}, {300, 600})
        for task in resumes:
            self.assertFalse(task['historical_hint']['resume_accepted'])
            self.assertFalse(task['historical_hint']['automatic_restart_from_zero'])
        self.assertFalse(self.plan['selection']['automatic_skip_from_plan'])

    def test_cleanup_requires_results_archive_and_every_checkpoint_consumer(self):
        for task in self.plan['tasks']:
            if task['kind'] != 'cleanup':
                continue
            consumers = suite.checkpoint_consumers(self.plan, task['checkpoint_task_id'])
            self.assertEqual(task['required_consumers'], consumers)
            self.assertTrue(set(consumers) <= set(task['dependencies']))
            self.assertIn(suite.task_id('archive', task['model_id'], 'results'), task['dependencies'])
            self.assertTrue(task['checkpoint_only'])
            self.assertTrue(task['keep_small_evidence'])
            self.assertTrue(task['fresh_local_and_oss_sha256_required'])

    def test_ready_task_order_releases_sft_before_starting_next_model(self):
        completed, order = set(), []
        while len(completed) < len(self.plan['tasks']):
            task = next(t for t in self.plan['tasks'] if t['id'] not in completed and set(t['dependencies']) <= completed)
            completed.add(task['id']); order.append(task['id'])
        self.assertLess(order.index('cleanup__ouro_r1_s17__sft'), order.index('train__ouro_r2_s17__alignment'))
        self.assertLess(order.index('cleanup__ouro_r4_s17__sft'), order.index('train__smollm2_s17__alignment'))
        self.assertGreater(order.index('cleanup__ouro_r4_s17__alignment'), order.index('train__ouro_r4_frozen_vision_s17__sft'))

    def test_config_bytes_are_verified_before_any_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            shutil.copytree(suite.EXPERIMENT_ROOT / 'configs/train/suite', root / 'configs/train/suite')
            suite.validate_plan(self.plan, config_root=root)
            selected = self.by_id['train__ouro_r4_f4_s17__sft']['config']['path']
            p = root / selected
            p.write_bytes(p.read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'Recipe bytes differ'):
                suite.validate_plan(self.plan, config_root=root)

    def test_modified_recipe_cannot_pass_by_repinning_its_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            shutil.copytree(suite.EXPERIMENT_ROOT / 'configs/train/suite', root / 'configs/train/suite')
            inventory_path = root / suite.RECIPE_INVENTORY
            inventory = json.loads(inventory_path.read_text())
            entry = next(r for r in inventory['files'] if r['model_id'] == 'smollm2_s17' and r['stage'] == 'sft')
            p = root / entry['path']; raw = json.loads(p.read_text()); raw['training']['learning_rate'] = .01
            p.write_text(json.dumps(raw)); entry.update(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            inventory_path.write_text(json.dumps(inventory))
            forged = suite.build_plan(root)
            with self.assertRaisesRegex(ValueError, 'schedule/precision/batch'):
                suite.validate_plan(forged, config_root=root)

    def test_incomplete_or_unsafe_plans_fail_closed(self):
        def mutate(field, value):
            plan = deepcopy(self.plan); plan[field] = value; return plan
        bad = [mutate('models', self.plan['models'][:-1]), mutate('tasks', self.plan['tasks'][:-1]),
               mutate('selection', {**self.plan['selection'], 'heldout_tuning': True})]
        for key, value in (('priority', 'normal'), ('nodes', 2), ('gpus', 1), ('restart_policy', 'OnFailure')):
            bad.append(mutate('allocation', {**self.plan['allocation'], key: value}))
        plan = deepcopy(self.plan); plan['tasks'][1]['dependencies'] = [plan['tasks'][1]['id']]; bad.append(plan)
        plan = deepcopy(self.plan)
        cleanup = next(t for t in plan['tasks'] if t['id'] == 'cleanup__ouro_r4_s17__alignment')
        cleanup['dependencies'].remove('train__ouro_r4_frozen_vision_s17__sft'); bad.append(plan)
        for plan in bad:
            with self.subTest(plan_sha=suite.plan_sha256(plan)), self.assertRaises(ValueError):
                suite.validate_plan(plan)

    def test_optional_research_and_unavailable_benchmarks_are_not_fake_tasks(self):
        optional = self.plan['scope']['conditional_work_not_claimed_complete']
        self.assertIn('mvbench_licensed_media_and_scope', optional)
        self.assertIn('h20_matched_hardware', optional)
        self.assertIn('selective_refresh_budget_kernel_codesign', optional)
        self.assertEqual({t['benchmark'] for t in self.plan['tasks'] if t['kind'] == 'evaluate'}, set(suite.BENCHMARKS))


if __name__ == '__main__':
    unittest.main()
