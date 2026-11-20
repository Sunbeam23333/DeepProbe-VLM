"""Synthetic CPU integration only: fixtures never certify real jobs or assets."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from deepprobe_vlm import experiment_suite as suite, suite_bindings as bindings, suite_runner as runner
from deepprobe_vlm import suite_workflow as workflow


class SuiteWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.plan = suite.default_plan()
        self.task = next(t for t in self.plan['tasks'] if t['id'] == 'train__smollm2_s29__alignment')
        self.original = self.pin('SYNTHETIC_SOURCE_AUDIT.json', {'scope': 'SYNTHETIC_UNIT_TEST_ONLY'})

    def pin(self, name, document):
        path = self.root / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document)); return runner.artifact(path)

    def entry(self):
        names = {'deepprobe_config.json', 'training_provenance.json', 'projector.safetensors',
                 'language_model/config.json', 'vision_model/config.json', 'language_model/model.safetensors',
                 'vision_model/model.safetensors', 'tokenizer/tokenizer_config.json',
                 'image_processor/preprocessor_config.json', 'trainer_state.json', 'optimizer.pt', 'scheduler.pt',
                 'training_args.bin'} | {f'rng_state_{i}.pth' for i in range(8)}
        return {'mode': 'resume', 'checkpoint_step': 350, 'source_root': str(self.root),
                'config': self.original, 'frozen_source_manifest': self.original,
                'checkpoint': {'path': str(self.root / 'old-run/checkpoint-350'),
                 'oss_prefix': 'oss://ss-devbox-stream/project/runs/synthetic/training/checkpoint-350/',
                 'files': [{'path': n, 'bytes': 1, 'sha256': '1' * 64} for n in sorted(names)]}}

    def review(self, entry, production=True):
        return self.pin('SYNTHETIC_REVIEW.json', {
            'status': 'SUITE_HISTORY_IMPORT_REVIEW_ACCEPTED', 'approved': True,
            'production_evidence': production, 'task_id': self.task['id'], 'task_sha256': runner.digest(self.task),
            'entry_sha256': runner.digest(entry), 'evidence': [self.original],
            'verified_facts': {'source_root': entry['source_root'], 'config_sha256': entry['config']['sha256'],
                'frozen_source_manifest_sha256': entry['frozen_source_manifest']['sha256'],
                'checkpoint_inventory_sha256': runner.digest(entry['checkpoint']['files']),
                'checkpoint_oss_prefix': entry['checkpoint']['oss_prefix'], 'optimizer_step': 350,
                'archive_verified': True, 'resume_metadata_verified': True,
                'optimizer_scheduler_rng_deserialized': False}})

    def test_cp350_explicit_binding_overrides_old_candidate_hint_without_claiming_restore(self):
        entry = self.entry()
        value = bindings.import_history_entry(self.plan, self.task['id'], entry, self.review(entry), self.root / 'IMPORTED.json')
        verified = bindings.verify_history_binding(self.plan, self.task, value)
        self.assertEqual(verified['optimizer_step'], 350)
        self.assertFalse(verified['runtime_resume_verified'])
        report = bindings.remaining_summary(self.plan, {'history': {self.task['id']: value}})
        row = next(r for r in report['tasks'] if r['task_id'] == self.task['id'])
        self.assertEqual(row['remaining_optimizer_updates'], 368)
        self.assertEqual(row['disposition'], 'restore_and_resume')
        self.assertEqual(report['operational_tasks_omitted'], 0)
        self.assertFalse((self.root / 'old-run').exists())

    def test_unreviewed_or_nonproduction_history_never_publishes_acceptance(self):
        entry = self.entry(); target = self.root / 'DENIED.json'
        with self.assertRaisesRegex(ValueError, 'production history import review'):
            bindings.import_history_entry(self.plan, self.task['id'], entry, self.review(entry, production=False), target)
        self.assertFalse(target.exists())
        with self.assertRaises(ValueError):
            bindings.validate_history_shape(self.task, {'mode': 'accepted'})

    def test_native_resume_requires_archival_and_complete_rng_before_import(self):
        entry = self.entry(); review = self.review(entry)
        document = runner.read(review['path']); document['verified_facts']['archive_verified'] = False
        review = self.pin('BAD_ARCHIVE_REVIEW.json', document)
        with self.assertRaisesRegex(ValueError, 'reviewed facts'):
            bindings.import_history_entry(self.plan, self.task['id'], entry, review, self.root / 'DENIED.json')
        broken = deepcopy(entry); broken['checkpoint']['files'] = [r for r in broken['checkpoint']['files'] if r['path'] != 'rng_state_7.pth']
        with self.assertRaisesRegex(ValueError, 'native optimizer'):
            bindings.validate_history_shape(self.task, {**broken, 'acceptance': self.original})
        self.assertFalse((self.root / 'DENIED.json').exists())

    def test_tampered_original_evidence_invalidates_reuse(self):
        entry = self.entry(); value = bindings.import_history_entry(self.plan, self.task['id'], entry, self.review(entry), self.root / 'IMPORTED.json')
        Path(self.original['path']).write_text('changed')
        report = bindings.remaining_summary(self.plan, {'history': {self.task['id']: value}})
        self.assertIn(self.task['id'], report['blocked_history_tasks'])
        self.assertEqual(report['verified_historical_counts'], {})

    def test_default_one_command_has_no_asset_network_admission_or_scheduler_effect(self):
        candidate = {'schema_version': 1, 'environment': {'history': {}, 'storage': {'home_limit_bytes': 150000000000,
                    'oss_limit_bytes': 150000000000}}, 'state_root': str(self.root / 'state')}
        with mock.patch.object(workflow, 'prepare_benchmark', side_effect=AssertionError('asset work from plan')):
            result = workflow.prepare_workflow(self.plan, candidate, self.root / 'control')
        self.assertEqual(result['status'], 'SUITE_WORKFLOW_PLANNED_NOT_EXECUTED')
        self.assertFalse(result['production_admitted'])
        self.assertEqual(result['task_count'], 185)
        self.assertFalse((self.root / 'state').exists())

    def test_evaluation_history_rejects_incomplete_scored_rows(self):
        task = next(t for t in self.plan['tasks'] if t['id'] == 'evaluate__ouro_r1_s17__videomme899')
        result = self.pin('INCOMPLETE_EVAL.json', {'status': 'SUITE_EVALUATION_COMPLETE', 'suite_protocol': task['protocol'],
            'task_sha256': runner.digest(task), 'qa': 2697, 'failed_units': 0, 'rows': []})
        acceptance = self.pin('INCOMPLETE_ACCEPTANCE.json', {'schema_version': 1, 'status': 'SUITE_HISTORY_ACCEPTED',
            'production_evidence': True, 'task_id': task['id'], 'task_sha256': runner.digest(task), 'mode': 'accepted',
            'evidence': [self.original], 'evaluation_sha256': result['sha256'], 'checkpoint_inventory_sha256': 'a' * 64,
            'protocol': task['protocol'], 'qa': 2697, 'failed_units': 0})
        with self.assertRaisesRegex(ValueError, 'complete unique scored rows'):
            bindings.verify_history_binding(self.plan, task, {'mode': 'accepted', 'acceptance': acceptance,
                'evaluation': result, 'checkpoint_inventory_sha256': 'a' * 64})

    def test_preparation_wires_real_cache_descriptor_into_binding_and_cpu_admission(self):
        # Adapter boundaries are synthetic; the actual workflow order and
        # propagation run. This test deliberately produces CPU_GATED, never a
        # production admission or a scheduler operation.
        env = {'suite_id': 'synthetic-workflow', 'experiment_root': str(self.root), 'history': {},
               'storage': {'roots': [str(self.root)]}, 'assets': {'benchmarks': {}, 'training': {},
               'development': {'source_metadata': self.original}}, 'source_files': []}
        request = {'benchmark': 'videomme899', 'frames': 4, 'source_manifest': self.original,
                   'oss_prefix': 'synthetic', 'max_png_bytes': 100, 'metadata_reserve_bytes': 100}
        candidate = {'schema_version': 1, 'environment': env, 'state_root': str(self.root / 'state'),
                     'benchmark_preparations': [request]}
        descriptor = {'cache': {'synthetic': True}, 'media_root': str(self.root / 'frames'), 'max_bytes': 100}
        seen = []
        def build(plan, environment, **kwargs):
            self.assertEqual(environment['assets']['benchmarks']['videomme899']['4'], descriptor)
            seen.append('binding')
            return {'source_files': environment['source_files']}
        def preflight(plan, binding, environment, **kwargs):
            self.assertEqual(seen, ['cache', 'binding'])
            self.assertFalse(kwargs['prepare_media'])
            seen.append('preflight')
            return {'status': 'SUITE_CPU_GATED', 'scope': 'synthetic integration fixture'}
        def cache(*args, **kwargs):
            seen.append('cache'); return descriptor, {'scope': 'synthetic integration fixture'}
        with mock.patch.object(workflow, 'prepare_benchmark', side_effect=cache), \
             mock.patch.object(bindings, 'remaining_summary', return_value={'blocked_history_tasks': []}), \
             mock.patch.object(bindings, 'build_binding', side_effect=build), \
             mock.patch('deepprobe_vlm.formal_integrity.source_inventory', return_value={'SYNTHETIC_SOURCE_AUDIT.json': self.original['sha256']}), \
             mock.patch('deepprobe_vlm.suite_preflight.inspect', side_effect=preflight):
            result = workflow.prepare_workflow(self.plan, candidate, self.root / 'control', prepare_assets=True, cpu_admission=True)
        self.assertEqual(seen, ['cache', 'binding', 'preflight'])
        self.assertFalse(result['cpu_admitted'])
        self.assertFalse(result['gpu_submitted'])
        self.assertFalse((self.root / 'state').exists())

    def test_token_file_never_puts_value_in_returned_receipt(self):
        token = self.root / 'private-token'; token.write_text('synthetic-token-value'); token.chmod(0o600)
        environment = {}
        result = bindings.load_modelscope_token_file(token, environ=environment)
        self.assertEqual(environment['MODELSCOPE_API_TOKEN'], 'synthetic-token-value')
        self.assertNotIn('synthetic-token-value', json.dumps(result))
        token.chmod(0o644)
        with self.assertRaises(ValueError): bindings.load_modelscope_token_file(token, environ={})

    def test_capacity_gate_is_actionable_append_only_and_precedes_asset_work(self):
        plan_pin = self.pin('RETIREMENT_PLAN.json', {'status': 'review required', 'automatic_delete': False})
        candidate = {'schema_version': 1, 'environment': {'history': {}, 'storage': {'schema_version': 2,
                    'home_limit_bytes': 150000000000, 'oss_limit_bytes': 150000000000}},
                    'state_root': str(self.root / 'state'), 'capacity_recovery_plan': plan_pin}
        gate = {'status': 'SUITE_PREPARATION_CAPACITY_GATED', 'blocked_domains': ['oss'],
                'fresh_snapshot': {'utc': 'first'}, 'recovery_plan': plan_pin}
        with mock.patch.object(workflow, 'preparation_capacity', return_value=gate), \
             mock.patch.object(workflow, 'prepare_benchmark', side_effect=AssertionError('gated asset transfer')):
            first = workflow.prepare_workflow(self.plan, candidate, self.root / 'control', prepare_assets=True)
            gate['fresh_snapshot']['utc'] = 'second'
            second = workflow.prepare_workflow(self.plan, candidate, self.root / 'control', prepare_assets=True, resume_assets=True)
        self.assertEqual(first['status'], 'SUITE_WORKFLOW_CAPACITY_GATED')
        self.assertEqual(first['capacity_recovery_plan'], plan_pin)
        self.assertNotEqual(first['capacity_gate']['path'], second['capacity_gate']['path'])
        self.assertEqual(runner.read(first['capacity_gate']['path'])['fresh_snapshot']['utc'], 'first')
        self.assertFalse((self.root / 'state').exists())

    def test_preparation_capacity_accepts_unlimited_oss_with_real_shared_budget(self):
        from deepprobe_vlm import suite_storage as storage
        environment = {'storage': {'schema_version': 2, 'roots': [str(self.root)],
            'ledger_root': str(self.root / 'ledger'), 'home_limit_bytes': 100_000_000,
            'oss_limit_bytes': None, 'home_safety_bytes': 1_000_000, 'oss_safety_bytes': 1_000_000,
            'free_guard_bytes': 0, 'oss_prefixes': ['oss://ss-devbox-stream/fixture/project/']},
            'tools': {'ossutil': '/bin/true', 'oss_config': self.original['path']}}
        with mock.patch.object(storage.OssutilInventory, '__call__', side_effect=AssertionError('quota scan')):
            value = workflow.preparation_capacity(environment)
        self.assertEqual(value['status'], 'SUITE_PREPARATION_CAPACITY_AVAILABLE')
        self.assertEqual(value['blocked_domains'], [])
        self.assertIsNone(value['fresh_snapshot']['oss']['bytes'])
        self.assertFalse(value['gpu_submitted'])

    def test_history_import_creates_parent_and_reuses_exact_completed_import(self):
        entry = self.entry(); review = self.review(entry); target = self.root / 'new-parent/accepted.json'
        value = bindings.import_history_entry(self.plan, self.task['id'], entry, review, target)
        self.assertEqual(bindings.import_history_entry(self.plan, self.task['id'], entry, review, target), value)

    def test_legacy_original_archive_requires_exact_audit_mapping_and_receipt(self):
        from deepprobe_vlm import suite_media as media
        key = 'jinhaopeng/Paper/DeepProbe-VLM/benchmark-cache/original/videos_chunked_02.zip'
        sha = 'a' * 64; etag = 'original-etag'; size = 345
        raw = media.encode({'binding': {'bucket': 'ss-devbox-stream', 'key': key, 'sha256': sha, 'size_bytes': size},
            'source_stream_sha256': sha, 'source_bytes': size,
            'oss_readback': {'bytes': size, 'sha256': sha, 'verification': 'complete_OSS_GET_SHA256_and_size'}})
        class Store:
            def head(self, observed):
                assert observed == key; return {'size_bytes': size, 'etag': etag}
            def get(self, observed, *args):
                assert observed == key + '.CACHE_VERIFIED.json'; return raw
        receipt = {'complete': True, 'entries': [{'archive_path': 'videos_chunked_02.zip', 'archive_sha256': sha}]}
        row = {'key': key, 'etag': etag, 'source_sha256': sha, 'receipt_sha256': media.digest(raw)}
        catalog = workflow.source_catalog(receipt, Store(), {'checked_archives': [row]})
        bound = workflow.bind_legacy_archive_identities(receipt, catalog)
        self.assertEqual(bound['entries'][0]['archive_key'], key)
        self.assertNotIn('archive_key', receipt['entries'][0])
        with self.assertRaisesRegex(ValueError, 'exact pinned archive audit mapping'):
            workflow.source_catalog(receipt, Store(), {'checked_archives': [row, dict(row)]})
        with self.assertRaisesRegex(ValueError, 'verification receipt changed'):
            workflow.source_catalog(receipt, Store(), {'checked_archives': [{**row, 'receipt_sha256': 'b' * 64}]})

    def test_separate_checkpoint_metadata_must_match_exact_payload_inventory(self):
        entry = self.entry(); value = {**entry, 'acceptance': self.original}
        row = next(r for r in entry['checkpoint']['files'] if r['path'] == 'trainer_state.json')
        value['checkpoint_metadata'] = {'trainer_state.json': {**row, 'path': str(self.root / 'retained-trainer-state.json')}}
        bindings.validate_history_shape(self.task, value)
        value['checkpoint_metadata']['trainer_state.json']['sha256'] = 'd' * 64
        with self.assertRaisesRegex(ValueError, 'metadata pin differs'): bindings.validate_history_shape(self.task, value)

    def test_global_asset_resume_starts_untouched_shape_and_rejects_unowned_objects(self):
        store = mock.Mock(); store.keys.side_effect = lambda prefix: (row for row in [])
        self.assertFalse(workflow.benchmark_resume_mode(self.root / 'new-f16', 'exact/', store, resume_requested=True))
        store.keys.side_effect = lambda prefix: (row for row in ['exact/unowned.png'])
        with self.assertRaisesRegex(ValueError, 'reconcile exact ownership'):
            workflow.benchmark_resume_mode(self.root / 'new-f16', 'exact/', store, resume_requested=True)
        output = self.root / 'started-f4'; output.mkdir(); (output / 'BUILD_PLAN.json').write_text('{}')
        self.assertTrue(workflow.benchmark_resume_mode(output, 'exact/', store, resume_requested=True))
        with self.assertRaisesRegex(ValueError, 'explicit asset resume'):
            workflow.benchmark_resume_mode(output, 'exact/', store, resume_requested=False)

    def test_cache_empty_listing_still_requires_complete_generator_success(self):
        def failing_listing(prefix):
            yield from ()
            raise OSError('synthetic OSS listing interrupted')
        store = mock.Mock(); store.keys.side_effect = failing_listing
        with self.assertRaisesRegex(OSError, 'listing interrupted'):
            workflow.benchmark_resume_mode(self.root / 'never-started', 'exact/', store, resume_requested=False)
        self.assertFalse((self.root / 'never-started').exists())


if __name__ == '__main__': unittest.main()
