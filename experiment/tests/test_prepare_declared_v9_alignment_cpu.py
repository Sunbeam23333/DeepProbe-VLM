"""Fresh alignment CPU preparation: guarded real gates with external doubles."""
import pytest
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import test_prepare_declared_v9_cpu as cpu_fixture
from test_prepare_declared_v9_cpu import ROOT, digest, load, put, r


@pytest.mark.private_evidence
class AlignmentPreparationTests(unittest.TestCase):
    def setUp(self):
        self.f = cpu_fixture.PrepareDeclaredV9CPUTests(methodName='test_complete_readonly_release_source_and_all17_config_bindings')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        f = self.f
        f.thaw_release()
        path = f.experiment / 'scripts/operations/prepare_declared_v9_alignment_cpu.py'
        shutil.copyfile(ROOT / 'scripts/operations/prepare_declared_v9_alignment_cpu.py', path)
        (f.experiment / 'scripts/operations/inspect_declared_v9_capacity.py').write_text('# Nonexecuted capacity boundary\n')
        self.m = load('alignment_preparation_fixture', path)
        m = self.m
        m.h = f.m
        m.OUTPUT = f.m.OUTPUT
        f.raw = json.loads((f.experiment / m.CONFIG_RELATIVE).read_text())
        f.runtime = copy.deepcopy(f.raw)
        for label, key in (('language', 'language_model'), ('vision', 'vision_model')):
            f.runtime['model'][key] = f.offline[label]['resolved_path']
        self.preview = {'cells': [{'config': m.CONFIG_RELATIVE, 'stage': 'alignment', 'seed': 29,
                                  'capacity_admitted': True, 'plan': {'required_additional_bytes': 18_070_438_560}}],
                        'ready_for_gpu_submission': False}
        self.precheck = Mock(side_effect=lambda _: copy.deepcopy(self.preview))
        patch.object(m, 'load', return_value=SimpleNamespace(capacity_inspection=self.precheck)).start()
        f.module.prepare_offline_config.side_effect = self.prepare_offline
        put(f.release / 'P1_SOURCE_MANIFEST.json', {'files': r.source_inventory(f.experiment)})
        f.freeze_release()

    def prepare_offline(self, config, target, hub, stage):
        f = self.f
        self.assertEqual((config, hub, stage), (f.experiment / self.m.CONFIG_RELATIVE, f.m.HUB, 'alignment'))
        put(target, f.runtime)
        offline = copy.deepcopy(f.offline)
        if f.fault == 'offline':
            offline['language']['files']['model.safetensors']['sha256'] = 'c' * 64
        return r.validate_recipe(f.runtime, 'alignment'), offline

    def run_main(self):
        self.f.process.side_effect = self.f.token_process
        return self.m.main(['--release-manifest-sha256', self.f.manifest_sha])

    def test_fresh_alignment_prepare_is_single_token_process_and_remains_gated(self):
        self.assertEqual(self.run_main(), 0)
        self.precheck.assert_called_once_with(self.f.module)
        self.f.process.assert_called_once()
        self.assertEqual(self.f.module.verify_formal_inputs.call_count, 2)
        receipt = json.loads((self.m.OUTPUT / 'CPU_PREPARED_PENDING_SCHEDULER_GATE.json').read_text())
        self.assertEqual(receipt['selected_cell'], 'ouro_r4_alignment_s29')
        self.assertEqual(receipt['initialization']['seed'], 29)
        self.assertEqual(receipt['initialization']['method'], 'two_stage')
        self.assertEqual(receipt['initialization']['kind'], 'pretrained_backbones_random_connector')
        for key in ('alignment_final', 'resume_checkpoint', 'actual_connector_state_sha256'):
            self.assertIsNone(receipt['initialization'][key])
        for key in ('ready_for_gpu_submission', 'gpu_job_submitted', 'scheduler_gate_checked', 'scheduler_dry_run_executed', 'oss_checked', 'oss_mutated'):
            self.assertIs(receipt[key], False)
        self.assertEqual(receipt['capacity']['plan']['required_additional_bytes'], 18_070_438_560)

    def test_initial_capacity_failure_prevents_full_input_model_and_token_work(self):
        self.preview['cells'][0]['capacity_admitted'] = False
        self.assertEqual(self.run_main(), 2)
        self.f.module.verify_formal_inputs.assert_not_called()
        self.f.module.prepare_offline_config.assert_not_called()
        self.f.process.assert_not_called()
        self.assertTrue((self.m.OUTPUT / 'CAPACITY_PRECHECK.json').is_file())
        self.assertTrue((self.m.OUTPUT / 'CPU_PREPARATION_FAILED.json').is_file())
        self.assertFalse((self.m.OUTPUT / 'CPU_INPUTS_VERIFIED_PENDING_CAPACITY.json').exists())

    def test_precheck_must_bind_exact_selected_alignment_cell(self):
        self.preview['cells'][0]['config'] = 'configs/train/declared_v9/ouro_r4_alignment_s43.json'
        self.assertEqual(self.run_main(), 2)
        self.f.module.verify_formal_inputs.assert_not_called()
        self.f.process.assert_not_called()

    def test_selected_config_rejects_wrong_stage_seed_or_model_identity(self):
        f = self.f
        path = f.experiment / self.m.CONFIG_RELATIVE
        binding_path = f.release / 'CONFIG_BINDINGS.json'
        binding = json.loads(binding_path.read_text())
        for key, value in (('num_loops', 2), ('language_model', 'HuggingFaceTB/SmolLM2-1.7B-Instruct')):
            changed = copy.deepcopy(f.raw)
            changed['model'][key] = value
            f.replace_json(path, changed)
            edited = copy.deepcopy(binding)
            for row in edited['configs']:
                if row['config'] == self.m.CONFIG_RELATIVE:
                    row['sha256'] = digest(path)
            f.replace_json(binding_path, edited)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.m.selected_config(f.module)

    def test_full_offline_or_token_failures_never_create_success(self):
        for fault in ('offline', 'token'):
            with self.subTest(fault=fault):
                self.f.fault = fault
                self.m.OUTPUT = self.f.p1 / ('alignment-' + fault)
                self.f.m.OUTPUT = self.m.OUTPUT
                self.f.process.reset_mock()
                self.assertEqual(self.run_main(), 2)
                self.assertEqual(self.f.process.call_count, 0 if fault == 'offline' else 1)
                self.assertFalse((self.m.OUTPUT / 'CPU_PREPARED_PENDING_SCHEDULER_GATE.json').exists())
                receipt = json.loads((self.m.OUTPUT / 'CPU_PREPARATION_FAILED.json').read_text())
                self.assertIs(receipt['ready_for_gpu_submission'], False)

    def test_final_capacity_is_remeasured_and_uses_alignment_limit1(self):
        self.m.OUTPUT.mkdir()
        result = self.m.capacity(self.f.module, self.f.runtime)
        self.assertEqual(result['projected_bytes'], result['combined_accounted_bytes'] + 18_070_438_560 + 7_000_000_000)
        self.f.module.estimate_stage_storage.assert_called_once_with(self.f.runtime, 'alignment')
        self.m.OUTPUT = self.f.p1 / 'final-capacity-failure'
        self.m.OUTPUT.mkdir()
        with patch.object(r.declared, 'staging_bytes', return_value={str(self.f.base): 73_929_561_440, str(self.f.p1): 0}):
            with self.assertRaises(RuntimeError):
                self.m.capacity(self.f.module, self.f.runtime)
        self.assertEqual(json.loads((self.m.OUTPUT / 'CAPACITY_SNAPSHOT.json').read_text())['projected_bytes'], 99_000_000_000)

    def test_untrusted_release_or_gpu_environment_rejected_before_attempt(self):
        with self.assertRaises(ValueError):
            self.m.main(['--release-manifest-sha256', '0' * 64])
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0'}), self.assertRaises(ValueError):
            self.run_main()
        self.assertFalse(self.m.OUTPUT.exists())
        self.f.process.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
