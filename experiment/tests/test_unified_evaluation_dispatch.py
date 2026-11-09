"""Exercise explicit R4/Smol dispatch without loading models or benchmark media."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from deepprobe_vlm import evaluate, evaluate_tempcompass
from unified_evaluation_fixtures import fixture, m, write
import run_tempcompass_eval as runner


class UnifiedTempCompassDispatchTests(unittest.TestCase):
    def run_worker(self, root, family, loops, seed, *, wrong_seed=False):
        binding = fixture(root / 'training', family=family, loops=loops, seed=seed)
        name = binding['model_id']
        accepted = write(root / 'accepted.json', {'synthetic_fixture_only': True})
        plan = {'run_root': str(root / 'evaluation'), 'model_sequence': [name],
                'model_bindings': {name: binding}, 'accepted_data': accepted,
                'media_root': '/synthetic/no-media-loaded'}
        plan_ref = write(root / 'PLAN.json', plan)
        provenance = m.verify_registered_checkpoint(binding)
        source_sha = 'f' * 64
        provenance.update(model=name, plan_sha256=plan_ref['sha256'], protocol=runner.PROTOCOL,
                          reference_engine_protocol='reference', manifest_sha256=runner.PINS['manifest'],
                          source_files_sha256=source_sha, accepted_data_sha256=accepted['sha256'])
        if wrong_seed:
            provenance['training_seed'] = 43
        proof_path = root / 'evaluation' / name / 'MODEL_PROVENANCE.json'
        write(proof_path, provenance)
        data = {'records': [], 'training_sources': [], 'manifest_path': '/synthetic/manifest.jsonl'}
        environment = {'TEMP_COMPASS_ADMITTED_PLAN_SHA256': plan_ref['sha256'], 'RANK': '0', 'WORLD_SIZE': '8'}
        with patch.dict(os.environ, environment), \
                patch.object(runner, 'validate_plan', return_value=plan), \
                patch.object(runner, 'verify_sources', return_value=source_sha), \
                patch.object(runner, 'pinned', return_value=Path(accepted['path'])), \
                patch.object(runner, 'verify_accepted_data', return_value=data), \
                patch.object(runner, 'check_training_sources') as contamination, \
                patch.object(runner, 'json_once') as receipt, \
                patch.object(evaluate, 'main', return_value=73) as engine:
            argv = ['--plan', plan_ref['path'], '--plan-sha256', plan_ref['sha256'],
                    '--model', name, '--provenance', str(proof_path)]
            if wrong_seed:
                with self.assertRaisesRegex(ValueError, 'runner_model_provenance_changed'):
                    evaluate_tempcompass.main(argv)
                engine.assert_not_called()
                receipt.assert_not_called()
            else:
                self.assertEqual(evaluate_tempcompass.main(argv), 73)
                contamination.assert_called_once_with(binding['checkpoint']['path'], [], [])
                engine.assert_called_once_with([
                    '--checkpoint', binding['checkpoint']['path'], '--manifest', data['manifest_path'],
                    '--media-root', plan['media_root'], '--output',
                    str(root / 'evaluation' / name / 'raw/predictions.jsonl'),
                    '--max-new-tokens', '16', '--protocol', 'reference', '--distributed-env',
                    '--run-provenance', str(proof_path)])
                self.assertEqual(receipt.call_args.args[1]['max_new_tokens'], 16)

    def test_r4_and_smol_retain_reference_generation_protocol(self):
        for family, loops, seed in [('ouro', 4, 17), ('smollm2', 1, 29)]:
            with self.subTest(family=family), tempfile.TemporaryDirectory() as folder:
                self.run_worker(Path(folder).resolve(), family, loops, seed)

    def test_model_provenance_cannot_relabel_training_seed(self):
        with tempfile.TemporaryDirectory() as folder:
            self.run_worker(Path(folder).resolve(), 'smollm2', 1, 29, wrong_seed=True)


if __name__ == '__main__':
    unittest.main()
