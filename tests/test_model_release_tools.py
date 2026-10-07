"""Configuration and orchestration checks requiring no model downloads."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from transformers import TrainingArguments

from deepprobe_vlm.modeling import DeepProbeConfig

ROOT = Path(__file__).resolve().parents[1]


class ReleaseToolsTest(unittest.TestCase):
    def test_all_training_configs_are_accepted(self):
        for path in (ROOT/"configs/train").glob("*.json"):
            raw = json.loads(path.read_text())
            DeepProbeConfig.from_dict(raw)
            settings = dict(raw["training"])
            settings.pop("stage")
            # Validate recipe syntax without pretending this CPU has BF16 CUDA.
            settings.update(bf16=False, use_cpu=True)
            TrainingArguments(output_dir="unused", **settings)

    def test_ablation_generator_never_emits_fake_ordinary_loops(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, PYTHONPATH=str(ROOT/"src"))
            subprocess.run([sys.executable, "scripts/make_ablation_configs.py", "--base",
                            "configs/train/smollm2_sft.json", "--output-dir", directory],
                           cwd=ROOT, env=env, check=True, capture_output=True)
            paths = list(Path(directory).glob("*.json"))
            self.assertEqual(len(paths), 27)
            for path in paths:
                config = DeepProbeConfig.from_dict(json.loads(path.read_text()))
                self.assertEqual(config.num_loops, 1)

    def test_cpu_preflight_cannot_claim_expected_gpu(self):
        spec = importlib.util.spec_from_file_location("preflight_test", ROOT/"scripts/preflight.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from unittest.mock import patch
        with patch("torch.cuda.is_available", return_value=False):
            report = module.inspect(allow_cpu=True, expected_gpu="H20")
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("Expected H20" in item for item in report["errors"]))


if __name__ == "__main__":
    unittest.main()
