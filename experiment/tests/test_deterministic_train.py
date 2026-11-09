"""P0-05 rejects nondeterministic kernels and selects math attention."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/operations/deterministic_train.py"
SPEC = importlib.util.spec_from_file_location("deterministic_train", SCRIPT)
DETERMINISTIC_TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DETERMINISTIC_TRAIN)


class DeterministicTrainingSettingsTest(unittest.TestCase):
    def test_strict_determinism_and_math_only_sdpa(self):
        cuda = torch.backends.cuda
        cudnn = torch.backends.cudnn
        original = {
            "algorithms": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_sdp": cuda.cudnn_sdp_enabled(),
            "flash_sdp": cuda.flash_sdp_enabled(),
            "mem_efficient_sdp": cuda.mem_efficient_sdp_enabled(),
            "math_sdp": cuda.math_sdp_enabled(),
            "matmul_tf32": cuda.matmul.allow_tf32,
            "cudnn_tf32": cudnn.allow_tf32,
            "cudnn_benchmark": cudnn.benchmark,
            "cudnn_deterministic": cudnn.deterministic,
            "matmul_precision": torch.get_float32_matmul_precision(),
        }
        try:
            with mock.patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                                           "PYTHONHASHSEED": "17"}):
                settings = DETERMINISTIC_TRAIN.configure()
            self.assertTrue(settings["deterministic_algorithms"])
            self.assertFalse(settings["deterministic_warn_only"])
            self.assertFalse(settings["cudnn_sdp_enabled"])
            self.assertFalse(settings["flash_sdp_enabled"])
            self.assertFalse(settings["mem_efficient_sdp_enabled"])
            self.assertTrue(settings["math_sdp_enabled"])
            self.assertFalse(settings["cuda_matmul_allow_tf32"])
            self.assertFalse(settings["cudnn_allow_tf32"])
            self.assertTrue(settings["cudnn_deterministic"])
        finally:
            torch.use_deterministic_algorithms(original["algorithms"], warn_only=original["warn_only"])
            cuda.enable_cudnn_sdp(original["cudnn_sdp"])
            cuda.enable_flash_sdp(original["flash_sdp"])
            cuda.enable_mem_efficient_sdp(original["mem_efficient_sdp"])
            cuda.enable_math_sdp(original["math_sdp"])
            cuda.matmul.allow_tf32 = original["matmul_tf32"]
            cudnn.allow_tf32 = original["cudnn_tf32"]
            cudnn.benchmark = original["cudnn_benchmark"]
            cudnn.deterministic = original["cudnn_deterministic"]
            torch.set_float32_matmul_precision(original["matmul_precision"])


if __name__ == "__main__":
    unittest.main()
