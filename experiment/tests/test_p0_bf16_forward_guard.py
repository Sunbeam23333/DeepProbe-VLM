"""P0-05 accepts observed CUDA BF16 forwards from every expected rank."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


OPERATIONS = Path(__file__).resolve().parents[1] / "scripts/operations"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, OPERATIONS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TRAIN = load_script("deterministic_train")
RUNNER = load_script("run_p0_05")


class ToyModel:
    training = True

    def forward(self, *args, **kwargs):
        return "forward_called"


class P0BF16ForwardGuardTest(unittest.TestCase):
    @staticmethod
    def fake_torch(rank=0, world_size=1, *, enabled=True, dtype=None):
        bf16 = object()
        distributed = SimpleNamespace(
            is_available=lambda: True, is_initialized=lambda: world_size > 1,
            get_rank=lambda: rank, get_world_size=lambda: world_size)
        return SimpleNamespace(
            bfloat16=bf16, distributed=distributed, is_grad_enabled=lambda: True,
            is_autocast_enabled=lambda device: enabled and device == "cuda",
            get_autocast_dtype=lambda device: bf16 if dtype is None else dtype)

    @staticmethod
    def input_ids(device="cuda"):
        return SimpleNamespace(device=SimpleNamespace(type=device))

    def test_eight_observed_ranks_are_required_and_hash_inventoried(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            for rank in range(8):
                torch = self.fake_torch(rank=rank, world_size=8)
                observed, original = TRAIN.install_bf16_forward_guard(ToyModel, torch, run_dir)
                try:
                    self.assertEqual(ToyModel().forward(input_ids=self.input_ids()), "forward_called")
                    self.assertEqual(observed["training_forwards"], 1)
                finally:
                    ToyModel.forward = original
                if rank == 6:
                    with self.assertRaisesRegex(RuntimeError, "missing"):
                        RUNNER.verify_bf16_forward_receipts(run_dir, 8)
            result = RUNNER.verify_bf16_forward_receipts(run_dir, 8)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(set(result["ranks"]), {str(rank) for rank in range(8)})
            bad = run_dir / "P0_BF16_FORWARD_RANK7.json"
            tampered = json.loads(bad.read_text())
            tampered["cuda_autocast_dtype"] = "torch.float16"
            bad.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(RuntimeError, "content differs"):
                RUNNER.verify_bf16_forward_receipts(run_dir, 8)

    def test_runtime_guard_rejects_fp16_disabled_autocast_and_cpu(self):
        for enabled, dtype, device in ((False, None, "cuda"),
                                       (True, "torch.float16", "cuda"),
                                       (True, None, "cpu")):
            with self.subTest(enabled=enabled, dtype=dtype, device=device), \
                    tempfile.TemporaryDirectory() as directory:
                torch = self.fake_torch(enabled=enabled, dtype=dtype)
                observed, original = TRAIN.install_bf16_forward_guard(
                    ToyModel, torch, Path(directory))
                try:
                    with self.assertRaisesRegex(RuntimeError, "CUDA BF16 autocast"):
                        ToyModel().forward(input_ids=self.input_ids(device))
                    self.assertEqual(observed["training_forwards"], 0)
                    self.assertEqual(list(Path(directory).iterdir()), [])
                finally:
                    ToyModel.forward = original

    def test_entry_rejects_non_bf16_recipe_before_importing_trainer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for training in ({"bf16": False}, {"bf16": True, "fp16": True},
                             {"bf16": "true", "fp16": False}):
                with self.subTest(training=training):
                    config = root / "config.json"
                    config.write_text(json.dumps({"training": training}))
                    argv = ["deterministic_train.py", "--config", str(config),
                            "--output-dir", str(root / "out")]
                    with mock.patch.object(TRAIN, "configure", return_value={}), \
                            mock.patch.object(sys, "argv", argv), \
                            contextlib.redirect_stdout(io.StringIO()), \
                            self.assertRaisesRegex(RuntimeError, "bf16=true and fp16=false"):
                        TRAIN.main()


if __name__ == "__main__":
    unittest.main()
