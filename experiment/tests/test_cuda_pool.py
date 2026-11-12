"""CPU mathematical/API contracts; real Triton tests require explicit opt-in."""
import argparse
import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from deepprobe_vlm import benchmark_cuda_pool as benchmark
from deepprobe_vlm import cuda_pool


def direct_bin_reference(states, weights, grid):
    """Independent weighted-sum formula, including overlapping adaptive bins."""
    side = int(states.shape[1] ** .5)
    x = states.float().reshape(states.shape[0], side, side, states.shape[2])
    w = weights.reshape(states.shape[0], side, side, 1)
    result = []
    for y in range(grid):
        for z in range(grid):
            y0, z0 = y * side // grid, z * side // grid
            y1, z1 = ((y + 1) * side + grid - 1) // grid, ((z + 1) * side + grid - 1) // grid
            values = x[:, y0:y1, z0:z1]
            selected = w[:, y0:y1, z0:z1]
            result.append((values * selected).sum((1, 2)) / selected.sum((1, 2)))
    return torch.stack(result, dim=1)


class CudaPoolContractTest(unittest.TestCase):
    def test_separable_reference_matches_direct_bins_for_both_dtypes_and_edges(self):
        for side, grid, channels in ((7, 3, 5), (8, 4, 37), (27, 4, 17), (7, 1, 1), (7, 7, 9)):
            for dtype in (torch.bfloat16, torch.float32):
                with self.subTest(side=side, grid=grid, dtype=dtype):
                    torch.manual_seed(side)
                    states = torch.randn(2, side * side, channels, dtype=dtype)
                    weights = torch.rand(2, side * side, 1).clamp_min(1e-6)
                    actual = cuda_pool.reference_weighted_spatial_pool(states, weights, grid)
                    expected = direct_bin_reference(states, weights, grid)
                    self.assertEqual(actual.dtype, torch.float32)
                    self.assertEqual(tuple(actual.shape), (2, grid * grid, channels))
                    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6)

    def test_overlap_is_preserved_and_weights_are_normalized(self):
        states = torch.zeros(1, 49, 1)
        states[0, 2 * 7 + 2, 0] = 1
        weights = torch.ones(1, 49, 1)
        actual = cuda_pool.reference_weighted_spatial_pool(states, weights, 3)
        self.assertEqual(int((actual > 0).sum()), 4)
        constant = torch.full((1, 49, 1), 3.5)
        unequal = torch.arange(1, 50).float().reshape(1, 49, 1)
        torch.testing.assert_close(cuda_pool.reference_weighted_spatial_pool(constant, unequal, 3),
                                   torch.full((1, 9, 1), 3.5), rtol=2e-6, atol=2e-6)

    def test_reference_preserves_gradient_and_fp32_autocast_contract(self):
        states = torch.randn(2, 49, 5, requires_grad=True)
        weights = torch.rand(2, 49, 1, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = cuda_pool.reference_weighted_spatial_pool(states, weights, 3)
        expected = direct_bin_reference(states, weights, 3)
        self.assertEqual(actual.dtype, torch.float32)
        upstream = torch.randn_like(actual)
        got = torch.autograd.grad(actual, (states, weights), upstream)
        want = torch.autograd.grad(expected, (states, weights), upstream)
        for a, b in zip(got, want):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=3e-6)

    def test_explicit_inference_and_cuda_required_without_importing_backend(self):
        states, weights = torch.randn(2, 9, 5), torch.ones(2, 9, 1)
        with patch.object(cuda_pool.importlib, "import_module") as imported:
            with self.assertRaisesRegex(RuntimeError, "inference-only"):
                cuda_pool.weighted_spatial_pool(states, weights, 2)
            with torch.inference_mode(), self.assertRaisesRegex(ValueError, "requires CUDA"):
                cuda_pool.weighted_spatial_pool(states, weights, 2)
            imported.assert_not_called()
        self.assertIsNone(cuda_pool.last_launch_metadata())

    def test_bad_shapes_dtypes_layouts_and_grids_are_rejected(self):
        states, weights = torch.randn(2, 9, 5), torch.ones(2, 9, 1)
        examples = [(states, weights, 0), (states, weights, True), (states, weights, 4),
                    (states[:, :8], weights[:, :8], 2), (states, weights[:, :, 0], 2),
                    (states.half(), weights, 2), (states, weights.bfloat16(), 2),
                    (states[:, :, ::2], weights, 2), (torch.empty(0, 9, 5), torch.empty(0, 9, 1), 2)]
        for s, w, grid in examples:
            with self.subTest(shape=s.shape, grid=grid), torch.inference_mode():
                with self.assertRaises((ValueError, TypeError)):
                    cuda_pool.weighted_spatial_pool(s, w, grid)

    def test_missing_triton_is_visible_and_does_not_run_reference(self):
        # Dispatch-error test only: bypass the unavailable local CUDA device.
        # It deliberately does not pretend to exercise a real kernel.
        states, weights = torch.randn(1, 9, 5), torch.ones(1, 9, 1)
        with torch.inference_mode(), patch.object(cuda_pool, "_validate_inputs", return_value=3), \
                patch.object(cuda_pool.importlib, "import_module", side_effect=ModuleNotFoundError("triton")), \
                patch.object(cuda_pool, "reference_weighted_spatial_pool") as reference:
            with self.assertRaisesRegex(RuntimeError, "no fallback"):
                cuda_pool.weighted_spatial_pool(states, weights, 2)
            reference.assert_not_called()
        self.assertIsNone(cuda_pool.last_launch_metadata())

    def test_requires_grad_tensors_rejected_even_inside_no_grad(self):
        states, weights = torch.randn(1, 9, 5, requires_grad=True), torch.ones(1, 9, 1)
        with torch.no_grad(), patch.object(cuda_pool, "_validate_inputs", return_value=3):
            with self.assertRaisesRegex(RuntimeError, "requiring gradients"):
                cuda_pool.weighted_spatial_pool(states, weights, 2)

    def test_cpu_import_is_triton_lazy_and_kernel_source_parses(self):
        source_root = str(Path(cuda_pool.__file__).resolve().parents[1])
        code = "import sys; sys.path.insert(0, " + repr(source_root) + "); from deepprobe_vlm import cuda_pool; assert 'triton' not in sys.modules; assert not cuda_pool.torch.cuda.is_initialized()"
        subprocess.run([sys.executable, "-I", "-B", "-c", code], check=True, capture_output=True)
        ast.parse(Path(cuda_pool.__file__).with_name("_triton_pool.py").read_text())


class CudaPoolBenchmarkTest(unittest.TestCase):
    def settings(self, output=None):
        return argparse.Namespace(frames=8, side=27, channels=1152, grid=4,
                                  dtype="bfloat16", warmup=25, repeats=100, profile=False, output=output)

    def test_plan_only_has_real_shape_and_no_gpu_or_timing_claim(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            self.assertEqual(benchmark.main([]), 0)
        data = json.loads(stream.getvalue())
        self.assertEqual(data["status"], "MICROBENCHMARK_PLAN_ONLY")
        self.assertFalse(data["cuda_used"])
        self.assertEqual(data["settings"]["shape"], [8, 729, 1152])
        self.assertFalse(data["settings"]["end_to_end_model_latency_measured"])
        self.assertNotIn("speedup", data)

    def test_plan_cli_does_not_import_torch_or_triton(self):
        root = str(Path(benchmark.__file__).resolve().parents[1])
        code = "import sys; sys.path.insert(0, " + repr(root) + "); from deepprobe_vlm import benchmark_cuda_pool; benchmark_cuda_pool.main([]); assert 'torch' not in sys.modules and 'triton' not in sys.modules"
        subprocess.run([sys.executable, "-I", "-B", "-c", code], check=True, capture_output=True)

    def test_settings_and_quantile_boundaries(self):
        args = self.settings()
        self.assertEqual(benchmark._validate_settings(args)["shape"], [8, 729, 1152])
        for key, value in (("side", 0), ("grid", 28), ("warmup", 0), ("repeats", 10001), ("channels", 100000)):
            changed = self.settings()
            setattr(changed, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                benchmark._validate_settings(changed)
        result = benchmark._summary([1, 2, 3, 4])
        self.assertEqual(result["median"], 2.5)
        self.assertAlmostEqual(result["p95"], 3.85)

    def test_unavailable_cuda_retains_failure_and_refuses_output_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.settings(Path(temp).resolve() / "attempt")
            settings = benchmark._validate_settings(args)
            with patch.object(torch.cuda, "is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
                    benchmark.execute(args, settings)
            self.assertTrue((args.output / "INTENT.json").is_file())
            self.assertTrue((args.output / "FAILURE.json").is_file())
            self.assertFalse((args.output / "SUMMARY.json").exists())
            with self.assertRaises(FileExistsError):
                benchmark.execute(args, settings)


@unittest.skipUnless(os.environ.get("DEEPPROBE_TEST_TRITON_POOL") == "1",
                     "real CUDA kernels require explicit allocated-job opt-in")
class CudaPoolRealGpuTest(unittest.TestCase):
    def test_real_kernel_numerics_overlap_channel_tail_and_model_shape(self):
        if not torch.cuda.is_available():
            self.fail("explicit GPU test requested without a CUDA device")
        torch.backends.cuda.matmul.allow_tf32 = False
        for frames, side, channels, grid in ((2, 7, 37, 3), (8, 27, 1152, 4), (1, 27, 113, 1), (1, 7, 19, 7)):
            for dtype in (torch.bfloat16, torch.float32):
                with self.subTest(shape=(frames, side, channels), dtype=dtype), torch.inference_mode():
                    torch.manual_seed(17)
                    states = torch.randn(frames, side * side, channels, device="cuda", dtype=dtype)
                    weights = torch.randn(frames, side * side, 1, device="cuda", dtype=dtype).sigmoid().float().clamp_min(1e-6)
                    expected = cuda_pool.reference_weighted_spatial_pool(states, weights, grid)
                    actual = cuda_pool.weighted_spatial_pool(states, weights, grid)
                    torch.cuda.synchronize()
                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
                    self.assertFalse(cuda_pool.last_launch_metadata()["fallback_used"])
                    self.assertEqual(cuda_pool.last_launch_metadata()["backend"], "triton")


if __name__ == "__main__":
    unittest.main()
