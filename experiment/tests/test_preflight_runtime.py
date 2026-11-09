"""Hardware identity and multi-rank failure reporting, without GPU dependencies."""
import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("preflight_runtime", ROOT / "scripts/preflight.py")
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


class PreflightRuntimeTest(unittest.TestCase):
    def test_b300_alias_requires_architecture_and_memory(self):
        cases = [
            ("NVIDIA B300", [10, 3], 268.0, True),
            ("NVIDIA-L20D", [10, 3], 268.0, True),
            ("NVIDIA-L20D", [8, 9], 48.0, False),
            ("NVIDIA-L20D", [12, 0], 268.0, False),
            ("NVIDIA-L20D", [10, 3], 32.0, False),
            ("NVIDIA GeForce RTX 5090", [12, 0], 32.0, False),
            ("NVIDIA B300", [12, 0], 32.0, False),
            ("Unknown device", [10, 3], 268.0, False),
        ]
        for name, capability, memory, expected in cases:
            with self.subTest(name=name, capability=capability, memory=memory):
                gpu = {"name": name, "compute_capability": capability, "total_memory_gib": memory}
                original = copy.deepcopy(gpu)
                self.assertEqual(PREFLIGHT.matches_expected_gpu(gpu, "B300"), expected)
                self.assertEqual(gpu, original)

    def test_h20_identity_remains_supported(self):
        self.assertTrue(PREFLIGHT.matches_expected_gpu({"name": "NVIDIA H20"}, "H20"))
        self.assertFalse(PREFLIGHT.matches_expected_gpu({"name": "NVIDIA B300"}, "H20"))

    def distributed_context(self, before=None, after=None, rank=0, cuda=True):
        """Simulate the states contributed by seven other workers."""
        world_size = 8
        report = {"world_size": world_size, "local_rank": rank, "errors": [],
                  "warnings": [], "attention_forward_backward": "pass"}
        dist = types.ModuleType("torch.distributed")
        dist.init_process_group = Mock()
        dist.is_initialized = Mock(return_value=True)
        dist.get_rank = Mock(return_value=rank)
        dist.new_group = Mock(return_value="nccl-group")
        dist.all_reduce = Mock()
        dist.destroy_process_group = Mock()
        calls = []

        def gather(states, local):
            overrides = before if not calls else after
            calls.append(copy.deepcopy(local))
            for peer in range(world_size):
                states[peer] = {"rank": peer, "local_rank": peer, "cuda_available": cuda,
                                "errors": [], "attention_forward_backward": "pass",
                                "distributed_all_reduce": "not_run" if len(calls) == 1 else "pass"}
                states[peer].update((overrides or {}).get(peer, {}))
            states[rank] = local

        dist.all_gather_object = Mock(side_effect=gather)
        torch = types.ModuleType("torch")
        torch.distributed = dist
        torch.tensor = Mock(return_value=types.SimpleNamespace(item=lambda: world_size))
        return report, dist, patch.dict(sys.modules, {"torch": torch, "torch.distributed": dist})

    def test_nonzero_rank_kernel_failure_is_reported_by_rank_zero(self):
        failure = {3: {"errors": ["Kernel gate failed: unsupported architecture"],
                       "attention_forward_backward": "failed"}}
        report, dist, modules = self.distributed_context(before=failure, after=failure)
        with modules:
            PREFLIGHT.distributed_gate(report, "cuda:0", True)
        self.assertEqual(report["distributed_all_reduce"], "skipped_due_to_rank_errors")
        self.assertIn("Rank 3: Kernel gate failed: unsupported architecture", report["errors"])
        self.assertEqual(len(report["rank_reports"]), 8)
        dist.new_group.assert_not_called()
        dist.all_reduce.assert_not_called()
        self.assertLessEqual(dist.init_process_group.call_args.kwargs["timeout"].total_seconds(), 60)

    def test_nonzero_rank_collective_failure_overrides_local_pass(self):
        failure = {7: {"errors": ["Collective gate failed: communication error"],
                       "distributed_all_reduce": "failed"}}
        report, dist, modules = self.distributed_context(after=failure)
        with modules:
            PREFLIGHT.distributed_gate(report, "cuda:0", True)
        self.assertEqual(report["distributed_all_reduce"], "failed")
        self.assertIn("Rank 7: Collective gate failed: communication error", report["errors"])
        dist.all_reduce.assert_called_once()

    def test_nonzero_rank_receives_global_failure_too(self):
        failure = {0: {"errors": ["Missing dependency: av"]}}
        report, dist, modules = self.distributed_context(before=failure, after=failure, rank=7)
        with modules:
            PREFLIGHT.distributed_gate(report, "cuda:7", True)
        self.assertIn("Rank 0: Missing dependency: av", report["errors"])
        dist.all_reduce.assert_not_called()

    def test_cuda_and_cpu_workers_cannot_claim_a_distributed_pass(self):
        report, dist, modules = self.distributed_context(before={7: {"cuda_available": False}})
        with modules:
            PREFLIGHT.distributed_gate(report, "cuda:0", True)
        self.assertTrue(any("disagree on CUDA availability" in error for error in report["errors"]))
        dist.all_reduce.assert_not_called()

    def test_successful_cpu_gate_uses_gloo_and_collects_all_ranks(self):
        report, dist, modules = self.distributed_context(cuda=False)
        with modules:
            PREFLIGHT.distributed_gate(report, "cpu", False)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["distributed_all_reduce"], "pass")
        self.assertEqual(len(report["rank_reports"]), 8)
        self.assertEqual(dist.init_process_group.call_args.args, ("gloo",))
        dist.new_group.assert_not_called()
        dist.all_reduce.assert_called_once()

    def test_failed_error_gather_cannot_claim_success(self):
        report, dist, modules = self.distributed_context()
        dist.all_gather_object.side_effect = RuntimeError("control channel timeout")
        with modules:
            PREFLIGHT.distributed_gate(report, "cuda:0", True)
        self.assertTrue(any("control channel timeout" in error for error in report["errors"]))
        dist.all_reduce.assert_not_called()
        dist.destroy_process_group.assert_called_once()


class TrainingLauncherTest(unittest.TestCase):
    def launch_arguments(self, **overrides):
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "torchrun"
            fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
            fake.chmod(0o755)
            env = dict(os.environ)
            for key in ("MASTER_ADDR", "MASTER_PORT", "DEEP_PROBE_GPUS"):
                env.pop(key, None)
            env.update(overrides)
            env["PATH"] = directory + os.pathsep + env.get("PATH", "")
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/launch/train.sh"), "config with spaces.json",
                 "--output-dir", "output with spaces"],
                check=True, env=env, capture_output=True, text=True)
            return result.stdout.splitlines()

    def test_default_rendezvous_uses_loopback_and_preserves_arguments(self):
        args = self.launch_arguments()
        self.assertNotIn("--standalone", args)
        self.assertIn("--master-addr=127.0.0.1", args)
        self.assertIn("--master-port=29500", args)
        self.assertIn("--nproc-per-node=8", args)
        self.assertEqual(args[-6:], ["-m", "deepprobe_vlm.train", "--config",
                                   "config with spaces.json", "--output-dir", "output with spaces"])

    def test_explicit_rendezvous_and_gpu_count_are_honored(self):
        args = self.launch_arguments(MASTER_ADDR="127.0.0.2", MASTER_PORT="29483", DEEP_PROBE_GPUS="1")
        self.assertIn("--master-addr=127.0.0.2", args)
        self.assertIn("--master-port=29483", args)
        self.assertIn("--nproc-per-node=1", args)


if __name__ == "__main__":
    unittest.main()
