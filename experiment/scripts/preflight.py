#!/usr/bin/env python3
"""Record actual hardware and test a small BF16 attention forward/backward.

This is a compatibility gate, not a model performance benchmark.
No network, credentials, environment-variable dumps, or model downloads.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def matches_expected_gpu(gpu, expected_gpu):
    """Validate the observed device without replacing its reported identity."""
    name = gpu["name"].upper()
    if expected_gpu.upper() == "B300":
        # This cluster reports B300 as NVIDIA-L20D. The alias alone is not
        # evidence: require the datacenter Blackwell architecture and memory.
        known_name = "B300" in name or name.replace("-", " ") == "NVIDIA L20D"
        return (known_name and gpu["compute_capability"] == [10, 3]
                and gpu["total_memory_gib"] >= 240)
    return expected_gpu.upper() in name


def distributed_gate(report, device, cuda, timeout_seconds=60):
    """Collect every rank's errors, including failures before NCCL can run."""
    import torch
    import torch.distributed as dist

    timeout = timedelta(seconds=timeout_seconds)

    def gather_states():
        state = {"rank": dist.get_rank(), "local_rank": report["local_rank"],
                 "cuda_available": cuda, "errors": list(report["errors"]),
                 "attention_forward_backward": report.get("attention_forward_backward", "not_run"),
                 "distributed_all_reduce": report.get("distributed_all_reduce", "not_run")}
        states = [None] * report["world_size"]
        dist.all_gather_object(states, state)
        return states

    try:
        # CPU control messages still work when a rank's CUDA kernel has failed.
        dist.init_process_group("gloo", timeout=timeout)
        states = gather_states()
        if len({state["cuda_available"] for state in states}) != 1:
            report["errors"].append("Distributed ranks disagree on CUDA availability.")
        if any(state["errors"] for state in states) or report["errors"]:
            report["distributed_all_reduce"] = "skipped_due_to_rank_errors"
        else:
            collective_group = None
            try:
                if cuda:
                    collective_group = dist.new_group(backend="nccl", timeout=timeout)
                value = torch.tensor([1.], device=device)
                dist.all_reduce(value, group=collective_group)
                if int(value.item()) != report["world_size"]:
                    raise RuntimeError("Distributed all-reduce returned an unexpected result")
                report["distributed_all_reduce"] = "pass"
            except Exception as exc:
                report["distributed_all_reduce"] = "failed"
                report["errors"].append(f"Collective gate failed: {type(exc).__name__}: {exc}")
            finally:
                if collective_group is not None:
                    try:
                        dist.destroy_process_group(collective_group)
                    except Exception as exc:
                        report["errors"].append(f"NCCL cleanup failed: {type(exc).__name__}: {exc}")
        report["rank_reports"] = gather_states()
        report["errors"] = [f"Rank {state['rank']}: {error}"
                            for state in report["rank_reports"] for error in state["errors"]]
        if report["errors"] and report["distributed_all_reduce"] == "pass":
            report["distributed_all_reduce"] = "failed"
    except Exception as exc:
        report["errors"].append(f"Distributed control gate failed: {type(exc).__name__}: {exc}")
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as exc:
                report["warnings"].append(f"Gloo cleanup failed: {type(exc).__name__}: {exc}")


def inspect(allow_cpu=False, expected_gpu=None):
    import torch
    report = {"status": "not_tested", "utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "platform": platform.system(),
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "gpus": [], "warnings": [], "errors": []}
    report["packages"] = {}
    for package in ("transformers", "accelerate", "huggingface-hub", "av", "numpy", "Pillow"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["errors"].append(f"Missing dependency: {package}")
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        report["nvidia_smi"] = p.stdout.strip() if p.returncode == 0 else "unavailable"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        report["nvidia_smi"] = "unavailable"
    cuda = torch.cuda.is_available()
    if not cuda and not allow_cpu:
        report["errors"].append("CUDA unavailable; GPU experiments must not fall back silently to CPU.")
    if not cuda and expected_gpu:
        report["errors"].append(f"Expected {expected_gpu}, but no CUDA device is available.")
    if cuda:
        report["compiled_arches"] = torch.cuda.get_arch_list()
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu = {"index": i, "name": props.name,
                   "compute_capability": list(torch.cuda.get_device_capability(i)),
                   "total_memory_gib": round(props.total_memory / 2**30, 2)}
            report["gpus"].append(gpu)
            if expected_gpu:
                gpu["expected_gpu"] = expected_gpu
                gpu["expected_gpu_matches"] = matches_expected_gpu(gpu, expected_gpu)
                if expected_gpu == "B300":
                    gpu["identity_rule"] = "B300 or known NVIDIA-L20D label, compute capability 10.3, >=240 GiB"
                if not gpu["expected_gpu_matches"]:
                    report["errors"].append(
                        f"GPU {i} is {props.name}, compute capability {gpu['compute_capability']}, "
                        f"{gpu['total_memory_gib']} GiB; does not match expected {expected_gpu}.")
        if not torch.cuda.is_bf16_supported():
            report["errors"].append("BF16 is not supported by this torch/GPU combination.")
    report["world_size"] = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    report["local_rank"] = local_rank
    device = torch.device(f"cuda:{local_rank}" if cuda else "cpu")
    dtype = torch.bfloat16 if cuda else torch.float32
    if cuda or allow_cpu:
        try:
            if cuda:
                torch.cuda.set_device(device)
            # Exercise library kernels; architecture lists alone can omit compatible PTX.
            q = torch.randn(2, 4, 64, 32, device=device, dtype=dtype, requires_grad=True)
            k = torch.randn_like(q, requires_grad=True)
            v = torch.randn_like(q, requires_grad=True)
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            y.float().square().mean().backward()
            finite = all(bool(torch.isfinite(t.grad).all()) for t in (q, k, v))
            if not finite:
                raise RuntimeError("Nonfinite gradients")
            report["attention_forward_backward"] = "pass"
            if cuda:
                torch.cuda.synchronize()
        except Exception as exc:
            report["attention_forward_backward"] = "failed"
            report["errors"].append(f"Kernel gate failed: {type(exc).__name__}: {exc}")
    if report["world_size"] > 1:
        distributed_gate(report, device, cuda)
    report["status"] = "failed" if report["errors"] else ("gpu_gate_passed" if cuda else "cpu_only")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--expected-gpu", choices=["H20", "B300"])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = inspect(args.allow_cpu, args.expected_gpu)
    if int(os.environ.get("RANK", "0")) == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
