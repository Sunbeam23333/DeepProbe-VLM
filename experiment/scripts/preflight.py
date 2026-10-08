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
from datetime import datetime, timezone
from pathlib import Path


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
            report["gpus"].append({"index": i, "name": props.name,
                                   "compute_capability": list(torch.cuda.get_device_capability(i)),
                                   "total_memory_gib": round(props.total_memory / 2**30, 2)})
            if expected_gpu and expected_gpu.lower() not in props.name.lower():
                report["errors"].append(f"GPU {i} is {props.name}, not expected {expected_gpu}.")
        if not torch.cuda.is_bf16_supported():
            report["errors"].append("BF16 is not supported by this torch/GPU combination.")
    report["world_size"] = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
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
            if report["world_size"] > 1:
                import torch.distributed as dist
                dist.init_process_group("nccl" if cuda else "gloo")
                value = torch.tensor([1.], device=device)
                dist.all_reduce(value)
                if int(value.item()) != report["world_size"]:
                    raise RuntimeError("Distributed all-reduce returned an unexpected result")
                report["distributed_all_reduce"] = "pass"
                dist.destroy_process_group()
        except Exception as exc:
            report["errors"].append(f"Kernel/collective gate failed: {type(exc).__name__}: {exc}")
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
