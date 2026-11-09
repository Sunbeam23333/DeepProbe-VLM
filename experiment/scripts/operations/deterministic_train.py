#!/usr/bin/env python3
"""Run the normal training CLI with reproducibility-focused CUDA settings.

This entry point is reserved for the P0-05 equivalence experiment.  It does not
change model, optimizer, data or schedule settings.  The surrounding runner
also fixes ``PYTHONHASHSEED``, cuBLAS workspace configuration and NCCL order
before Python starts.
"""
from __future__ import annotations

import argparse
import os
import json
from pathlib import Path


PRECISION_RECEIPT_PREFIX = "P0_BF16_FORWARD_RANK"


def install_bf16_forward_guard(model_class, torch_module, output_dir: Path):
    """Prove actual CUDA BF16 autocast during every P0 training forward.

    A config flag or saved TrainingArguments alone cannot establish the dtype
    used by Accelerate's forward wrapper.  The first observation from each DDP
    rank is written separately so the outer P0 runner can require all ranks.
    """
    original_forward = model_class.forward
    observed = {"training_forwards": 0}

    def guarded_forward(model, *args, **kwargs):
        if model.training and torch_module.is_grad_enabled():
            input_ids = kwargs.get("input_ids") if "input_ids" in kwargs else (args[0] if args else None)
            input_device = getattr(getattr(input_ids, "device", None), "type", None)
            enabled = torch_module.is_autocast_enabled("cuda")
            dtype = torch_module.get_autocast_dtype("cuda")
            if input_device != "cuda" or not enabled or dtype != torch_module.bfloat16:
                raise RuntimeError(
                    "P0-05 training forward requires CUDA input and active CUDA BF16 autocast; "
                    f"observed input_device={input_device}, autocast_enabled={enabled}, dtype={dtype}"
                )
            if observed["training_forwards"] == 0:
                distributed = torch_module.distributed
                if distributed.is_available() and distributed.is_initialized():
                    rank, world_size = distributed.get_rank(), distributed.get_world_size()
                else:
                    rank, world_size = 0, 1
                output_dir.mkdir(parents=True, exist_ok=True)
                receipt = output_dir / f"{PRECISION_RECEIPT_PREFIX}{rank}.json"
                with receipt.open("x", encoding="utf-8") as stream:
                    json.dump({
                        "schema_version": 1,
                        "status": "cuda_bf16_autocast_observed_on_training_forward",
                        "rank": rank,
                        "world_size": world_size,
                        "input_device": "cuda",
                        "cuda_autocast_enabled": True,
                        "cuda_autocast_dtype": "torch.bfloat16",
                        "output_dir": str(output_dir.resolve()),
                    }, stream, indent=2, sort_keys=True, allow_nan=False)
                    stream.write("\n")
            observed["training_forwards"] += 1
        return original_forward(model, *args, **kwargs)

    model_class.forward = guarded_forward
    return observed, original_forward


def configure() -> dict[str, object]:
    # CUBLAS_WORKSPACE_CONFIG must be present before a CUDA context is created.
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("P0-05 requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    if os.environ.get("PYTHONHASHSEED") is None:
        raise RuntimeError("P0-05 requires an explicit PYTHONHASHSEED before interpreter start")
    import torch

    # Abort if a CUDA kernel lacks a deterministic implementation.  The
    # tensor/state comparison remains the acceptance gate after training.
    torch.use_deterministic_algorithms(True, warn_only=False)
    # Force scaled-dot-product attention through the math backend.  cuDNN
    # attention backward can be nondeterministic even with cudnn.deterministic.
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    return {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
            "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "python_hash_seed": os.environ["PYTHONHASHSEED"],
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def main() -> int:
    settings = configure()
    print("P0_05_REPRODUCIBILITY " + json.dumps(settings, sort_keys=True), flush=True)
    cli = argparse.ArgumentParser(add_help=False)
    cli.add_argument("--config", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path, required=True)
    options, _ = cli.parse_known_args()
    recipe = json.loads(options.config.read_text())
    training = recipe.get("training", {})
    if training.get("bf16") is not True or training.get("fp16", False) is not False:
        raise RuntimeError("P0-05 training requires bf16=true and fp16=false")
    import torch
    from deepprobe_vlm.modeling import DeepProbeVLM
    from deepprobe_vlm.train import main as train_main

    observed, original_forward = install_bf16_forward_guard(
        DeepProbeVLM, torch, options.output_dir)
    try:
        status = train_main()
    finally:
        DeepProbeVLM.forward = original_forward
    if status == 0 and observed["training_forwards"] == 0:
        raise RuntimeError("P0-05 training finished without an observed CUDA BF16 forward")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
