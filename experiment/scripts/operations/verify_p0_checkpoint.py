#!/usr/bin/env python3
"""Single-sample P0 engineering diagnostics; never a quality measurement.

Use local, already staged assets. Runs one SFT forward/backward without an
optimizer or training step, then compares fixed-length cached/uncached greedy
decoding. An existing checkpoint is loaded twice sequentially for a bounded
logits reproducibility check; this does not certify a fresh save round trip.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--media-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True, help="New JSON report; never overwritten")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                   help="CPU is explicit and intended for tiny tests; no automatic fallback")
    p.add_argument("--max-new-tokens", type=int, default=4, help="Fixed decoding length, 2–16 tokens")
    return p


def validate_arguments(args):
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite diagnostic report: {args.output}")
    if not 2 <= args.max_new_tokens <= 16:
        raise ValueError("max-new-tokens must be 2–16 to exercise cache with a bounded workload")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Run this diagnostic as a single process on one GPU, without torchrun")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gradient_report(model):
    """All effective parameters must receive finite gradients; each group flows."""
    import torch
    result = {}
    for group, module in (("language", model.language_model), ("vision", model.vision_model),
                          ("pool_projector", model.projector)):
        values = {"trainable_parameters": 0, "trainable_tensors": 0, "nonzero_gradient_tensors": 0,
                  "missing_gradient_names": [], "nonfinite_gradient_names": [],
                  "zero_gradient_names": [], "frozen_parameter_names": [],
                  "unexpected_frozen_gradient_names": [], "max_abs_gradient": 0.0}
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                values["frozen_parameter_names"].append(name)
                if parameter.grad is not None:
                    values["unexpected_frozen_gradient_names"].append(name)
                continue
            values["trainable_parameters"] += parameter.numel()
            values["trainable_tensors"] += 1
            grad = parameter.grad
            if grad is None:
                values["missing_gradient_names"].append(name)
            elif not bool(torch.isfinite(grad).all()):
                values["nonfinite_gradient_names"].append(name)
            else:
                maximum = grad.detach().abs().max().item()
                values["max_abs_gradient"] = max(values["max_abs_gradient"], maximum)
                if maximum > 0:
                    values["nonzero_gradient_tensors"] += 1
                else:
                    values["zero_gradient_names"].append(name)
        invalid = any(values[key] for key in ("missing_gradient_names", "nonfinite_gradient_names",
                                               "unexpected_frozen_gradient_names"))
        if not values["trainable_tensors"]:
            expected_frozen = group == "vision" and model.deep_config.freeze_vision
            values["status"] = "expected_frozen" if expected_frozen and not invalid else "failed"
        else:
            values["status"] = "passed" if not invalid and values["nonzero_gradient_tensors"] else "failed"
        result[group] = values
    return result


def loaded_model_checks(model, train_batch, prompt_batch, *, bf16, max_new_tokens, checkpointing, result=None):
    """Keep model loading separate so tiny, local models can exercise the gate."""
    import torch
    if result is None:
        result = {}

    def precision():
        return torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext()

    model.set_stage("sft")
    model.train()
    if checkpointing:
        model.gradient_checkpointing_enable({"use_reentrant": False})
    model.zero_grad(set_to_none=True)
    with precision():
        output = model(**train_batch)
    finite_loss = bool(torch.isfinite(output.loss))
    result.update(loss_finite=finite_loss,
                  single_sample_sft_loss=output.loss.item() if finite_loss else None,
                  loss_interpretation="Engineering finite-loss check only; not training progress or quality.",
                  optimizer_steps=0, gradient_checkpointing=checkpointing)
    if not finite_loss:
        raise ValueError("SFT forward produced a nonfinite loss")
    output.loss.backward()
    result["gradient_groups"] = gradient_report(model)
    # Release the backward graph and gradients before either inference pass.
    del output
    model.zero_grad(set_to_none=True)
    for module in (model.language_model, model.vision_model):
        disable = getattr(module, "gradient_checkpointing_disable", None)
        if disable is not None:
            disable()
    model.eval()
    with torch.no_grad(), precision():
        reference = model(**prompt_batch).logits[:, -1].detach().float().cpu()
        cached = model.greedy_generate(**prompt_batch, max_new_tokens=max_new_tokens,
                                       eos_token_id=None, use_cache=True).cpu()
        uncached = model.greedy_generate(**prompt_batch, max_new_tokens=max_new_tokens,
                                         eos_token_id=None, use_cache=False).cpu()
    result["cache_equivalence"] = {
        "status": "passed" if torch.equal(cached, uncached) else "failed",
        "fixed_new_tokens": max_new_tokens, "eos_early_stop": False,
        "cached_token_ids": cached.tolist(), "uncached_token_ids": uncached.tolist(),
        "comparison": "Exact greedy token IDs on one prompt; not answer correctness or a speed benchmark."}
    if not bool(torch.isfinite(reference).all()):
        raise ValueError("Prompt logits are nonfinite")
    result["prompt_logits_finite"] = True
    return result, reference


def run(args, report):
    # Set offline mode before importing Transformers or Hugging Face Hub.
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    import torch
    from transformers import AutoImageProcessor, AutoTokenizer, set_seed
    from deepprobe_vlm.data import MultimodalCollator, VideoManifestDataset
    from deepprobe_vlm.manifest import load_manifest
    from deepprobe_vlm.modeling import DeepProbeConfig, DeepProbeVLM

    raw = json.loads(args.config.read_text())
    config = DeepProbeConfig.from_dict(raw)
    settings = raw.get("training", {})
    if settings.get("stage", "sft") != "sft":
        raise ValueError("P0 gradient diagnosis requires an SFT configuration")
    if args.checkpoint:
        saved = DeepProbeConfig.from_dict(json.loads((args.checkpoint / "deepprobe_config.json").read_text()))
        if saved != config:
            raise ValueError("Checkpoint and requested model configuration must match exactly")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing to run the GPU diagnostic on CPU")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    bf16 = bool(settings.get("bf16", True))
    if bf16 and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 recipe needs a compatible CUDA GPU; tiny CPU configs must set bf16=false")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    set_seed(settings.get("seed", 17))
    records = load_manifest(args.train_manifest, media_root=args.media_root)
    if any(record["split"] != "train" for record in records):
        raise ValueError("Only a train manifest is allowed; do not diagnose gradients on validation/test sources")
    record = records[0]
    report.update(model=asdict(config), config_sha256=sha256_file(args.config),
                  manifest_sha256=sha256_file(args.train_manifest),
                  sample={key: record[key] for key in ("sample_id", "source_video_id", "split")},
                  selection="First manifest record, without replacement or token truncation",
                  checkpoint=str(args.checkpoint) if args.checkpoint else None,
                  device=str(device), torch=torch.__version__, cuda_runtime=torch.version.cuda,
                  parameter_dtype="float32", autocast_dtype="bfloat16" if bf16 else None)
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        report["gpu"] = {"name": properties.name, "total_memory_bytes": properties.total_memory,
                         "compute_capability": list(torch.cuda.get_device_capability(device))}
    model = DeepProbeVLM.from_backbones(config, checkpoint=args.checkpoint, torch_dtype=torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.checkpoint / "tokenizer") if args.checkpoint else config.language_model,
        revision=None if args.checkpoint else config.language_revision,
        trust_remote_code=False, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(
        str(args.checkpoint / "image_processor") if args.checkpoint else config.vision_model,
        revision=None if args.checkpoint else config.vision_revision, use_fast=False, local_files_only=True)
    dataset = VideoManifestDataset([record], processor, config.num_frames, args.media_root)
    example = dataset[0]
    batches = []
    for include_answer in (True, False):
        collator = MultimodalCollator(tokenizer, num_visual_tokens=config.num_visual_tokens,
                                     max_text_tokens=config.max_text_tokens, include_answer=include_answer)
        batches.append({key: value.to(device) for key, value in collator([example]).items()})
    train_batch, prompt_batch = batches
    report["input"] = {"batch_size": 1, "num_frames": config.num_frames,
                       "visual_tokens": config.num_visual_tokens,
                       "sft_sequence_tokens": train_batch["input_ids"].shape[1],
                       "prompt_sequence_tokens": prompt_batch["input_ids"].shape[1],
                       "supervised_tokens": int((train_batch["labels"][:, 1:] != -100).sum()),
                       "generation_prompt_includes_gold_answer": False}
    report["parameter_counts"] = model.parameter_counts()
    report["checks"] = {}
    checks, reference = loaded_model_checks(
        model, train_batch, prompt_batch, bf16=bf16, max_new_tokens=args.max_new_tokens,
        checkpointing=bool(settings.get("gradient_checkpointing", True)), result=report["checks"])
    for group, value in checks["gradient_groups"].items():
        if value["status"] == "failed":
            report["errors"].append(f"Gradient diagnostics failed for {group}")
    if checks["cache_equivalence"]["status"] != "passed":
        report["errors"].append("Cached and uncached greedy token IDs differ")
    # Never hold two full models or save another weights directory for this gate.
    del model, train_batch, batches, example, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    report["checkpoint_reload"] = {"status": "not_requested", "fresh_save_roundtrip_verified": False}
    if args.checkpoint:
        restored = DeepProbeVLM.from_pretrained(args.checkpoint, torch_dtype=torch.float32).to(device).eval()
        precision = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext()
        with torch.no_grad(), precision:
            actual = restored(**prompt_batch).logits[:, -1].detach().float().cpu()
        finite = bool(torch.isfinite(actual).all())
        # Both loads use identical weights and precision; small kernel variations
        # are tolerated, while the greedy cache comparison remains exact.
        atol, rtol = (1e-3, 1e-3) if bf16 else (1e-6, 1e-6)
        equal = finite and bool(torch.allclose(reference, actual, atol=atol, rtol=rtol))
        report["checkpoint_reload"] = {
            "status": "passed" if equal else "failed", "logits_finite": finite,
            "scope": "Final prompt-position logits from two sequential loads of the same existing checkpoint",
            "max_abs_difference": (reference - actual).abs().max().item() if finite else None,
            "atol": atol, "rtol": rtol, "fresh_save_roundtrip_verified": False,
            "extra_weight_files_written": 0}
        if not equal:
            report["errors"].append("Existing-checkpoint reload logits differ")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)


def main(argv=None):
    args = parser().parse_args(argv)
    validate_arguments(args)
    started = time.perf_counter()
    report = {"status": "running", "purpose": "P0_ENGINEERING_DIAGNOSTIC_NOT_A_QUALITY_RESULT",
              "utc": datetime.now(timezone.utc).isoformat(), "num_samples": 1, "optimizer_steps": 0,
              "quality_metrics_computed": False, "errors": [],
              "limitations": ["One train example does not establish convergence or evaluation accuracy.",
                              "Raw runtime and memory are diagnostics, not a throughput benchmark.",
                              "Existing-checkpoint reload does not compare against pre-save in-memory weights."]}
    try:
        run(args, report)
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["status"] = "failed" if report["errors"] else "engineering_checks_passed"
    report["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "report": str(args.output), "errors": report["errors"]}))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
