#!/usr/bin/env python3
"""Measure a fixed tiny train subset before/after a short overfit run.

This is an engineering sanity check, not a validation or benchmark metric.  It
uses teacher forcing on an explicitly frozen train-only manifest and requires
every sample's supervised NLL to improve, a bounded aggregate loss ratio, and a
minimum supervised-token top-1 accuracy.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--train-manifest", type=Path, required=True,
                   help="Frozen tiny train-only manifest used by the overfit run")
    p.add_argument("--media-root", type=Path, required=True)
    p.add_argument("--initial-checkpoint", type=Path, required=True)
    p.add_argument("--final-checkpoint", type=Path, required=True)
    p.add_argument("--training-output", type=Path, required=True,
                   help="Trainer output directory containing run_manifest/trainer_state")
    p.add_argument("--expected-samples", type=int, default=4)
    p.add_argument("--expected-world-size", type=int, default=1)
    p.add_argument("--expected-final-step", type=int, required=True)
    p.add_argument("--max-loss-ratio", type=float, default=0.5,
                   help="Final token-weighted NLL / initial token-weighted NLL")
    p.add_argument("--min-final-token-accuracy", type=float, default=0.95)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--output", type=Path, required=True, help="New JSON report; never overwritten")
    return p


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def measure(checkpoint: Path, *, config, records, tokenizer, processor, media_root: Path,
            device, bf16: bool) -> dict[str, Any]:
    import torch
    from torch.nn import functional as F
    from deepprobe_vlm.data import MultimodalCollator, VideoManifestDataset
    from deepprobe_vlm.modeling import DeepProbeVLM

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = DeepProbeVLM.from_pretrained(checkpoint, torch_dtype=torch.float32).to(device)
    model.set_stage("sft")
    model.eval()
    dataset = VideoManifestDataset(records, processor, config.num_frames, media_root)
    collator = MultimodalCollator(tokenizer, num_visual_tokens=config.num_visual_tokens,
                                 max_text_tokens=config.max_text_tokens, include_answer=True)
    rows: list[dict[str, Any]] = []
    total_nll, total_tokens, total_correct = 0.0, 0, 0
    precision = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext()
    started = time.perf_counter()
    for index, record in enumerate(records):
        batch = {key: value.to(device) for key, value in collator([dataset[index]]).items()}
        with torch.no_grad(), precision:
            output = model(**batch)
            logits = output.logits[:, :-1].float()
        labels = batch["labels"][:, 1:]
        mask = labels != -100
        count = int(mask.sum().item())
        if count <= 0:
            raise ValueError(f"No supervised answer tokens for {record['sample_id']}")
        # NaN/Inf logits are evidence of failure, not valid inputs to either
        # cross-entropy or argmax accuracy.  Keep the report JSON finite.
        logits_finite = bool(torch.isfinite(logits).all().item())
        losses = None
        nll_sum, correct = None, None
        if logits_finite:
            losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                     ignore_index=-100, reduction="none").reshape_as(labels)
            candidate_nll = float(losses[mask].sum().item())
            if math.isfinite(candidate_nll):
                nll_sum = candidate_nll
                correct = int((logits.argmax(-1)[mask] == labels[mask]).sum().item())
        finite = nll_sum is not None
        rows.append({"sample_id": record["sample_id"], "source_video_id": record["source_video_id"],
                     "supervised_tokens": count, "nll_sum": nll_sum,
                     "mean_nll": nll_sum / count if finite else None,
                     "correct_tokens": correct,
                     "token_accuracy": correct / count if finite else None,
                     "finite": finite,
                     "nonfinite_reason": None if finite else (
                         "logits" if not logits_finite else "supervised_nll")})
        if finite:
            total_nll += nll_sum
            total_correct += correct
        total_tokens += count
        del batch, output, logits, labels, mask, losses
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    all_finite = all(row["finite"] for row in rows) and math.isfinite(total_nll)
    result = {"checkpoint": str(checkpoint.resolve()), "samples": rows,
              "sample_count": len(rows), "supervised_tokens": total_tokens,
              "nll_sum": total_nll if all_finite else None,
              "token_weighted_mean_nll": total_nll / total_tokens if all_finite else None,
              "correct_tokens": total_correct if all_finite else None,
              "teacher_forced_token_accuracy": total_correct / total_tokens if all_finite else None,
              "all_finite": all_finite, "elapsed_seconds": elapsed}
    if device.type == "cuda":
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    del model, dataset, collator
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def evaluate_acceptance(before: dict[str, Any], after: dict[str, Any], *, max_loss_ratio: float,
                        min_final_token_accuracy: float) -> dict[str, Any]:
    def finite_number(value: Any) -> bool:
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value))

    def finite_or_none(value: Any) -> int | float | None:
        return value if finite_number(value) else None

    if [row["sample_id"] for row in before["samples"]] != [row["sample_id"] for row in after["samples"]]:
        raise ValueError("Before/after sample order differs")
    initial = before["token_weighted_mean_nll"]
    final = after["token_weighted_mean_nll"]
    initial_accuracy = before["teacher_forced_token_accuracy"]
    final_accuracy = after["teacher_forced_token_accuracy"]
    finite_aggregates = (finite_number(initial) and finite_number(final)
                         and initial > 0 and final >= 0)
    ratio = final / initial if finite_aggregates else None
    if ratio is not None and not math.isfinite(ratio):
        ratio = None
        finite_aggregates = False
    per_sample = []
    for left, right in zip(before["samples"], after["samples"]):
        left_nll, right_nll = left["mean_nll"], right["mean_nll"]
        improved = bool(left["finite"] is True and right["finite"] is True
                        and finite_number(left_nll) and finite_number(right_nll)
                        and right_nll >= 0 and right_nll < left_nll)
        per_sample.append({"sample_id": left["sample_id"],
                           "initial_mean_nll": finite_or_none(left_nll),
                           "final_mean_nll": finite_or_none(right_nll),
                           "strictly_improved": improved})
    valid_accuracy = (finite_number(initial_accuracy) and finite_number(final_accuracy)
                      and 0 <= initial_accuracy <= 1 and 0 <= final_accuracy <= 1)
    valid_token_counts = (type(before["supervised_tokens"]) is int
                          and type(after["supervised_tokens"]) is int
                          and before["supervised_tokens"] > 0 and after["supervised_tokens"] > 0)
    checks = {
        "all_metrics_finite": bool(before["all_finite"] is True and after["all_finite"] is True
                                   and finite_aggregates and valid_accuracy
                                   and all(row["initial_mean_nll"] is not None
                                           and row["final_mean_nll"] is not None
                                           and row["initial_mean_nll"] >= 0
                                           and row["final_mean_nll"] >= 0 for row in per_sample)),
        "same_supervised_token_count": bool(valid_token_counts and
                                              before["supervised_tokens"] == after["supervised_tokens"]),
        "all_samples_strictly_improved": bool(per_sample) and all(
            row["strictly_improved"] for row in per_sample),
        "aggregate_loss_ratio_within_bound": bool(
            finite_aggregates and ratio is not None and ratio <= max_loss_ratio),
        "final_token_accuracy_reaches_threshold": bool(
            valid_accuracy and final_accuracy >= min_final_token_accuracy),
    }
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "initial_token_weighted_mean_nll": finite_or_none(initial),
            "final_token_weighted_mean_nll": finite_or_none(final),
            "final_to_initial_loss_ratio": ratio, "max_loss_ratio": max_loss_ratio,
            "initial_teacher_forced_token_accuracy": finite_or_none(initial_accuracy),
            "final_teacher_forced_token_accuracy": finite_or_none(final_accuracy),
            "min_final_token_accuracy": min_final_token_accuracy, "per_sample": per_sample,
            "interpretation": (
                "Teacher-forced memorization on a fixed tiny training subset; this is neither held-out "
                "quality nor free-running answer accuracy.")}


def _replace_nonfinite_json_numbers(value: Any, path: str = "$") -> tuple[Any, list[str]]:
    """Keep an otherwise useful failure report serializable without hiding invalid metrics."""
    if isinstance(value, float) and not math.isfinite(value):
        return None, [path]
    if isinstance(value, dict):
        result, invalid = {}, []
        for key, item in value.items():
            result[key], found = _replace_nonfinite_json_numbers(item, f"{path}.{key}")
            invalid.extend(found)
        return result, invalid
    if isinstance(value, (list, tuple)):
        result, invalid = [], []
        for index, item in enumerate(value):
            clean, found = _replace_nonfinite_json_numbers(item, f"{path}[{index}]")
            result.append(clean)
            invalid.extend(found)
        return result, invalid
    return value, []


def run(args: argparse.Namespace, report: dict[str, Any]) -> None:
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite report: {args.output}")
    if args.expected_samples <= 0 or args.expected_samples > 8:
        raise ValueError("expected-samples must be in 1..8")
    if args.expected_world_size <= 0 or args.expected_final_step <= 0:
        raise ValueError("expected world size/final step must be positive")
    if not 0 < args.max_loss_ratio < 1:
        raise ValueError("max-loss-ratio must be strictly between 0 and 1")
    if not 0 <= args.min_final_token_accuracy <= 1:
        raise ValueError("min-final-token-accuracy must be in 0..1")
    for path in (args.initial_checkpoint, args.final_checkpoint, args.training_output):
        if not path.is_dir():
            raise FileNotFoundError(path)

    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    import torch
    from transformers import AutoImageProcessor, AutoTokenizer, set_seed
    from deepprobe_vlm.manifest import load_manifest
    from deepprobe_vlm.modeling import DeepProbeConfig

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing to relabel a CPU run as the GPU overfit gate")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    raw = load_json(args.config)
    config = DeepProbeConfig.from_dict(raw)
    settings = raw.get("training", {})
    if args.device == "cuda" and (settings.get("bf16") is not True
                                  or settings.get("fp16", False) is not False):
        raise ValueError("P0-05 CUDA overfit verification requires bf16=true and fp16=false")
    bf16 = settings.get("bf16", True) is True
    if bf16 and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 overfit recipe requires compatible CUDA")
    records = load_manifest(args.train_manifest, media_root=args.media_root)
    if len(records) != args.expected_samples:
        raise ValueError(f"Expected exactly {args.expected_samples} fixed samples, found {len(records)}")
    if any(row["split"] != "train" for row in records):
        raise ValueError("Short-overfit manifest must contain train records only")
    if len({row["source_video_id"] for row in records}) != len(records):
        raise ValueError("Short-overfit manifest must contain one fixed sample from each distinct source")

    for checkpoint in (args.initial_checkpoint, args.final_checkpoint):
        saved = DeepProbeConfig.from_dict(load_json(checkpoint / "deepprobe_config.json"))
        if saved != config:
            raise ValueError(f"Checkpoint/model config mismatch: {checkpoint}")
    run_manifest = load_json(args.training_output / "run_manifest.json")
    state = load_json(args.training_output / "trainer_state.json")
    contract = run_manifest.get("training_contract", {})
    if state.get("global_step") != args.expected_final_step:
        raise ValueError("Overfit run did not reach the expected optimizer step")
    if contract.get("world_size") != args.expected_world_size:
        raise ValueError("Overfit world size differs from the declared gate")
    if contract.get("train_manifest_sha256") != sha256_file(args.train_manifest):
        raise ValueError("Overfit run used a different tiny manifest")
    init_value = run_manifest.get("init_checkpoint")
    if not init_value or Path(init_value).resolve() != args.initial_checkpoint.resolve():
        raise ValueError("Overfit run did not initialize from the declared before checkpoint")
    provenance = load_json(args.final_checkpoint / "training_provenance.json")
    missing_sources = sorted({row["source_video_id"] for row in records} -
                             set(provenance.get("training_sources", [])))
    if missing_sources:
        raise ValueError(f"Final checkpoint provenance is missing tiny-subset sources: {missing_sources}")

    set_seed(int(settings.get("seed", 17)))
    tokenizer = AutoTokenizer.from_pretrained(args.initial_checkpoint / "tokenizer",
                                              trust_remote_code=False, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(args.initial_checkpoint / "image_processor",
                                                   use_fast=False, local_files_only=True)
    report.update(model=asdict(config), config_sha256=sha256_file(args.config),
                  fixed_manifest={"path": str(args.train_manifest.resolve()),
                                  "sha256": sha256_file(args.train_manifest),
                                  "sample_ids": [row["sample_id"] for row in records],
                                  "source_video_ids": [row["source_video_id"] for row in records],
                                  "selection": (f"One earliest manifest record from each of the first "
                                                f"{args.expected_samples} distinct train sources")},
                  device=str(device), world_size=args.expected_world_size,
                  optimizer_steps=args.expected_final_step, quality_result=False)
    before = measure(args.initial_checkpoint, config=config, records=records, tokenizer=tokenizer,
                     processor=processor, media_root=args.media_root, device=device, bf16=bf16)
    report["before"] = before
    after = measure(args.final_checkpoint, config=config, records=records, tokenizer=tokenizer,
                    processor=processor, media_root=args.media_root, device=device, bf16=bf16)
    report["after"] = after
    acceptance = evaluate_acceptance(before, after, max_loss_ratio=args.max_loss_ratio,
                                     min_final_token_accuracy=args.min_final_token_accuracy)
    report.update(acceptance=acceptance,
                  short_overfit_engineering_gate_passed=acceptance["status"] == "passed")
    report["status"] = ("short_overfit_engineering_gate_passed" if
                        report["short_overfit_engineering_gate_passed"] else
                        "short_overfit_engineering_gate_failed")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report: dict[str, Any] = {
        "status": "running", "purpose": "P0_05_SHORT_OVERFIT_ENGINEERING_EVIDENCE_NOT_A_QUALITY_RESULT",
        "utc": datetime.now(timezone.utc).isoformat(), "errors": [],
    }
    try:
        run(args, report)
    except Exception as exc:
        report["status"] = "error"
        report["short_overfit_engineering_gate_passed"] = False
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report, invalid_paths = _replace_nonfinite_json_numbers(report)
    if invalid_paths:
        if report["status"] != "error":
            report["status"] = "short_overfit_engineering_gate_failed"
        report["short_overfit_engineering_gate_passed"] = False
        report["errors"].append(
            f"{len(invalid_paths)} non-finite metric value(s) converted to null: " +
            ", ".join(invalid_paths[:20]))
        acceptance = report.get("acceptance")
        if isinstance(acceptance, dict):
            acceptance["status"] = "failed"
            if isinstance(acceptance.get("checks"), dict):
                acceptance["checks"]["all_metrics_finite"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.output.is_symlink():
        print(json.dumps({"status": "error", "error": f"output exists: {args.output}"}))
        return 2
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    except FileExistsError:
        print(json.dumps({"status": "error", "error": f"output exists: {args.output}"}))
        return 2
    print(json.dumps({"status": report["status"], "report": str(args.output),
                      "errors": report["errors"]}))
    return 0 if report.get("short_overfit_engineering_gate_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
