#!/usr/bin/env python3
"""Check exact train/eval text-token budgets without model weights or media decode.

Default: report every over-budget sample and fail. No truncation, rewriting,
filtering, weight download, image processor, or CUDA initialization is requested.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from deepprobe_vlm.data import MultimodalCollator
from deepprobe_vlm.manifest import load_manifest, validate_disjoint


def check_records(records, collator):
    """Pure counting entry point: suitable for tiny-tokenizer CPU tests."""
    lengths, overflow = [], []
    for row in records:
        prefix, suffix, answer = collator.tokenize_record(row, check_budget=False)
        count = len(prefix) + len(suffix) + len(answer)
        lengths.append(count)
        if count > collator.max_text_tokens:
            overflow.append({"sample_id": row["sample_id"], "split": row["split"],
                             "text_tokens": count, "prompt_tokens": len(prefix) + len(suffix),
                             "answer_tokens_with_eos": len(answer),
                             "visual_plus_text_tokens": count + collator.num_visual_tokens,
                             "over_budget_by": count - collator.max_text_tokens})
    if not lengths:
        raise ValueError("cannot check an empty record set")
    ordered = sorted(lengths)
    quantiles = {name: ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]
                 for name, q in (("p50", .50), ("p95", .95), ("p99", .99))}
    return {"passed": not overflow, "samples": len(lengths),
            "splits": dict(Counter(row["split"] for row in records)),
            "max_text_tokens": collator.max_text_tokens,
            "num_visual_tokens": collator.num_visual_tokens,
            "include_answer": collator.include_answer,
            "observed_text_tokens": {"min": ordered[0], "max": ordered[-1], **quantiles},
            "observed_max_visual_plus_text_tokens": ordered[-1] + collator.num_visual_tokens,
            "overflow_count": len(overflow), "overflow_samples": overflow,
            "next_step_if_failed": "Explicitly increase model.max_text_tokens in the run config after checking context/GPU memory, or document and apply a separate curation policy. This tool never truncates or drops samples.",
            "limitations": "Text preflight only. Media existence/decoding, backbone context limit, generation reserve, and GPU memory are not certified."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifests", nargs="+", type=Path)
    p.add_argument("--config", required=True, type=Path, help="The exact training/evaluation model config")
    p.add_argument("--mode", choices=["training", "inference"], default="training",
                   help="training counts gold answer+EOS (also validation loss); inference excludes them")
    p.add_argument("--checkpoint", type=Path,
                   help="Use saved checkpoint tokenizer, matching --init-checkpoint/--resume training")
    p.add_argument("--report", type=Path, help="Optional JSON report; refuses to overwrite")
    p.add_argument("--local-files-only", action="store_true", help="Require tokenizer already cached/local")
    args = p.parse_args()
    if args.report and args.report.exists():
        p.error("report exists; choose a new filename")
    raw_config = json.loads(args.config.read_text())
    model = raw_config.get("model", raw_config)
    for key in ("max_text_tokens", "num_frames", "pool_grid"):
        if isinstance(model.get(key), bool) or not isinstance(model.get(key), int) or model[key] <= 0:
            p.error(f"config model.{key} must be an explicit positive integer")
    revision = model.get("language_revision", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        p.error("config language_revision must be an immutable 40-character SHA")
    groups = [load_manifest(path, check_media=False) for path in args.manifests]
    validate_disjoint(*groups)
    records = [row for group in groups for row in group]
    if args.mode == "training" and any(row["split"] == "test" for row in records):
        p.error("held-out test is not training-time validation; use --mode inference for test preflight")
    tokenizer_location = model["language_model"]
    tokenizer_revision = revision
    if args.checkpoint:
        saved = json.loads((args.checkpoint / "deepprobe_config.json").read_text())
        for key in ("language_model", "language_revision"):
            if saved.get(key) != model[key]:
                p.error(f"checkpoint/config mismatch: {key}")
        tokenizer_location = str(args.checkpoint / "tokenizer")
        tokenizer_revision = None
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_location, revision=tokenizer_revision,
                                              trust_remote_code=False,
                                              local_files_only=args.local_files_only)
    collator = MultimodalCollator(tokenizer,
                                 num_visual_tokens=model["num_frames"] * model["pool_grid"] ** 2,
                                 max_text_tokens=model["max_text_tokens"],
                                 include_answer=args.mode == "training")
    report = check_records(records, collator)
    report.update(config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  prompt_source_sha256=hashlib.sha256((ROOT / "src/deepprobe_vlm/data.py").read_bytes()).hexdigest(),
                  language_model=model["language_model"], language_revision=revision,
                  tokenizer_source=tokenizer_location, mode=args.mode,
                  manifest_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.manifests},
                  media_decoded=False, model_weights_loaded=False)
    if args.checkpoint:
        report["checkpoint_tokenizer_sha256"] = {
            str(path.relative_to(args.checkpoint / "tokenizer")): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((args.checkpoint / "tokenizer").rglob("*")) if path.is_file()}
    value = json.dumps(report, ensure_ascii=False, indent=2)
    print(value)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(value + "\n")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
