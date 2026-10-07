#!/usr/bin/env python3
"""Deterministic source-group training split with explicit eval-source exclusion."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from deepprobe_vlm.manifest import load_manifest, validate_disjoint


def group_split(records, validation_fraction=0.05, seed=42, excluded_sources=()):
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    excluded = set(excluded_sources)
    train, validation, omitted = [], [], []
    for row in records:
        if row["split"] != "train":
            raise ValueError("only training data may be repartitioned; never use evaluation data for SFT")
        if row["source_video_id"] in excluded:
            omitted.append(row["sample_id"])
            continue
        u = int(hashlib.sha256(f"{seed}:{row['source_video_id']}".encode()).hexdigest(), 16) / 2**256
        target = validation if u < validation_fraction else train
        target.append({**row, "split": "validation" if u < validation_fraction else "train"})
    if not train or not validation:
        raise ValueError("empty split after source-grouping; increase data or adjust fraction (never split individual questions)")
    validate_disjoint(train, validation)
    return train, validation, omitted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--media-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--validation-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exclude-manifest", action="append", type=Path, default=[], help="Eval manifest(s); all matching original source IDs excluded")
    args = p.parse_args()
    rows = load_manifest(args.input, args.media_root)
    evaluation = [load_manifest(path, check_media=False) for path in args.exclude_manifest]
    excluded = {r["source_video_id"] for group in evaluation for r in group}
    train, val, omitted = group_split(rows, args.validation_fraction, args.seed, excluded)
    validate_disjoint(train, val, *evaluation)
    paths = [args.output_dir / n for n in ("train.jsonl", "validation.jsonl", "split_report.json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("split output already exists; select a new directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, group in zip(paths, (train, val)):
        with path.open("x", encoding="utf-8") as stream:
            for row in group:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"seed": args.seed, "validation_fraction": args.validation_fraction,
              "train_samples": len(train), "validation_samples": len(val),
              "excluded_eval_source_count": len(excluded), "excluded_sample_ids": omitted,
              "limitation": "Source-ID checks do not identify renamed/re-encoded duplicate videos; audit underlying video provenance before publication."}
    with paths[2].open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
