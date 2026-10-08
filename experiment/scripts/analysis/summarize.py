#!/usr/bin/env python3
"""Measured evaluation only: source-cluster bootstrap and paired accuracy delta.

Uses an equal-weight question point estimate but resamples source videos, not
questions. Latency is descriptive cold/mixed CLI timing, never warmed throughput.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty prediction file")
    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample_id in evaluation output")
    for r in rows:
        if r.get("correct") not in (0, 1):
            raise ValueError("Expected an observed binary correct field")
        if not r.get("source_video_id") or not r.get("metric"):
            raise ValueError("Missing source or metric")
    return rows


def bootstrap(rows, field="correct", repeats=2000, seed=17):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_video_id"]].append(float(row[field]))
    values = list(grouped.values())
    point = float(np.mean([x for group in values for x in group]))
    if len(values) < 2:
        return {"estimate": point, "ci95": None, "sources": len(values),
                "questions": len(rows), "warning": "At least two source videos required for a CI"}
    sums = np.array([sum(x) for x in values])
    counts = np.array([len(x) for x in values])
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repeats):
        indices = rng.integers(0, len(values), size=len(values))
        samples.append(float(sums[indices].sum() / counts[indices].sum()))
    return {"estimate": point, "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
            "sources": len(values), "questions": len(rows)}


def paired_rows(current, reference):
    ref = {r["sample_id"]: r for r in reference}
    if set(ref) != {r["sample_id"] for r in current}:
        raise ValueError("Paired runs must contain exactly the same sample IDs; no silent intersection")
    result = []
    for row in current:
        old = ref[row["sample_id"]]
        for key in ["source_video_id", "answer", "metric", "split"]:
            if row.get(key) != old.get(key):
                raise ValueError(f"Paired record mismatch: {row['sample_id']} / {key}")
        result.append(dict(row, delta=float(row["correct"]) - float(old["correct"])))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Output exists; refusing overwrite")
    if args.bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap resamples")
    rows = read_rows(args.predictions)
    paired = paired_rows(rows, read_rows(args.reference)) if args.reference else None
    report = {"status": "MEASURED_INPUTS_ONLY", "bootstrap_unit": "source_video_id",
              "bootstrap_resamples": args.bootstrap, "seed": args.seed, "by_metric": {},
              "notes": ["No official judge score is inferred from normalized exact match.",
                        "Confidence intervals do not remove cross-model pretraining confounds."]}
    for metric in sorted({r["metric"] for r in rows}):
        group = [r for r in rows if r["metric"] == metric]
        entry = bootstrap(group, repeats=args.bootstrap, seed=args.seed)
        if paired:
            entry["paired_accuracy_delta"] = bootstrap([r for r in paired if r["metric"] == metric],
                                                       "delta", args.bootstrap, args.seed)
        for field in ["task", "duration"]:
            entry[f"by_{field}"] = {}
            for value in sorted({str(r[field]) for r in group if r.get(field) is not None}):
                entry[f"by_{field}"][value] = bootstrap(
                    [r for r in group if str(r.get(field)) == value], repeats=args.bootstrap, seed=args.seed)
        report["by_metric"][metric] = entry
    timings = [r["model_seconds"] for r in rows if "model_seconds" in r
               and r.get("timing_status") != "profiled_not_benchmark"]
    if any(not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0 for value in timings):
        raise ValueError("Timing values must be finite nonnegative seconds")
    report["timing_examples"] = len(timings)
    report["profiled_timing_examples_excluded"] = sum(r.get("timing_status") == "profiled_not_benchmark" for r in rows)
    report["timing_status"] = "COLD_OR_MIXED_DESCRIPTIVE_ONLY_NOT_A_WARMED_BENCHMARK"
    report["model_seconds_median"] = float(np.median(timings)) if timings else None
    report["model_seconds_p95"] = float(np.quantile(timings, 0.95)) if timings else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
