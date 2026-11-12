"""Compare completed, source-bound latency runs only after exact token parity.

This reads small measurement artifacts, never loads a model or submits a job.
Runtime isolation and stability still need experimental review; a ratio is a
descriptive paired measurement, not a statistical speedup significance claim.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import benchmark_latency as latency


HARDWARE_AND_SOFTWARE = (
    "packages", "cuda_runtime", "cudnn", "gpu_name", "gpu_total_memory",
    "compute_capability", "gpu_uuid", "driver", "measured_device",
    "actual_parameter_dtypes", "tf32", "cudnn_benchmark", "torch_num_threads",
    "thread_environment", "machine", "platform", "python",
)


def require_implementation(record, plan, label):
    expected = latency.implementation_options(plan) or {"last_token_logits": False, "pool_backend": "reference"}
    # Original v1 artifacts predate this additive metadata. If present, it must
    # still match the reference implementation; v2 must always declare it.
    if plan["schema_version"] == 2 or "implementation" in record:
        latency.require(latency.canonical(record.get("implementation")) == latency.canonical(expected),
                        f"{label} implementation differs from plan")


def validate_rows(rows, plan, plan_sha):
    expected_order = [(sample["sample_id"], phase, repeat)
                      for sample in plan["samples"]
                      for phase, count in (("warmup", 3), ("measured", 10))
                      for repeat in range(count)]
    latency.require([(r.get("sample_id"), r.get("phase"), r.get("repeat_index")) for r in rows]
                    == expected_order, "request order, coverage or pairing differs from plan")
    record_hashes = {sample["sample_id"]: sample["record_sha256"] for sample in plan["samples"]}
    for row in rows:
        latency.require(row.get("plan_sha256") == plan_sha, "request bound to a different plan")
        latency.require(row.get("record_sha256") == record_hashes[row["sample_id"]], "request input record differs from plan")
        latency.require(type(row["repeat_index"]) is int, "invalid repeat index")
        require_implementation(row, plan, "request")
        ids = row.get("generated_token_ids")
        latency.require(isinstance(ids, list) and 1 <= len(ids) <= latency.SETTINGS["max_new_tokens"]
                        and all(type(token) is int and token >= 0 for token in ids)
                        and type(row.get("generated_tokens")) is int and row["generated_tokens"] == len(ids),
                        "invalid actual generated token IDs or count")
        latency.require(all(type(row.get(k)) in (int, float) and math.isfinite(row[k]) and row[k] >= 0
                            for k in latency.TIMINGS) and row["model_seconds"] > 0,
                        "invalid request timing")
        latency.require(math.isclose(row["request_seconds"], sum(row[k] for k in latency.TIMINGS[:3]),
                                     rel_tol=1e-9, abs_tol=1e-9), "request timing boundaries disagree")
        memory = row.get("memory", {})
        latency.require(all(type(memory.get(k)) is int and memory[k] >= 0 for k in (
            "baseline_allocated_bytes", "baseline_reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes")),
            "invalid request memory counters")
        inputs = row.get("input", {})
        latency.require(isinstance(inputs, dict)
                        and latency.canonical(inputs.get("model_config")) == latency.canonical(plan["model_config"]),
                        "request model config differs from plan")
        latency.require(isinstance(row.get("prediction"), str), "prediction must be text")


def read_run(directory):
    root = latency.safe(directory)
    for name in ("PLAN.json", "SUMMARY.json", "RUNTIME.json", "requests.jsonl"):
        latency.require(latency.safe(root / name).is_file(), "missing measurement artifact")
    plan = latency.validate_plan(json.loads((root / "PLAN.json").read_text()))
    summary = json.loads((root / "SUMMARY.json").read_text())
    runtime = json.loads((root / "RUNTIME.json").read_text())
    latency.require(summary.get("status") == "DEVELOPMENT_LATENCY_MEASUREMENTS_COMPLETE",
                    "completed timing summary required")
    for name, key in (("PLAN.json", "plan_sha256"), ("RUNTIME.json", "runtime_sha256"),
                      ("requests.jsonl", "requests_sha256")):
        latency.require(latency.sha(root / name) == summary.get(key), "measurement artifact hash differs")
    latency.require(summary.get("comparison_identity") == plan["comparison_identity"],
                    "summary comparison identity differs")
    latency.require(summary.get("protocol") == plan["protocol"] and summary.get("cache") is plan["cache"],
                    "summary protocol or cache differs from plan")
    latency.require(summary.get("measurement_scope") == latency.SCOPE, "timing measurement scope differs")
    require_implementation(summary, plan, "summary")
    require_implementation(runtime, plan, "runtime")
    latency.require(runtime.get("profiler_enabled") is False, "profiled requests cannot be benchmarked")
    latency.require(all(key in runtime for key in HARDWARE_AND_SOFTWARE), "incomplete runtime identity")
    latency.require(isinstance(runtime["gpu_uuid"], str) and runtime["gpu_uuid"].strip()
                    and runtime["gpu_uuid"].lower() not in {"none", "unavailable", "unknown"},
                    "actual GPU UUID required for comparison")
    provenance = next(r for r in plan["checkpoint"]["files"] if r["path"] == "training_provenance.json")
    latency.require(runtime.get("checkpoint_training_provenance_sha256") == provenance["sha256"],
                    "runtime checkpoint provenance differs from plan")
    rows = [json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()]
    validate_rows(rows, plan, summary["plan_sha256"])
    sample_ids = [row["sample_id"] for row in plan["samples"]]
    recomputed = latency.summarize(rows, sample_ids)
    for key, value in recomputed.items():
        latency.require(summary.get(key) == value, "timing summary differs from complete request rows")
    return plan, runtime, summary, rows


def compare(reference, candidate):
    ref_plan, ref_runtime, ref_summary, ref_rows = read_run(reference)
    new_plan, new_runtime, new_summary, new_rows = read_run(candidate)
    latency.require(ref_plan["comparison_identity"] == new_plan["comparison_identity"],
                    "inputs, checkpoint, protocol or executable sources differ")
    latency.require(all(ref_runtime[k] == new_runtime[k] for k in HARDWARE_AND_SOFTWARE),
                    "GPU or software runtime differs")
    key = lambda row: (row["sample_id"], row["phase"], row["repeat_index"])
    expected = {key(row): row for row in ref_rows}
    actual = {key(row): row for row in new_rows}
    latency.require(set(expected) == set(actual), "request pairing differs")
    latency.require(all(latency.canonical(expected[k]["input"]) == latency.canonical(actual[k]["input"]) for k in expected),
                    "prepared input shapes, dtypes or prompt metadata differ")
    mismatches = [{"sample_id": k[0], "phase": k[1], "repeat_index": k[2],
                   "reference_tokens": expected[k]["generated_token_ids"],
                   "candidate_tokens": actual[k]["generated_token_ids"]}
                  for k in sorted(expected)
                  if expected[k]["generated_token_ids"] != actual[k]["generated_token_ids"]]
    ratios = None
    if not mismatches:
        ratios = {}
        for timing in ("model_seconds", "request_seconds"):
            ratios[timing] = {}
            for stat in ("median", "p95"):
                denominator = new_summary["timings"][timing][stat]
                latency.require(denominator > 0, "positive candidate timing required")
                ratios[timing][stat] = ref_summary["timings"][timing][stat] / denominator
    return {
        "status": "TOKEN_PARITY_PASSED" if not mismatches else "TOKEN_PARITY_FAILED",
        "comparison_identity": ref_plan["comparison_identity"],
        "reference_plan_sha256": ref_summary["plan_sha256"],
        "candidate_plan_sha256": new_summary["plan_sha256"],
        "reference_implementation": latency.implementation_options(ref_plan),
        "candidate_implementation": latency.implementation_options(new_plan),
        "reference_cache": ref_plan["cache"], "candidate_cache": new_plan["cache"],
        "requests_compared_including_warmups": len(expected),
        "token_mismatch_count": len(mismatches), "mismatches": mismatches,
        "reference_over_candidate_timing_ratio": ratios,
        "timing_ratio_interpretation": "greater than one means lower candidate latency; no significance or serving-throughput claim",
        "heldout_quality_claimed": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = compare(args.reference, args.candidate)
    if args.output:
        latency.write_json(args.output, result)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0 if result["status"] == "TOKEN_PARITY_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
