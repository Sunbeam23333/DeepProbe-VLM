#!/usr/bin/env python3
"""Derive a cache-only candidate from pinned completed-preparation snapshots.

This command reads small local JSON files and writes a new local candidate and
derivation receipt. It does not connect to OSS, copy checkpoint payloads, build
caches, verify a deployment, or admit/submit any CPU/GPU execution. The later
workflow rechecks live history pins before building the remaining caches.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from deepprobe_vlm import experiment_suite, suite_runner as runner


PREPARATION_FIELDS = ("recovery_preparations", "history_preparations")
TEMPLATE_FIELDS = ("job_template", "job_template_source", "job_template_document")
RECEIPT_NAMES = {"prepared": "RECOVERY_PREPARED.json",
                 "original_request": "RECOVERY_REQUEST.json",
                 "archive_acceptance": "ARCHIVE_ACCEPTANCE.json",
                 "history_acceptance": "HISTORY_ACCEPTANCE.json",
                 "native_metadata": "NATIVE_METADATA_VERIFICATION.json"}


def remote_path(value):
    """Validate a recorded POSIX path without inspecting the developer machine."""
    text = str(value)
    path = PurePosixPath(text)
    runner.require(path.is_absolute() and str(path) == text and ".." not in path.parts
                   and "\\" not in text and "\x00" not in text and not text.startswith("//"),
                   "Canonical absolute remote POSIX path required")
    return path


def overlaps(left, right):
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def pinned_document(path, expected_sha):
    pin = runner.artifact(runner.safe(Path(path).absolute()))
    runner.require(pin["sha256"] == expected_sha, "Input SHA differs from its explicit pin")
    return runner.read(pin["path"]), pin


def captured_document(snapshot, pin):
    """Authenticate captured canonical write_once bytes, without live reads."""
    runner.require({k: snapshot.get(k) for k in pin} == pin,
                   "Captured receipt identity differs from verification pin")
    raw = runner.canonical(snapshot["content"]) + b"\n"
    runner.require(len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest() == pin["sha256"],
                   "Captured receipt content differs from its original bytes")
    return snapshot["content"]


def derive(candidate, candidate_pin, verified, snapshot, *, experiment_root, workflow_output):
    """Preserve historical bindings and prepare only remaining benchmark caches."""
    runner.require(candidate.get("schema_version") == 1, "Unsupported original candidate")
    for document in (verified, snapshot):
        runner.require(all(document["candidate"][k] == candidate_pin[k] for k in ("bytes", "sha256")),
                       "Verification does not bind this exact original candidate")
    runner.require(snapshot["candidate"]["content"] == candidate, "Captured original candidate differs")
    runner.require(verified.get("status") == "COMPLETED_HISTORY_PREPARATIONS_REUSE_VERIFIED"
                   and verified.get("new_checkpoint_payload_gets") == 0
                   and verified.get("remote_mutations") is False,
                   "Complete metadata-only reuse verification required")
    requests = [r for field in PREPARATION_FIELDS for r in candidate.get(field, [])]
    ids = [r["task_id"] for r in requests]
    runner.require(ids and len(ids) == len(set(ids)), "Preparation tasks must be nonempty and unique")
    runner.require(set(ids) == set(verified["history"]) == set(verified["receipts"]),
                   "Reuse verification must cover every preparation task exactly")
    env = candidate["environment"]
    runner.require(not env.get("source_files"), "Derivation requires an unbound preparation candidate")
    runner.require(not set(ids).intersection(env.get("history", {}))
                   and not set(ids).intersection(candidate.get("history_imports", {})),
                   "Prepared tasks overlap existing history or imports")
    tasks = {task["id"]: task for task in experiment_suite.default_plan()["tasks"]}
    for request in requests:
        tid = request["task_id"]
        history, receipts = verified["history"][tid], verified["receipts"][tid]
        runner.require(tid in tasks and tasks[tid]["kind"] == "train", "Expected an exact training task")
        # The pinned verification attests live shape/pin/source checks already.
        # Repeating filesystem checks against remote /home paths on a Mac would
        # inspect the wrong host (where /home itself may be a symlink).
        remote_path(history["source_root"])
        remote_path(history["checkpoint"]["path"])
        for name in ("config", "frozen_source_manifest", "acceptance", "archive_acceptance"):
            remote_path(history[name]["path"])
        captured = {}
        for key, name in RECEIPT_NAMES.items():
            runner.require(receipts[key]["path"] == str(remote_path(request["output_dir"]) / name),
                           "Preparation receipt escaped its exact original output")
            captured[key] = captured_document(snapshot["prepared"][tid][name], receipts[key])
        runner.require(captured["original_request"] == request, "Original preparation request changed")
        prepared = captured["prepared"]
        runner.require(prepared.get("status") == "SUITE_RECOVERY_PREPARED"
                       and prepared["history"] == history
                       and prepared["checkpoint_step"] == history["checkpoint_step"]
                       and prepared["remaining_optimizer_updates"] == tasks[tid]["expected_optimizer_steps"] - history["checkpoint_step"]
                       and prepared.get("gpu_submitted") is False
                       and prepared.get("runtime_resume_verified") is False,
                       "Prepared history identity or measurement scope differs")
        checkpoint = history["checkpoint"]
        runner.require(receipts["history_eligible"] is True
                       and receipts["scientific_recipe_equivalent"] is True
                       and receipts["archive_full_get_is_existing_measurement_not_this_review"] is True
                       and receipts["checkpoint_files"] == len(checkpoint["files"])
                       and receipts["checkpoint_bytes"] == sum(row["bytes"] for row in checkpoint["files"])
                       and receipts["checkpoint_step"] == history["checkpoint_step"]
                       and receipts["mode"] == history["mode"]
                       and receipts["history_acceptance"] == history["acceptance"]
                       and receipts["archive_acceptance"] == history["archive_acceptance"],
                       "Reuse summary differs from the exact history binding")
        archive, native = captured["archive_acceptance"], captured["native_metadata"]
        inventory_sha = runner.digest(checkpoint["files"])
        runner.require(archive.get("status") == "SUITE_RECOVERY_ARCHIVE_VERIFIED"
                       and archive.get("production_evidence") is True
                       and archive.get("all_objects_full_get_sha256_verified") is True
                       and archive["checkpoint_inventory_sha256"] == inventory_sha
                       and archive["checkpoint_oss_prefix"] == checkpoint["oss_prefix"]
                       and archive["objects"] == [{"uri": checkpoint["oss_prefix"] + row["path"],
                            "bytes": row["bytes"], "sha256": row["sha256"]} for row in checkpoint["files"]],
                       "Existing archive measurement differs from checkpoint inventory")
        runner.require(native.get("status") == "SUITE_NATIVE_METADATA_VERIFIED"
                       and native["checkpoint_inventory_sha256"] == inventory_sha
                       and native["optimizer_scheduler_rng_deserialized"] is False
                       and native["runtime_resume_verified"] is False,
                       "Native metadata scope differs")

    old_source, new_source = remote_path(env["experiment_root"]), remote_path(experiment_root)
    output = remote_path(workflow_output)
    original_control = remote_path(verified["candidate"]["path"]).parent
    protected = [old_source, original_control, remote_path(candidate["state_root"])]
    protected += [remote_path(r["output_dir"]) for r in requests]
    protected += [remote_path(r[k]) for r in candidate.get("benchmark_preparations", [])
                  for k in ("output_dir", "media_root")]
    retained_history = {**env.get("history", {}), **verified["history"]}
    for history in retained_history.values():
        if history.get("source_root"):
            protected.append(remote_path(history["source_root"]))
        if history.get("checkpoint"):
            protected.append(remote_path(history["checkpoint"]["path"]))
    runner.require(not any(overlaps(new_source, root) for root in protected),
                   "Use a separate new immutable source release")
    runner.require(not any(overlaps(output, root) for root in [*protected, new_source]),
                   "Use a separate new workflow output directory")

    result = deepcopy(candidate)
    result["environment"]["experiment_root"] = str(new_source)
    runtime = result["environment"]["runtime_environment"]
    if "PYTHONPATH" in runtime:
        runner.require(runtime["PYTHONPATH"] == str(old_source / "src"), "Unexpected original PYTHONPATH")
        runtime["PYTHONPATH"] = str(new_source / "src")
    result["environment"]["history"].update(deepcopy(verified["history"]))
    for field in PREPARATION_FIELDS:
        result.pop(field, None)
        result.get("input_indexes", {}).pop(field, None)
    for field in TEMPLATE_FIELDS:
        result.pop(field, None)
    result.pop("recovery_candidates", None)
    result["discovery"] = {"status": "OFFLINE_CACHE_CONTINUATION_NOT_ADMITTED",
        "historical_training_preparations_reused": len(ids),
        "evaluation_imports_pending": len(result.get("history_imports", {})),
        "cache_build_executed": False, "source_deployment_verified": False,
        "cpu_admitted": False, "gpu_submitted": False, "launch_template_omitted": True}
    # Full equality deliberately preserves every old source/config/archive path.
    runner.require(all(result["environment"]["history"][tid] == verified["history"][tid] for tid in ids),
                   "Historical bindings changed during derivation")
    runner.require(result.get("history_imports") == candidate.get("history_imports")
                   and result.get("benchmark_preparations") == candidate.get("benchmark_preparations"),
                   "Evaluation imports or cache requests changed during derivation")
    detail = {"reused_tasks": ids, "verification_observed_utc": verified["observed_utc"],
        "original_remote_candidate": verified["candidate"], "original_source_root": str(old_source),
        "new_source_root": str(new_source), "workflow_output": str(output),
        "original_discovery": candidate.get("discovery"),
        "original_preparation_indexes": {k: candidate.get("input_indexes", {}).get(k) for k in PREPARATION_FIELDS},
        "original_recovery_candidates": candidate.get("recovery_candidates"),
        "launch_fields_omitted": [k for k in TEMPLATE_FIELDS if k in candidate],
        "preserved_execution_profile": candidate["environment"].get("execution_profile"),
        "existing_preparation_receipts": deepcopy(verified["receipts"]),
        "new_checkpoint_payload_gets": 0, "source_deployment_verified": False,
        "cache_build_executed": False, "cpu_admitted": False, "gpu_submitted": False,
        "claim_limits": ["Only pinned historical snapshots were inspected; live metadata will be rechecked by the workflow.",
            "No current OSS availability, new full GET, runtime resume, or GPU admission is established.",
            "Use the new source release for the actual Python invocation; old code retains the generator bug.",
            "Run cache preparation only after the separately authorized a6 admission finishes."]}
    return result, detail


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("candidate", "verified-reuse", "captured-receipts"):
        parser.add_argument("--" + name, required=True, type=Path)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--workflow-output", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    candidate, candidate_pin = pinned_document(args.candidate, args.candidate_sha256)
    verified, verified_pin = pinned_document(args.verified_reuse, args.verified_reuse_sha256)
    snapshot, snapshot_pin = pinned_document(args.captured_receipts, args.captured_receipts_sha256)
    derived, detail = derive(candidate, candidate_pin, verified, snapshot,
        experiment_root=args.experiment_root, workflow_output=args.workflow_output)
    output = runner.safe(args.output_dir.absolute())
    runner.require(not output.exists(), "Use a new local derivation output directory")
    output.mkdir(parents=True)
    runner.write_once(output / "CANDIDATE-CACHE-CONTINUATION.json", derived)
    result = {"status": "OFFLINE_CACHE_CONTINUATION_DERIVED_NOT_EXECUTED", **detail,
        "inputs": {"candidate": candidate_pin, "verified_reuse": verified_pin, "captured_receipts": snapshot_pin},
        "candidate": runner.artifact(output / "CANDIDATE-CACHE-CONTINUATION.json")}
    runner.write_once(output / "CACHE_CONTINUATION_DERIVATION.json", result)
    print(json.dumps({"candidate": result["candidate"],
        "receipt": runner.artifact(output / "CACHE_CONTINUATION_DERIVATION.json"),
        "gpu_submitted": False, "cache_build_executed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
