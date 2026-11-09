#!/usr/bin/env python3
"""One-shot CPU preparation followed by one independently gated low GPU submit.

This is a bounded process for one authorized experiment, not a recurring task.
Incomplete media never allocates GPUs; any failed audit or submission stops it.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

BASE = Path("/home/jinhaopeng/.cache/Paper/DeepProbe-VLM")
P1 = Path(str(BASE) + "-p1")
RUN_ID = "dp-benchmark-videomme-r1-r2-s17-a1"
PROTOCOL = "videomme-899-no-subtitles-no-audio-v1"
RESTORE_SHA = "fda1b11e76b9fb0fbbd3910b912a67082e91c08eadba9890f283cee7c9efa6fa"
ORIGINAL_READY_SHA = "1c08b9e18d7e9f2bcb9664ae81225603c94585ffabfcebe5b7b9bd01fc9ca66c"
STAGE_CWD = "/home/jinhaopeng"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify(item):
    path = Path(item["path"])
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or digest(path) != item["sha256"]:
        raise ValueError(f"Pinned pipeline input changed: {path}")
    return path


def artifact(path):
    return {"path": str(path), "sha256": digest(path)}


def media_failure_marker(media_root, configured=None):
    """Select one attempt's failure marker without hiding an escaped path."""
    media_root = Path(media_root)
    marker = Path(configured) if configured is not None else media_root / "COORDINATOR_FAILURE.json"
    if (not media_root.is_absolute() or not marker.is_absolute()
            or ".." in marker.parts or marker == media_root
            or not marker.is_relative_to(media_root)
            or not marker.resolve().is_relative_to(media_root.resolve())):
        raise ValueError("Media failure marker must be an absolute path inside media_root")
    for path in (marker, *marker.parents):
        if path.is_symlink():
            raise ValueError("Symlink media failure marker path is forbidden")
        if path == media_root:
            break
    if marker.exists() and not marker.is_file():
        raise ValueError("Media failure marker must identify a file")
    return marker


def await_complete_inputs(paths, root, timeout_seconds=86400, *, failure_marker=None):
    """Require stable receipts and the media worker's atomic SHA sentinel."""
    previous, start = None, time.monotonic()
    while time.monotonic() - start < timeout_seconds:
        marker = media_failure_marker(paths[0].parent, failure_marker)
        if marker.exists():
            raise RuntimeError(f"Media acquisition failed; inspect {marker}")
        current = []
        for path in paths:
            if path.is_symlink():
                raise ValueError(f"Symlink waiting input: {path}")
            if not path.is_file():
                break
            stat = path.stat()
            current.append((stat.st_size, stat.st_mtime_ns))
        if len(current) == len(paths) and current == previous:
            media, training, sentinel = map(load, paths)
            if (media.get("complete") is not True or training.get("passed") is not True
                    or sentinel.get("complete") is not True or sentinel.get("source_count") != 899
                    or sentinel.get("frame_count") != 7192
                    or sentinel.get("receipt_sha256") != digest(paths[0])):
                raise ValueError("Media completion or training fingerprint check did not pass")
            save(root / "INPUTS_COMPLETE.json", {"observed_utc": datetime.now(timezone.utc).isoformat(),
                 "media_failure_marker": str(marker),
                 "inputs": [artifact(path) for path in paths]})
            return
        previous = current
        time.sleep(30)
    raise TimeoutError("One-shot media preparation wait exceeded 24h")


def run_stage(root, name, argv, env):
    save(root / (name + ".argv.json"), {"argv": argv})
    print(json.dumps({"event": "START", "stage": name}), flush=True)
    with (root / (name + ".stdout")).open("xb") as out, (root / (name + ".stderr")).open("xb") as err:
        result = subprocess.run(argv, stdout=out, stderr=err, env=env, cwd=STAGE_CWD)
    save(root / (name + ".exit.json"), {"returncode": result.returncode,
         "stdout_sha256": digest(root / (name + ".stdout")), "stderr_sha256": digest(root / (name + ".stderr"))})
    if result.returncode:
        raise RuntimeError(f"{name} failed with exit {result.returncode}; inspect its saved output")
    print(json.dumps({"event": "COMPLETE", "stage": name}), flush=True)


def prepare_and_submit(config, root):
    experiment = Path(config["experiment_root"])
    if not experiment.is_absolute() or experiment.is_symlink():
        raise ValueError("Expected frozen absolute experiment root")
    for item in config["pinned_files"]:
        verify(item)
    original_ready = P1 / "formal-source-final-20261009-v1/FORMAL_READY.json"
    verify({"path": str(original_ready), "sha256": ORIGINAL_READY_SHA})
    restore = P1 / "benchmark-eval-assets-20261012/checkpoints/CHECKPOINT_RESTORE_VERIFIED.json"
    verify({"path": str(restore), "sha256": RESTORE_SHA})
    media_root = P1 / "benchmark-media-20261012/videomme"
    media_receipt = media_root / "FRAME_CACHE_RECEIPT.json"
    training_receipt = P1 / "benchmark-content-training-20261012-attempt1/TRAINING_FINGERPRINTS.json"
    print(json.dumps({"event": "WAITING_FOR_COMPLETE_MEDIA_AND_TRAINING_FINGERPRINTS", "gpu_submitted": False}), flush=True)
    marker = media_failure_marker(media_root, config.get("media_failure_marker"))
    await_complete_inputs([media_receipt, training_receipt, media_root / "FRAME_CACHE_COMPLETE.json"], root,
                          failure_marker=marker)
    for item in config["pinned_files"]:
        verify(item)
    execution = load(verify(config["audit_execution_plan"]))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    env["PATH"] = "/home/jinhaopeng/.local/bin:" + env.get("PATH", "")
    run_stage(root, "content_audit", execution["audit_argv"], env)
    run_stage(root, "formal_evaluation_gate", execution["gate_argv"], env)
    ready = Path(execution["new_evaluation_ready"])
    if load(ready).get("formal_evaluation_ready") is not True:
        raise ValueError("Fresh formal evaluation gate is not ready")
    verify({"path": str(original_ready), "sha256": ORIGINAL_READY_SHA})
    manifest_root = P1 / "benchmark-eval-assets-20261012/videomme-evaluation-v1"
    annotation = BASE / "assets/videomme/ead1408f75b618502df9a1d8e0950166bf0a2a0b/videomme/test-00000-of-00001.parquet"
    run_stage(root, "manifest", [str(BASE / "env/bin/python"), str(experiment / "scripts/operations/prepare_videomme_evaluation.py"),
        "--annotation", str(annotation), "--source-registry", str(original_ready.parent / "benchmark_source_registry.json"),
        "--frame-cache-receipt", str(media_receipt), "--frame-cache-sha256", digest(media_receipt),
        "--media-root", str(media_root), "--output-dir", str(manifest_root)], env)
    plan = {"schema_version": 1, "run_id": RUN_ID,
            "run_root": str(BASE / "evaluations" / RUN_ID), "protocol": PROTOCOL,
            "manifest": artifact(manifest_root / "videomme.jsonl"), "media_root": str(media_root),
            "manifest_binding": artifact(manifest_root / "MANIFEST_BINDING.json"),
            "formal_ready": artifact(ready), "checkpoint_restore_receipt": artifact(restore),
            "source_files": config["evaluation_source_files"], "storage_roots": [str(BASE), str(P1)],
            "storage_limit_bytes": 99_000_000_000, "working_reserve_bytes": 1_000_000_000}
    plan_path = root / "EVALUATION_PLAN.json"
    save(plan_path, plan)
    for item in config["pinned_files"]:
        verify(item)
    # This helper owns the fresh CPU, OSS, scheduler, YAML checks and durable
    # one-submit intent. No retry is made if the client outcome is ambiguous.
    submit = verify(config["submit_helper"])
    run_stage(root, "submit_once", [str(BASE / "env/bin/python"), str(submit),
        "--plan", str(plan_path), "--plan-sha256", digest(plan_path),
        "--output-dir", str(root / "submission"), "--execute"], env)
    save(root / "PIPELINE_SUBMITTED.json", {"status": "SUBMISSION_HELPER_COMPLETED",
         "run_id": RUN_ID, "plan": artifact(plan_path), "submission_evidence": str(root / "submission"),
         "observed_utc": datetime.now(timezone.utc).isoformat(), "inference_started": False})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    config_path = verify({"path": str(args.config), "sha256": args.config_sha256})
    root = args.output_dir
    if not root.is_absolute() or not root.is_relative_to(P1) or root.is_symlink():
        raise ValueError("Pipeline output must be a new path under the project P1 root")
    lock_path = P1 / "benchmark-evaluation-pipeline.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir(parents=True, exist_ok=False)
        save(root / "PIPELINE_STARTED.json", {"pid": os.getpid(), "run_id": RUN_ID,
             "config": artifact(config_path), "script": artifact(Path(__file__).resolve()),
             "observed_utc": datetime.now(timezone.utc).isoformat()})
        try:
            prepare_and_submit(load(config_path), root)
        except BaseException as exc:
            save(root / "PIPELINE_FAILED.json", {"error": f"{type(exc).__name__}: {exc}",
                 "traceback": traceback.format_exc(), "never_retry_submission": True,
                 "observed_utc": datetime.now(timezone.utc).isoformat()})
            raise


if __name__ == "__main__":
    main()
