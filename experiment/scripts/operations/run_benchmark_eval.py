#!/usr/bin/env python3
"""Verify frozen inputs, then evaluate explicitly bound models sequentially on eight GPUs.

This entry point never submits a scheduler job, modifies model/data inputs, or
chooses models/prompts from benchmark scores. --preflight-only uses CPU only.
The run directory can resume only with exactly the same plan and identities.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

EXPERIMENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENT / "src"))
VIDEOMME_ANNOTATION_SHA256 = "7fffab8ed38ecc2f9f0eca4c44d8a11636f0ee96116ede83580c8b9ae0faf986"
VIDEOMME_REGISTRY_SHA256 = "a143d6a364221738b727d3944527039060315e04373599c5829ab3ea60eb0226"

from model_registry import model_sequence, verify_registered_checkpoint

from deepprobe_vlm.evaluate import (VIDEOMME_PROTOCOL, canonical_sha256, check_provenance,
    read_prediction_rows, sha256_file, shard_output, shard_records, summarise_groups,
    summarise_metrics, validate_videomme_subset, write_json_once)
from deepprobe_vlm.data import MultimodalCollator
from deepprobe_vlm.manifest import load_manifest, safe_relative_path, resolve_media_path


def read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def verify_artifact(item):
    path = Path(item["path"])
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected an absolute regular artifact: {path}")
    if sha256_file(path) != item["sha256"]:
        raise ValueError(f"Artifact SHA mismatch: {path}")
    return path


def tree_files(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Expected regular directory: {root}")
    result = {}
    def onerror(error):
        raise error
    for directory, dirs, files in os.walk(root, onerror=onerror, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError(f"Symlink inside input/output tree: {path}")
        for name in files:
            path = Path(directory) / name
            if not path.is_file():
                raise ValueError(f"Not a regular file: {path}")
            result[path.relative_to(root).as_posix()] = path
    return result


def verify_checkpoint(binding):
    return verify_registered_checkpoint(binding)


def staging_bytes(root_names):
    """Account both roots while allowing their existing env/HF symlinks.

    Internal targets are counted by their physical traversal, external regular
    file targets are charged conservatively once per reference. Directory links
    must stay within these roots. Broken links and walk errors fail closed.
    """
    roots = [Path(name).resolve(strict=True) for name in root_names]
    totals = {}
    def onerror(error):
        raise error
    for root in roots:
        total = 0
        for directory, dirs, files in os.walk(root, followlinks=False, onerror=onerror):
            for name in list(dirs):
                path = Path(directory) / name
                if path.is_symlink():
                    target = path.resolve(strict=True)
                    if not target.is_dir() or not any(target.is_relative_to(other) for other in roots):
                        raise ValueError(f"Directory symlink escapes staging roots: {path}")
                    dirs.remove(name)
            for name in files:
                path = Path(directory) / name
                if path.is_symlink():
                    target = path.resolve(strict=True)
                    if not target.is_file():
                        raise ValueError(f"Not a regular symlink target: {path}")
                    if not any(target.is_relative_to(other) for other in roots):
                        total += target.stat().st_size
                else:
                    if not stat.S_ISREG(path.stat().st_mode):
                        raise ValueError(f"Not a regular staging file: {path}")
                    total += path.stat().st_size
        totals[str(root)] = total
    return totals


def validate_plan(plan):
    model_sequence(plan)
    if plan.get("schema_version") != 1 or plan.get("protocol") != VIDEOMME_PROTOCOL:
        raise ValueError("Unsupported frozen evaluation plan")
    if plan.get("storage_limit_bytes") != 99_000_000_000 or plan.get("working_reserve_bytes", 0) < 1_000_000_000:
        raise ValueError("Plan must retain the 99GB limit and at least 1GB working reserve")
    for key in ("run_root", "media_root"):
        if not Path(plan[key]).is_absolute() or Path(plan[key]).is_symlink():
            raise ValueError(f"Unsafe {key}")
    if not plan["run_id"] or Path(plan["run_root"]).name != plan["run_id"]:
        raise ValueError("run_root must end with the declared run_id")
    roots = [Path(root).resolve() for root in plan["storage_roots"]]
    if len(roots) != 2 or any(a == b or a.is_relative_to(b) for i, a in enumerate(roots) for j, b in enumerate(roots) if i != j):
        raise ValueError("Two distinct, nonnested staging roots are required")
    if not any(Path(plan["run_root"]).is_relative_to(root) for root in roots):
        raise ValueError("Evaluation output must be included in the staging capacity audit")


def verify_media_gate(ready_path, records, media_root):
    ready = read_json(ready_path)
    if ready.get("formal_evaluation_ready") is not True:
        raise ValueError("Formal benchmark media/content gate is not ready")
    if ready.get("benchmark_media_gate") != "passed" or ready.get("content_duplicate_gate") != "passed":
        raise ValueError("Benchmark media/content gate statuses must both pass")
    artifacts = ready.get("artifact_sha256")
    if not isinstance(artifacts, dict) or "benchmark_media_inventory.json" not in artifacts:
        raise ValueError("Formal gate lacks pinned benchmark media inventory")
    for relative, expected_sha in artifacts.items():
        path = ready_path.parent / safe_relative_path(relative)
        verify_artifact({"path": str(path), "sha256": expected_sha})
    inventory = read_json(ready_path.parent / "benchmark_media_inventory.json")
    if (inventory.get("kind") != "uniform8_lossless_frame_cache" or inventory.get("complete") is not True
            or inventory.get("scope") != "uniform8_sampled_frames"):
        raise ValueError("Expected complete lossless uniform-eight-frame benchmark inventory")
    if Path(inventory["root"]).resolve() != Path(media_root).resolve():
        raise ValueError("Benchmark media root differs from accepted inventory")
    verify_artifact({"path": inventory["receipt_path"], "sha256": inventory["receipt_sha256"]})
    entries, originals = inventory["entries"], inventory["original_entries"]
    if len(entries) != 7192 or len(originals) != 899:
        raise ValueError("Benchmark media inventory must contain 7192 frames / 899 original videos")
    entry_paths = set()
    for entry in entries:
        relative = safe_relative_path(entry["relative_path"])
        if relative in entry_paths:
            raise ValueError("Duplicate benchmark frame inventory entry")
        entry_paths.add(relative)
        path = resolve_media_path(relative, media_root)
        if path.is_symlink() or not path.is_file() or path.stat().st_size != entry["size_bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Current benchmark frame differs from accepted inventory: {relative}")
    by_source = {}
    for row in inventory["records"]:
        source = row["source_video_id"]
        if source in by_source or len(row["image_paths"]) != 8:
            raise ValueError("Invalid source/frame acquisition mapping")
        by_source[source] = row
    if len(by_source) != 899:
        raise ValueError("Expected all 899 source mappings")
    for row in records:
        source = row["source_video_id"]
        if source not in by_source or row.get("image_paths") != by_source[source]["image_paths"]:
            raise ValueError(f"Evaluation image order differs from source-bound frame cache: {row['sample_id']}")
    if entry_paths != {path for row in records for path in row["image_paths"]}:
        raise ValueError("Manifest frame set differs from pinned media inventory")
    return {"ready_sha256": sha256_file(ready_path), "inventory_sha256": sha256_file(ready_path.parent / "benchmark_media_inventory.json"),
            "frame_cache_receipt_sha256": inventory["receipt_sha256"], "num_frames": 7192,
            "num_sources": 899, "current_frame_sha256_verified": True,
            "content_audit_scope": "uniform8_sampled_frames",
            "content_audit_limitation": "No content-disjointness claim for unsampled video portions."}


def verify_manifest_binding(binding_path, manifest_item, records, media_root, media_identity):
    binding = read_json(binding_path)
    expected = {"schema_version": 1, "status": "EVALUATION_MANIFEST_BOUND", "protocol": VIDEOMME_PROTOCOL,
                "repo_id": "lmms-lab/Video-MME", "revision": "ead1408f75b618502df9a1d8e0950166bf0a2a0b",
                "raw_questions": 2700, "selected_questions": 2697, "selected_sources": 899,
                "excluded_question_ids": ["490-1", "490-2", "490-3"],
                "subtitles": False, "audio": False, "all_selected_qa_bound_to_original_annotation": True}
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError("Manifest binding status/protocol/counts differ from frozen Video-MME subset")
    if binding["annotation"]["sha256"] != VIDEOMME_ANNOTATION_SHA256 or binding["source_registry"]["sha256"] != VIDEOMME_REGISTRY_SHA256:
        raise ValueError("Manifest binding annotation/source-registry identity changed")
    verify_artifact(binding["annotation"])
    verify_artifact(binding["source_registry"])
    if (binding["manifest"]["sha256"] != manifest_item["sha256"] or
            Path(binding["manifest"]["path"]).resolve() != Path(manifest_item["path"]).resolve()):
        raise ValueError("Manifest binding does not refer to this exact manifest")
    media = binding["media_binding"]
    if (media["sha256"] != media_identity["frame_cache_receipt_sha256"] or media["frame_count"] != 7192
            or Path(media["media_root"]).resolve() != Path(media_root).resolve()):
        raise ValueError("Manifest binding frame-cache identity/root differs from accepted media")
    verify_artifact({"path": media["path"], "sha256": media["sha256"]})
    qa_path = verify_artifact({"path": str(binding_path.parent / "QA_BINDINGS.json"),
                               "sha256": binding["qa_bindings_sha256"]})
    verify_artifact({"path": str(binding_path.parent / "EXCLUSIONS.json"), "sha256": binding["exclusions_sha256"]})
    qa = json.loads(qa_path.read_text())
    if not isinstance(qa, list) or len(qa) != 2697 or len(records) != 2697:
        raise ValueError("QA binding must cover all 2697 questions")
    for proof, row in zip(qa, records):
        if (proof.get("sample_id") != row["sample_id"] or proof.get("source_video_id") != row["source_video_id"]
                or proof.get("output_record_sha256") != canonical_sha256(row)
                or proof.get("annotation_sha256") != VIDEOMME_ANNOTATION_SHA256):
            raise ValueError(f"QA binding differs from current manifest row: {row['sample_id']}")
    return {"manifest_binding_sha256": sha256_file(binding_path), "qa_bindings_sha256": binding["qa_bindings_sha256"],
            "manifest_sha256": manifest_item["sha256"], "annotation_sha256": VIDEOMME_ANNOTATION_SHA256,
            "source_registry_sha256": VIDEOMME_REGISTRY_SHA256, "all_output_records_bound": True}


def preflight(plan, plan_sha):
    validate_plan(plan)
    from transformers import AutoTokenizer
    # Every executable source in the package is pinned, including this runner.
    required = {str(path.resolve()) for path in (EXPERIMENT / "src").rglob("*.py")}
    required |= {str(Path(__file__).resolve()), str(Path(__file__).with_suffix(".sh").resolve()),
                 str(Path(__file__).with_name("model_registry.py").resolve())}
    supplied = [str(verify_artifact(item).resolve()) for item in plan["source_files"]]
    if len(supplied) != len(set(supplied)) or not required <= set(supplied):
        raise ValueError("Frozen source_files must include the complete executable package and runner")
    manifest = verify_artifact(plan["manifest"])
    ready_path = verify_artifact(plan["formal_ready"])
    records = load_manifest(manifest, media_root=plan["media_root"])
    coverage = validate_videomme_subset(records)
    media_identity = verify_media_gate(ready_path, records, plan["media_root"])
    qa_identity = verify_manifest_binding(verify_artifact(plan["manifest_binding"]), plan["manifest"],
                                          records, plan["media_root"], media_identity)
    checkpoints = {}
    for name in model_sequence(plan):
        item = verify_checkpoint(plan["model_bindings"][name])
        root = Path(item["checkpoint"])
        item["contamination_check"] = check_provenance(root, records)
        tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", trust_remote_code=False, local_files_only=True)
        collator = MultimodalCollator(tokenizer, num_visual_tokens=8 * 4 * 4,
            max_text_tokens=item["config"]["max_text_tokens"], include_answer=False)
        lengths = [sum(map(len, collator.tokenize_record(row))) for row in records]
        item["token_budget"] = {"num_examples": len(lengths), "max_prompt_text_tokens": max(lengths),
                                "max_text_tokens": collator.max_text_tokens, "num_over_budget": 0,
                                "include_answer": False, "truncation": False}
        item["plan_sha256"] = plan_sha
        item["manifest_sha256"] = plan["manifest"]["sha256"]
        item["media_identity"] = media_identity
        item["qa_identity"] = qa_identity
        checkpoints[name] = item
    totals = staging_bytes(plan["storage_roots"])
    projection = sum(totals.values()) + plan["working_reserve_bytes"]
    if projection >= plan["storage_limit_bytes"]:
        raise ValueError(f"Capacity gate failed: {projection} >= {plan['storage_limit_bytes']}")
    return {"status": "CPU_PREFLIGHT_PASSED", "plan_sha256": plan_sha,
            "coverage": coverage, "media_identity": media_identity, "qa_identity": qa_identity,
            "checkpoints": checkpoints, "storage_bytes": totals,
            "projected_bytes": projection, "storage_limit_bytes": plan["storage_limit_bytes"],
            "protocol": VIDEOMME_PROTOCOL, "greedy_max_new_tokens": 16,
            "num_gpus": 8, "sequence": model_sequence(plan), "subtitles": False, "audio": False}


def merge_shards(model_root, records, provenance_path):
    output = model_root / "predictions.jsonl"
    provenance = read_json(provenance_path)
    rows_by_id, shard_receipts = {}, []
    for rank in range(8):
        path = shard_output(output, rank, 8)
        protocol_path = path.with_suffix(".protocol.json")
        summary_path = path.with_suffix(".summary.json")
        protocol = read_json(protocol_path)
        if protocol.get("run_provenance_sha256") != sha256_file(provenance_path) or protocol.get("shard_index") != rank or protocol.get("num_shards") != 8:
            raise ValueError("Shard protocol/provenance mismatch")
        expected_protocol = {"protocol": VIDEOMME_PROTOCOL, "max_new_tokens": 16, "cache": True,
                             "decoding": "greedy_argmax", "subtitles": False, "audio": False,
                             "checkpoint": provenance["checkpoint"], "checkpoint_config": provenance["config"],
                             "manifest_sha256": provenance["manifest_sha256"]}
        if any(protocol.get(key) != value for key, value in expected_protocol.items()):
            raise ValueError("Shard inference protocol differs from frozen benchmark protocol")
        assigned = shard_records(records, rank, 8)
        rows = read_prediction_rows(path, assigned, canonical_sha256(protocol))
        if len(rows) != len(assigned):
            raise ValueError(f"Incomplete shard {rank}: {len(rows)}/{len(assigned)}")
        summary = read_json(summary_path)
        if summary.get("predictions_sha256") != sha256_file(path) or summary.get("num_examples") != len(assigned):
            raise ValueError("Shard completion summary mismatch")
        for row in rows:
            if row.get("shard_index") != rank or row.get("num_shards") != 8:
                raise ValueError("Prediction row belongs to a different shard")
            if row["sample_id"] in rows_by_id:
                raise ValueError("Duplicate sample while merging shards")
            rows_by_id[row["sample_id"]] = row
        shard_receipts.append({"rank": rank, "path": path.name, "sha256": sha256_file(path),
                               "protocol_sha256": canonical_sha256(protocol), "num_examples": len(rows)})
    if set(rows_by_id) != {row["sample_id"] for row in records}:
        raise ValueError("Merged sample set differs from frozen manifest")
    rows = [rows_by_id[record["sample_id"]] for record in records]
    encoded = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for row in rows).encode()
    if output.is_symlink():
        raise ValueError("Unsafe merged output")
    if output.exists():
        if output.read_bytes() != encoded:
            raise ValueError("Existing merged predictions differ")
    else:
        with tempfile.NamedTemporaryFile(dir=model_root, prefix=".pending-merge-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    summary = {"status": "DECLARED_VIDEOMME_SUBSET_COMPLETE_REFERENCE_SCORER", "protocol": VIDEOMME_PROTOCOL,
               "num_examples": len(rows), "num_sources": len({row["source_video_id"] for row in rows}),
               "by_metric": summarise_metrics(rows), "by_duration": summarise_groups(rows, "duration"),
               "by_task": summarise_groups(rows, "task"), "coverage": validate_videomme_subset(records),
               "predictions_sha256": sha256_file(output), "shards": shard_receipts,
               "provenance": read_json(provenance_path), "subtitles": False, "audio": False,
               "content_audit_scope": "uniform8_sampled_frames",
               "content_audit_limitation": "No content-disjointness claim for unsampled video portions.",
               "timing_note": "Cold/mixed inference; not a warmed latency or throughput benchmark."}
    write_json_once(output.with_suffix(".summary.json"), summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    verify_artifact({"path": str(args.plan), "sha256": args.plan_sha256})
    plan = read_json(args.plan)
    result = preflight(plan, args.plan_sha256)
    if args.preflight_only:
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise ValueError("Exactly eight visible CUDA GPUs are required")
    if os.environ.get("UNIFIED_EVALUATION_ADMITTED_PLAN_SHA256") != args.plan_sha256:
        raise ValueError("parent_gpu_admission_required")
    root = Path(plan["run_root"])
    lock_dir = Path("/home/jinhaopeng/.cache/Paper/DeepProbe-VLM/locks")
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / "p1-stage.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if root.is_symlink() or (root.exists() and not args.resume):
            raise FileExistsError("Run root already exists; use --resume only for an identical frozen plan")
        root.mkdir(parents=True, exist_ok=True)
        if args.resume and any(root.iterdir()) and not (root / "PLAN.json").is_file():
            raise ValueError("Cannot resume a run lacking its frozen plan")
        write_json_once(root / "PLAN.json", plan)
        if (root / "EVALUATION_COMPLETE.json").is_file() and (root / "ARCHIVE_MANIFEST.json").is_file():
            complete = read_json(root / "EVALUATION_COMPLETE.json")
            if complete.get("plan_sha256") != args.plan_sha256:
                raise ValueError("Completed run has different plan identity")
            archive = read_json(root / "ARCHIVE_MANIFEST.json")
            for row in archive["files"]:
                path = root / safe_relative_path(row["path"])
                if path.is_symlink() or path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
                    raise ValueError("Completed run archive payload changed")
            print(json.dumps(complete, indent=2), flush=True)
            return 0
        # Capacity values change as artifacts accumulate; retain each actual check.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        write_json_once(root / f"PREFLIGHT-{stamp}.json", result)
        records = load_manifest(plan["manifest"]["path"], media_root=plan["media_root"])
        env = dict(os.environ, PYTHONPATH=str(EXPERIMENT / "src"), PYTHONNOUSERSITE="1",
                   PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                   CUBLAS_WORKSPACE_CONFIG=":4096:8")
        summaries = {}
        for name in plan["model_sequence"]:
            model_root = root / name
            model_root.mkdir(exist_ok=True)
            provenance = model_root / "CHECKPOINT_PROVENANCE.json"
            write_json_once(provenance, result["checkpoints"][name])
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                       "--nproc_per_node=8", "--max_restarts=0", "-m", "deepprobe_vlm.evaluate",
                       "--checkpoint", result["checkpoints"][name]["checkpoint"],
                       "--manifest", plan["manifest"]["path"], "--media-root", plan["media_root"],
                       "--output", str(model_root / "predictions.jsonl"), "--max-new-tokens", "16",
                       "--protocol", VIDEOMME_PROTOCOL, "--distributed-env", "--run-provenance", str(provenance)]
            if args.resume:
                command.append("--resume")
            print(json.dumps({"event": "START_BENCHMARK", "model": name, "command": command}), flush=True)
            with (root / f"{name}-{stamp}.log").open("xb") as log:
                process = subprocess.Popen(command, cwd=EXPERIMENT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                for line in iter(process.stdout.readline, b""):
                    log.write(line)
                    log.flush()
                    sys.stdout.buffer.write(line)
                    sys.stdout.buffer.flush()
                code = process.wait()
            write_json_once(root / f"{name}-EXIT-{stamp}.json", {"exit_code": code, "command": command})
            if code:
                return code
            summaries[name] = merge_shards(model_root, records, provenance)
        complete = {"status": "DECLARED_MODELS_BENCHMARK_COMPLETE", "plan_sha256": args.plan_sha256,
                    "num_models": len(summaries), "num_predictions": len(records) * len(summaries), "protocol": VIDEOMME_PROTOCOL,
                    "summaries": {name: {"path": f"{name}/predictions.summary.json",
                        "sha256": sha256_file(root / name / "predictions.summary.json")} for name in summaries}}
        write_json_once(root / "EVALUATION_COMPLETE.json", complete)
        # Parent-owned archival uploads this inventory and exact payloads. No
        # input model/media files are copied into the results archive.
        files = [{"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                 for relative, path in sorted(tree_files(root).items()) if relative != "ARCHIVE_MANIFEST.json"]
        write_json_once(root / "ARCHIVE_MANIFEST.json", {"run_id": plan["run_id"],
            "plan_sha256": args.plan_sha256, "files": files, "file_count": len(files),
            "total_bytes": sum(row["bytes"] for row in files)})
        print(json.dumps(complete, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
