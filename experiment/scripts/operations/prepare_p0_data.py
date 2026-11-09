#!/usr/bin/env python3
"""Prepare a bounded, source-disjoint real-data P0 pilot from local pinned assets.

No downloads. The benchmark output is an annotation-only source registry, NOT an
evaluation manifest or evidence that benchmark media is available. A READY.json
is written only after the existing media validator and offline token-budget CLI
both pass. Never truncate answers or replace a failed example to satisfy a cap.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from deepprobe_vlm.manifest import resolve_media_path, validate_disjoint, validate_records

_spec = importlib.util.spec_from_file_location("p0_official_converter", ROOT / "scripts/data/convert.py")
convert = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(convert)

LLAVA_ANNOTATION = "0_30_s_youtube_v0_1/0_30_s_youtube_oe_v0_1_qa_processed.json"
LLAVA_ARCHIVE = "0_30_s_youtube_v0_1/0_30_s_youtube_v0_1_videos_1.tar.gz"
VIDEOMME_ANNOTATION = "videomme/test-00000-of-00001.parquet"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_receipt(path, dataset):
    spec = json.loads((ROOT / "configs/datasets.json").read_text())["datasets"][dataset]
    receipt = json.loads(path.read_text())
    if receipt.get("repo_id") != spec["repo_id"] or receipt.get("revision") != spec["revision"]:
        raise ValueError(f"{dataset}: receipt does not match the configured immutable repository revision")
    if receipt.get("downloaded") is not True:
        raise ValueError(f"{dataset}: a completed downloaded:true receipt is required")
    files = receipt.get("files", [])
    names = [entry["path"] for entry in files]
    if len(names) != len(set(names)):
        raise ValueError(f"{dataset}: duplicate receipt file paths")
    return {"dataset": dataset, "repo_id": spec["repo_id"], "revision": spec["revision"],
            "receipt_sha256": sha256_file(path), "files": {entry["path"]: entry for entry in files}}


def verify_local_file(root, relative, receipt):
    """Authenticate local bytes against the pinned downloader's recorded hashes."""
    if relative not in receipt["files"]:
        raise ValueError(f"file absent from completed download receipt: {relative}")
    entry = receipt["files"][relative]
    path = resolve_media_path(relative, root)
    expected = entry.get("local_sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"receipt lacks verified local SHA256: {relative}")
    if not path.is_file() or path.stat().st_size != entry.get("size_bytes"):
        raise ValueError(f"downloaded file missing or size mismatch: {relative}")
    actual = sha256_file(path)
    if actual != expected or (entry.get("lfs_sha256") and actual != entry["lfs_sha256"]):
        raise ValueError(f"downloaded file hash mismatch: {relative}")
    if not entry.get("lfs_sha256") and entry.get("git_blob_sha1"):
        digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry["git_blob_sha1"]:
            raise ValueError(f"downloaded Git blob hash mismatch: {relative}")
    return path, {"dataset": receipt["dataset"], "repo_id": receipt["repo_id"],
                  "revision": receipt["revision"], "relative_path": relative,
                  "size_bytes": path.stat().st_size, "sha256": actual,
                  "receipt_sha256": receipt["receipt_sha256"]}


def benchmark_entries(dataset, relative, rows, revision, task=None, allow_unresolved=False):
    """Extract identities only; do not invoke converters that require test media."""
    entries = []
    for index, row in enumerate(rows, 1):
        unresolved = False
        if dataset == "videomme":
            video = row["videoID"]  # Numeric video_id is not the original source.
            if not isinstance(video, str) or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video):
                if not allow_unresolved or not isinstance(video, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", video):
                    raise ValueError(f"invalid Video-MME original videoID at annotation row {index}")
                unresolved = True
            source = "youtube:" + video
        elif dataset == "mvbench":
            video = row["video"]
            source = convert.source_id(video, family=convert.MV_TASKS[task][1])
        else:
            raise ValueError("unsupported benchmark registry")
        entries.append({"dataset": dataset, "dataset_revision": revision,
                        "annotation_relative_path": relative, "annotation_row_1based": index,
                        "annotation_row_sha256": canonical_sha(row), "task": task,
                        "upstream_question_id": row.get("question_id"),
                        "upstream_media_id": video, "source_video_id": source,
                        "source_identity_unresolved": unresolved})
    if not entries:
        raise ValueError(f"empty benchmark annotation: {relative}")
    return entries


def llava_candidates(path, relative, revision, source_map, *, unresolved_omissions=None, media_root=None):
    rows = []
    presence = {}
    for index, upstream in enumerate(convert.rows_from(path), 1):
        video = convert.safe_relative_path(upstream["video"])
        try:
            convert.source_id(video, source_map=source_map)
        except ValueError:
            if unresolved_omissions is None or video in source_map:
                raise
            turns = upstream["conversations"]
            if not turns or len(turns) % 2:
                raise ValueError("Unresolved-source row also has invalid conversation structure")
            if video not in presence:
                presence[video] = resolve_media_path(video, media_root).is_file()
            for qa in range(len(turns) // 2):
                human, answer = turns[2 * qa:2 * qa + 2]
                if human["from"] not in {"human", "user"} or answer["from"] not in {"gpt", "assistant"}:
                    raise ValueError("Unresolved-source row has nonalternating conversations")
                identity = f"{revision}|{relative}|row:{index}|qa:{qa}"
                unresolved_omissions.append({"sample_id": "llava:" + hashlib.sha256(identity.encode()).hexdigest()[:24],
                    "source_video_id": None, "video_path": video, "annotation_row_1based": index,
                    "annotation_row_sha256": canonical_sha(upstream), "qa_index_0based": qa,
                    "media_present": presence[video], "selected_split": None, "decision": "omitted",
                    "reasons": ["unresolved_original_training_source"],
                    "policy": "Exclude unidentified sources entirely; never fabricate an original video ID"})
            continue
        converted = list(convert.convert_llava(upstream, SimpleNamespace(input=path, split="train"), source_map))
        if not converted:
            raise ValueError(f"LLaVA annotation row {index} contains no QA pairs")
        for qa_index, row in enumerate(converted):
            if not re.fullmatch(r"youtube:[A-Za-z0-9_-]{11}", row["source_video_id"]):
                raise ValueError("P0 requires the explicit short-YouTube shard; non-YouTube source encountered")
            identity = f"{revision}|{relative}|row:{index}|qa:{qa_index}"
            rows.append({**row, "sample_id": "llava:" + hashlib.sha256(identity.encode()).hexdigest()[:24],
                         "dataset_revision": revision,
                         "annotation_relative_path": relative, "annotation_row_1based": index,
                         "annotation_row_sha256": canonical_sha(upstream),
                         "upstream_annotation_id": upstream.get("id"), "qa_index_0based": qa_index})
    validate_records(rows, check_media=False)
    return rows


def select_source_groups(records, media_root, excluded_sources, *, seed, train_max_qa, validation_max_qa):
    """Choose whole source groups, validation first, using a stable seeded hash.

    If any candidate clip for a source is missing, omit the entire source. This
    preserves all of its annotated QA together instead of making media loss an
    undocumented within-source curation rule. Caps are maxima, not exact targets.
    """
    if train_max_qa <= 0 or validation_max_qa <= 0:
        raise ValueError("QA caps must be positive")
    validate_records(records, check_media=False)
    groups, present, decisions = defaultdict(list), {}, {}
    # Hundreds of thousands of QA rows share media; avoid repeated CPFS metadata
    # walks. Existing candidates still receive the same resolved-root guard.
    media_root = Path(media_root).resolve()
    media_present = {}
    for row in records:
        groups[row["source_video_id"]].append(row)
        relative = row["video_path"]
        if relative not in media_present:
            exists = (media_root / relative).is_file()
            if exists:
                exists = resolve_media_path(relative, media_root).is_file()
            media_present[relative] = exists
        present[row["sample_id"]] = media_present[relative]
    eligible = {}
    for source, group in groups.items():
        reasons = []
        if source in excluded_sources:
            reasons.append("benchmark_source_excluded")
        if not all(present[row["sample_id"]] for row in group):
            reasons.append("source_group_has_missing_media")
        if len(group) > max(train_max_qa, validation_max_qa):
            reasons.append("source_group_exceeds_both_qa_caps")
        if reasons:
            decisions[source] = (None, reasons)
        else:
            eligible[source] = group
    ordered = sorted(eligible, key=lambda source: (hashlib.sha256(
        f"p0-source-selection-v1:{seed}:{source}".encode()).hexdigest(), source))
    selected = {"train": [], "validation": []}
    for split, cap in (("validation", validation_max_qa), ("train", train_max_qa)):
        for source in ordered:
            if source in decisions or len(selected[split]) + len(eligible[source]) > cap:
                continue
            decisions[source] = (split, [])
            selected[split].extend({**row, "split": split} for row in eligible[source])
    for source in ordered:
        if source not in decisions:
            decisions[source] = (None, ["source_group_does_not_fit_remaining_qa_budgets"])
    coverage = []
    for row in sorted(records, key=lambda row: row["sample_id"]):
        split, reasons = decisions[row["source_video_id"]]
        coverage.append({key: row[key] for key in (
            "sample_id", "source_video_id", "video_path", "annotation_row_1based", "qa_index_0based")})
        coverage[-1].update(media_present=present[row["sample_id"]], selected_split=split,
                            decision="selected" if split else "omitted", reasons=reasons)
    for split in selected:
        selected[split].sort(key=lambda row: (row["source_video_id"], row["sample_id"]))
    if selected["train"] and selected["validation"]:
        validate_disjoint(selected["train"], selected["validation"])
    return selected, coverage


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("llava", "videomme", "mvbench"):
        p.add_argument(f"--{name}-root", type=Path, required=True, help="Local downloaded repository root")
        p.add_argument(f"--{name}-plan", type=Path, required=True, help="Completed downloader/ASSET_COMMITTED receipt")
    p.add_argument("--llava-annotation", default=LLAVA_ANNOTATION)
    p.add_argument("--llava-archive", default=LLAVA_ARCHIVE)
    p.add_argument("--media-root", type=Path, required=True, help="Media extracted from the specified LLaVA shard")
    p.add_argument("--output-dir", type=Path, required=True, help="Must not already exist, even if empty")
    p.add_argument("--config", type=Path, required=True, help="Exact config for mandatory offline token preflight")
    p.add_argument("--checkpoint", type=Path, help="Saved tokenizer for --init-checkpoint/--resume, if applicable")
    p.add_argument("--source-map", type=Path, help="Optional audited LLaVA media-path to original-source mapping")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--train-max-qa", type=int, default=256)
    p.add_argument("--validation-max-qa", type=int, default=32)
    p.add_argument("--allow-unresolved-benchmark-sources-for-p0", action="store_true",
                   help="Explicit engineering-only exception: retain malformed upstream IDs and block formal source clearance")
    p.add_argument("--exclude-unresolved-training-sources", action="store_true",
                   help="Explicit pilot curation: record and exclude every QA whose original source cannot be identified")
    return p


def run_check(name, command, output):
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "HF_DATASETS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(command, capture_output=True, text=True, env=environment, check=False)
    for suffix, text in (("stdout.txt", result.stdout), ("stderr.txt", result.stderr)):
        with (output / f"{name}.{suffix}").open("x", encoding="utf-8") as stream:
            stream.write(text)
    return {"command": command, "returncode": result.returncode,
            "stdout": f"{name}.stdout.txt", "stderr": f"{name}.stderr.txt"}


def main(argv=None):
    args = parser().parse_args(argv)
    output = args.output_dir
    if output.exists() or output.is_symlink():
        raise FileExistsError("Output directory already exists; use a fresh directory")
    if not 0 < args.train_max_qa <= 256 or not 0 < args.validation_max_qa <= 32:
        raise ValueError("P0 QA caps must be positive and no larger than train=256, validation=32")
    if "youtube" not in args.llava_annotation or "youtube" not in args.llava_archive:
        raise ValueError("Only an explicitly named LLaVA YouTube shard is supported for P0")
    config_hash = sha256_file(args.config)
    print(json.dumps({"phase": "verify_pinned_assets"}), flush=True)
    source_map = json.loads(args.source_map.read_text()) if args.source_map else {}
    if not isinstance(source_map, dict):
        raise ValueError("source-map must be a JSON object")
    receipts = {key: read_receipt(path, key) for key, path in (
        ("llava_video_178k", args.llava_plan), ("videomme", args.videomme_plan), ("mvbench", args.mvbench_plan))}
    llava_path, llava_proof = verify_local_file(args.llava_root, args.llava_annotation, receipts["llava_video_178k"])
    _, archive_proof = verify_local_file(args.llava_root, args.llava_archive, receipts["llava_video_178k"])
    vm_path, vm_proof = verify_local_file(args.videomme_root, VIDEOMME_ANNOTATION, receipts["videomme"])
    proofs = [llava_proof, archive_proof, vm_proof]
    registry_rows = benchmark_entries("videomme", VIDEOMME_ANNOTATION, convert.rows_from(vm_path),
                                      receipts["videomme"]["revision"],
                                      allow_unresolved=args.allow_unresolved_benchmark_sources_for_p0)
    for task in sorted(convert.MV_TASKS):
        relative = f"json/{task}.json"
        path, proof = verify_local_file(args.mvbench_root, relative, receipts["mvbench"])
        proofs.append(proof)
        registry_rows.extend(benchmark_entries("mvbench", relative, convert.rows_from(path),
                                               receipts["mvbench"]["revision"], task))
    excluded = {row["source_video_id"] for row in registry_rows}
    unresolved = [row for row in registry_rows if row["source_identity_unresolved"]]
    print(json.dumps({"phase": "convert_llava", "unresolved_benchmark_entries": len(unresolved)}), flush=True)
    unresolved_training = []
    candidates = llava_candidates(llava_path, args.llava_annotation, receipts["llava_video_178k"]["revision"], source_map,
                                 unresolved_omissions=unresolved_training if args.exclude_unresolved_training_sources else None,
                                 media_root=args.media_root)
    print(json.dumps({"phase": "select_whole_sources", "identified_qa": len(candidates),
                      "unresolved_training_qa": len(unresolved_training)}), flush=True)
    selected, coverage = select_source_groups(candidates, args.media_root, excluded, seed=args.seed,
                                               train_max_qa=args.train_max_qa, validation_max_qa=args.validation_max_qa)
    coverage = sorted(coverage + unresolved_training, key=lambda row: row["sample_id"])
    output.mkdir(parents=True, exist_ok=False)
    registry = {"kind": "annotation_only_benchmark_source_registry", "is_evaluation_manifest": False,
                "benchmark_media_checked": False, "benchmark_media_complete": "unknown",
                "annotation_tasks": sorted(convert.MV_TASKS), "entries": registry_rows,
                "unique_source_count": len(excluded), "sources": sorted(excluded),
                "all_source_identities_resolved": not unresolved, "unresolved_entries": unresolved,
                "limitation": "Source-ID exclusion only; this is not a usable test manifest or a content-duplicate audit."}
    write_json(output / "benchmark_source_registry.json", registry)
    write_jsonl(output / "coverage.jsonl", coverage)
    for split, rows in selected.items():
        write_jsonl(output / f"{split}.jsonl", rows)
    selected_media = []
    for relative in sorted({row["video_path"] for group in selected.values() for row in group}):
        path = resolve_media_path(relative, args.media_root)
        selected_media.append({"video_path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(output / "selected_media.json", selected_media)
    report = {"status": "prepared_checks_pending", "purpose": "P0_CONNECTIVITY_PILOT_NOT_A_QUALITY_RESULT",
              "seed": args.seed, "selection_policy": "whole_source_groups_seeded_sha256_validation_first_v1",
              "sample_id_policy": "SHA256(revision|annotation_relative_path|row_1based|qa_index); upstream IDs are not unique",
              "qa_caps": {"train": args.train_max_qa, "validation": args.validation_max_qa},
              "counts": {split: {"qa": len(rows), "sources": len({row["source_video_id"] for row in rows})}
                         for split, rows in selected.items()},
              "candidate_qa": len(candidates) + len(unresolved_training),
              "identified_source_candidate_qa": len(candidates),
              "unresolved_training_source_qa_excluded": len(unresolved_training),
              "candidate_sources": len({row["source_video_id"] for row in candidates}),
              "missing_media_qa": sum(not row["media_present"] for row in coverage),
              "omitted_qa_by_reason": dict(Counter(reason for row in coverage for reason in row["reasons"])),
              "input_files": proofs, "media_root": str(args.media_root.resolve()),
              "config_sha256": config_hash, "source_map_sha256": sha256_file(args.source_map) if args.source_map else None,
              "output_sha256": {path.name: sha256_file(path) for path in sorted(output.iterdir())},
              "media_decoded": False, "benchmark_media_checked": False, "checks": [],
              "formal_training_ready": False, "unresolved_benchmark_source_entries": len(unresolved),
              "benchmark_source_identity_gate": "blocked" if unresolved else "passed",
              "limitations": ["Available-media whole-source subset is not a representative full-dataset result.",
                              "Media root is supplied as the extraction of the recorded shard; selected files are hashed, not byte-compared with archive members.",
                              "Source identities cannot detect renamed/re-encoded duplicates or upstream model pretraining overlap."]}
    if not selected["train"] or not selected["validation"]:
        report["status"] = "failed_empty_split"
        write_json(output / "preparation_report.json", report)
        return 2
    train, validation = output / "train.jsonl", output / "validation.jsonl"
    commands = [("media_validation", [sys.executable, str(ROOT / "scripts/data/validate.py"), str(train), str(validation),
                                        "--media-root", str(args.media_root)]),
                ("token_budget", [sys.executable, str(ROOT / "scripts/data/check_token_budget.py"), str(train), str(validation),
                                   "--config", str(args.config), "--local-files-only", "--report", str(output / "token_budget.json")])]
    if args.checkpoint:
        commands[1][1].extend(["--checkpoint", str(args.checkpoint)])
    for name, command in commands:
        check = run_check(name, command, output)
        report["checks"].append(check)
        if name == "token_budget" and check["returncode"] == 0:
            token_report = output / "token_budget.json"
            if not token_report.is_file() or json.loads(token_report.read_text()).get("passed") is not True:
                check["returncode"] = 2
                check["error"] = "Token CLI returned success without an explicit passed:true report"
        report["output_sha256"] = {path.name: sha256_file(path) for path in sorted(output.iterdir())}
        if check["returncode"]:
            report["status"] = f"failed_{name}"
            write_json(output / "preparation_report.json", report)
            print(json.dumps({"status": report["status"], "report": str(output / "preparation_report.json")}))
            return 2
    report["status"] = "p0_media_and_offline_token_preflight_passed"
    write_json(output / "preparation_report.json", report)
    write_json(output / "READY.json", {"status": report["status"], "report_sha256": sha256_file(output / "preparation_report.json"),
                                        "manifest_sha256": {path.name: sha256_file(path) for path in (train, validation)},
                                        "purpose": "P0_ENGINEERING_ONLY", "formal_training_ready": False,
                                        "benchmark_source_identity_gate": report["benchmark_source_identity_gate"],
                                        "media_decoded": False, "gpu_validated": False})
    print(json.dumps({"status": report["status"], "counts": report["counts"], "output_dir": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
