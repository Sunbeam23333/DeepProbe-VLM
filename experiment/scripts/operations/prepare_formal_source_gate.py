#!/usr/bin/env python3
"""Build the fail-closed source gate for formal DeepProbe experiments.

This command does not download data and does not repair upstream identifiers.
For the pinned Video-MME revision it excludes the complete benchmark group 490
(`490-1`, `490-2`, and `490-3`) because its 13-character ``videoID`` is not a
valid YouTube ID.  The fixed evaluation subset is therefore 2,697 questions
from 899 videos.  The raw invalid value is retained only as audit evidence; it
is never truncated or emitted as ``source_video_id``.

``FORMAL_READY.json`` separates training readiness from final-evaluation
readiness.  Formal training may proceed once the pinned source registry, fixed
subset, complete training-candidate coverage, and declared-ID separation pass.
Missing benchmark media or a content audit keeps final evaluation blocked and
must never be described as content-level decontamination.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from deepprobe_vlm.manifest import (  # noqa: E402
    ManifestError,
    load_manifest,
    resolve_media_path,
    validate_disjoint,
)

_convert_spec = importlib.util.spec_from_file_location(
    "formal_source_gate_converter", ROOT / "scripts/data/convert.py"
)
convert = importlib.util.module_from_spec(_convert_spec)
assert _convert_spec.loader is not None
_convert_spec.loader.exec_module(convert)

VIDEOMME_ANNOTATION = "videomme/test-00000-of-00001.parquet"
VIDEOMME_EXPECTED_QUESTIONS = 2700
VIDEOMME_EXPECTED_GROUPS = 900
VIDEOMME_FIXED_QUESTIONS = 2697
VIDEOMME_FIXED_GROUPS = 899
LLAVA_EXPECTED_ANNOTATION_ROWS = 420_200
LLAVA_EXPECTED_CANDIDATE_QA = 1_914_467
LLAVA_EXPECTED_UNRESOLVED_QA = 101_385
EXCLUDED_GROUP = "490"
EXCLUDED_QUESTION_IDS = ("490-1", "490-2", "490-3")
EXCLUDED_RAW_VIDEO_ID = "azZZZbSwLQght"
FORBIDDEN_GUESSED_YOUTUBE_IDS = frozenset(
    EXCLUDED_RAW_VIDEO_ID[offset:offset + 11]
    for offset in range(len(EXCLUDED_RAW_VIDEO_ID) - 10)
)
YOUTUBE_ID = re.compile(r"[A-Za-z0-9_-]{11}")
SHA256 = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def formatted_json_sha(value) -> str:
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def write_jsonl(path: Path, rows) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _dataset_spec(dataset: str) -> dict:
    config = json.loads((ROOT / "configs/datasets.json").read_text(encoding="utf-8"))
    return config["datasets"][dataset]


def read_receipt(path: Path, dataset: str) -> dict:
    """Validate a completed immutable downloader receipt."""
    spec = _dataset_spec(dataset)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("repo_id") != spec["repo_id"] or receipt.get("revision") != spec["revision"]:
        raise ValueError(f"{dataset}: receipt does not match configured immutable revision")
    if receipt.get("downloaded") is not True:
        raise ValueError(f"{dataset}: downloaded:true receipt required")
    files = receipt.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"{dataset}: non-empty receipt files required")
    by_path = {}
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(f"{dataset}: invalid receipt file entry")
        if entry["path"] in by_path:
            raise ValueError(f"{dataset}: duplicate receipt path {entry['path']}")
        by_path[entry["path"]] = entry
    return {
        "dataset": dataset,
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "receipt_sha256": sha256_file(path),
        "files": by_path,
    }


def verify_receipt_file(root: Path, relative: str, receipt: dict) -> tuple[Path, dict]:
    if relative not in receipt["files"]:
        raise ValueError(f"{receipt['dataset']}: receipt missing {relative}")
    entry = receipt["files"][relative]
    expected = entry.get("local_sha256")
    if not isinstance(expected, str) or not SHA256.fullmatch(expected):
        raise ValueError(f"{receipt['dataset']}: verified SHA256 missing for {relative}")
    path = resolve_media_path(relative, root)
    if not path.is_file() or path.stat().st_size != entry.get("size_bytes"):
        raise ValueError(f"{receipt['dataset']}: missing/size-mismatched {relative}")
    actual = sha256_file(path)
    if actual != expected or (entry.get("lfs_sha256") and entry["lfs_sha256"] != actual):
        raise ValueError(f"{receipt['dataset']}: hash mismatch for {relative}")
    return path, {
        "dataset": receipt["dataset"],
        "repo_id": receipt["repo_id"],
        "revision": receipt["revision"],
        "relative_path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": actual,
        "receipt_sha256": receipt["receipt_sha256"],
    }


def canonical_video_group(value) -> str:
    if isinstance(value, bool):
        raise ValueError("Video-MME video_id must be a decimal benchmark index")
    text = str(value).strip()
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError(f"invalid Video-MME video_id: {value!r}")
    return str(int(text))


def canonical_source_id(value) -> str:
    """Require the repository's explicit ``namespace:identity`` representation."""
    if not isinstance(value, str) or value != value.strip() or ":" not in value:
        raise ValueError(f"non-canonical source_video_id: {value!r}")
    namespace, identity = value.split(":", 1)
    if not re.fullmatch(r"[a-z][a-z0-9_]*", namespace) or not identity or ":" in identity:
        raise ValueError(f"non-canonical source_video_id: {value!r}")
    if namespace == "youtube" and not YOUTUBE_ID.fullmatch(identity):
        raise ValueError(f"invalid canonical YouTube source_video_id: {value!r}")
    if namespace == "youtube" and identity in FORBIDDEN_GUESSED_YOUTUBE_IDS:
        raise ValueError(f"forbidden guessed/truncated Video-MME group-490 identity: {value!r}")
    return value


def source_policy(videomme_revision: str, mvbench_revision: str) -> dict:
    return {
        "schema_version": 1,
        "policy_id": "deepprobe-formal-benchmark-source-v1",
        "fail_closed": True,
        "videomme": {
            "repo_id": _dataset_spec("videomme")["repo_id"],
            "revision": videomme_revision,
            "annotation_relative_path": VIDEOMME_ANNOTATION,
            "expected_input": {"questions": VIDEOMME_EXPECTED_QUESTIONS, "video_groups": VIDEOMME_EXPECTED_GROUPS},
            "fixed_subset": {"questions": VIDEOMME_FIXED_QUESTIONS, "video_groups": VIDEOMME_FIXED_GROUPS},
            "source_identity": "youtube:<exact 11-character upstream videoID>",
            "exclusion": {
                "group_key": "canonical decimal upstream video_id",
                "group_value": EXCLUDED_GROUP,
                "question_ids": list(EXCLUDED_QUESTION_IDS),
                "observed_raw_videoID": EXCLUDED_RAW_VIDEO_ID,
                "action": "exclude_entire_source_group",
                "reason": "upstream videoID is 13 characters and is not a valid YouTube source identity",
                "raw_value_is_source_video_id": False,
            },
            "forbidden_repairs": ["truncate", "substring", "guess", "map to numeric video_id"],
            "forbidden_guessed_source_video_ids": [
                "youtube:" + value for value in sorted(FORBIDDEN_GUESSED_YOUTUBE_IDS)
            ],
        },
        "mvbench": {
            "repo_id": _dataset_spec("mvbench")["repo_id"],
            "revision": mvbench_revision,
            "annotation_tasks": sorted(convert.MV_TASKS),
            "use": "declared-source exclusion only; media completeness is a later evaluation gate",
        },
        "formal_training_ready_requires": [
            "pinned_annotations",
            "videomme_fixed_subset",
            "benchmark_source_identity",
            "training_manifests",
            "training_candidate_coverage",
            "training_data_attestation",
            "training_media_integrity",
            "declared_source_leakage",
            "artifact_integrity",
        ],
        "formal_evaluation_ready_also_requires": [
            "benchmark_media",
            "training_media",
            "exact_byte_overlap",
            "perceptual_content_audit",
        ],
        "limitation": "Declared IDs and exact hashes cannot detect renamed or re-encoded duplicates; a bound content audit is mandatory before final evaluation claims.",
    }


def build_videomme_subset(rows, revision: str, annotation_sha256: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Return selected registry rows, exclusion rows, and full coverage."""
    rows = list(rows)
    if len(rows) != VIDEOMME_EXPECTED_QUESTIONS:
        raise ValueError(f"Video-MME pinned annotation must contain {VIDEOMME_EXPECTED_QUESTIONS} QA rows")
    groups = defaultdict(list)
    seen_questions = set()
    for row_number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Video-MME row {row_number}: JSON object required")
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not re.fullmatch(r"[0-9]+-[1-3]", question_id):
            raise ValueError(f"Video-MME row {row_number}: invalid question_id")
        if question_id in seen_questions:
            raise ValueError(f"Video-MME duplicate question_id: {question_id}")
        seen_questions.add(question_id)
        group = canonical_video_group(row.get("video_id"))
        if canonical_video_group(question_id.rsplit("-", 1)[0]) != group:
            raise ValueError(f"Video-MME {question_id}: question/video group mismatch")
        groups[group].append((row_number, row))
    if len(groups) != VIDEOMME_EXPECTED_GROUPS:
        raise ValueError(f"Video-MME pinned annotation must contain {VIDEOMME_EXPECTED_GROUPS} video groups")
    for group, members in groups.items():
        suffixes = {row["question_id"].rsplit("-", 1)[1] for _, row in members}
        raw_ids = {row.get("videoID") for _, row in members}
        if len(members) != 3 or suffixes != {"1", "2", "3"}:
            raise ValueError(f"Video-MME group {group}: exactly question suffixes 1/2/3 required")
        if len(raw_ids) != 1:
            raise ValueError(f"Video-MME group {group}: inconsistent upstream videoID")

    excluded_members = groups.get(EXCLUDED_GROUP)
    if excluded_members is None:
        raise ValueError("Video-MME fixed exclusion group 490 missing")
    if {row["question_id"] for _, row in excluded_members} != set(EXCLUDED_QUESTION_IDS):
        raise ValueError("Video-MME group 490 does not match fixed question IDs")
    if {row.get("videoID") for _, row in excluded_members} != {EXCLUDED_RAW_VIDEO_ID}:
        raise ValueError("Video-MME group 490 raw videoID changed; review and version the policy instead of guessing")

    registry, exclusions, coverage = [], [], []
    for group in sorted(groups, key=lambda value: int(value)):
        for row_number, row in sorted(groups[group], key=lambda item: item[1]["question_id"]):
            base = {
                "dataset": "videomme",
                "dataset_revision": revision,
                "annotation_relative_path": VIDEOMME_ANNOTATION,
                "annotation_sha256": annotation_sha256,
                "annotation_row_1based": row_number,
                "annotation_row_sha256": canonical_sha(row),
                "question_id": row["question_id"],
                "benchmark_video_group": group,
                "upstream_videoID_raw": row.get("videoID"),
            }
            if group == EXCLUDED_GROUP:
                excluded = {**base, "source_video_id": None,
                            "decision": "excluded", "reason": "invalid_upstream_source_identity_entire_group"}
                exclusions.append(excluded)
                coverage.append(excluded)
                continue
            raw_id = row.get("videoID")
            if not isinstance(raw_id, str) or not YOUTUBE_ID.fullmatch(raw_id):
                raise ValueError(f"Video-MME group {group}: unresolved original source identity")
            canonical_source_id("youtube:" + raw_id)
            selected = {**base, "source_video_id": "youtube:" + raw_id,
                        "decision": "selected", "reason": None}
            registry.append({key: selected[key] for key in selected if key not in {"decision", "reason"}})
            coverage.append(selected)

    selected_groups = {row["benchmark_video_group"] for row in registry}
    selected_sources = {row["source_video_id"] for row in registry}
    if len(registry) != VIDEOMME_FIXED_QUESTIONS or len(selected_groups) != VIDEOMME_FIXED_GROUPS:
        raise ValueError("Video-MME fixed subset count mismatch")
    if len(selected_sources) != VIDEOMME_FIXED_GROUPS:
        raise ValueError("Video-MME selected groups do not map one-to-one to original source IDs")
    if len(exclusions) != 3 or any(row["source_video_id"] is not None for row in exclusions):
        raise ValueError("Video-MME exclusion must contain exactly three source-less rows")
    return registry, exclusions, coverage


def build_mvbench_registry(root: Path, receipt: dict) -> tuple[list[dict], list[dict]]:
    entries, proofs = [], []
    for task in sorted(convert.MV_TASKS):
        relative = f"json/{task}.json"
        path, proof = verify_receipt_file(root, relative, receipt)
        proofs.append(proof)
        seen_rows = 0
        for row_number, row in enumerate(convert.rows_from(path), 1):
            source = convert.source_id(row["video"], family=convert.MV_TASKS[task][1])
            canonical_source_id(source)
            entries.append({
                "dataset": "mvbench",
                "dataset_revision": receipt["revision"],
                "annotation_relative_path": relative,
                "annotation_sha256": proof["sha256"],
                "annotation_row_1based": row_number,
                "annotation_row_sha256": canonical_sha(row),
                "task": task,
                "source_video_id": source,
                "upstream_media_id": row["video"],
            })
            seen_rows += 1
        if not seen_rows:
            raise ValueError(f"MVBench task annotation is empty: {task}")
    return entries, proofs


def load_training_manifests(train_path: Path | None, validation_path: Path | None) -> tuple[dict, dict]:
    report = {
        "provided": bool(train_path and validation_path),
        "valid": False,
        "paths": {},
        "sha256": {},
        "counts": {},
        "errors": [],
    }
    groups = {}
    if not train_path or not validation_path:
        report["errors"].append("both --train-manifest and --validation-manifest are required")
        return groups, report
    for split, path in (("train", train_path), ("validation", validation_path)):
        report["paths"][split] = str(path.resolve())
        if not path.is_file():
            report["errors"].append(f"missing {split} manifest: {path}")
            continue
        report["sha256"][split] = sha256_file(path)
        try:
            rows = load_manifest(path, check_media=False)
        except (ManifestError, OSError, ValueError) as exc:
            report["errors"].append(f"{split}: {exc}")
            continue
        if any(row["split"] != split for row in rows):
            report["errors"].append(f"{split}: every row must declare split={split}")
        if any(row.get("source_identity_unresolved") is True for row in rows):
            report["errors"].append(f"{split}: unresolved source identities are forbidden")
        for row in rows:
            try:
                canonical_source_id(row["source_video_id"])
            except ValueError as exc:
                report["errors"].append(f"{split}: {exc}")
        groups[split] = rows
        report["counts"][split] = {
            "qa": len(rows),
            "source_videos": len({row["source_video_id"] for row in rows}),
            "media_paths": len({path for row in rows for path in ([row["video_path"]] if "video_path" in row else row["image_paths"])}),
        }
    if set(groups) == {"train", "validation"}:
        try:
            validate_disjoint(groups["train"], groups["validation"])
        except ManifestError as exc:
            report["errors"].append(str(exc))
    report["valid"] = not report["errors"] and set(groups) == {"train", "validation"}
    return groups, report


def load_training_coverage(path: Path | None, training: dict, benchmark_sources: set[str]) -> dict:
    """Verify that every selected QA is covered and every unknown source is omitted.

    The coverage file is the auditable candidate universe emitted by the data
    preparation step.  This checker binds all selected rows to the exact train
    or validation manifests and fails if an unresolved or benchmark source is
    selected.
    """
    report = {"provided": path is not None, "valid": False, "errors": [], "counts": {},
              "path": str(path.resolve()) if path else None,
              "sha256": sha256_file(path) if path and path.is_file() else None}
    if path is None:
        report["errors"].append("--training-coverage is required")
        return report
    if not path.is_file():
        report["errors"].append(f"missing training coverage: {path}")
        return report
    manifest_rows = {row["sample_id"]: row for rows in training.values() for row in rows}
    seen, selected, unknown_selected = set(), set(), 0
    unknown_omitted, unknown_without_reason, benchmark_omitted = 0, 0, 0
    decisions = Counter()
    try:
        with path.open(encoding="utf-8") as stream:
            for lineno, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                sample = row.get("sample_id")
                if not isinstance(sample, str) or not sample:
                    report["errors"].append(f"coverage line {lineno}: non-empty sample_id required")
                    continue
                if sample in seen:
                    report["errors"].append(f"coverage line {lineno}: duplicate sample_id {sample}")
                    continue
                seen.add(sample)
                decision = row.get("decision")
                reasons = row.get("reasons")
                source = row.get("source_video_id")
                if decision not in {"selected", "omitted"} or not isinstance(reasons, list):
                    report["errors"].append(f"coverage line {lineno}: invalid decision/reasons")
                    continue
                decisions[decision] += 1
                if decision == "selected":
                    selected.add(sample)
                    manifest = manifest_rows.get(sample)
                    if manifest is None:
                        report["errors"].append(f"coverage selected row absent from manifests: {sample}")
                        continue
                    if not isinstance(source, str) or not source.strip() or source != manifest["source_video_id"]:
                        report["errors"].append(f"coverage selected source mismatch/unresolved: {sample}")
                        if source is None:
                            unknown_selected += 1
                    if row.get("selected_split") != manifest["split"]:
                        report["errors"].append(f"coverage selected split mismatch: {sample}")
                    if source in benchmark_sources:
                        report["errors"].append(f"coverage selected benchmark source: {sample}")
                else:
                    if sample in manifest_rows:
                        report["errors"].append(f"coverage omitted row appears in manifests: {sample}")
                    if source is None:
                        unknown_omitted += 1
                        if "unresolved_original_training_source" not in reasons:
                            unknown_without_reason += 1
                            report["errors"].append(f"coverage unresolved omission lacks reason: {sample}")
                    if source in benchmark_sources:
                        benchmark_omitted += 1
                        if "benchmark_source_excluded" not in reasons:
                            report["errors"].append(f"coverage benchmark omission lacks reason: {sample}")
    except (OSError, json.JSONDecodeError) as exc:
        report["errors"].append(f"cannot parse training coverage: {exc}")
    missing = sorted(set(manifest_rows) - selected)
    if missing:
        report["errors"].append(f"{len(missing)} manifest rows absent from selected coverage")
        report["missing_manifest_sample_ids"] = missing[:100]
    report["counts"] = {"candidate_qa": len(seen), "selected_qa": decisions["selected"],
                        "omitted_qa": decisions["omitted"],
                        "unresolved_source_qa_selected": unknown_selected,
                        "unresolved_source_qa_omitted": unknown_omitted,
                        "benchmark_source_qa_omitted": benchmark_omitted}
    report["all_unresolved_sources_omitted"] = unknown_selected == 0 and unknown_without_reason == 0
    report["valid"] = not report["errors"] and selected == set(manifest_rows)
    return report


def validate_training_data_attestation(directory: Path | None, registry_sha256: str,
                                       training_report: dict, coverage_report: dict) -> dict:
    result = {"provided": directory is not None, "valid": False, "errors": [],
              "directory": str(directory.resolve()) if directory else None}
    if directory is None:
        result["errors"].append("--training-data-dir from prepare_formal_training_data.py is required")
        return result
    report_path, ready_path = directory / "preparation_report.json", directory / "READY.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["errors"].append(f"cannot read formal training-data attestation: {exc}")
        return result
    result["report_sha256"] = sha256_file(report_path)
    result["ready_sha256"] = sha256_file(ready_path)
    if ready.get("purpose") != "FORMAL_TRAINING_DATA" or ready.get("formal_data_ready") is not True:
        result["errors"].append("training READY must declare FORMAL_TRAINING_DATA/formal_data_ready:true")
    if report.get("status") != "formal_training_data_ready" or report.get("formal_data_ready") is not True:
        result["errors"].append("training preparation report is not ready")
    if ready.get("report_sha256") != result["report_sha256"]:
        result["errors"].append("training preparation report hash mismatch")
    if ready.get("benchmark_registry_sha256") != registry_sha256:
        result["errors"].append("training data used a different benchmark source registry")
    if report.get("benchmark_registry", {}).get("sha256") != registry_sha256:
        result["errors"].append("training report benchmark registry hash mismatch")
    if report.get("dataset_revision") != _dataset_spec("llava_video_178k")["revision"]:
        result["errors"].append("training data does not use the pinned LLaVA revision")
    annotation = report.get("annotation", {})
    if annotation.get("dataset") != "llava_video_178k" or \
            annotation.get("relative_path") != "0_30_s_youtube_v0_1/0_30_s_youtube_oe_v0_1_qa_processed.json" or \
            not isinstance(annotation.get("sha256"), str) or not SHA256.fullmatch(annotation["sha256"]):
        result["errors"].append("training report lacks the pinned LLaVA annotation proof")
    if ready.get("annotation_sha256") != annotation.get("sha256"):
        result["errors"].append("training READY annotation hash mismatch")
    expected = {"annotation_rows": LLAVA_EXPECTED_ANNOTATION_ROWS,
                "candidate_qa": LLAVA_EXPECTED_CANDIDATE_QA,
                "unresolved_source_qa": LLAVA_EXPECTED_UNRESOLVED_QA}
    if report.get("expected_pinned_counts") != expected:
        result["errors"].append("training report pinned candidate-count contract mismatch")
    if report.get("scan", {}).get("annotation_rows") != LLAVA_EXPECTED_ANNOTATION_ROWS or \
            report.get("scan", {}).get("candidate_qa") != LLAVA_EXPECTED_CANDIDATE_QA or \
            report.get("scan", {}).get("unresolved_source_qa") != LLAVA_EXPECTED_UNRESOLVED_QA:
        result["errors"].append("training report did not traverse the complete pinned candidate universe")
    if coverage_report.get("counts", {}).get("candidate_qa") != LLAVA_EXPECTED_CANDIDATE_QA:
        result["errors"].append("coverage row count is not the complete pinned QA expansion")
    counts = report.get("counts", {})
    if counts.get("candidate_qa") != LLAVA_EXPECTED_CANDIDATE_QA or \
            counts.get("selected_qa") != sum(value.get("qa", 0) for value in training_report.get("counts", {}).values()):
        result["errors"].append("training report/manifest counts mismatch")
    files = {"train.jsonl": directory / "train.jsonl", "validation.jsonl": directory / "validation.jsonl",
             "coverage.jsonl": directory / "coverage.jsonl", "source_groups.jsonl": directory / "source_groups.jsonl",
             "training_media_inventory.json": directory / "training_media_inventory.json"}
    ready_hashes = {**ready.get("manifest_sha256", {}),
                    "coverage.jsonl": ready.get("coverage_sha256"),
                    "source_groups.jsonl": ready.get("source_groups_sha256"),
                    "training_media_inventory.json": ready.get("training_media_inventory_sha256")}
    report_hashes = report.get("output_sha256", {})
    for name, path in files.items():
        if not path.is_file():
            result["errors"].append(f"training data artifact missing: {name}")
            continue
        actual = sha256_file(path)
        if ready_hashes.get(name) != actual or report_hashes.get(name) != actual:
            result["errors"].append(f"training data artifact hash mismatch: {name}")
    if ready.get("coverage_sha256") != coverage_report.get("sha256"):
        result["errors"].append("validated coverage hash differs from training READY")
    for split in ("train", "validation"):
        if ready.get("manifest_sha256", {}).get(f"{split}.jsonl") != training_report.get("sha256", {}).get(split):
            result["errors"].append(f"validated {split} manifest hash differs from training READY")
    inventory_path = files["training_media_inventory.json"]
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["errors"].append(f"cannot read training media inventory: {exc}")
        inventory = {}
    entries = inventory.get("entries")
    inventory_report = report.get("training_media_inventory", {})
    if inventory.get("schema_version") != 1 or \
            inventory.get("kind") != "formal_training_media_sha256_inventory" or \
            not isinstance(entries, list) or not entries:
        result["errors"].append("training media inventory schema/entries are invalid")
        entries = []
    paths = set()
    entry_errors = []
    for index, entry in enumerate(entries):
        relative = entry.get("relative_path") if isinstance(entry, dict) else None
        source = entry.get("source_video_id") if isinstance(entry, dict) else None
        size = entry.get("size_bytes") if isinstance(entry, dict) else None
        digest = entry.get("sha256") if isinstance(entry, dict) else None
        try:
            if not isinstance(relative, str):
                raise ValueError("relative_path missing")
            convert.safe_relative_path(relative)
            canonical_source_id(source)
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ValueError("positive size_bytes required")
            if not isinstance(digest, str) or not SHA256.fullmatch(digest):
                raise ValueError("SHA-256 required")
            if relative in paths:
                raise ValueError("duplicate relative_path")
            paths.add(relative)
        except (ManifestError, ValueError) as exc:
            entry_errors.append(f"entry {index}: {exc}")
    if entry_errors:
        result["errors"].append("invalid training media inventory entries: " + "; ".join(entry_errors[:10]))
    entries_sha256 = canonical_sha(entries)
    total_bytes = sum(entry.get("size_bytes", 0) for entry in entries if isinstance(entry, dict)
                      and isinstance(entry.get("size_bytes"), int) and not isinstance(entry.get("size_bytes"), bool))
    inventory_sha256 = sha256_file(inventory_path) if inventory_path.is_file() else None
    if (inventory.get("file_count") != len(entries)
            or inventory.get("total_bytes") != total_bytes
            or inventory.get("entries_sha256") != entries_sha256):
        result["errors"].append("training media inventory counts/content digest mismatch")
    if (inventory_report.get("relative_path") != "training_media_inventory.json"
            or inventory_report.get("sha256") != inventory_sha256
            or inventory_report.get("file_count") != len(entries)
            or inventory_report.get("total_bytes") != total_bytes
            or inventory_report.get("entries_sha256") != entries_sha256):
        result["errors"].append("training report media inventory attestation mismatch")
    if (ready.get("training_media_inventory_sha256") != inventory_sha256
            or ready.get("training_media_entries_sha256") != entries_sha256
            or ready.get("training_media_file_count") != len(entries)
            or ready.get("training_media_total_bytes") != total_bytes):
        result["errors"].append("training READY media inventory attestation mismatch")
    result["media_inventory"] = {
        "path": str(inventory_path.resolve()), "sha256": inventory_sha256,
        "entries_sha256": entries_sha256, "file_count": len(entries),
        "total_bytes": total_bytes,
    }
    result["valid"] = not result["errors"]
    return result


def leakage_report(training: dict, benchmark_entries: list[dict], training_report: dict) -> dict:
    benchmark_sources = {row["source_video_id"] for row in benchmark_entries}
    overlaps = []
    split_sources = {split: {row["source_video_id"] for row in rows} for split, rows in training.items()}
    for split, sources in sorted(split_sources.items()):
        for source in sorted(sources & benchmark_sources):
            overlaps.append({"split": split, "source_video_id": source})
    cross_split = sorted(split_sources.get("train", set()) & split_sources.get("validation", set()))
    checked = training_report["valid"]
    return {
        "checked": checked,
        "passed": checked and not overlaps and not cross_split,
        "benchmark_source_count": len(benchmark_sources),
        "training_source_counts": {key: len(value) for key, value in split_sources.items()},
        "benchmark_overlaps": overlaps,
        "cross_train_validation_sources": cross_split,
        "method": "exact declared source_video_id equality",
        "limitation": "Does not detect renamed or re-encoded content.",
    }


def media_inventory(rows: list[dict], media_root: Path | None, *, benchmark: bool = False,
                    video_prefix: str = "videos") -> dict:
    if media_root is None:
        expected = len({row["source_video_id"] for row in rows}) if benchmark else len({
            path for row in rows for path in ([row["video_path"]] if "video_path" in row else row["image_paths"])
        })
        return {"checked": False, "complete": False, "expected_files": expected,
                "present_files": 0, "missing": [], "entries": [],
                "errors": [], "reason": "media root not provided"}
    paths = {}
    conflicts = []
    if benchmark:
        for row in rows:
            source = row["source_video_id"]
            relative = f"{video_prefix.rstrip('/')}/{source.removeprefix('youtube:')}.mp4"
            previous = paths.setdefault(relative, source)
            if previous != source:
                conflicts.append(f"{relative}: {previous} != {source}")
    else:
        for row in rows:
            for relative in ([row["video_path"]] if "video_path" in row else row["image_paths"]):
                previous = paths.setdefault(relative, row["source_video_id"])
                if previous != row["source_video_id"]:
                    conflicts.append(f"{relative}: {previous} != {row['source_video_id']}")
    entries, missing, errors = [], [], list(conflicts)
    for relative in sorted(paths):
        path = resolve_media_path(relative, media_root)
        if not path.is_file():
            missing.append(relative)
            continue
        if path.stat().st_size <= 0:
            errors.append(f"empty media file: {relative}")
            continue
        entries.append({"relative_path": relative, "source_video_id": paths[relative],
                        "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {
        "checked": True,
        "complete": not missing and not errors and len(entries) == len(paths),
        "root": str(media_root.resolve()),
        "expected_files": len(paths),
        "present_files": len(entries),
        "missing": missing,
        "errors": errors,
        "entries": entries,
    }


def validate_attested_training_media(attestation: dict, actual: dict) -> dict:
    """Require current selected media bytes to equal the preparation inventory."""
    result = {"checked": actual.get("checked") is True, "passed": False, "errors": []}
    metadata = attestation.get("media_inventory")
    if attestation.get("valid") is not True or not isinstance(metadata, dict):
        result["errors"].append("formal training-data media attestation is invalid")
        return result
    if actual.get("checked") is not True:
        result["errors"].append("--training-media-root is required for formal training readiness")
        return result
    if actual.get("complete") is not True:
        result["errors"].append(
            f"current training media is incomplete: missing={len(actual.get('missing', []))}, "
            f"errors={actual.get('errors', [])[:10]}"
        )
        return result
    try:
        inventory = json.loads(Path(metadata["path"]).read_text(encoding="utf-8"))
    except (KeyError, OSError, json.JSONDecodeError) as exc:
        result["errors"].append(f"cannot reload attested training media inventory: {exc}")
        return result
    expected = inventory.get("entries")
    if expected != actual.get("entries"):
        result["errors"].append("current training media size/SHA-256 inventory differs from attested bytes")
        return result
    if (metadata.get("entries_sha256") != canonical_sha(expected)
            or metadata.get("file_count") != len(expected)
            or metadata.get("total_bytes") != sum(entry["size_bytes"] for entry in expected)):
        result["errors"].append("attested training media inventory metadata changed")
        return result
    result.update({"passed": True, "file_count": len(expected),
                   "total_bytes": metadata["total_bytes"],
                   "entries_sha256": metadata["entries_sha256"]})
    return result


def exact_overlap_report(training_inventory: dict, benchmark_inventory: dict) -> dict:
    checked = training_inventory["complete"] and benchmark_inventory["complete"]
    matches = []
    if checked:
        training_hashes = defaultdict(list)
        for entry in training_inventory["entries"]:
            training_hashes[entry["sha256"]].append(entry)
        for benchmark in benchmark_inventory.get("original_entries", benchmark_inventory["entries"]):
            for training in training_hashes.get(benchmark["sha256"], []):
                matches.append({"sha256": benchmark["sha256"],
                                "training_path": training["relative_path"],
                                "benchmark_path": benchmark["relative_path"]})
    return {"checked": checked, "passed": checked and not matches,
            "duplicate_pairs": matches, "method": "original complete-file SHA-256 equality"}


def audit_binding(annotation_sha256: str, selected_sources: set[str], training_report: dict,
                  coverage_report: dict) -> dict:
    return {
        "videomme_annotation_sha256": annotation_sha256,
        "videomme_selected_source_ids_sha256": canonical_sha(sorted(selected_sources)),
        "training_manifest_sha256": training_report["sha256"],
        "training_counts": training_report["counts"],
        "training_coverage_sha256": coverage_report["sha256"],
        "videomme_fixed_questions": VIDEOMME_FIXED_QUESTIONS,
        "videomme_fixed_video_groups": VIDEOMME_FIXED_GROUPS,
    }


def content_audit_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_content_audit", ROOT / "scripts/operations/audit_benchmark_content.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def validate_content_audit(path: Path | None, required_binding: dict,
                           training_inventory: dict | None = None,
                           benchmark_inventory: dict | None = None) -> dict:
    result = {"provided": path is not None, "passed": False, "errors": [],
              "required_binding": required_binding}
    if path is None:
        result["errors"].append("--content-audit is required")
        return result
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["errors"].append(f"cannot read content audit: {exc}")
        return result
    result["path"] = str(path.resolve())
    result["sha256"] = sha256_file(path)
    if audit.get("schema_version") != 1:
        result["errors"].append("content audit schema_version must be 1")
    if audit.get("binding") != required_binding:
        result["errors"].append("content audit binding does not match current annotations/manifests")
    method = audit.get("method")
    if not isinstance(method, str) or not method.strip():
        result["errors"].append("content audit must describe its method")
    for key in ("exact_duplicate_pairs", "perceptual_duplicate_pairs"):
        if audit.get(key) != []:
            result["errors"].append(f"content audit {key} must be an explicit empty list")
    if audit.get("passed") is not True:
        result["errors"].append("content audit passed:true required")
    if training_inventory is not None and benchmark_inventory is not None:
        content = content_audit_module()
        if audit.get("media_binding") != content.media_binding(training_inventory, benchmark_inventory):
            result["errors"].append("content audit media SHA inventories do not match current bytes")
        if audit.get("policy") != content.POLICY or audit.get("scope") != "uniform8_sampled_frames":
            result["errors"].append("explicit supported sampled-frame audit policy required")
        if audit.get("scan_complete") is not True or audit.get("errors") != [] or audit.get("uninformative_sources") != []:
            result["errors"].append("content audit has missing, failed, or inconclusive scans")
        expected_sources = benchmark_inventory.get("expected_source_videos", benchmark_inventory["expected_files"])
        if (audit.get("training_file_count") != training_inventory["expected_files"]
                or audit.get("benchmark_source_count") != expected_sources):
            result["errors"].append("content audit media counts mismatch")
        for name in ("training_fingerprints_receipt", "benchmark_fingerprints"):
            artifact = audit.get("artifacts", {}).get(name, {})
            try:
                artifact_path = Path(artifact["path"])
                if not artifact_path.is_absolute():
                    artifact_path = resolve_media_path(artifact["path"], path.parent)
                if sha256_file(artifact_path) != artifact.get("sha256"):
                    raise ValueError("SHA mismatch")
            except (KeyError, OSError, ValueError) as exc:
                result["errors"].append(f"content audit {name} evidence invalid: {exc}")
        result["scope"] = "uniform8_sampled_frames"
        result["limitation"] = content.POLICY["limitation"]
    result["passed"] = not result["errors"]
    return result


def gate(status: bool | None, evidence: dict | str | None = None) -> dict:
    state = "blocked" if status is None else ("passed" if status else "failed")
    result = {"status": state}
    if evidence is not None:
        result["evidence"] = evidence
    return result


def readiness_gates(training_report: dict, coverage_report: dict, attestation: dict,
                    training_media_attestation: dict, leakage: dict,
                    benchmark_media: dict, training_media: dict, exact_overlap: dict,
                    content_audit: dict) -> tuple[dict, dict]:
    training_state = True if training_report["valid"] else None if not training_report["provided"] else False
    leakage_state = leakage["passed"] if leakage["checked"] else None
    benchmark_media_state = benchmark_media["complete"] if benchmark_media["checked"] else None
    training_media_state = training_media["complete"] if training_media["checked"] else None
    exact_state = exact_overlap["passed"] if exact_overlap["checked"] else None
    audit_state = content_audit["passed"] if content_audit["provided"] else None
    coverage_state = True if coverage_report["valid"] else None if not coverage_report["provided"] else False
    attestation_state = True if attestation["valid"] else None if not attestation["provided"] else False
    media_attestation_state = (True if training_media_attestation["passed"] else
                               None if not training_media_attestation["checked"] else False)
    training_gates = {
        "pinned_annotations": gate(True),
        "videomme_fixed_subset": gate(True, {"questions": VIDEOMME_FIXED_QUESTIONS, "video_groups": VIDEOMME_FIXED_GROUPS}),
        "benchmark_source_identity": gate(True),
        "training_manifests": gate(training_state, training_report["errors"]),
        "training_candidate_coverage": gate(coverage_state, coverage_report["errors"]),
        "training_data_attestation": gate(attestation_state, attestation["errors"]),
        "training_media_integrity": gate(media_attestation_state,
                                         training_media_attestation["errors"]),
        "declared_source_leakage": gate(leakage_state, {"benchmark_overlaps": len(leakage["benchmark_overlaps"]),
                                                         "cross_split": len(leakage["cross_train_validation_sources"])}),
        "artifact_integrity": gate(True, "all generated artifacts are SHA-256 bound in FORMAL_READY.json"),
    }
    evaluation_gates = {
        "benchmark_media": gate(benchmark_media_state, {"present": benchmark_media["present_files"],
                                                         "expected": benchmark_media["expected_files"]}),
        "training_media": gate(training_media_state, {"present": training_media["present_files"],
                                                       "expected": training_media["expected_files"]}),
        "exact_byte_overlap": gate(exact_state, {"duplicate_pairs": len(exact_overlap["duplicate_pairs"])}),
        "perceptual_content_audit": gate(audit_state, content_audit["errors"]),
    }
    return training_gates, evaluation_gates


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--videomme-root", type=Path, required=True)
    p.add_argument("--videomme-plan", type=Path, required=True)
    p.add_argument("--mvbench-root", type=Path, required=True)
    p.add_argument("--mvbench-plan", type=Path, required=True)
    p.add_argument("--train-manifest", type=Path)
    p.add_argument("--validation-manifest", type=Path)
    p.add_argument("--training-coverage", type=Path,
                   help="Complete candidate coverage JSONL binding selected/omitted QA to train/dev")
    p.add_argument("--training-data-dir", type=Path,
                   help="Immutable output of prepare_formal_training_data.py; preferred and required for readiness")
    p.add_argument("--training-media-root", type=Path)
    p.add_argument("--videomme-media-root", type=Path)
    p.add_argument("--videomme-video-prefix", default="videos")
    p.add_argument("--videomme-frame-cache-receipt", type=Path,
                   help="Verified source-bound exact uniform8 lossless PNG cache; requires media root")
    p.add_argument("--content-audit", type=Path,
                   help="Bound exact+perceptual duplicate audit; source-ID checks alone cannot replace it")
    p.add_argument("--output-dir", type=Path, default=Path("cache/formal-data-v1"))
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError("output directory already exists; use a fresh immutable directory")
    if bool(args.train_manifest) != bool(args.validation_manifest):
        raise ValueError("provide both train and validation manifests, or neither")
    if args.training_data_dir and any((args.train_manifest, args.validation_manifest, args.training_coverage)):
        raise ValueError("--training-data-dir cannot be combined with individual training artifact paths")

    vm_receipt = read_receipt(args.videomme_plan, "videomme")
    mv_receipt = read_receipt(args.mvbench_plan, "mvbench")
    vm_path, vm_proof = verify_receipt_file(args.videomme_root, VIDEOMME_ANNOTATION, vm_receipt)
    vm_registry, exclusions, coverage = build_videomme_subset(
        convert.rows_from(vm_path), vm_receipt["revision"], vm_proof["sha256"]
    )
    mv_registry, mv_proofs = build_mvbench_registry(args.mvbench_root, mv_receipt)
    all_registry = vm_registry + mv_registry
    benchmark_sources = {row["source_video_id"] for row in all_registry}
    if any(row.get("source_video_id") == "youtube:" + EXCLUDED_RAW_VIDEO_ID for row in all_registry):
        raise AssertionError("invalid Video-MME value leaked into source registry")

    registry_document = {
        "schema_version": 1,
        "kind": "annotation_only_benchmark_source_registry",
        "is_evaluation_manifest": False,
        "contains_questions_or_answers": False,
        "all_source_identities_resolved": True,
        "datasets": {
            "videomme": {"questions": len(vm_registry),
                         "video_groups": len({row["benchmark_video_group"] for row in vm_registry}),
                         "source_videos": len({row["source_video_id"] for row in vm_registry})},
            "mvbench": {"annotation_rows": len(mv_registry),
                        "source_videos": len({row["source_video_id"] for row in mv_registry}),
                        "tasks": len({row["task"] for row in mv_registry})},
        },
        "unique_source_count": len(benchmark_sources),
        "sources": sorted(benchmark_sources),
        "entries": all_registry,
        "limitation": "Annotation registry only; media completeness and content-duplicate checks are separate gates.",
    }
    registry_sha256 = formatted_json_sha(registry_document)
    if args.training_data_dir:
        train_path = args.training_data_dir / "train.jsonl"
        validation_path = args.training_data_dir / "validation.jsonl"
        coverage_path = args.training_data_dir / "coverage.jsonl"
    else:
        train_path, validation_path, coverage_path = (
            args.train_manifest, args.validation_manifest, args.training_coverage
        )
    training, training_report = load_training_manifests(train_path, validation_path)
    coverage_report = load_training_coverage(coverage_path, training, benchmark_sources)
    attestation = validate_training_data_attestation(
        args.training_data_dir, registry_sha256, training_report, coverage_report
    )
    leak = leakage_report(training, all_registry, training_report)
    flat_training = [row for split in ("train", "validation") for row in training.get(split, [])]
    if args.videomme_frame_cache_receipt:
        if args.videomme_media_root is None:
            raise ValueError("frame-cache receipt requires --videomme-media-root")
        benchmark_media = content_audit_module().frame_cache_inventory(
            args.videomme_frame_cache_receipt, args.videomme_media_root,
            {row["source_video_id"] for row in vm_registry},
        )
    else:
        benchmark_media = media_inventory(vm_registry, args.videomme_media_root, benchmark=True,
                                          video_prefix=args.videomme_video_prefix)
    training_media = media_inventory(flat_training, args.training_media_root)
    training_media_attestation = validate_attested_training_media(attestation, training_media)
    exact_overlap = exact_overlap_report(training_media, benchmark_media)
    binding = audit_binding(vm_proof["sha256"], {row["source_video_id"] for row in vm_registry},
                            training_report, coverage_report)
    content_audit = validate_content_audit(args.content_audit, binding, training_media, benchmark_media)
    training_gates, evaluation_gates = readiness_gates(
        training_report, coverage_report, attestation, training_media_attestation, leak,
        benchmark_media, training_media,
        exact_overlap, content_audit
    )
    formal_training_ready = all(item["status"] == "passed" for item in training_gates.values())
    formal_evaluation_ready = formal_training_ready and all(
        item["status"] == "passed" for item in evaluation_gates.values()
    )

    policy = source_policy(vm_receipt["revision"], mv_receipt["revision"])
    exclusion_document = {
        "schema_version": 1,
        "policy_id": policy["policy_id"],
        "excluded_group_count": 1,
        "excluded_question_count": len(exclusions),
        "excluded_source_video_ids": [],
        "group": policy["videomme"]["exclusion"],
        "rows": exclusions,
        "assertion": "No original source_video_id is assigned or guessed for this excluded group.",
    }
    report = {
        "evaluation_content_scope": content_audit.get("scope"),
        "evaluation_content_limitation": content_audit.get("limitation"),
        "benchmark_input_num_frames": 8 if args.videomme_frame_cache_receipt else None,
        "schema_version": 1,
        "status": "ready_for_evaluation" if formal_evaluation_ready else
                  "ready_for_training" if formal_training_ready else "blocked",
        "formal_training_ready": formal_training_ready,
        "formal_evaluation_ready": formal_evaluation_ready,
        "benchmark_source_identity_gate": "passed_on_declared_subset",
        "training_readiness_gates": training_gates,
        "evaluation_readiness_gates": evaluation_gates,
        "input_proofs": [vm_proof, *mv_proofs],
        "training_manifests": training_report,
        "training_candidate_coverage": coverage_report,
        "training_data_attestation": attestation,
        "training_media_attestation": training_media_attestation,
        "required_content_audit_binding": binding,
        "counts": {
            "videomme_input_questions": len(coverage),
            "videomme_input_video_groups": len({row["benchmark_video_group"] for row in coverage}),
            "videomme_selected_questions": len(vm_registry),
            "videomme_selected_video_groups": len({row["benchmark_video_group"] for row in vm_registry}),
            "videomme_excluded_questions": len(exclusions),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "source_policy.json", policy)
    write_json(args.output_dir / "benchmark_source_registry.json", registry_document)
    write_json(args.output_dir / "source_exclusions.json", exclusion_document)
    write_jsonl(args.output_dir / "benchmark_coverage.jsonl", coverage)
    write_json(args.output_dir / "leakage_report.json", leak)
    write_json(args.output_dir / "benchmark_media_inventory.json", benchmark_media)
    write_json(args.output_dir / "training_media_inventory.json", training_media)
    write_json(args.output_dir / "exact_overlap_report.json", exact_overlap)
    write_json(args.output_dir / "content_audit_validation.json", content_audit)
    write_json(args.output_dir / "source_gate_report.json", report)
    artifact_hashes = {path.name: sha256_file(path) for path in sorted(args.output_dir.iterdir())}
    formal = {
        "evaluation_content_scope": content_audit.get("scope"),
        "evaluation_content_limitation": content_audit.get("limitation"),
        "benchmark_input_num_frames": 8 if args.videomme_frame_cache_receipt else None,
        "schema_version": 1,
        "status": "ready_for_evaluation" if formal_evaluation_ready else
                  "ready_for_training" if formal_training_ready else "blocked",
        "formal_training_ready": formal_training_ready,
        "formal_evaluation_ready": formal_evaluation_ready,
        "benchmark_source_identity_gate": "passed_on_declared_subset",
        "benchmark_media_gate": evaluation_gates["benchmark_media"]["status"],
        "content_duplicate_gate": "passed" if all(evaluation_gates[key]["status"] == "passed" for key in
                                                    ("exact_byte_overlap", "perceptual_content_audit")) else "blocked",
        "policy_id": policy["policy_id"],
        "training_readiness_gates": training_gates,
        "evaluation_readiness_gates": evaluation_gates,
        "artifact_sha256": artifact_hashes,
        "immutable_input_sha256": {
            "videomme_annotation": vm_proof["sha256"],
            "videomme_receipt": vm_receipt["receipt_sha256"],
            "mvbench_receipt": mv_receipt["receipt_sha256"],
            **{f"training_{key}": value for key, value in training_report["sha256"].items()},
            **({"training_coverage": coverage_report["sha256"]} if coverage_report["sha256"] else {}),
            **({"training_media_inventory": attestation.get("media_inventory", {}).get("sha256")}
               if attestation.get("media_inventory", {}).get("sha256") else {}),
        },
    }
    write_json(args.output_dir / "FORMAL_READY.json", formal)
    print(json.dumps({"status": formal["status"], "formal_training_ready": formal_training_ready,
                      "formal_evaluation_ready": formal_evaluation_ready,
                      "counts": report["counts"], "output_dir": str(args.output_dir)}, indent=2))
    return 0 if formal_training_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
