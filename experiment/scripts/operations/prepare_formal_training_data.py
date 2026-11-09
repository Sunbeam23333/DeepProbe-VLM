#!/usr/bin/env python3
"""Prepare the fixed, source-disjoint formal LLaVA training manifests.

The input is the pinned short-YouTube LLaVA annotation and its completed asset
receipt.  Every QA candidate is represented exactly once in ``coverage.jsonl``.
Unresolved original sources, any complete source group with missing media, and
all declared Video-MME/MVBench sources are omitted as complete source groups.
The remaining pool is split by a frozen source-level SHA-256 rule (seed 42,
validation fraction 0.05).  This command performs no tokenization or training.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]

_gate_spec = importlib.util.spec_from_file_location(
    "formal_source_gate_helpers", ROOT / "scripts/operations/prepare_formal_source_gate.py"
)
source_gate = importlib.util.module_from_spec(_gate_spec)
assert _gate_spec.loader is not None
_gate_spec.loader.exec_module(source_gate)
convert = source_gate.convert

LLAVA_ANNOTATION = "0_30_s_youtube_v0_1/0_30_s_youtube_oe_v0_1_qa_processed.json"
LLAVA_ARCHIVE = "0_30_s_youtube_v0_1/0_30_s_youtube_v0_1_videos_1.tar.gz"
EXPECTED_ANNOTATION_ROWS = 420_200
EXPECTED_CANDIDATE_QA = 1_914_467
EXPECTED_UNRESOLVED_QA = 101_385
SPLIT_SEED = 42
VALIDATION_FRACTION = 0.05
SELECTION_POLICY = "all_available_whole_sources_sha256_v1"
SPLIT_POLICY = "sha256(formal-source-split-v1:42:source_video_id)<0.05"


def iter_json_array(path: Path, chunk_chars: int = 1024 * 1024):
    """Stream a top-level JSON array without retaining the complete annotation."""
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as stream:
        buffer, position, eof = "", 0, False

        def refill():
            nonlocal buffer, position, eof
            if position:
                buffer = buffer[position:]
                position = 0
            chunk = stream.read(chunk_chars)
            if chunk:
                buffer += chunk
            else:
                eof = True

        def skip_space():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    return
                refill()

        refill()
        skip_space()
        if position >= len(buffer) or buffer[position] != "[":
            raise ValueError("LLaVA annotation must be a top-level JSON array")
        position += 1
        first = True
        while True:
            skip_space()
            if position >= len(buffer):
                raise ValueError("unterminated annotation JSON array")
            if buffer[position] == "]":
                position += 1
                break
            if not first:
                if buffer[position] != ",":
                    raise ValueError("expected comma between annotation rows")
                position += 1
                skip_space()
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError as exc:
                    if eof:
                        raise ValueError(f"invalid annotation JSON: {exc.msg}") from exc
                    refill()
            if not isinstance(value, dict):
                raise ValueError("every LLaVA annotation row must be an object")
            yield value
            position = end
            first = False
        while not eof:
            refill()
        if buffer[position:].strip():
            raise ValueError("unexpected data after annotation JSON array")


def annotation_rows(path: Path):
    if path.suffix != ".json":
        raise ValueError("formal LLaVA annotation must be the pinned JSON file")
    return iter_json_array(path)


def conversation_pairs(row: dict, row_number: int):
    video = convert.safe_relative_path(row["video"])
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or not conversations or len(conversations) % 2:
        raise ValueError(f"LLaVA row {row_number}: non-empty paired conversations required")
    for qa_index in range(len(conversations) // 2):
        human, answer = conversations[2 * qa_index:2 * qa_index + 2]
        if not isinstance(human, dict) or not isinstance(answer, dict):
            raise ValueError(f"LLaVA row {row_number}: conversation objects required")
        if human.get("from") not in {"human", "user"} or answer.get("from") not in {"gpt", "assistant"}:
            raise ValueError(f"LLaVA row {row_number}: conversations must alternate human/gpt")
        question = re.sub(r"<(?:image|video)>", "", str(human.get("value", ""))).strip()
        response = str(answer.get("value", "")).strip()
        if not question or not response:
            raise ValueError(f"LLaVA row {row_number}: empty question/answer")
        yield qa_index, video, question, response


def sample_id(revision: str, relative: str, row_number: int, qa_index: int) -> str:
    identity = f"{revision}|{relative}|row:{row_number}|qa:{qa_index}"
    return "llava:" + hashlib.sha256(identity.encode()).hexdigest()[:24]


def source_for_video(video: str) -> str | None:
    try:
        source = convert.source_id(video)
        source_gate.canonical_source_id(source)
        return source
    except ValueError:
        return None


def split_for_source(source: str) -> str:
    digest = hashlib.sha256(f"formal-source-split-v1:{SPLIT_SEED}:{source}".encode()).digest()
    unit_interval = int.from_bytes(digest, "big") / 2**256
    return "validation" if unit_interval < VALIDATION_FRACTION else "train"


def load_benchmark_registry(path: Path, ready_path: Path | None = None) -> tuple[set[str], dict]:
    if not path.is_file():
        raise FileNotFoundError(f"benchmark source registry missing: {path}")
    registry_sha = source_gate.sha256_file(path)
    if ready_path is not None:
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        expected = ready.get("artifact_sha256", {}).get("benchmark_source_registry.json")
        if expected != registry_sha:
            raise ValueError("benchmark registry hash does not match source-gate FORMAL_READY")
        if ready.get("benchmark_source_identity_gate") != "passed_on_declared_subset":
            raise ValueError("source-gate benchmark identity gate has not passed on the fixed subset")
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("kind") != "annotation_only_benchmark_source_registry":
        raise ValueError("unexpected benchmark registry kind")
    if registry.get("all_source_identities_resolved") is not True:
        raise ValueError("benchmark registry contains unresolved identities")
    datasets = registry.get("datasets", {})
    vm = datasets.get("videomme", {})
    mv = datasets.get("mvbench", {})
    if vm.get("questions") != 2697 or vm.get("source_videos") != 899:
        raise ValueError("benchmark registry is not the fixed Video-MME 2697-QA/899-video subset")
    if mv.get("tasks") != len(convert.MV_TASKS):
        raise ValueError("benchmark registry does not contain all 20 MVBench tasks")
    sources = registry.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("benchmark registry sources missing")
    source_set = set(sources)
    if len(source_set) != len(sources):
        raise ValueError("benchmark registry contains duplicate source list entries")
    for source in source_set:
        source_gate.canonical_source_id(source)
    entry_sources = {row.get("source_video_id") for row in registry.get("entries", [])}
    if entry_sources != source_set:
        raise ValueError("benchmark registry source list/entries mismatch")
    return source_set, {"path": str(path.resolve()), "sha256": registry_sha,
                        "source_count": len(source_set), "source_gate_ready": str(ready_path.resolve()) if ready_path else None,
                        "source_gate_ready_sha256": source_gate.sha256_file(ready_path) if ready_path else None}


def scan_groups(annotation: Path, media_root: Path, benchmark_sources: set[str]) -> tuple[dict, dict]:
    """First pass: validate all rows and determine whole-source decisions."""
    groups = {}
    media_present = {}
    counters = Counter()
    unresolved_paths = Counter()
    for row_number, row in enumerate(annotation_rows(annotation), 1):
        counters["annotation_rows"] += 1
        pairs = list(conversation_pairs(row, row_number))
        counters["candidate_qa"] += len(pairs)
        video = pairs[0][1]
        source = source_for_video(video)
        if video not in media_present:
            media_present[video] = source_gate.resolve_media_path(video, media_root).is_file()
        if source is None:
            counters["unresolved_source_qa"] += len(pairs)
            unresolved_paths[video] += len(pairs)
            continue
        state = groups.setdefault(source, {"paths": set(), "qa": 0, "all_media_present": True})
        state["paths"].add(video)
        state["qa"] += len(pairs)
        state["all_media_present"] = state["all_media_present"] and media_present[video]
    if counters["annotation_rows"] != EXPECTED_ANNOTATION_ROWS:
        raise ValueError(f"pinned LLaVA annotation row count changed: {counters['annotation_rows']}")
    if counters["candidate_qa"] != EXPECTED_CANDIDATE_QA:
        raise ValueError(f"pinned LLaVA QA expansion count changed: {counters['candidate_qa']}")
    if counters["unresolved_source_qa"] != EXPECTED_UNRESOLVED_QA:
        raise ValueError(f"pinned LLaVA unresolved-source QA count changed: {counters['unresolved_source_qa']}")
    decisions = {}
    for source, state in groups.items():
        reasons = []
        if source in benchmark_sources:
            reasons.append("benchmark_source_excluded")
        if not state["all_media_present"]:
            reasons.append("source_group_has_missing_media")
        decisions[source] = {"selected_split": None if reasons else split_for_source(source), "reasons": reasons,
                             "qa": state["qa"], "paths": sorted(state["paths"]),
                             "all_media_present": state["all_media_present"]}
    stats = dict(counters)
    stats.update({"identified_source_groups": len(groups), "unresolved_video_paths": len(unresolved_paths),
                  "media_paths_checked": len(media_present),
                  "benchmark_source_groups_present": sum(source in benchmark_sources for source in groups)})
    return {"decisions": decisions, "media_present": media_present,
            "unresolved_paths": dict(unresolved_paths)}, stats


def emit(annotation: Path, revision: str, media_root: Path, group_state: dict,
         output_dir: Path) -> tuple[dict, list[dict]]:
    manifests = {
        "train": (output_dir / "train.jsonl").open("x", encoding="utf-8"),
        "validation": (output_dir / "validation.jsonl").open("x", encoding="utf-8"),
    }
    coverage_stream = (output_dir / "coverage.jsonl").open("x", encoding="utf-8")
    counts = Counter()
    split_sources = defaultdict(set)
    try:
        for row_number, row in enumerate(annotation_rows(annotation), 1):
            pairs = conversation_pairs(row, row_number)
            video = convert.safe_relative_path(row["video"])
            source = source_for_video(video)
            if source is None:
                split, reasons = None, ["unresolved_original_training_source"]
            else:
                decision = group_state["decisions"][source]
                split, reasons = decision["selected_split"], decision["reasons"]
            row_sha256 = source_gate.canonical_sha(row)
            for qa_index, pair_video, question, answer in pairs:
                sid = sample_id(revision, LLAVA_ANNOTATION, row_number, qa_index)
                selected = split is not None
                coverage = {
                    "sample_id": sid,
                    "source_video_id": source,
                    "video_path": pair_video,
                    "annotation_row_1based": row_number,
                    "annotation_row_sha256": row_sha256,
                    "qa_index_0based": qa_index,
                    "media_present": group_state["media_present"][video],
                    "selected_split": split,
                    "decision": "selected" if selected else "omitted",
                    "reasons": reasons,
                }
                coverage_stream.write(json.dumps(coverage, ensure_ascii=False, sort_keys=True) + "\n")
                counts["coverage_qa"] += 1
                if not selected:
                    counts["omitted_qa"] += 1
                    for reason in reasons:
                        counts[f"omitted:{reason}"] += 1
                    continue
                record = {
                    "sample_id": sid,
                    "source_video_id": source,
                    "video_path": pair_video,
                    "question": question,
                    "answer": answer,
                    "split": split,
                    "dataset": "llava_video_178k",
                    "dataset_revision": revision,
                    "annotation_relative_path": LLAVA_ANNOTATION,
                    "annotation_row_1based": row_number,
                    "annotation_row_sha256": coverage["annotation_row_sha256"],
                    "upstream_annotation_id": row.get("id"),
                    "qa_index_0based": qa_index,
                }
                manifests[split].write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                counts[f"{split}_qa"] += 1
                split_sources[split].add(source)
    finally:
        coverage_stream.close()
        for stream in manifests.values():
            stream.close()
    if counts["coverage_qa"] != EXPECTED_CANDIDATE_QA:
        raise AssertionError("coverage does not contain the complete pinned QA expansion")
    if split_sources["train"] & split_sources["validation"]:
        raise AssertionError("source-level split leaked between train and validation")
    summary = {
        "candidate_qa": counts["coverage_qa"],
        "selected_qa": counts["train_qa"] + counts["validation_qa"],
        "omitted_qa": counts["omitted_qa"],
        "train": {"qa": counts["train_qa"], "source_videos": len(split_sources["train"])},
        "validation": {"qa": counts["validation_qa"], "source_videos": len(split_sources["validation"])},
        "omitted_qa_by_reason": {key.removeprefix("omitted:"): value for key, value in sorted(counts.items())
                                 if key.startswith("omitted:")},
    }
    source_rows = []
    for source, decision in sorted(group_state["decisions"].items()):
        source_rows.append({"source_video_id": source, **decision,
                            "decision": "selected" if decision["selected_split"] else "omitted"})
    for video, qa in sorted(group_state["unresolved_paths"].items()):
        source_rows.append({"source_video_id": None, "unresolved_video_path": video, "qa": qa,
                            "paths": [video], "all_media_present": group_state["media_present"][video],
                            "selected_split": None, "decision": "omitted",
                            "reasons": ["unresolved_original_training_source"]})
    return summary, source_rows


def build_media_inventory(group_state: dict, media_root: Path) -> dict:
    """Hash every unique media file selected by the frozen source decision.

    File existence alone is insufficient for a reproducible formal run: bytes at
    the same relative path can otherwise change after manifest preparation.  The
    source gate independently re-hashes this exact inventory before authorizing
    training.
    """
    by_path: dict[str, str] = {}
    for source, decision in group_state["decisions"].items():
        if decision["selected_split"] is None:
            continue
        for relative in decision["paths"]:
            previous = by_path.setdefault(relative, source)
            if previous != source:
                raise ValueError(
                    f"selected media path maps to multiple original sources: {relative}"
                )
    entries = []
    for relative, source in sorted(by_path.items()):
        path = source_gate.resolve_media_path(relative, media_root)
        if not path.is_file():
            raise FileNotFoundError(f"selected formal-training media disappeared: {path}")
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(f"selected formal-training media is empty: {path}")
        entries.append({
            "relative_path": relative,
            "source_video_id": source,
            "size_bytes": size,
            "sha256": source_gate.sha256_file(path),
        })
    if not entries:
        raise ValueError("formal training selected no media files")
    return {
        "schema_version": 1,
        "kind": "formal_training_media_sha256_inventory",
        "media_root": str(media_root.resolve()),
        "file_count": len(entries),
        "total_bytes": sum(entry["size_bytes"] for entry in entries),
        "entries_sha256": source_gate.canonical_sha(entries),
        "entries": entries,
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--llava-root", type=Path, required=True)
    p.add_argument("--llava-plan", type=Path, required=True)
    p.add_argument("--media-root", type=Path, required=True)
    benchmark = p.add_mutually_exclusive_group(required=True)
    benchmark.add_argument("--source-gate-dir", type=Path,
                           help="Directory containing hash-bound benchmark registry and FORMAL_READY.json")
    benchmark.add_argument("--benchmark-source-registry", type=Path,
                           help="Standalone registry; hash is recorded but no source-gate receipt is implied")
    p.add_argument("--output-dir", type=Path, default=Path("cache/formal-training-data-v1"))
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError("output directory already exists; use a fresh immutable directory")
    receipt = source_gate.read_receipt(args.llava_plan, "llava_video_178k")
    annotation, annotation_proof = source_gate.verify_receipt_file(args.llava_root, LLAVA_ANNOTATION, receipt)
    _, archive_proof = source_gate.verify_receipt_file(args.llava_root, LLAVA_ARCHIVE, receipt)
    if args.source_gate_dir:
        registry_path = args.source_gate_dir / "benchmark_source_registry.json"
        ready_path = args.source_gate_dir / "FORMAL_READY.json"
    else:
        registry_path, ready_path = args.benchmark_source_registry, None
    benchmark_sources, benchmark_proof = load_benchmark_registry(registry_path, ready_path)
    state, scan = scan_groups(annotation, args.media_root, benchmark_sources)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary, source_rows = emit(annotation, receipt["revision"], args.media_root, state, args.output_dir)
    if source_gate.sha256_file(annotation) != annotation_proof["sha256"]:
        raise RuntimeError("pinned LLaVA annotation changed during the two-pass preparation")
    source_gate.write_jsonl(args.output_dir / "source_groups.jsonl", source_rows)
    media_inventory = build_media_inventory(state, args.media_root)
    source_gate.write_json(args.output_dir / "training_media_inventory.json", media_inventory)
    train_path, validation_path = args.output_dir / "train.jsonl", args.output_dir / "validation.jsonl"
    # Schema/source disjointness is checked without decoding media; media presence
    # was checked once per path during the complete first pass.
    train = source_gate.load_manifest(train_path, check_media=False)
    validation = source_gate.load_manifest(validation_path, check_media=False)
    source_gate.validate_disjoint(train, validation)
    if {row["source_video_id"] for row in train + validation} & benchmark_sources:
        raise AssertionError("benchmark source entered formal training manifests")
    output_hashes = {path.name: source_gate.sha256_file(path) for path in sorted(args.output_dir.iterdir())}
    report = {
        "schema_version": 1,
        "status": "formal_training_data_ready",
        "formal_data_ready": True,
        "dataset": "llava_video_178k",
        "dataset_revision": receipt["revision"],
        "annotation": annotation_proof,
        "archive": archive_proof,
        "benchmark_registry": benchmark_proof,
        "media_root": str(args.media_root.resolve()),
        "selection_policy": SELECTION_POLICY,
        "split_policy": SPLIT_POLICY,
        "split_seed": SPLIT_SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "expected_pinned_counts": {"annotation_rows": EXPECTED_ANNOTATION_ROWS,
                                   "candidate_qa": EXPECTED_CANDIDATE_QA,
                                   "unresolved_source_qa": EXPECTED_UNRESOLVED_QA},
        "scan": scan,
        "counts": summary,
        "source_groups": {
            "total_identified": len(state["decisions"]),
            "selected": sum(row["decision"] == "selected" for row in source_rows),
            "omitted": sum(row["decision"] == "omitted" for row in source_rows),
            "unresolved_path_groups": len(state["unresolved_paths"]),
        },
        "training_media_inventory": {
            "relative_path": "training_media_inventory.json",
            "sha256": output_hashes["training_media_inventory.json"],
            "file_count": media_inventory["file_count"],
            "total_bytes": media_inventory["total_bytes"],
            "entries_sha256": media_inventory["entries_sha256"],
        },
        "output_sha256": output_hashes,
        "limitations": [
            "Declared source IDs do not detect renamed or re-encoded benchmark media.",
            "The selected count is determined by actual media availability and is recorded, not forced to an estimate.",
            "This artifact performs no token-budget, model, or GPU validation.",
        ],
    }
    source_gate.write_json(args.output_dir / "preparation_report.json", report)
    ready = {
        "schema_version": 1,
        "status": report["status"],
        "purpose": "FORMAL_TRAINING_DATA",
        "formal_data_ready": True,
        "formal_training_ready": False,
        "reason": "Source gate must independently verify this artifact before formal training.",
        "report_sha256": source_gate.sha256_file(args.output_dir / "preparation_report.json"),
        "manifest_sha256": {"train.jsonl": source_gate.sha256_file(train_path),
                            "validation.jsonl": source_gate.sha256_file(validation_path)},
        "coverage_sha256": source_gate.sha256_file(args.output_dir / "coverage.jsonl"),
        "source_groups_sha256": source_gate.sha256_file(args.output_dir / "source_groups.jsonl"),
        "training_media_inventory_sha256": source_gate.sha256_file(
            args.output_dir / "training_media_inventory.json"
        ),
        "training_media_entries_sha256": media_inventory["entries_sha256"],
        "training_media_file_count": media_inventory["file_count"],
        "training_media_total_bytes": media_inventory["total_bytes"],
        "annotation_sha256": annotation_proof["sha256"],
        "benchmark_registry_sha256": benchmark_proof["sha256"],
        "counts": summary,
    }
    source_gate.write_json(args.output_dir / "READY.json", ready)
    print(json.dumps({"status": ready["status"], "counts": summary,
                      "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
