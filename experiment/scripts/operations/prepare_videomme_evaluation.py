#!/usr/bin/env python3
"""Bind the predeclared Video-MME subset to pinned QA and acquired RGB frames.

Annotation-only mode prepares auditable QA identities while media acquisition
runs. Full mode creates a NEW manifest only after the complete frame receipt
exists. Neither mode changes an annotation, the old source gate, or a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_formal_source_gate as gate
from deepprobe_vlm.manifest import safe_relative_path, validate_records

REVISION = "ead1408f75b618502df9a1d8e0950166bf0a2a0b"
ANNOTATION_SHA = "7fffab8ed38ecc2f9f0eca4c44d8a11636f0ee96116ede83580c8b9ae0faf986"
REGISTRY_SHA = "a143d6a364221738b727d3944527039060315e04373599c5829ab3ea60eb0226"
PROTOCOL = "videomme-899-no-subtitles-no-audio-v1"


def read_pinned(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or gate.sha256_file(path) != expected:
        raise ValueError(f"Pinned artifact differs: {path}")
    return path


def build_qa(rows, accepted_registry):
    selected, excluded, _ = gate.build_videomme_subset(rows, REVISION, ANNOTATION_SHA)
    accepted = [row for row in accepted_registry["entries"] if row["dataset"] == "videomme"]
    if selected != accepted:
        raise ValueError("QA identities differ from pretraining source registry")
    by_question = {row["question_id"]: row for row in rows}
    converted, bindings = [], []
    for proof in selected:
        row = by_question[proof["question_id"]]
        options = row["options"]
        if len(options) != 4 or any(not option.startswith(chr(65 + i) + ". ")
                                    for i, option in enumerate(options)):
            raise ValueError("Pinned options must retain their original A/B/C/D order")
        record, = gate.convert.convert_videomme(row, SimpleNamespace(video_prefix="videos"), {})
        if record["source_video_id"] != proof["source_video_id"]:
            raise ValueError("Converter source differs from pinned registry")
        # Explicit comparisons are independent of the converter's field mapping.
        exact = {"sample_id": "videomme:" + row["question_id"],
                 "question": row["question"], "choices": [option[3:].strip() for option in options],
                 "answer": row["answer"].strip().upper(), "duration_category": row["duration"],
                 "task_type": row["task_type"], "domain": row["domain"]}
        if any(record.get(key) != value for key, value in exact.items()):
            raise ValueError("Converted QA text, choices, answer or metadata differs")
        if len(record["choices"]) != 4 or record["answer"] not in {"A", "B", "C", "D"}:
            raise ValueError("Expected four choices and one canonical gold letter")
        converted.append(record)
        bindings.append({**proof, "sample_id": record["sample_id"],
                         "qa_record_sha256": gate.canonical_sha(record)})
    validate_records(converted, check_media=False)
    duration_counts = Counter(row["duration_category"] for row in converted)
    if duration_counts != {"short": 900, "medium": 897, "long": 900}:
        raise ValueError("Predeclared duration coverage changed")
    return converted, bindings, excluded


def attach_frames(records, receipt):
    if (receipt.get("complete") is not True or receipt.get("kind") != "videomme_uniform8_frame_cache"
            or receipt.get("num_frames") != 8 or receipt.get("revision") != REVISION):
        raise ValueError("Expected complete pinned uniform8 frame cache")
    sources = {row["source_video_id"] for row in records}
    by_source = {}
    for entry in receipt["entries"]:
        source = entry["source_video_id"]
        if source in by_source or entry.get("decode_complete") is not True:
            raise ValueError("Duplicate or incomplete frame source")
        frames = entry["frames"]
        if len(frames) != 8:
            raise ValueError("Each source requires exactly eight ordered frame slots")
        paths = [safe_relative_path(frame["relative_path"]) for frame in frames]
        if entry.get("image_paths") != paths:
            raise ValueError("Ordered source image_paths differ from frame slots")
        if len(set(paths)) != 8:
            raise ValueError("Eight distinct frame slots required, including repeated frame indices")
        by_source[source] = paths
    if set(by_source) != sources or len(by_source) != 899:
        raise ValueError("Frame source set differs from all 899 fixed sources")
    output = []
    for row in records:
        item = {key: value for key, value in row.items() if key != "video_path"}
        item["image_paths"] = by_source[row["source_video_id"]]
        output.append(item)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--source-registry", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--annotation-only", action="store_true")
    parser.add_argument("--frame-cache-receipt", type=Path)
    parser.add_argument("--frame-cache-sha256")
    parser.add_argument("--media-root", type=Path)
    args = parser.parse_args(argv)
    if not args.annotation_only and not all((args.frame_cache_receipt, args.frame_cache_sha256, args.media_root)):
        parser.error("Full conversion requires frame receipt, its SHA and media root")
    if args.annotation_only and any((args.frame_cache_receipt, args.frame_cache_sha256, args.media_root)):
        parser.error("Annotation-only mode has no media inputs")
    annotation = read_pinned(args.annotation, ANNOTATION_SHA)
    registry_path = read_pinned(args.source_registry, REGISTRY_SHA)
    raw = list(gate.convert.rows_from(annotation))
    records, bindings, exclusions = build_qa(raw, json.loads(registry_path.read_text()))
    media_binding = None
    if not args.annotation_only:
        receipt_path = read_pinned(args.frame_cache_receipt, args.frame_cache_sha256)
        records = attach_frames(records, json.loads(receipt_path.read_text()))
        validate_records(records, args.media_root, check_media=True)
        media_binding = {"path": str(receipt_path.resolve()), "sha256": args.frame_cache_sha256,
                         "media_root": str(args.media_root.resolve()), "frame_count": 7192}
    for record, proof in zip(records, bindings):
        proof["output_record_sha256"] = gate.canonical_sha(record)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = args.output_dir / ("annotation_only.jsonl" if args.annotation_only else "videomme.jsonl")
    gate.write_jsonl(manifest, records)
    gate.write_json(args.output_dir / "QA_BINDINGS.json", bindings)
    gate.write_json(args.output_dir / "EXCLUSIONS.json", exclusions)
    receipt = {"schema_version": 1, "status": "ANNOTATION_BOUND" if args.annotation_only else "EVALUATION_MANIFEST_BOUND",
               "protocol": PROTOCOL, "repo_id": "lmms-lab/Video-MME", "revision": REVISION,
               "annotation": {"path": str(annotation.resolve()), "sha256": ANNOTATION_SHA},
               "source_registry": {"path": str(registry_path.resolve()), "sha256": REGISTRY_SHA},
               "manifest": {"path": str(manifest.resolve()), "sha256": gate.sha256_file(manifest)},
               "qa_bindings_sha256": gate.sha256_file(args.output_dir / "QA_BINDINGS.json"),
               "exclusions_sha256": gate.sha256_file(args.output_dir / "EXCLUSIONS.json"),
               "raw_questions": 2700, "selected_questions": len(records), "selected_sources": 899,
               "excluded_question_ids": ["490-1", "490-2", "490-3"],
               "by_duration": dict(sorted(Counter(row["duration_category"] for row in records).items())),
               "media_binding": media_binding, "subtitles": False, "audio": False,
               "all_selected_qa_bound_to_original_annotation": True,
               "script_sha256": gate.sha256_file(Path(__file__)),
               "converter_sha256": gate.sha256_file(Path(gate.convert.__file__)),
               "subset_builder_sha256": gate.sha256_file(Path(gate.__file__))}
    gate.write_json(args.output_dir / "MANIFEST_BINDING.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
