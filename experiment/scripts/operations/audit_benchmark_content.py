#!/usr/bin/env python3
"""CPU evidence for exact-byte and uniform-eight-frame benchmark overlap checks.

This is a bounded sampled-frame audit, not a proof of full-content separation.
Original video SHA-256, complete sequential decode counts, the exact eight RGB
frames, and the immutable training gate are bound into the output. Any missing,
changed, undecodable, uninformative, or suspected overlapping source fails closed.
It never downloads media, deletes files, mutates an old gate, or submits GPU work.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from deepprobe_vlm.data import uniform_indices
from deepprobe_vlm.manifest import resolve_media_path

SHA = re.compile(r"[a-f0-9]{64}")
ARCHIVE_DIRECTORY_SHA256 = "ca3c186c375286500e0576b80eac0012e6fcf61abe9dd8714be731fe88292ac0"
VIDEOMME_REVISION = "ead1408f75b618502df9a1d8e0950166bf0a2a0b"
POLICY = {
    "policy_id": "deepprobe-uniform8-overlap-v1",
    "num_frames": 8,
    "scope": "uniform8_sampled_frames",
    "sampling": "deepprobe_vlm.data.uniform_indices",
    "phash_bits": 63,
    "phash_max_hamming": 6,
    "dhash_bits": 64,
    "dhash_max_hamming": 10,
    "minimum_grayscale_std": 8.0,
    "candidate_rule": "any informative frame pair within both hash thresholds; all candidates block until separately reviewed",
    "limitation": "No assertion is made about unsampled frames, strong crops, overlays, temporal transformations, or arbitrary near duplicates.",
}


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
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2)
        stream.write("\n")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_gate(directory, expected_sha):
    directory = Path(directory)
    ready_path = directory / "FORMAL_READY.json"
    if not SHA.fullmatch(expected_sha) or sha256_file(ready_path) != expected_sha:
        raise ValueError("original FORMAL_READY SHA mismatch")
    ready = load_json(ready_path)
    if ready.get("formal_training_ready") is not True:
        raise ValueError("original formal training gate is not ready")
    for relative, digest in ready["artifact_sha256"].items():
        if sha256_file(resolve_media_path(relative, directory)) != digest:
            raise ValueError(f"original gate artifact SHA mismatch: {relative}")
    report = load_json(directory / "source_gate_report.json")
    registry = load_json(directory / "benchmark_source_registry.json")
    source_ids = {row["source_video_id"] for row in registry["entries"]
                  if row.get("dataset") == "videomme"}
    if len(source_ids) != 899 or report["counts"]["videomme_selected_questions"] != 2697:
        raise ValueError("original fixed Video-MME subset is not 899 sources / 2697 QA")
    inventory = load_json(directory / "training_media_inventory.json")
    if inventory.get("complete") is not True or not inventory.get("entries"):
        raise ValueError("original training media inventory is incomplete")
    return {"ready_sha256": expected_sha, "directory": str(directory.resolve()),
            "binding": report["required_content_audit_binding"], "sources": source_ids,
            "training_inventory": inventory}


def frame_cache_inventory(receipt_path, root, selected_sources):
    """Validate pinned original identities and all exact uniform-eight PNG bytes."""
    receipt_path, root = Path(receipt_path), Path(root)
    receipt = load_json(receipt_path)
    errors, entries, original_entries, records = [], [], [], []
    archives = {}
    directory_path = root / "videomme_archive_directory.json"
    try:
        if sha256_file(directory_path) != ARCHIVE_DIRECTORY_SHA256:
            raise ValueError("pinned archive directory SHA mismatch")
        archives = {item["path"]: item for item in load_json(directory_path)}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"pinned media provenance missing: {exc}")
    required = (receipt.get("schema_version") == 1
                and receipt.get("kind") == "videomme_uniform8_frame_cache"
                and receipt.get("complete") is True
                and receipt.get("num_frames") == 8
                and receipt.get("sampling") == POLICY["sampling"]
                and receipt.get("failures") == []
                and receipt.get("archive_directory_sha256") == ARCHIVE_DIRECTORY_SHA256)
    if not required:
        errors.append("frame-cache receipt schema/completeness/sampling failed")
    rows = receipt.get("entries", [])
    if not isinstance(rows, list):
        rows = []
        errors.append("frame-cache entries must be a list")
    found, paths_seen = set(), set()
    from PIL import Image
    for row in rows:
        source = row.get("source_video_id")
        try:
            if source not in selected_sources or source in found:
                raise ValueError("unknown or repeated source identity")
            found.add(source)
            archive = archives.get(row.get("archive_path"))
            member_path = "data/" + source.removeprefix("youtube:") + ".mp4"
            if (archive is None or row.get("archive_sha256") != archive["sha256"]
                    or row.get("repo_id") != "lmms-lab/Video-MME"
                    or row.get("revision") != VIDEOMME_REVISION
                    or row.get("member_path") != member_path):
                raise ValueError("original source is not bound to the pinned media archive")
            members = {item["path"]: item for item in archive["members"]}
            if member_path not in members or members[member_path]["size_bytes"] != row.get("original_size_bytes"):
                raise ValueError("original source size/member mismatch with pinned ZIP directory")
            archive_receipt = load_json(root / "archives" / (archive["path"] + ".verified.json"))
            if (archive_receipt.get("verified") is not True
                    or archive_receipt.get("archive_sha256") != archive["sha256"]
                    or archive_receipt.get("size_bytes") != archive["size_bytes"]
                    or archive_receipt.get("repo_id") != "lmms-lab/Video-MME"
                    or archive_receipt.get("revision") != VIDEOMME_REVISION):
                raise ValueError("pinned archive download verification receipt failed")
            identity = source.removeprefix("youtube:")
            source_receipt_path = root / "entries" / (identity + ".json")
            completion = load_json(root / "archives" / (archive["path"] + ".complete.json"))
            if (completion.get("archive_sha256") != archive["sha256"]
                    or source not in completion.get("source_video_ids", [])
                    or completion.get("source_receipt_sha256", {}).get(identity) != sha256_file(source_receipt_path)
                    or load_json(source_receipt_path) != row):
                raise ValueError("source cache entry differs from archive completion evidence")
            count = row.get("decoded_frame_count")
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError("positive full-decoded frame count required")
            passes = row.get("full_decode_passes", 2)
            if (isinstance(passes, bool) or passes not in (1, 2)
                    or (passes == 1 and (row.get("container_declared_frame_count") != count
                                         or row.get("second_pass_frame_count") is not None))
                    or (passes == 2 and row.get("second_pass_frame_count") != count)
                    or row.get("decode_complete") is not True
                    or row.get("selected_indices") != uniform_indices(count, 8)):
                raise ValueError("complete sequential decode count and exact uniform-eight indices required")
            original_sha, size = row.get("original_sha256"), row.get("original_size_bytes")
            if not isinstance(original_sha, str) or not SHA.fullmatch(original_sha) or not isinstance(size, int) or size <= 0:
                raise ValueError("original source SHA/size missing")
            frame_rows, image_paths = row.get("frames"), row.get("image_paths")
            if not isinstance(frame_rows, list) or len(frame_rows) != 8 or not isinstance(image_paths, list) or len(image_paths) != 8:
                raise ValueError("exactly eight frame slots required")
            if image_paths != [frame.get("relative_path") for frame in frame_rows]:
                raise ValueError("image_paths differ from ordered frame metadata")
            for slot, frame in enumerate(frame_rows):
                relative = frame["relative_path"]
                if frame.get("source_frame_index") != row["selected_indices"][slot]:
                    raise ValueError(f"source frame index/slot mismatch: {relative}")
                if relative in paths_seen:
                    raise ValueError("each frame slot must have a distinct cached path")
                paths_seen.add(relative)
                path = resolve_media_path(relative, root)
                if not path.is_file() or path.stat().st_size <= 0:
                    raise ValueError(f"missing/empty frame: {relative}")
                if path.stat().st_size != frame.get("size_bytes") or sha256_file(path) != frame.get("sha256"):
                    raise ValueError(f"frame SHA/size mismatch: {relative}")
                with Image.open(path) as image:
                    image.load()
                    if image.format != "PNG" or image.mode != "RGB":
                        raise ValueError(f"exact RGB lossless PNG required: {relative}")
                    if (image.size != (frame.get("width"), frame.get("height"))
                            or hashlib.sha256(image.tobytes()).hexdigest() != frame.get("rgb_sha256")):
                        raise ValueError(f"decoded frame dimensions/RGB SHA mismatch: {relative}")
                entries.append({"relative_path": relative, "source_video_id": source,
                                "size_bytes": frame["size_bytes"], "sha256": frame["sha256"]})
            original_entries.append({"relative_path": source.removeprefix("youtube:") + ".mp4",
                                     "source_video_id": source, "size_bytes": size, "sha256": original_sha})
            records.append(row)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            errors.append(f"{source}: {exc}")
    missing = sorted(set(selected_sources) - found)
    if missing:
        errors.append(f"missing {len(missing)} source groups")
    complete = not errors and len(records) == len(selected_sources) and len(entries) == 8 * len(selected_sources)
    return {"checked": True, "complete": complete, "kind": "uniform8_lossless_frame_cache",
            "root": str(root.resolve()), "receipt_path": str(receipt_path.resolve()),
            "receipt_sha256": sha256_file(receipt_path),
            "archive_directory_sha256": ARCHIVE_DIRECTORY_SHA256, "expected_files": 8 * len(selected_sources),
            "present_files": len(entries), "expected_source_videos": len(selected_sources),
            "present_source_videos": len(records), "missing": missing, "errors": errors,
            "entries": sorted(entries, key=lambda x: x["relative_path"]),
            "original_entries": sorted(original_entries, key=lambda x: x["relative_path"]),
            "records": sorted(records, key=lambda x: x["source_video_id"]),
            "scope": POLICY["scope"]}


def fingerprint_image(image):
    import numpy as np
    from PIL import Image
    rgb = image.convert("RGB")
    gray = rgb.convert("L")
    small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    k = np.arange(8)[:, None]
    n = np.arange(32)[None, :]
    basis = np.cos(np.pi * (2 * n + 1) * k / 64)
    coefficients = (basis @ small @ basis.T).reshape(-1)[1:]
    bits = coefficients > np.median(coefficients)
    phash = sum(int(bit) << i for i, bit in enumerate(bits))
    horizontal = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    dhash = sum(int(bit) << i for i, bit in enumerate((horizontal[:, 1:] > horizontal[:, :-1]).reshape(-1)))
    rgb_digest = hashlib.sha256((f"RGB:{rgb.width}x{rgb.height}:".encode()) + rgb.tobytes()).hexdigest()
    return {"phash": f"{phash:016x}", "dhash": f"{dhash:016x}",
            "rgb_sha256": rgb_digest, "width": rgb.width, "height": rgb.height,
            "grayscale_std": round(float(small.std()), 6),
            "informative": float(small.std()) >= POLICY["minimum_grayscale_std"]}


def decode_uniform_images(path):
    import av
    with av.open(str(path), options={"err_detect": "explode"}) as container:
        container.streams.video[0].codec_context.thread_count = 1
        count = sum(1 for _ in container.decode(video=0))
    indices = uniform_indices(count, 8)
    wanted, selected = set(indices), {}
    second_count = 0
    with av.open(str(path), options={"err_detect": "explode"}) as container:
        container.streams.video[0].codec_context.thread_count = 1
        for index, frame in enumerate(container.decode(video=0)):
            second_count += 1
            if index in wanted:
                selected[index] = frame.to_image().convert("RGB")
    if second_count != count or len(selected) != len(wanted):
        raise ValueError("inconsistent complete video decode")
    return [selected[i].copy() for i in indices], count, indices


def fingerprint_training_entry(entry, root):
    result = dict(entry)
    try:
        path = resolve_media_path(entry["relative_path"], root)
        if not path.is_file() or path.stat().st_size != entry["size_bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError("training media differs from original attested SHA/size")
        images, count, indices = decode_uniform_images(path)
        fingerprints = [fingerprint_image(image) for image in images]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError("training media changed while decoding")
        result.update({"decode_complete": True, "decoded_frame_count": count,
                       "selected_indices": indices, "frames": fingerprints,
                       "informative_frames": sum(frame["informative"] for frame in fingerprints)})
    except Exception as exc:
        result.update({"decode_complete": False, "error": f"{type(exc).__name__}: {exc}"})
    return result


def versions():
    import av
    import numpy
    import PIL
    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "pillow": PIL.__version__, "pyav": av.__version__,
            "libav": {key: list(value) for key, value in av.library_versions.items()},
            "scanner_sha256": sha256_file(Path(__file__)),
            "frame_sampling_source_sha256": sha256_file(ROOT / "src/deepprobe_vlm/data.py"),
            "decoder_threads_per_worker": 1}


def training(args):
    gate = load_gate(args.formal_gate_dir, args.formal_ready_sha256)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    entries = gate["training_inventory"]["entries"]
    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool, (args.output_dir / "training_fingerprints.jsonl").open("x") as stream:
        for index, row in enumerate(pool.map(lambda entry: fingerprint_training_entry(entry, args.training_media_root), entries), 1):
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            if row.get("decode_complete") is not True:
                errors.append({"path": row["relative_path"], "error": row["error"]})
            if index % 100 == 0 or index == len(entries):
                print(json.dumps({"fingerprinted": index, "total": len(entries), "errors": len(errors)}), flush=True)
    receipt = {"schema_version": 1, "kind": "formal_training_uniform8_fingerprints",
               "passed": not errors and len(rows) == len(entries), "errors": errors,
               "binding": gate["binding"], "original_formal_ready_sha256": gate["ready_sha256"],
               "training_entries_sha256": canonical_sha(entries), "file_count": len(rows),
               "fingerprints_sha256": sha256_file(args.output_dir / "training_fingerprints.jsonl"),
               "policy": POLICY, "environment": versions()}
    write_json(args.output_dir / "TRAINING_FINGERPRINTS.json", receipt)
    return 0 if receipt["passed"] else 2


def candidate_pairs(training_rows, benchmark_rows):
    """Exact candidate recall for pHash Hamming<=6 using four16-bit blocks.

    If<=6 bits differ, at least one of four blocks differs by<=1. Searching
    that block and all16 one-bit neighbors cannot omit a threshold candidate.
    """
    flat = []
    buckets = defaultdict(list)
    for video in training_rows:
        for slot, frame in enumerate(video.get("frames", [])):
            if not frame["informative"]:
                continue
            idx = len(flat)
            phash, dhash = int(frame["phash"], 16), int(frame["dhash"], 16)
            flat.append((video, slot, frame, phash, dhash))
            for block in range(4):
                buckets[block, (phash >> (16 * block)) & 65535].append(idx)
    result = {}
    for benchmark in benchmark_rows:
        for slot, frame in enumerate(benchmark["frames"]):
            if not frame["informative"]:
                continue
            phash, dhash = int(frame["phash"], 16), int(frame["dhash"], 16)
            candidates = set()
            for block in range(4):
                value = (phash >> (16 * block)) & 65535
                for mask in (0, *(1 << bit for bit in range(16))):
                    candidates.update(buckets.get((block, value ^ mask), ()))
            for index in sorted(candidates):
                train, train_slot, train_frame, train_phash, train_dhash = flat[index]
                p_distance, d_distance = (phash ^ train_phash).bit_count(), (dhash ^ train_dhash).bit_count()
                if p_distance > POLICY["phash_max_hamming"] or d_distance > POLICY["dhash_max_hamming"]:
                    continue
                key = train["relative_path"], benchmark["source_video_id"]
                pair = result.setdefault(key, {"training_path": key[0],
                        "training_source_video_id": train["source_video_id"],
                        "benchmark_source_video_id": key[1], "classification": "suspected_overlap_requires_review",
                        "frame_matches": []})
                pair["frame_matches"].append({"training_slot": train_slot, "benchmark_slot": slot,
                    "phash_hamming": p_distance, "dhash_hamming": d_distance,
                    "same_decoded_rgb": train_frame["rgb_sha256"] == frame["rgb_sha256"]})
    return [result[key] for key in sorted(result)]


def media_binding(training_inventory, benchmark_inventory):
    return {"training_entries_sha256": canonical_sha(training_inventory["entries"]),
            "benchmark_entries_sha256": canonical_sha(benchmark_inventory["entries"]),
            "benchmark_original_entries_sha256": canonical_sha(benchmark_inventory.get("original_entries", benchmark_inventory["entries"])),
            "frame_cache_receipt_sha256": benchmark_inventory.get("receipt_sha256")}


def audit(args):
    gate = load_gate(args.formal_gate_dir, args.formal_ready_sha256)
    cache = frame_cache_inventory(args.benchmark_frame_cache_receipt, args.benchmark_media_root, gate["sources"])
    receipt = load_json(args.training_fingerprints / "TRAINING_FINGERPRINTS.json")
    fingerprint_path = args.training_fingerprints / "training_fingerprints.jsonl"
    if receipt.get("passed") is not True or receipt.get("binding") != gate["binding"] or receipt.get("policy") != POLICY:
        raise ValueError("training fingerprint receipt failed or has different binding/policy")
    if receipt.get("original_formal_ready_sha256") != gate["ready_sha256"] or receipt.get("training_entries_sha256") != canonical_sha(gate["training_inventory"]["entries"]):
        raise ValueError("training fingerprint original gate/media mismatch")
    if sha256_file(fingerprint_path) != receipt.get("fingerprints_sha256"):
        raise ValueError("training fingerprints SHA mismatch")
    training_rows = [json.loads(line) for line in fingerprint_path.read_text().splitlines()]
    keys = ("relative_path", "source_video_id", "size_bytes", "sha256")
    if [{key: row[key] for key in keys} for row in training_rows] != gate["training_inventory"]["entries"]:
        raise ValueError("training fingerprint rows differ from attested inventory")
    if any(row.get("decode_complete") is not True or len(row.get("frames", [])) != 8 for row in training_rows):
        raise ValueError("training fingerprints do not cover full decode and eight frame slots")
    errors = list(cache["errors"])
    from PIL import Image
    benchmark_rows = []
    for record in cache["records"]:
        fingerprints = []
        for relative in record["image_paths"]:
            with Image.open(resolve_media_path(relative, args.benchmark_media_root)) as image:
                fingerprints.append(fingerprint_image(image))
        benchmark_rows.append({"source_video_id": record["source_video_id"], "frames": fingerprints})
    training_hashes = defaultdict(list)
    for row in training_rows:
        training_hashes[row["sha256"]].append(row["relative_path"])
    exact = [{"training_path": relative, "benchmark_source_video_id": row["source_video_id"], "sha256": row["sha256"]}
             for row in cache["original_entries"] for relative in training_hashes.get(row["sha256"], [])]
    perceptual = candidate_pairs(training_rows, benchmark_rows)
    uninformative = ([{"training_path": row["relative_path"]} for row in training_rows
                      if not any(frame["informative"] for frame in row["frames"])]
                    + [{"benchmark_source_video_id": row["source_video_id"]} for row in benchmark_rows
                       if not any(frame["informative"] for frame in row["frames"])])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "benchmark_fingerprints.json", benchmark_rows)
    report = {"schema_version": 1, "kind": "deterministic_uniform8_content_audit",
        "passed": cache["complete"] and not errors and not exact and not perceptual and not uninformative,
        "binding": gate["binding"], "media_binding": media_binding(gate["training_inventory"], cache),
        "original_formal_ready_sha256": gate["ready_sha256"], "policy": POLICY,
        "method": "Original complete-file SHA-256 equality plus exact-recall pHash<=6 AND dHash<=10 search over every informative uniform-eight decoded frame pair; all candidates block.",
        "scope": POLICY["scope"], "scan_complete": cache["complete"] and not errors,
        "training_file_count": len(training_rows), "benchmark_source_count": len(benchmark_rows),
        "exact_duplicate_pairs": exact, "perceptual_duplicate_pairs": perceptual,
        "uninformative_sources": uninformative, "errors": errors,
        "artifacts": {"training_fingerprints_receipt": {"path": str((args.training_fingerprints / "TRAINING_FINGERPRINTS.json").resolve()), "sha256": sha256_file(args.training_fingerprints / "TRAINING_FINGERPRINTS.json")},
            "benchmark_fingerprints": {"path": "benchmark_fingerprints.json", "sha256": sha256_file(args.output_dir / "benchmark_fingerprints.json")}},
        "environment": versions()}
    write_json(args.output_dir / "CONTENT_AUDIT.json", report)
    print(json.dumps({key: report[key] for key in ("passed", "scan_complete", "training_file_count", "benchmark_source_count")}), flush=True)
    return 0 if report["passed"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("training", "audit"):
        sub = commands.add_parser(name)
        sub.add_argument("--formal-gate-dir", type=Path, required=True)
        sub.add_argument("--formal-ready-sha256", required=True)
        sub.add_argument("--output-dir", type=Path, required=True)
        if name == "training":
            sub.add_argument("--training-media-root", type=Path, required=True)
            sub.add_argument("--workers", type=int, default=4, choices=range(1, 17))
        else:
            sub.add_argument("--training-fingerprints", type=Path, required=True)
            sub.add_argument("--benchmark-frame-cache-receipt", type=Path, required=True)
            sub.add_argument("--benchmark-media-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError("use a new immutable output directory")
    return training(args) if args.command == "training" else audit(args)


if __name__ == "__main__":
    raise SystemExit(main())
