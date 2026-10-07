#!/usr/bin/env python3
"""Offline deterministic frame cache; preserves split/source IDs and clip bounds.

Use a fresh --output-dir. PNG outputs are lossless, unresized selected frames.
Output manifest paths are relative to that new directory. No model is loaded.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from deepprobe_vlm.data import load_frames
from deepprobe_vlm.manifest import load_manifest, resolve_media_path, validate_records


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_cache(records, media_root, output_dir, num_frames, max_bytes):
    if output_dir.exists():
        raise FileExistsError(f"cache directory already exists; refusing overwrite: {output_dir}")
    validate_records(records, media_root)
    versions = {}
    for package in ("av", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    decoder_hash = sha256_file(ROOT / "src/deepprobe_vlm/data.py")
    output_dir.mkdir(parents=True)
    digest_cache, frame_cache, output_rows = {}, {}, []
    written_bytes = 0
    for record in records:
        paths = [record["video_path"]] if "video_path" in record else record["image_paths"]
        source_files = []
        source_states = []
        for relative in paths:
            path = resolve_media_path(relative, media_root)
            stat = path.stat()
            state = (str(path), stat.st_size, stat.st_mtime_ns)
            if state not in digest_cache:
                digest_cache[state] = sha256_file(path)
            source_files.append({"path": relative, "sha256": digest_cache[state]})
            source_states.append(state)
        provenance = {"source_files": source_files, "num_frames": num_frames,
                      "start_seconds": record.get("start_seconds"),
                      "end_seconds": record.get("end_seconds"),
                      "decoder_sha256": decoder_hash, "package_versions": versions,
                      "format": "PNG_RGB_lossless_original_resolution"}
        key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        if key not in frame_cache:
            images = load_frames(record, media_root, num_frames)
            for path_string, size, mtime in source_states:
                state = Path(path_string).stat()
                if (state.st_size, state.st_mtime_ns) != (size, mtime):
                    raise ValueError(f"source changed while caching: {path_string}")
            names, hashes = [], []
            for index, image in enumerate(images):
                encoded = io.BytesIO()
                image.convert("RGB").save(encoded, format="PNG", optimize=False)
                value = encoded.getvalue()
                if written_bytes + len(value) > max_bytes:
                    raise ValueError("frame cache exceeds --max-gb; partial cache retained, no completed manifest written")
                relative = f"frames/{key}/{index:04d}.png"
                destination = output_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as stream:
                    stream.write(value)
                written_bytes += len(value)
                names.append(relative)
                hashes.append(hashlib.sha256(value).hexdigest())
            frame_cache[key] = {"image_paths": names, "frame_sha256": hashes, "provenance": provenance}
        cached = frame_cache[key]
        output = {k: v for k, v in record.items() if k not in {"video_path", "image_paths"}}
        output.update(image_paths=cached["image_paths"], frame_cache_fingerprint=key)
        # start/end remain as provenance. The selected image sequence already
        # implements them; consumers must not apply the bounds a second time.
        output["bounds_applied_to_cached_frames"] = "start_seconds" in record
        output_rows.append(output)
    validate_records(output_rows, output_dir)
    with (output_dir / "manifest.jsonl").open("x", encoding="utf-8") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"completed": True, "samples": len(output_rows), "unique_decodes": len(frame_cache),
              "num_frames": num_frames, "written_png_bytes": written_bytes,
              "decoder_sha256": decoder_hash, "package_versions": versions, "entries": frame_cache}
    with (output_dir / "cache_index.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    return {k: v for k, v in report.items() if k != "entries"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--media-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--max-gb", type=float, required=True, help="Finite PNG cache budget, decimal GB")
    args = p.parse_args()
    if args.num_frames <= 0 or not math.isfinite(args.max_gb) or args.max_gb <= 0:
        p.error("--num-frames and --max-gb must be finite and positive")
    records = load_manifest(args.input, args.media_root)
    print(json.dumps(build_cache(records, args.media_root, args.output_dir, args.num_frames,
                                 int(args.max_gb * 1e9)), indent=2))


if __name__ == "__main__":
    main()
