#!/usr/bin/env python3
"""Plan exact-revision Hub downloads. No downloads without --download --max-gb."""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from deepprobe_vlm.manifest import resolve_media_path, safe_relative_path


def choose_files(siblings, patterns):
    selected = []
    for pattern in patterns:
        if not any(fnmatch.fnmatch(f.rfilename, pattern) for f in siblings):
            raise ValueError(f"pattern matched no files at the pinned revision: {pattern}")
    for item in siblings:
        if not any(fnmatch.fnmatch(item.rfilename, p) for p in patterns):
            continue
        safe_relative_path(item.rfilename)
        if item.size is None or item.size < 0:
            raise ValueError(f"missing file size; refusing an unbudgeted download: {item.rfilename}")
        lfs = item.lfs
        sha = getattr(lfs, "sha256", None) if lfs is not None else None
        if isinstance(lfs, dict):
            sha = lfs.get("sha256")
        selected.append({"path": item.rfilename, "size_bytes": item.size,
                         "git_blob_sha1": item.blob_id, "lfs_sha256": sha,
                         "expected_etag": sha or item.blob_id})
    if not selected:
        raise ValueError("no files selected")
    return sorted(selected, key=lambda f: f["path"])


def verify_download(path, entry):
    if path.stat().st_size != entry["size_bytes"]:
        raise ValueError(f"size mismatch: {path}")
    sha = hashlib.sha256()
    git = hashlib.sha1(f"blob {entry['size_bytes']}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
            git.update(chunk)
    digest = sha.hexdigest()
    if entry["lfs_sha256"] and digest != entry["lfs_sha256"]:
        raise ValueError(f"SHA256 mismatch: {path}")
    if not entry["lfs_sha256"] and entry["git_blob_sha1"] and git.hexdigest() != entry["git_blob_sha1"]:
        raise ValueError(f"Git blob hash mismatch: {path}")
    entry["local_sha256"] = digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", choices=json.loads((ROOT / "configs/datasets.json").read_text())["datasets"])
    p.add_argument("--include", action="append", help="Explicit repository-relative glob; repeat. Defaults are annotations only.")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--download", action="store_true")
    modes.add_argument("--dry-run", action="store_true", help="Default: inspect metadata without downloading")
    modes.add_argument("--list", action="store_true", help="List all remote files; no downloads")
    p.add_argument("--max-gb", type=float, help="Required for download; decimal GB, total selected bytes")
    p.add_argument("--local-dir", type=Path)
    p.add_argument("--manifest", type=Path, help="Optional download-plan JSON; never overwrite an existing file")
    args = p.parse_args()
    if args.max_gb is not None and (not math.isfinite(args.max_gb) or args.max_gb <= 0):
        p.error("--max-gb must be positive and finite")
    if args.download and (args.max_gb is None or args.local_dir is None):
        p.error("--download requires positive --max-gb and --local-dir")
    from huggingface_hub import HfApi, snapshot_download
    spec = json.loads((ROOT / "configs/datasets.json").read_text())["datasets"][args.dataset]
    if not re.fullmatch(r"[0-9a-f]{40}", spec["revision"]):
        raise ValueError("config revision must be an immutable 40-character commit SHA")
    info = HfApi().dataset_info(spec["repo_id"], revision=spec["revision"], files_metadata=True)
    if info.sha != spec["revision"]:
        raise ValueError("Hub resolved a different revision")
    patterns = ["*"] if args.list else args.include or spec["default_patterns"]
    files = choose_files(info.siblings, patterns)
    total = sum(f["size_bytes"] for f in files)
    if args.max_gb is not None and total > args.max_gb * 1_000_000_000:
        raise ValueError(f"selected {total / 1e9:.3f} GB exceeds cap {args.max_gb}; narrow --include")
    plan = {"repo_id": spec["repo_id"], "revision": info.sha, "dataset": args.dataset,
            "created_utc": datetime.now(timezone.utc).isoformat(), "rights": spec["rights"],
            "total_bytes": total, "downloaded": False, "files": files}
    print(json.dumps(plan, indent=2))
    if args.manifest and args.manifest.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {args.manifest}")
    if args.download:
        args.local_dir.mkdir(parents=True, exist_ok=True)
        for entry in files:
            resolve_media_path(entry["path"], args.local_dir)
        snapshot_download(repo_id=spec["repo_id"], repo_type="dataset", revision=info.sha,
                          allow_patterns=[x["path"] for x in files], local_dir=args.local_dir)
        for entry in files:
            verify_download(resolve_media_path(entry["path"], args.local_dir), entry)
        plan["downloaded"] = True
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open("x", encoding="utf-8") as stream:
            json.dump(plan, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
