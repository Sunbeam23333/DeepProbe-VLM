#!/usr/bin/env python3
"""Bounded zip/tar extraction: regular files only, no links, no overwrite.

Default is a listing. Multipart tar fragments are NOT standalone archives.
"""
from __future__ import annotations
import argparse
import math
import stat
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from deepprobe_vlm.manifest import resolve_media_path, safe_relative_path


def extract_archive(archive: Path, output: Path, max_bytes: int, extract: bool = False):
    """Preflight all member paths/types/sizes, then copy only regular files."""
    is_zip = zipfile.is_zipfile(archive)
    handle = zipfile.ZipFile(archive) if is_zip else tarfile.open(archive, "r:*")
    with handle:
        entries = handle.infolist() if is_zip else handle.getmembers()
        plan, seen, total = [], set(), 0
        for entry in entries:
            name = entry.filename if is_zip else entry.name
            if name in {".", "./"}:
                continue
            name = safe_relative_path(name)
            dest = resolve_media_path(name, output)
            directory = entry.is_dir() if is_zip else entry.isdir()
            if is_zip:
                mode = entry.external_attr >> 16
                kind = stat.S_IFMT(mode)
                if kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
                    raise ValueError(f"archive links/special files forbidden: {name}")
                size = entry.file_size
            else:
                if not (entry.isfile() or entry.isdir()):
                    raise ValueError(f"archive links/special files forbidden: {name}")
                size = entry.size
            if name in seen:
                raise ValueError(f"duplicate archive member: {name}")
            seen.add(name)
            if size < 0:
                raise ValueError(f"negative archive size: {name}")
            if not directory:
                total += size
                if total > max_bytes:
                    raise ValueError(f"uncompressed archive size exceeds cap ({max_bytes} bytes)")
                if extract and dest.exists():
                    raise FileExistsError(f"refusing to overwrite extracted file: {dest}")
            plan.append((entry, name, directory, size))
        if extract:
            output.mkdir(parents=True, exist_ok=True)
            for entry, name, directory, size in plan:
                dest = resolve_media_path(name, output)
                if directory:
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Re-resolve after mkdir to reject pre-existing escaped symlinks.
                dest = resolve_media_path(name, output)
                source = handle.open(entry) if is_zip else handle.extractfile(entry)
                with source, dest.open("xb") as target:
                    remaining = size
                    while remaining:
                        chunk = source.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError(f"truncated archive member: {name}")
                        target.write(chunk)
                        remaining -= len(chunk)
                    if source.read(1):
                        raise ValueError(f"archive member exceeded declared size: {name}")
        return {"files": sum(not d for _, _, d, _ in plan), "uncompressed_bytes": total,
                "extracted": extract, "members": [n for _, n, _, _ in plan]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-gb", type=float, required=True, help="Uncompressed decimal-GB cap")
    p.add_argument("--extract", action="store_true", help="Actually extract; otherwise inspect only")
    args = p.parse_args()
    if not math.isfinite(args.max_gb) or args.max_gb <= 0:
        p.error("--max-gb must be positive and finite")
    import json
    print(json.dumps(extract_archive(args.archive, args.output, int(args.max_gb * 1e9), args.extract), indent=2))


if __name__ == "__main__":
    main()
