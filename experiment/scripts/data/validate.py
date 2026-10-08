#!/usr/bin/env python3
"""Validate one or more manifests and check cross-split source-video leakage."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from deepprobe_vlm.manifest import load_manifest, validate_disjoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--schema-only", action="store_true", help="Explicitly skip media existence checks (does not certify GPU readiness)")
    args = parser.parse_args()
    if not args.schema_only and args.media_root is None:
        parser.error("--media-root required unless --schema-only is explicit")
    groups = [load_manifest(p, args.media_root, check_media=not args.schema_only) for p in args.manifests]
    validate_disjoint(*groups)
    print(json.dumps({"samples": sum(map(len, groups)), "splits": dict(Counter(
        r["split"] for g in groups for r in g)), "source_videos": len({
        r["source_video_id"] for g in groups for r in g}), "media_checked": not args.schema_only}, indent=2))


if __name__ == "__main__":
    main()
