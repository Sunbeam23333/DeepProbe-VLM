#!/usr/bin/env python3
"""Hydrate a byte-pinned benchmark frame cache on its Linux evaluation node."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from deepprobe_vlm.suite_media import hydrate_main

if __name__ == "__main__":
    raise SystemExit(hydrate_main())
