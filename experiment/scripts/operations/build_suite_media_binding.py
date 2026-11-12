#!/usr/bin/env python3
"""Bind original archive members using accepted decode evidence and ZIP metadata."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from deepprobe_vlm.suite_media import binding_main

if __name__ == "__main__":
    raise SystemExit(binding_main())
