#!/usr/bin/env python3
"""Seal, verify or publish a curated result bundle; token only via environment."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from deepprobe_vlm.result_publication import main

if __name__ == "__main__":
    raise SystemExit(main())
