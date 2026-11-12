#!/usr/bin/env python3
"""Expand all declared suite tasks into a frozen, machine-bound execution file."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from deepprobe_vlm.suite_bindings import main

if __name__ == "__main__":
    raise SystemExit(main())
