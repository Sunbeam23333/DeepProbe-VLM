#!/usr/bin/env python3
"""Prepare an immutable equivalence receipt for a new low allocation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from deepprobe_vlm.suite_relocation import main

if __name__ == '__main__':
    raise SystemExit(main())
