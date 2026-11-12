#!/usr/bin/env python3
"""Prepare/server-dry-run/explicit one-shot launch of the serial suite."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from deepprobe_vlm.suite_launch import main

if __name__ == '__main__':
    raise SystemExit(main())
