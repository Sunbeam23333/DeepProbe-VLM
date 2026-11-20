#!/usr/bin/env python3
"""Verify prepared CPU caches and publish immutable fleet READY receipts."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from deepprobe_vlm.suite_cache_publisher import main

if __name__ == '__main__':
    raise SystemExit(main())
