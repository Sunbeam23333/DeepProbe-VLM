#!/usr/bin/env python3
"""Default preparation / explicit one-shot launch for a pinned B300 fleet."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from deepprobe_vlm.suite_fleet_launch import main
if __name__ == '__main__':
    raise SystemExit(main())
