#!/usr/bin/env python3
"""Plan or explicitly prepare/admit/launch the complete remaining experiment suite."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from deepprobe_vlm.suite_workflow import main

if __name__ == "__main__":
    raise SystemExit(main())
