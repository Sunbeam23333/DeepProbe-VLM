#!/usr/bin/env python3
"""Create original-video F4/F8/F16 caches in stream OSS; dry planning by default."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from deepprobe_vlm.suite_media import main

if __name__ == "__main__":
    raise SystemExit(main())
