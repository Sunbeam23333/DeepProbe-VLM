#!/usr/bin/env python3
"""Directly verify that final/ and the last full Trainer checkpoint match."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from verify_resume_equivalence import compare_model_state


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--final", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite report: {args.output}")
    report = {"status": "running", "purpose": "EXACT_DUPLICATE_MODEL_SAVE_CHECK",
              "utc": datetime.now(timezone.utc).isoformat(), "errors": []}
    try:
        result = compare_model_state(args.checkpoint, args.final)
        report["comparison"] = result
        report["exact_model_copy_proven"] = result["status"] == "passed"
        report["status"] = "exact_model_copy_passed" if report["exact_model_copy_proven"] else "failed"
    except Exception as exc:
        report["status"] = "error"
        report["exact_model_copy_proven"] = False
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "report": str(args.output),
                      "errors": report["errors"]}))
    return 0 if report["exact_model_copy_proven"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
