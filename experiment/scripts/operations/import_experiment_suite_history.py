#!/usr/bin/env python3
"""Import one separately reviewed history entry; do not infer acceptance from hints."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from deepprobe_vlm import experiment_suite, suite_bindings, suite_runner

def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", type=Path, default=experiment_suite.DEFAULT_PLAN)
    parser.add_argument("--task", required=True)
    for name in ("entry", "review"):
        parser.add_argument("--" + name, required=True, type=Path)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    for name in ("entry", "review"):
        suite_runner.verify_pin({"path": str(getattr(args, name)), "sha256": getattr(args, name + "_sha256")})
    output = suite_bindings.import_history_entry(experiment_suite.read_plan(args.plan), args.task,
        suite_runner.read(args.entry), suite_runner.artifact(args.review), args.output)
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
