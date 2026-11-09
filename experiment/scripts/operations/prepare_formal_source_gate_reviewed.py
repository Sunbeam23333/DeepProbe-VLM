#!/usr/bin/env python3
"""Run the frozen formal gate with an explicit separate candidate review.

The accepted frozen source builder, raw audit, candidate list and thresholds
remain untouched. The sole adapter replaces content validation with the version
that verifies every candidate decision and its SHA-bound original frame proof.
"""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_benchmark_content_review as review

FROZEN_GATE = Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/benchmark-evaluation-code-20261013-v2/experiment/scripts/operations/prepare_formal_source_gate.py')
FROZEN_GATE_SHA = 'ebbc68ccbffe563b4c0af03b4db7983c83fabf54b3a8deae9e2700bae93c20a4'


def load_gate():
    review.verify(FROZEN_GATE, FROZEN_GATE_SHA)
    spec = importlib.util.spec_from_file_location('frozen_formal_source_gate_review_adapter', FROZEN_GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--content-review', type=Path, required=True)
    parser.add_argument('--content-review-sha256', required=True)
    parser.add_argument('--content-review-independent', type=Path, required=True)
    parser.add_argument('--content-review-independent-sha256', required=True)
    args, rest = parser.parse_known_args(argv)
    review.verify(args.content_review, args.content_review_sha256)
    review.verify(args.content_review_independent, args.content_review_independent_sha256)
    gate = load_gate()
    def validate(path, binding, training=None, benchmark=None):
        return review.validate_content_audit(path, binding, training, benchmark,
            content_module=gate.content_audit_module(), review_path=args.content_review,
            review_sha256=args.content_review_sha256,
            independent_path=args.content_review_independent,
            independent_sha256=args.content_review_independent_sha256)
    gate.validate_content_audit = validate
    return gate.main(rest)


if __name__ == '__main__':
    raise SystemExit(main())
