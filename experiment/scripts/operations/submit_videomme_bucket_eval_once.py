#!/usr/bin/env python3
"""Reuse the frozen one-submit helper with one non-project CPU-row fix.

Only a '-' GPU field on a syntactically named, non-DeepProbe personal job is
normalized to0 for the original strict parser. Project jobs, statuses, row counts,
admission, dry-run, durable intent and ambiguous-outcome handling are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

FROZEN_HELPER = Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/benchmark-evaluation-code-20261013-v2/experiment/scripts/operations/submit_benchmark_eval_once.py')
FROZEN_HELPER_SHA = 'b31cf9e72c0536462c4e86f2f2733348e8ce92f22b2e04e0efc320436df00943'
PROJECT_PREFIX = 'jinhaopeng-dpvlm-'


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_nonproject_cpu_rows(output):
    lines = output.splitlines()
    titles = [index for index, line in enumerate(lines) if line.strip() == 'Job List:']
    # Let the frozen parser report malformed/ambiguous document structure.
    if len(titles) != 1: return output
    for index in range(titles[0] + 2, len(lines)):
        fields = lines[index].strip().split(maxsplit=7)
        if (len(fields) in {7, 8} and re.fullmatch(r'jinhaopeng-[a-z0-9-]+', fields[0])
                and not fields[0].startswith(PROJECT_PREFIX) and fields[1] == '-'):
            fields[1] = '0'
            lines[index] = ' '.join(fields)
    return '\n'.join(lines) + ('\n' if output.endswith('\n') else '')


def load_frozen_helper():
    require(not FROZEN_HELPER.is_symlink() and FROZEN_HELPER.is_file() and
            sha(FROZEN_HELPER) == FROZEN_HELPER_SHA, 'Frozen submission helper changed')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(FROZEN_HELPER.parent))
    spec = importlib.util.spec_from_file_location('accepted_benchmark_submission_helper', FROZEN_HELPER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    original = module.jobs_in_list
    module.jobs_in_list = lambda output: original(normalize_nonproject_cpu_rows(output))
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    known, _ = parser.parse_known_args(argv)
    require(known.plan.is_absolute() and not known.plan.is_symlink() and sha(known.plan) == known.plan_sha256,
            'Frozen evaluation plan changed')
    plan = json.loads(known.plan.read_text())
    require(plan.get('evaluation_scope') == 'videomme_only' and plan.get('all_benchmark_assets_ready') is False and
            plan.get('mvbench_evaluation_ready') is False, 'Explicit Video-MME-only plan required')
    own = Path(__file__).resolve()
    pins = [row for row in plan['source_files'] if Path(row['path']).resolve() == own]
    require(len(pins) == 1 and pins[0]['sha256'] == sha(own), 'Plan must pin this exact CPU-row compatibility wrapper')
    return load_frozen_helper().main(argv)


if __name__ == '__main__': raise SystemExit(main())
