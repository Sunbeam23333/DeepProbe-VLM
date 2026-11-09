#!/usr/bin/env python3
"""Submit only the fresh reviewed a2 plan through the frozen one-submit helper.

Preserves global lock/durable intent, empty exact OSS prefix, project scheduler
and Kubernetes admission, low/stream/one-node/eight-GPU/restartNever dry-run, and
read-only discovery after ambiguous clients. Never starts the old coordinator.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
FROZEN_HELPER = Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/benchmark-evaluation-code-20261013-v2/experiment/scripts/operations/submit_benchmark_eval_once.py')
FROZEN_HELPER_SHA = 'b31cf9e72c0536462c4e86f2f2733348e8ce92f22b2e04e0efc320436df00943'
RUN_ID = 'dp-benchmark-videomme-r1-r2-s17-a2'
JOB_SHORT = 'dpvlm-videomme-r1r2-s17-a2'
PROJECT_PREFIX = 'jinhaopeng-dpvlm-'
ORIGINAL_AUDIT_SHA = '4366a474e4aad03a4526d1ea24e087b8dd8bfac84bcfd368f2c39e08c95b9197'


def require(value, message):
    if not value: raise ValueError(message)


def safe(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
            'Unsafe control artifact path')
    return path


def sha(path):
    digest = hashlib.sha256()
    with safe(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(item):
    path = safe(item['path'])
    require(path.is_file() and sha(path) == item['sha256'], 'Pinned review/control artifact changed')
    return path


def normalize_nonproject_cpu_rows(output):
    lines = output.splitlines()
    titles = [i for i, line in enumerate(lines) if line.strip() == 'Job List:']
    if len(titles) != 1: return output
    for i in range(titles[0] + 2, len(lines)):
        fields = lines[i].strip().split(maxsplit=7)
        if (len(fields) in {7, 8} and re.fullmatch(r'jinhaopeng-[a-z0-9-]+', fields[0]) and
                not fields[0].startswith(PROJECT_PREFIX) and fields[1] == '-'):
            fields[1] = '0'; lines[i] = ' '.join(fields)
    return '\n'.join(lines) + ('\n' if output.endswith('\n') else '')


def load_frozen_helper():
    verify({'path': str(FROZEN_HELPER), 'sha256': FROZEN_HELPER_SHA})
    sys.path.insert(0, str(FROZEN_HELPER.parent))
    spec = importlib.util.spec_from_file_location('frozen_reviewed_a2_submission', FROZEN_HELPER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    original = module.jobs_in_list
    module.jobs_in_list = lambda output: original(normalize_nonproject_cpu_rows(output))
    module.JOB_SHORT = JOB_SHORT
    module.JOB_PREFIX = 'jinhaopeng-' + JOB_SHORT + '-n1-'
    return module


def validate_review_plan(plan):
    require(plan.get('run_id') == RUN_ID and Path(plan.get('run_root', '')).name == RUN_ID,
            'Only the new a2 attempt may use this submission adapter')
    require(plan.get('evaluation_scope') == 'videomme_only' and plan.get('all_benchmark_assets_ready') is False and
            plan.get('mvbench_evaluation_ready') is False and plan.get('benchmark_score_based_model_selection') is False,
            'Explicit fixed Video-MME-only plan required')
    require(plan.get('original_content_audit', {}).get('sha256') == ORIGINAL_AUDIT_SHA,
            'Original failed content scan identity changed')
    original = verify(plan['original_content_audit'])
    require(json.loads(original.read_text()).get('passed') is False, 'Original scanner report must remain unchanged')
    review_path = verify(plan['content_review']); independent_path = verify(plan['independent_content_review'])
    validation_path = verify(plan['reviewed_content_validation']); validation = json.loads(validation_path.read_text())
    require(validation.get('provided') is True and validation.get('passed') is True and validation.get('errors') == [] and
            validation.get('path') == str(original) and validation.get('sha256') == ORIGINAL_AUDIT_SHA and
            validation.get('admission_mode') == 'original_scan_plus_bound_candidate_review_v1', 'Reviewed content validation failed')
    proof = validation['adjudication']
    require(proof.get('passed') is True and proof.get('original_candidates_preserved') is True and
            proof.get('original_audit_passed') is False and proof.get('candidate_count') == 12 and
            proof.get('matched_frame_count') == 30 and proof.get('review') == plan['content_review'] and
            all(proof.get('independent_review', {}).get(k) == v for k, v in plan['independent_content_review'].items()),
            'Independent review must bind all original candidates')
    rows = plan['source_files']; supplied = {row['path']: row for row in rows}
    require(len(supplied) == len(rows), 'Duplicate source/control pin')
    for item in proof['artifacts']:
        require(supplied.get(item['path']) == item, 'All reviewed content evidence must be pinned for GPU revalidation')
        verify(item)
    for item in (plan['original_content_audit'], plan['content_review'], plan['independent_content_review']):
        require(supplied.get(item['path']) == item, 'Original scan/review pin missing')
    own = Path(__file__).resolve()
    require(supplied.get(str(own)) == {'path': str(own), 'sha256': sha(own)}, 'Plan must pin exact a2 submit adapter')
    return validation


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    known, _ = parser.parse_known_args(argv)
    plan_path = verify({'path': str(known.plan), 'sha256': known.plan_sha256})
    plan = json.loads(plan_path.read_text()); validate_review_plan(plan)
    # sslaunch itself resolves kubectl via PATH; use the already installed binary.
    os.environ['PATH'] = '/home/jinhaopeng/.local/bin:' + os.environ.get('PATH', '')
    return load_frozen_helper().main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
