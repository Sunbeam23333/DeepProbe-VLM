#!/usr/bin/env python3
"""Submit a3 with an explicit local rendezvous address, retaining the a2 inputs.

The accepted inference package and its torchrun command stay byte-identical.
PyTorch's documented PET_LOCAL_ADDR option selects 127.0.0.1 for the single
node's rendezvous address. No scores or model/data/prompt choices change.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
P1 = Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1')
BASE_PLAN = P1 / 'videomme-reviewed-cpu-20261021-a1/EVALUATION_PLAN.json'
BASE_PLAN_SHA = 'b39fa893e6389982e6a87b7b580b05f97db82df48b2c9a1c2fb32f3bbeeb3fa7'
ADAPTER = P1 / 'videomme-reviewed-control-20261021-v1/submit_videomme_reviewed_eval_once.py'
ADAPTER_SHA = '8d80e54fbd39bf82254fee02e8647f278490399cf431e0f1a46ef3a7c5909fbc'
RUN_ID = 'dp-benchmark-videomme-r1-r2-s17-a3'
JOB_SHORT = 'dpvlm-videomme-r1r2-s17-a3'
ENVIRONMENT = {'PET_LOCAL_ADDR': '127.0.0.1'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    p = Path(path)
    require(p.is_absolute() and '..' not in p.parts and not any(x.is_symlink() for x in (p, *p.parents)), 'Unsafe pinned artifact')
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(row):
    require(sha(row['path']) == row['sha256'], 'Pinned artifact changed')
    return Path(row['path'])


def validate_transport_only_delta(plan):
    original = json.loads(verify({'path': str(BASE_PLAN), 'sha256': BASE_PLAN_SHA}).read_text())
    require(plan.get('run_id') == RUN_ID and plan.get('run_root') == str(Path(original['run_root']).with_name(RUN_ID)), 'Only fresh a3 is allowed')
    require(plan.get('distributed_runtime_environment') == ENVIRONMENT, 'Only the fixed local rendezvous address is allowed')
    require(plan.get('recovery_of') == {'run_id': original['run_id'], 'plan': {'path': str(BASE_PLAN), 'sha256': BASE_PLAN_SHA},
            'exact_job': 'jinhaopeng-dpvlm-videomme-r1r2-s17-a2-n1-261021164247', 'observed_state': 'Failed', 'prediction_count': 0}, 'Wrong failed-attempt recovery binding')
    changed = {'run_id', 'run_root', 'source_files', 'distributed_runtime_environment', 'recovery_of', 'recovery_evidence'}
    require({k: v for k, v in plan.items() if k not in changed} == {k: v for k, v in original.items() if k not in changed}, 'Non-transport evaluation protocol or data changed')
    original_rows = {x['path']: x for x in original['source_files']}
    rows = {x['path']: x for x in plan['source_files']}
    require(len(rows) == len(plan['source_files']) and all(rows.get(p) == r for p, r in original_rows.items()), 'Original source/control pins must remain unchanged')
    evidence = plan.get('recovery_evidence')
    require(isinstance(evidence, list) and len(evidence) >= 3, 'Failure/archive and CPU smoke evidence is required')
    additions = {str(Path(__file__).resolve()): {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__).resolve())},
                 str(BASE_PLAN): {'path': str(BASE_PLAN), 'sha256': BASE_PLAN_SHA}}
    for row in evidence:
        require(set(row) == {'path', 'sha256'} and row['path'] not in additions and row['path'] not in original_rows, 'Duplicate or malformed recovery evidence')
        verify(row)
        additions[row['path']] = row
    require(rows == {**original_rows, **additions}, 'Unexpected recovery source additions')
    for row in additions.values():
        verify(row)
    return original


def load_adapter():
    verify({'path': str(ADAPTER), 'sha256': ADAPTER_SHA})
    spec = importlib.util.spec_from_file_location('frozen_a2_adapter_for_a3', ADAPTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RUN_ID = RUN_ID
    module.JOB_SHORT = JOB_SHORT
    original_loader = module.load_frozen_helper

    def load_with_loopback():
        helper = original_loader()
        previous = helper.runner_argv
        helper.runner_argv = lambda path, plan_sha: ['env', 'PET_LOCAL_ADDR=127.0.0.1', *previous(path, plan_sha)]
        return helper

    module.load_frozen_helper = load_with_loopback
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args, _ = parser.parse_known_args(argv)
    plan = json.loads(verify({'path': str(args.plan), 'sha256': args.plan_sha256}).read_text())
    validate_transport_only_delta(plan)
    return load_adapter().main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
