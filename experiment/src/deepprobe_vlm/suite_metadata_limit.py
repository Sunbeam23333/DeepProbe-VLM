"""Bound one private training task's metadata by its existing shared token.

This adds no reservation or lifetime quota. Finished metadata remains in the
normal home inventory. Only the four exact per-task metadata directories are
checked; native training logs stay inside the archived private run. A small
headroom covers the runner's final command/acceptance journal writes.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import stat

from . import suite_runner as runner
from .suite_storage import local_inventory

CONTROL_HEADROOM_BYTES = 1024 * 1024


def task_roots(state, task_id):
    runner.require(isinstance(task_id, str) and runner.ID.fullmatch(task_id), 'Invalid metadata task identity')
    state = runner.safe(state)
    return [state / name / task_id for name in ('artifacts', 'tasks', 'checkpoint-exports', 'durable-metadata')]


def check_metadata(state, task_id, *, additional_bytes=0, environ=None):
    """Read the live exact task token; caller holds the common budget lock."""
    runtime = os.environ if environ is None else environ
    state = runner.safe(state)
    runner.require(type(additional_bytes) is int and additional_bytes >= 0, 'Invalid metadata growth')
    runner.require(runtime.get('DP_SUITE_FLEET_STATE') == str(state)
                   and runtime.get('DP_SUITE_TASK_ID') == task_id, 'Metadata accounting task/state differs')
    token = runtime.get('DP_SUITE_STORAGE_TOKEN')
    claim = runner.read(state / 'tasks' / task_id / 'CLAIM.json')
    runner.require(isinstance(claim.get('storage_token'), dict)
                   and claim['storage_token'].get('home') == token
                   and claim['storage_token'].get('scratch') == runtime.get('DP_SUITE_SCRATCH_TOKEN'),
                   'Metadata accounting requires both exact held task tokens')
    ledger = runner.read(runner.safe(runtime['DP_SUITE_STORAGE_LEDGER']) / 'LEDGER.json')
    row = ledger.get('reservations', {}).get(token)
    runner.require(ledger.get('schema_version') == 2 and isinstance(row, dict)
                   and row.get('state') == 'ACTIVE' and row.get('owner') == claim.get('owner')
                   and row['owner'].get('worker_id') == runtime.get('DP_SUITE_WORKER_ID')
                   and row['owner'].get('task_id') == task_id,
                   'Metadata growth requires the active exact shared reservation owner')
    limit = row.get('home_bytes')
    runner.require(type(limit) is int and limit > 0, 'Shared metadata token has no byte budget')
    roots = [str(path) for path in task_roots(state, task_id) if path.exists()]
    observed = local_inventory(roots)
    runner.require(not observed['external_symlinks_not_followed'], 'Shared task metadata escapes its accounted directories')
    projected = observed['bytes'] + additional_bytes + CONTROL_HEADROOM_BYTES
    runner.require(projected <= limit,
                   'Shared training metadata exceeds its held token; preserve artifacts and stop')
    return {'status': 'SHARED_TRAIN_METADATA_WITHIN_HELD_TOKEN', 'task_id': task_id,
            'storage_token': token, 'held_home_bytes': limit, 'observed_bytes': observed['bytes'],
            'additional_peak_bytes': additional_bytes, 'control_headroom_bytes': CONTROL_HEADROOM_BYTES,
            'projected_bytes': projected, 'inventory_sha256': observed['inventory_sha256'],
            'scope': 'this_task_all_savepoints_and_shared_receipts', 'lifetime_quota_claimed': False}


@contextmanager
def metadata_write(path, additional_bytes=0):
    """Guard shared writes only inside a real private training claim.

    Ordinary v2 actions, CPU proposals and synthetic standalone export tests
    have no scratch token and retain their existing behavior.
    """
    if not os.environ.get('DP_SUITE_SCRATCH_TOKEN'):
        yield None
        return
    state = runner.safe(os.environ['DP_SUITE_FLEET_STATE'])
    task_id = os.environ['DP_SUITE_TASK_ID']
    roots = task_roots(state, task_id)
    path = runner.safe(path)
    if not any(path == root or path.is_relative_to(root) for root in roots):
        yield None
        return
    claim = runner.read(state / 'tasks' / task_id / 'CLAIM.json')
    intent = runner.read(runner.safe(claim['attempt']) / 'TASK_INTENT.json')
    runner.require(intent.get('task_id') == task_id, 'Metadata task intent differs')
    if intent.get('kind') != 'train':
        yield None
        return
    lock = runner.safe(os.environ['DP_SUITE_STORAGE_LEDGER']) / 'budget.lock'
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        runner.require(stat.S_ISREG(os.fstat(fd).st_mode) and os.fstat(fd).st_nlink == 1,
                       'Unsafe shared metadata budget lock')
        fcntl.flock(fd, fcntl.LOCK_EX)
        before = check_metadata(state, task_id, additional_bytes=additional_bytes)
        yield before
        check_metadata(state, task_id)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
