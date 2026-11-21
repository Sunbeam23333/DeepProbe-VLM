"""Retain bounded failure diagnostics before an owned private Pod exits.

These byte snapshots never attest training success or process quiescence.
Large or changing logs receive only a range/hash for the bytes actually read;
no complete source-file hash is inferred from a tail or a prior exit receipt.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat

from . import suite_runner as runner

CONTROL_BYTES = 128 * 1024
LOG_TAIL_BYTES = 256 * 1024
PAYLOAD_LIMIT_BYTES = 2 * CONTROL_BYTES + 2 * LOG_TAIL_BYTES
REPORT_LIMIT_BYTES = 32 * 1024


def _identity(value):
    return {name: getattr(value, 'st_' + name) for name in ('dev', 'ino', 'size', 'mtime_ns', 'ctime_ns')}


def _capture(path, limit):
    """Read a finite range from one exact regular file, without following links."""
    runner.safe(path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return {'source_path': str(path), 'status': 'MISSING'}, None
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        runner.require(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid(),
                       'Training diagnostic must be an owned regular file')
        start = max(0, before.st_size - limit)
        stream.seek(start)
        payload = stream.read(min(limit, before.st_size - start))
        after = os.fstat(stream.fileno())
    try:
        final = path.lstat()
    except FileNotFoundError:
        final = None
    stable = final is not None and _identity(before) == _identity(after) == _identity(final)
    complete = stable and start == 0 and len(payload) == before.st_size
    value = {'source_path': str(path), 'status': 'CAPTURED',
             'source_stat_before': _identity(before), 'source_stat_after': _identity(after),
             'source_path_stat_after': _identity(final) if final is not None else None,
             'source_stable_during_capture': stable, 'complete_source_file': complete,
             'range_start_inclusive': start, 'range_end_exclusive': start + len(payload),
             'read_limit_bytes': limit, 'captured_bytes': len(payload),
             'captured_sha256': hashlib.sha256(payload).hexdigest(),
             'truncated': start != 0 or len(payload) != before.st_size}
    if complete:
        value['source_bytes'] = before.st_size
        value['source_sha256'] = value['captured_sha256']
    return value, payload


def retain_failure(logs_prefix, *, private_run_root, destination, error_type):
    """Copy at most 768 KiB plus a bounded report under the held task token.

    The caller catches retention failures and re-raises its original training
    error. Each invocation directory is immutable; nothing here retries a job,
    rewrites a source receipt, consumes a checkpoint, or releases a budget.
    """
    prefix, private = runner.safe(logs_prefix), runner.safe(private_run_root)
    root = runner.safe(destination)
    runner.require(prefix.parent == private / '.suite-logs'
                   and root != private and not root.is_relative_to(private)
                   and not private.is_relative_to(root), 'Failure diagnostics escape their owned source/destination')
    runtime = os.environ
    if runtime.get('DP_SUITE_FLEET_STATE'):
        task = runtime.get('DP_SUITE_TASK_ID')
        runner.require(isinstance(task, str) and runner.ID.fullmatch(task)
                       and root == runner.safe(runtime['DP_SUITE_FLEET_STATE']) / 'durable-metadata' / task,
                       'Failure diagnostics must use the exact task durable-metadata directory')
    target = root / ('training-failure-' + runner.digest(str(prefix))[:24])
    runner.require(not target.exists(), 'Training failure diagnostics already retained; never overwrite')
    records, payloads = {}, {}
    for name in ('intent.json', 'exit.json', 'stdout', 'stderr'):
        limit = CONTROL_BYTES if name.endswith('.json') else LOG_TAIL_BYTES
        record, payload = _capture(prefix.with_name(prefix.name + '.' + name), limit)
        records[name] = record
        if payload is not None:
            payloads[name] = payload
            record['retained'] = {'path': str(target / name), 'bytes': len(payload),
                                  'sha256': record['captured_sha256']}
    total = sum(map(len, payloads.values()))
    runner.require(total <= PAYLOAD_LIMIT_BYTES, 'Failure diagnostic payload exceeds its fixed bound')
    report = {'schema_version': 1, 'status': 'TRAINING_FAILURE_DIAGNOSTICS_RETAINED',
              'utc': runner.utc(), 'original_error_type': error_type,
              'private_run_root': str(private), 'logs_prefix': str(prefix), 'files': records,
              'retained_payload_bytes': total, 'payload_limit_bytes': PAYLOAD_LIMIT_BYTES,
              'training_success': False, 'process_quiescence_claimed': False,
              'automatic_retry': False, 'source_files_modified': False}
    runner.require(len(runner.canonical(report)) + 1 <= REPORT_LIMIT_BYTES, 'Failure diagnostic report exceeds its fixed bound')
    from .suite_metadata_limit import metadata_write
    # Conservatively charge twice the payload/report size for publication.
    with metadata_write(target, 2 * (total + REPORT_LIMIT_BYTES)):
        target.mkdir(parents=True, exist_ok=False)
        for name, payload in payloads.items():
            with (target / name).open('xb') as stream:
                stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        runner.write_once(target / 'DIAGNOSTICS.json', report)
        runner._fsync_directory(target)
    return runner.artifact(target / 'DIAGNOSTICS.json')
