"""Retain exact small receipt bytes in shared state before emptyDir disappears."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import uuid

from . import suite_runner as runner


def retain_pin(pin, run_root, shared_root, *, max_bytes=64 * 1024 * 1024):
    runner.verify_pin(pin)
    source = runner.safe(pin['path'])
    run_root, shared_root = runner.safe(run_root), runner.safe(shared_root)
    if not source.is_relative_to(run_root):
        return dict(pin)
    size = source.stat().st_size
    runner.require(size <= max_bytes and not source.name.endswith(('.safetensors', '.pt', '.pth', '.bin')),
                   'Only bounded metadata may be copied from private scratch into shared state')
    runner.require(not shared_root.is_relative_to(run_root) and not run_root.is_relative_to(shared_root),
                   'Durable metadata destination overlaps private scratch')
    destination = shared_root / (runner.digest(str(source))[:16] + '--' + pin['sha256'] + '--' + source.name)
    from .suite_metadata_limit import metadata_write
    # Count the temporary hardlink publication peak and its provenance receipt.
    with metadata_write(destination, 2 * size + 16384):
        return _retain(pin, source, shared_root, destination)


def _retain(pin, source, shared_root, destination):
    shared_root.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        runner.verify_pin({**pin, 'path': str(destination)})
    else:
        pending = destination.with_name(destination.name + '.pending-' + uuid.uuid4().hex)
        with source.open('rb') as input_stream, pending.open('xb') as output:
            shutil.copyfileobj(input_stream, output, 1024 * 1024)
            output.flush(); os.fsync(output.fileno())
        runner.verify_pin({**pin, 'path': str(pending)})
        os.link(pending, destination)
        pending.unlink()
        runner._fsync_directory(shared_root)
    result = runner.artifact(destination)
    mapping = {'schema_version': 1, 'source': runner.artifact(source), 'retained': result,
               'bytes_preserved': True, 'source_owner_or_manifest_rewritten': False}
    receipt = destination.with_name(destination.name + '.SOURCE.json')
    if receipt.exists():
        runner.require(runner.read(receipt) == mapping, 'Durable metadata source provenance changed')
    else:
        runner.write_once(receipt, mapping)
    return result
