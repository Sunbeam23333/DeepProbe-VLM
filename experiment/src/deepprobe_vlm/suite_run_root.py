"""Prepare the declared suite container without claiming or modifying task runs."""
from __future__ import annotations

import os
from pathlib import Path
import stat

from . import suite_runner as runner


METADATA_RESERVE_BYTES = 16 * 1024 * 1024


def _identity(path, uid):
    info = runner.safe(path).lstat()
    runner.require(stat.S_ISDIR(info.st_mode) and info.st_uid == uid,
                   'Suite run-root ancestry must contain only directories owned by the preparation UID')
    runner.require(os.access(path, os.W_OK | os.X_OK),
                   'Suite run-root directory is not writable/searchable: ' + str(path))
    return {'path': str(path), 'uid': info.st_uid, 'device': info.st_dev, 'inode': info.st_ino}


def _location(environment):
    raw = environment['run_root']
    path = runner.safe(raw)
    runner.require(Path(raw).is_absolute() and str(path) == raw,
                   'Declared suite run root must be a canonical absolute path')
    declared_roots = environment['storage']['roots']
    runner.require(isinstance(declared_roots, list) and declared_roots, 'Accounted storage roots are required')
    roots = [runner.safe(root) for root in declared_roots]
    runner.require(all(str(root) == declared and str(root) != '/'
                       for root, declared in zip(roots, declared_roots)),
                   'Accounted storage roots must be canonical project directories')
    matches = [root for root in roots if path != root and path.is_relative_to(root)]
    runner.require(len(matches) == 1, 'Suite run root must be strictly inside one accounted storage root')
    uid = os.getuid()
    machine = environment.get('machine_identity')
    runner.require(os.geteuid() == uid and (machine is None or
                   (machine['uid'] == uid and machine['gid'] == os.getgid() == os.getegid())),
                   'Suite run-root preparation must use the declared machine UID/GID')
    root = matches[0]
    observed = {root: _identity(root, uid)}
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if os.path.lexists(current):
            observed[current] = _identity(current, uid)
    ledger_value = environment['storage'].get('ledger_root')
    if ledger_value:
        ledger = runner.safe(ledger_value)
        runner.require(ledger != path and not ledger.is_relative_to(path),
                       'Storage ledger must be outside the declared suite run root')
        missing = [parent for parent in (path, *path.parents)
                   if parent.is_relative_to(root) and parent not in observed]
        runner.require(not any(ledger == parent or ledger.is_relative_to(parent) for parent in missing),
                       'Ledger initialization overlaps missing run-root ancestry; initialize the ledger separately first')
    return root, path, uid, observed


def validate_run_root_location(environment):
    """Pure path/identity check, including before any ledger initialization."""
    root, path, _, observed = _location(environment)
    return {'run_root': str(path), 'accounted_root': str(root), 'exists': path in observed}


def inspect_run_root(environment):
    """Read-only CPU gate; never initialize a ledger, run owner or directory."""
    _, path, _, observed = _location(environment)
    runner.require(path in observed,
                   'Declared suite run root is missing; use --prepare-assets before CPU admission')
    return {'status': 'SUITE_RUN_ROOT_CHECKED', 'directory': observed[path],
            'created_directories': [], 'run_ownership_claimed': False, 'existing_contents_modified': False}


def prepare_run_root(environment):
    """Create only missing ancestors of the declared container under a reservation.

    An existing safe container is checked without touching its contents. Child
    task ownership remains exclusively governed by checkpoint_retention.
    """
    root, path, uid, observed = _location(environment)
    if path in observed:
        return inspect_run_root(environment)
    runner.require(environment['storage'].get('schema_version') == 2,
                   'Automatic run-root creation requires shared storage v2; prepare the directory explicitly')
    from .suite_storage import budget_for
    budget = budget_for(environment)
    owner = {'suite_id': environment['suite_id'], 'operation': 'prepare-run-root', 'run_root': str(path)}
    token = budget.reserve(owner, home_bytes=METADATA_RESERVE_BYTES, oss_bytes=0)
    created, descriptor = [], None
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = os.open(root, flags)
        current = root
        for part in (None, *path.relative_to(root).parts):
            if part is not None:
                current /= part
                if current not in observed:
                    # A concurrent appearance must fail; never adopt a raced path.
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                    created.append(str(current))
                child = os.open(part, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            info = os.fstat(descriptor)
            actual = _identity(current, uid)
            runner.require(stat.S_ISDIR(info.st_mode) and info.st_uid == uid
                           and (actual['device'], actual['inode']) == (info.st_dev, info.st_ino)
                           and (current not in observed or actual == observed[current]),
                           'Suite run-root directory identity changed during preparation')
        os.fsync(descriptor)
        checked = inspect_run_root(environment)
        runner.require(checked['directory'] == actual, 'Prepared run-root path changed')
    finally:
        if descriptor is not None:
            os.close(descriptor)
        # mkdir/fsync above are synchronous and start no worker or upload.
        # On failure keep any created directories; never remove existing data.
        budget.release(token, quiescent=lambda value: {
            'owner': value, 'active_writers': [], 'active_uploads': [], 'synchronous_metadata_only': True})
    return {**checked, 'status': 'SUITE_RUN_ROOT_PREPARED', 'created_directories': created,
            'metadata_reservation_bytes': METADATA_RESERVE_BYTES, 'storage_reservation_released': True}
