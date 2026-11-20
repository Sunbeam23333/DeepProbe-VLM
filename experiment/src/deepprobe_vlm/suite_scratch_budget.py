"""Per-Pod private disk reservations, separate from the shared home ledger.

Only the owning worker scans its emptyDir. The durable journal lives in shared
state and binds its exact Job/Pod/node and local directory inode. Neither a
lease nor a reservation expires after a lost Pod. Kubernetes disk emptyDir
limits provide eviction accounting; this ledger provides admission, not an
ENOSPC-enforcing filesystem quota.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import shutil
import stat
import uuid

from . import suite_runner as runner
from .suite_storage import StorageError, _replace, local_inventory, nonnegative, require


class PrivateScratchBudget:
    def __init__(self, profile, worker_id, live, state_root, *, scanner=local_inventory):
        from .suite_fleet import validate_execution_profile, worker_identity
        from .suite_scratch import worker_run_root
        validate_execution_profile(profile)
        require(profile['schema_version'] == 3, 'Private disk needs an explicit scratch profile')
        worker_identity(profile, worker_id)
        require(live.get('worker_id') == worker_id and all(isinstance(live.get(k), str) and live[k]
                for k in ('job_uid', 'pod_uid', 'node')), 'Exact private worker allocation identity required')
        self.policy = profile['scratch']
        self.mount = runner.safe(self.policy['mount_path'])
        self.runs = worker_run_root(profile, worker_id)
        for path in (self.mount, self.runs):
            info = path.stat()
            require(stat.S_ISDIR(info.st_mode), 'Private scratch directory is unavailable')
        require(self.runs.stat().st_uid == os.getuid(), 'Private run container belongs to another UID')
        self.directory_identity = {'device': self.mount.stat().st_dev, 'inode': self.mount.stat().st_ino,
                                   'runs_device': self.runs.stat().st_dev, 'runs_inode': self.runs.stat().st_ino}
        self.identity = {'profile_sha256': runner.digest(profile), 'worker_id': worker_id,
                         **{k: live[k] for k in ('job_uid', 'pod_uid', 'node')},
                         'mount_path': str(self.mount), 'run_root': str(self.runs), **self.directory_identity}
        self.root = runner.safe(state_root) / 'workers' / worker_id
        require(self.root.is_dir(), 'Private budget requires an already registered shared worker directory')
        self.path = self.root / 'SCRATCH_BUDGET.json'
        self.scanner = scanner
        with self.lock():
            if self.path.exists():
                self._read()
            else:
                value = {'schema_version': 1, 'identity': self.identity, 'policy': self.policy,
                         'reservations': {}, 'events': []}
                self._admit(value)
                _replace(self.path, value)

    @contextmanager
    def lock(self):
        fd = os.open(runner.safe(self.root / 'scratch-budget.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            require(stat.S_ISREG(os.fstat(fd).st_mode) and os.fstat(fd).st_nlink == 1, 'Unsafe private budget lock')
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _read(self):
        value = runner.read(self.path)
        require(value.get('schema_version') == 1 and value.get('identity') == self.identity
                and value.get('policy') == self.policy,
                'Private scratch belongs to another exact Pod; explicit recovery is required')
        return value

    def _snapshot(self, value):
        observed = {'device': self.mount.stat().st_dev, 'inode': self.mount.stat().st_ino,
                    'runs_device': self.runs.stat().st_dev, 'runs_inode': self.runs.stat().st_ino}
        require(observed == self.directory_identity, 'Private scratch mount or run container changed')
        inventory = self.scanner([str(self.mount)])
        require(nonnegative(inventory.get('bytes')) and not inventory.get('external_symlinks_not_followed'),
                'Private scratch inventory escapes its accounted mount')
        active = [row for row in value['reservations'].values() if row['state'] == 'ACTIVE']
        reserved = sum(row['bytes'] for row in active)
        return {'utc': runner.utc(), 'identity': self.identity, 'inventory': inventory,
                'active_reservations': len(active), 'reserved_bytes': reserved,
                'projected_bytes': inventory['bytes'] + reserved + self.policy['safety_bytes'],
                'limit_bytes': self.policy['limit_bytes'], 'free_bytes': shutil.disk_usage(self.mount).free,
                'accounting_scope': 'this_exact_pod_private_emptydir', 'hard_filesystem_quota': False}

    def _admit(self, value):
        snapshot = self._snapshot(value)
        require(snapshot['projected_bytes'] < self.policy['limit_bytes'],
                'Private scratch capacity gated: actual plus reservations and safety reaches its limit')
        require(snapshot['free_bytes'] > snapshot['reserved_bytes'] + self.policy['free_guard_bytes'],
                'Private scratch filesystem free-space guard failed')
        return snapshot

    def snapshot(self):
        with self.lock():
            return self._snapshot(self._read())

    def reserve(self, owner, amount):
        require(isinstance(owner, dict) and owner.get('worker_id') == self.identity['worker_id']
                and owner.get('job_uid') == self.identity['job_uid'] and nonnegative(amount),
                'Private reservation requires its exact worker Job and nonnegative bytes')
        token = uuid.uuid4().hex
        with self.lock():
            value = self._read()
            require(not any(row['state'] == 'ACTIVE' and row['owner'] == owner
                            for row in value['reservations'].values()), 'Unreconciled private reservation exists')
            value['reservations'][token] = {'owner': owner, 'bytes': amount, 'state': 'ACTIVE', 'utc': runner.utc()}
            snapshot = self._admit(value)
            value['events'].append({'kind': 'RESERVED', 'token': token, 'snapshot': snapshot})
            _replace(self.path, value)
        return token

    def extend(self, token, amount):
        require(nonnegative(amount), 'Invalid private scratch reservation growth')
        with self.lock():
            value = self._read()
            row = value['reservations'].get(token)
            require(row and row['state'] == 'ACTIVE', 'Active private scratch reservation required')
            row['bytes'] = max(amount, row['bytes'])
            snapshot = self._admit(value)
            value['events'].append({'kind': 'EXTENDED', 'token': token, 'snapshot': snapshot})
            _replace(self.path, value)
            return snapshot

    def release(self, token, quiescent):
        require(callable(quiescent), 'Private reservation release needs exact child/upload closure')
        with self.lock():
            value = self._read()
            row = value['reservations'].get(token)
            require(row and row['state'] == 'ACTIVE', 'Active private scratch reservation required')
            proof = quiescent(row['owner'])
            require(isinstance(proof, dict) and proof.get('owner') == row['owner']
                    and proof.get('active_writers') == [] and proof.get('active_uploads') == [],
                    'Private scratch worker is not proven quiescent')
            snapshot = self._snapshot(value)
            row.update(state='RELEASED', released_utc=runner.utc(), quiescence=proof)
            value['events'].append({'kind': 'RELEASED', 'token': token, 'snapshot': snapshot})
            _replace(self.path, value)
            return snapshot


class TieredTaskBudget:
    """Route training payload reservations to its private disk, preserving home accounting."""
    def __init__(self, shared, private, tasks):
        self.shared, self.private, self.tasks = shared, private, tasks

    def snapshot(self):
        return {'shared': self.shared.snapshot(), 'private_scratch': self.private.snapshot()}

    def reserve(self, owner, home_bytes, oss_bytes):
        task = self.tasks.get(owner['task_id'], {})
        training = task.get('kind') == 'train'
        if owner['task_id'] == 'startup':
            return {'home': self.shared.reserve(owner, home_bytes, oss_bytes), 'scratch': None}
        # Consumers may use TMPDIR for transient decodes: conservatively reserve
        # their declared peak in both domains instead of exempting those bytes.
        private_token = self.private.reserve(owner, home_bytes + 4 * 64 * 1024 * 1024)
        try:
            home_token = self.shared.reserve(owner,
                self.private.policy['shared_home_reserve_bytes'] if training else home_bytes, oss_bytes)
        except BaseException:
            self.private.release(private_token, lambda exact: {'owner': exact, 'active_writers': [],
                'active_uploads': [], 'claim_not_dispatched': True})
            raise
        return {'home': home_token, 'scratch': private_token}

    def release(self, token, quiescent):
        # Preserve the shared reservation on a failed/uncertain private release.
        if token['scratch'] is not None:
            self.private.release(token['scratch'], quiescent)
        return self.shared.release(token['home'], quiescent)
