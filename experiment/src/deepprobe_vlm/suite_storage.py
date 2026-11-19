"""Shared, fail-closed storage admission for a one-to-four-node suite.

The two limits are independent: project home and project OSS each have their
own ceiling. A reservation is a conservative *additional peak*, not credit for
files that might later be deleted. Complete objects and uploaded multipart
parts count. Old versions count where readable; the shared-bucket permission
fallback explicitly reports its logical live-object scope and unknown legacy
version bytes, rather than claiming a physical bucket total. Crashed workers retain reservations until
an exact quiescence check and a new inventory permit explicit reconciliation.
All participating workers must use the same ledger on a filesystem supporting
cross-node POSIX flock/atomic replace. This is admission, not a filesystem quota.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import uuid

from .checkpoint_retention import canonical, digest, file_pin, safe, utc, valid_sha


class StorageError(ValueError):
    pass


class InventoryAccessDenied(StorageError):
    """A shared bucket may grant prefix operations without bucket-admin rights."""


def require(condition, message):
    if not condition:
        raise StorageError(message)


def nonnegative(value):
    return type(value) is int and value >= 0


def prefix(value):
    require(isinstance(value, str) and value.startswith('oss://ss-devbox-stream/')
            and value.endswith('/') and len(value.split('/')) >= 5
            and '..' not in value.split('/') and '\\' not in value
            and not any(c.isspace() for c in value), 'Exact owned stream prefix required')
    return value


def validate_config(config):
    fields = {'schema_version', 'roots', 'ledger_root', 'home_limit_bytes', 'oss_limit_bytes',
              'oss_prefixes', 'home_safety_bytes', 'oss_safety_bytes', 'free_guard_bytes'}
    require(isinstance(config, dict) and set(config) == fields and config['schema_version'] == 2,
            'Storage v2 requires explicit separate home and OSS budgets')
    roots = config['roots']
    require(isinstance(roots, list) and roots, 'Owned home roots required')
    paths = [safe(p) for p in roots]
    require(all(p.is_absolute() and str(p) == value and str(p) != '/'
                for p, value in zip(paths, roots)), 'Use canonical absolute owned roots')
    require(all(a != b and not a.is_relative_to(b) and not b.is_relative_to(a)
                for i, a in enumerate(paths) for b in paths[i + 1:]), 'Home roots overlap')
    ledger = safe(config['ledger_root'])
    require(ledger.is_absolute() and any(ledger.is_relative_to(p) for p in paths),
            'Shared ledger must be inside an accounted project root')
    prefixes = config['oss_prefixes']
    require(isinstance(prefixes, list) and prefixes, 'Owned OSS accounting prefixes required')
    for value in prefixes:
        prefix(value)
    require(all(not a.startswith(b) and not b.startswith(a)
                for i, a in enumerate(prefixes) for b in prefixes[i + 1:]), 'OSS prefixes overlap')
    for name in ('home_limit_bytes', 'oss_limit_bytes'):
        require(type(config[name]) is int and 0 < config[name] <= 150_000_000_000,
                'Each domain must stay within its independently authorized 150 GB ceiling')
    for name in ('home_safety_bytes', 'oss_safety_bytes', 'free_guard_bytes'):
        require(nonnegative(config[name]), 'Invalid storage safety reserve')
    return config


def local_inventory(roots, *, detailed=False):
    """Include tmp/cache/partials while tolerating concurrent atomic publication.

    A name may disappear between scandir and lstat as another reserved worker
    renames a journal. Record that observation; additions remain covered by the
    worker's peak reservation. Permission and other I/O errors are not absence.
    """
    rows, vanished = [], []
    owned = [safe(value) for value in roots]
    def walk_error(error):
        if isinstance(error, FileNotFoundError):
            vanished.append(str(error.filename))
            return
        raise StorageError('Home inventory traversal failed: ' + str(error.filename)) from error
    def link_row(path):
        before = path.lstat()
        target = path.resolve(strict=False)
        after = path.lstat()
        require((before.st_ino, before.st_mtime_ns, before.st_size) ==
                (after.st_ino, after.st_mtime_ns, after.st_size), 'Symlink changed during accounting')
        accounted = any(target.is_relative_to(root) for root in owned)
        return {'path': str(path), 'bytes': max(before.st_size, before.st_blocks * 512),
                'kind': 'owned_symlink' if accounted else 'external_symlink', 'target': str(target)}
    for root_value in roots:
        root = safe(root_value)
        require(root.is_dir(), 'Accounted home root is unavailable: ' + str(root))
        for directory, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
            for name in dirs:
                path = Path(directory) / name
                try:
                    if stat.S_ISLNK(path.lstat().st_mode):
                        rows.append(link_row(path))
                except FileNotFoundError:
                    vanished.append(str(path))
            for name in files:
                path = Path(directory) / name
                try:
                    info = path.lstat()
                    if stat.S_ISLNK(info.st_mode):
                        rows.append(link_row(path))
                        continue
                except FileNotFoundError:
                    vanished.append(str(path))
                    continue
                require(stat.S_ISREG(info.st_mode), 'Nonregular file makes home accounting ambiguous')
                # Count every directory entry. Hardlinks are conservatively counted
                # more than once; sparse files use max(logical, allocated) bytes.
                rows.append({'path': str(path), 'bytes': max(info.st_size, info.st_blocks * 512)})
    result = {'bytes': sum(r['bytes'] for r in rows), 'files': len(rows),
            'vanished_during_scan': sorted(set(vanished)),
            'external_symlinks_not_followed': [r for r in rows if r.get('kind') == 'external_symlink'],
            'inventory_sha256': digest(sorted(rows, key=lambda r: r['path']))}
    if detailed:
        result['by_root'] = {root: {'bytes': sum(r['bytes'] for r in rows if r['path'].startswith(root + '/')),
                                   'files': sum(r['path'].startswith(root + '/') for r in rows)} for root in roots}
        result['largest_files'] = sorted(rows, key=lambda r: r['bytes'], reverse=True)[:100]
    return result


def remote_usage(document, prefixes):
    require(isinstance(document, dict) and {'complete', 'objects', 'multipart'} <= set(document)
            and document['complete'] is True, 'Full object-version and multipart inventory required')
    seen, result = set(), {'object_bytes': 0, 'multipart_bytes': 0}
    for category in ('objects', 'multipart'):
        require(isinstance(document[category], list), 'Invalid OSS inventory rows')
        for row in document[category]:
            require(isinstance(row, dict) and isinstance(row.get('uri'), str)
                    and any(row['uri'].startswith(p) for p in prefixes)
                    and '..' not in row['uri'].split('/') and nonnegative(row.get('bytes')),
                    'OSS inventory contains an escaped or invalid entry')
            identity = (category, row['uri'], row.get('version_id', ''), row.get('upload_id', ''),
                        row.get('part_number', ''))
            require(identity not in seen, 'Duplicate OSS inventory entry')
            seen.add(identity)
            if category == 'multipart':
                require(isinstance(row.get('upload_id'), str) and row['upload_id']
                        and type(row.get('part_number')) is int and row['part_number'] > 0,
                        'Multipart part identity missing')
            result['object_bytes' if category == 'objects' else 'multipart_bytes'] += row['bytes']
    return {**result, 'bytes': result['object_bytes'] + result['multipart_bytes'],
            'objects': len(document['objects']), 'parts': len(document['multipart']),
            'accounting_scope': document.get('accounting_scope', 'all_object_versions_and_uploaded_multipart'),
            'legacy_version_bytes_unknown': document.get('legacy_version_bytes_unknown', False),
            'version_inventory_status': document.get('version_inventory_status', 'available'),
            'inventory_sha256': digest(document)}


def _replace(path, value):
    path = safe(path)
    pending = path.with_name(path.name + '.pending-' + uuid.uuid4().hex)
    with pending.open('xb') as handle:
        handle.write(canonical(value)); handle.flush(); os.fsync(handle.fileno())
    os.replace(pending, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class SharedBudget:
    """Durable global reservations; all API mutations hold the same flock."""
    def __init__(self, config, remote_inventory, *, local_scanner=local_inventory):
        self.config = validate_config(config)
        require(callable(remote_inventory), 'A live OSS inventory provider is required')
        self.remote_inventory, self.local_scanner = remote_inventory, local_scanner
        self.root = safe(config['ledger_root'])
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'LEDGER.json'
        with self.lock():
            if self.path.exists():
                self._read()
            else:
                self._write({'schema_version': 2, 'config_sha256': digest(config),
                             'reservations': {}, 'artifacts': {}, 'events': []})

    @contextmanager
    def lock(self):
        path = safe(self.root / 'budget.lock')
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            require(stat.S_ISREG(os.fstat(fd).st_mode) and os.fstat(fd).st_nlink == 1,
                    'Unsafe shared budget lock')
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)

    def _read(self):
        value = json.loads(safe(self.path).read_text())
        require(value.get('schema_version') == 2 and value.get('config_sha256') == digest(self.config),
                'All workers must use the exact same shared storage configuration')
        return value

    def _write(self, ledger):
        _replace(self.path, ledger)

    def _snapshot(self, ledger):
        home = self.local_scanner(self.config['roots'])
        oss = remote_usage(self.remote_inventory(), self.config['oss_prefixes'])
        require(nonnegative(home.get('bytes')), 'Home inventory unavailable')
        active = [r for r in ledger['reservations'].values() if r['state'] == 'ACTIVE']
        reserved = {domain: sum(r[domain + '_bytes'] for r in active) for domain in ('home', 'oss')}
        return {'utc': utc(), 'home': home, 'oss': oss, 'active_reservations': len(active),
                'reserved_bytes': reserved, 'projected_bytes': {
                    domain: observed['bytes'] + reserved[domain] + self.config[domain + '_safety_bytes']
                    for domain, observed in (('home', home), ('oss', oss))}}

    def _admit(self, ledger):
        snapshot = self._snapshot(ledger)
        for domain in ('home', 'oss'):
            require(snapshot['projected_bytes'][domain] < self.config[domain + '_limit_bytes'],
                    domain + ' capacity gated: actual usage + all reservations + safety reaches its limit')
        future = snapshot['reserved_bytes']['home'] + self.config['free_guard_bytes']
        require(all(shutil.disk_usage(p).free > future for p in self.config['roots']),
                'Filesystem free-space guard failed')
        return snapshot

    def snapshot(self):
        with self.lock():
            return self._snapshot(self._read())

    @contextmanager
    def asset_lock(self, identity):
        """One hydrator per exact shared cache; distinct caches can progress together."""
        directory = self.root / 'asset-locks'
        directory.mkdir(exist_ok=True)
        fd = os.open(safe(directory / (digest(identity) + '.lock')),
                     os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            require(stat.S_ISREG(os.fstat(fd).st_mode) and os.fstat(fd).st_nlink == 1,
                    'Unsafe shared asset lock')
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)

    def reserve(self, owner, home_bytes, oss_bytes):
        require(isinstance(owner, dict) and owner and all(isinstance(k, str) for k in owner)
                and nonnegative(home_bytes) and nonnegative(oss_bytes), 'Invalid reservation owner/bytes')
        token = uuid.uuid4().hex
        with self.lock():
            ledger = self._read()
            require(not any(r['state'] == 'ACTIVE' and r['owner'] == owner
                            for r in ledger['reservations'].values()),
                    'Unreconciled reservation already exists for this exact owner')
            ledger['reservations'][token] = {'owner': owner, 'home_bytes': home_bytes, 'oss_bytes': oss_bytes,
                                              'state': 'ACTIVE', 'created_utc': utc()}
            snapshot = self._admit(ledger)
            ledger['events'].append({'kind': 'RESERVED', 'token': token, 'snapshot': snapshot})
            self._write(ledger)
        return token

    def extend(self, token, home_bytes=0, oss_bytes=0):
        require(nonnegative(home_bytes) and nonnegative(oss_bytes), 'Invalid reservation growth')
        with self.lock():
            ledger = self._read()
            row = ledger['reservations'].get(token)
            require(row and row['state'] == 'ACTIVE', 'Active reservation token required')
            row['home_bytes'] = max(row['home_bytes'], home_bytes)
            row['oss_bytes'] = max(row['oss_bytes'], oss_bytes)
            snapshot = self._admit(ledger)
            ledger['events'].append({'kind': 'EXTENDED', 'token': token, 'snapshot': snapshot})
            self._write(ledger)
            return snapshot

    def release(self, token, quiescent):
        require(callable(quiescent), 'Exact stopped-worker proof required to release storage')
        with self.lock():
            ledger = self._read()
            row = ledger['reservations'].get(token)
            require(row and row['state'] == 'ACTIVE', 'Active reservation token required')
            proof = quiescent(row['owner'])
            require(isinstance(proof, dict) and proof.get('owner') == row['owner']
                    and proof.get('active_writers') == [] and proof.get('active_uploads') == [],
                    'Exact task processes/uploads have not been proven quiescent')
            snapshot = self._snapshot(ledger)
            row.update(state='RELEASED', released_utc=utc(), quiescence=proof)
            ledger['events'].append({'kind': 'RELEASED', 'token': token, 'snapshot': snapshot})
            self._write(ledger)
            return snapshot

    def register_artifact(self, artifact_id, *, kind, locations, references=()):
        require(isinstance(artifact_id, str) and artifact_id and kind in {'checkpoint', 'result'},
                'Invalid retained artifact')
        require(isinstance(locations, list) and locations, 'Exact retained artifact locations required')
        for row in locations:
            require(set(row) == {'path', 'bytes', 'sha256'} and nonnegative(row['bytes'])
                    and valid_sha(row['sha256']), 'Artifact location must be byte/SHA pinned')
            path = row['path']
            require((path.startswith('oss://') and any(path.startswith(p) for p in self.config['oss_prefixes']))
                    or (not path.startswith('oss://') and any(safe(path).is_relative_to(safe(p))
                                                              for p in self.config['roots'])),
                    'Artifact escapes project scope')
        require(all(isinstance(r, str) and r for r in references), 'Invalid artifact references')
        value = {'kind': kind, 'permanent': kind == 'result', 'locations': locations,
                 'references': sorted(set(references)), 'completed_references': {}}
        with self.lock():
            ledger = self._read()
            existing = ledger['artifacts'].get(artifact_id)
            require(not existing or all(existing[k] == value[k] for k in ('kind', 'permanent', 'locations')),
                    'Retained artifact identity changed')
            if not existing:
                ledger['artifacts'][artifact_id] = value
                self._write(ledger)

    def complete_reference(self, artifact_id, reference, evidence):
        file_pin_value = file_pin(evidence['path'])
        require(file_pin_value == evidence, 'Reference completion evidence changed')
        with self.lock():
            ledger = self._read()
            row = ledger['artifacts'].get(artifact_id)
            require(row and reference in row['references'], 'Unknown artifact reference')
            existing = row['completed_references'].get(reference)
            require(not existing or existing == evidence, 'Reference completion evidence changed')
            row['completed_references'][reference] = evidence
            self._write(ledger)

    def add_reference(self, artifact_id, reference):
        """Register an additional consumer/recovery hold before a task can use bytes."""
        require(isinstance(reference, str) and reference, 'Exact reference identity required')
        with self.lock():
            ledger = self._read()
            row = ledger['artifacts'].get(artifact_id)
            require(row and reference not in row['completed_references'],
                    'Unknown checkpoint or a completed reference cannot be reopened')
            row['references'] = sorted(set(row['references']) | {reference})
            self._write(ledger)

    @contextmanager
    def deletion_guard(self, artifact_ids):
        """Hold the registry lock through exact deletion to exclude new references."""
        with self.lock():
            ledger = self._read()
            for artifact_id in artifact_ids:
                row = ledger['artifacts'].get(artifact_id)
                require(row and not row['permanent'], 'Public accepted results are permanently retained')
                require(set(row['references']) == set(row['completed_references']),
                        'Downstream consumer or recovery references still need this checkpoint')
                for evidence in row['completed_references'].values():
                    require(file_pin(evidence['path']) == evidence, 'Reference evidence changed')
            yield


class OssutilInventory:
    """Read-only complete version/part accounting, without credentials in argv."""
    def __init__(self, executable, config_file, prefixes):
        self.executable, self.config_file = safe(executable), safe(config_file)
        self.prefixes = [prefix(p) for p in prefixes]

    def _json(self, args):
        result = subprocess.run([str(self.executable), 'api', *args, '-c', str(self.config_file),
                                 '--retry-times', '0', '--output-format', 'json', '-q'],
                                capture_output=True, timeout=60)
        if result.returncode and b'AccessDenied' in result.stderr and b'403' in result.stderr:
            raise InventoryAccessDenied('Shared-bucket version inventory is AccessDenied (403)')
        require(result.returncode == 0 and len(result.stdout) <= 16 * 1024 * 1024,
                'OSS inventory API failed; capacity remains gated')
        value = json.loads(result.stdout)
        require(isinstance(value, dict), 'Invalid OSS inventory response')
        return value

    @staticmethod
    def rows(value, key):
        result = value.get(key) or []
        return result if isinstance(result, list) else [result]

    def pages(self, operation, fixed, markers):
        state, seen = {}, set()
        while True:
            args = [operation, *fixed]
            for name, value in state.items():
                args += [name, value]
            value = self._json(args)
            yield value
            truncated = value.get('IsTruncated')
            require(truncated in (True, False, 'true', 'false'), 'Unknown OSS pagination state')
            if truncated in (False, 'false'):
                return
            state = {flag: str(value[field]) for flag, field in markers.items() if value.get(field) is not None}
            require(state and all(state.values()), 'Missing OSS continuation marker')
            identity = digest(state)
            require(identity not in seen, 'Repeated OSS continuation marker')
            seen.add(identity)

    def __call__(self):
        objects, parts, denied_versions = [], [], False
        for allowed in self.prefixes:
            bucket, key = 'ss-devbox-stream', allowed.removeprefix('oss://ss-devbox-stream/')
            fixed = ['--bucket', bucket, '--prefix', key]
            # Versions include old versions if bucket versioning is enabled or
            # suspended. Listing only current objects would undercount storage.
            version_rows = []
            try:
                for page in self.pages('list-object-versions', fixed + ['--max-keys', '1000'],
                                       {'--key-marker': 'NextKeyMarker', '--version-id-marker': 'NextVersionIdMarker'}):
                    require(page.get('Name') == bucket and page.get('Prefix') == key,
                            'Version inventory bucket/prefix differs')
                    for row in self.rows(page, 'Version'):
                        version_rows.append({'uri': 'oss://' + bucket + '/' + row['Key'],
                                        'bytes': int(row['Size']), 'version_id': row['VersionId'], 'etag': row['ETag']})
                    for row in self.rows(page, 'DeleteMarker'):
                        version_rows.append({'uri': 'oss://' + bucket + '/' + row['Key'], 'bytes': 0,
                                        'version_id': row['VersionId'], 'delete_marker': True})
                objects.extend(version_rows)
            except InventoryAccessDenied:
                # User-approved logical project quota for this shared bucket:
                # live objects + uploaded multipart bytes. Historic versions
                # remain explicitly unknown, never guessed to be zero. Suite
                # writes use unique immutable keys and never overwrite them.
                denied_versions = True
                for page in self.pages('list-objects-v2', fixed + ['--max-keys', '1000'],
                                       {'--continuation-token': 'NextContinuationToken'}):
                    require(page.get('Name') == bucket and page.get('Prefix') == key,
                            'Live inventory bucket/prefix differs')
                    for row in self.rows(page, 'Contents'):
                        objects.append({'uri': 'oss://' + bucket + '/' + row['Key'],
                                        'bytes': int(row['Size']), 'etag': row['ETag']})
            for page in self.pages('list-multipart-uploads', fixed + ['--max-uploads', '1000'],
                                   {'--key-marker': 'NextKeyMarker', '--upload-id-marker': 'NextUploadIdMarker'}):
                require(page.get('Bucket') == bucket and page.get('Prefix') == key,
                        'Multipart inventory bucket/prefix differs')
                for upload in self.rows(page, 'Upload'):
                    upload_key, upload_id = upload['Key'], upload['UploadId']
                    require(upload_key.startswith(key), 'Multipart upload escapes owned prefix')
                    for part_page in self.pages('list-parts', ['--bucket', bucket, '--key', upload_key,
                            '--upload-id', upload_id, '--max-parts', '1000'],
                            {'--part-number-marker': 'NextPartNumberMarker'}):
                        require(part_page.get('Bucket') == bucket and part_page.get('Key') == upload_key
                                and part_page.get('UploadId') == upload_id, 'Part inventory identity differs')
                        for row in self.rows(part_page, 'Part'):
                            parts.append({'uri': 'oss://' + bucket + '/' + upload_key, 'upload_id': upload_id,
                                          'part_number': int(row['PartNumber']), 'bytes': int(row['Size'])})
        return {'complete': True, 'objects': objects, 'multipart': parts,
                'accounting_scope': 'live_objects_and_uploaded_multipart' if denied_versions else
                                    'all_object_versions_and_uploaded_multipart',
                'legacy_version_bytes_unknown': denied_versions,
                'version_inventory_status': 'AccessDenied:403' if denied_versions else 'available'}


def budget_for(environment):
    config, tools = environment['storage'], environment['tools']
    return SharedBudget(config, OssutilInventory(tools['ossutil'], tools['oss_config'], config['oss_prefixes']))
