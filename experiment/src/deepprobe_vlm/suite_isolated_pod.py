"""Read-only legacy inputs with one writable, lifetime-budgeted home overlay."""
from __future__ import annotations

from copy import deepcopy
from pathlib import PurePosixPath

from . import suite_runner as runner

USER_HOME = PurePosixPath('/home/jinhaopeng')
HOST_HOME = PurePosixPath('/mnt/vepfs/base2/jinhaopeng')
VEPFS = PurePosixPath('/mnt/vepfs')
VOLUME = 'deepprobe-isolated-home'


def _path(raw):
    runner.require(isinstance(raw, str) and raw and '\\' not in raw, 'Invalid isolated Pod path')
    path = PurePosixPath(raw)
    runner.require(path.is_absolute() and '..' not in path.parts and str(path) == raw,
                   'Isolated Pod paths must be canonical absolute paths')
    return path


def _layout(environment):
    profile = environment['execution_profile']
    runner.require(profile.get('schema_version') == 3 and len(profile['workers']) == 1,
                   'Read-only input isolation requires exactly one private worker')
    namespace = _path(environment['isolated_storage']['namespace_root'])
    runner.require(namespace != USER_HOME and namespace.is_relative_to(USER_HOME),
                   'Writable namespace must be a strict child of the declared user home')
    return namespace, HOST_HOME / namespace.relative_to(USER_HOME)


def configure_isolated_pod(pod, environment):
    """Modify a copied template only; no filesystem or scheduler operations."""
    if 'isolated_storage' not in environment:
        return pod
    namespace, host_namespace = _layout(environment)
    runner.require(len(pod.get('containers', [])) == 1, 'One isolated execution container required')
    container = pod['containers'][0]
    for mount in container.get('volumeMounts', []):
        if mount.get('name') in {'vepfs', 'user-home'}:
            mount['readOnly'] = True
    volume = {'name': VOLUME, 'hostPath': {'path': str(host_namespace), 'type': 'Directory'}}
    mount = {'name': VOLUME, 'mountPath': str(namespace), 'readOnly': False}
    volumes, mounts = pod.setdefault('volumes', []), container.setdefault('volumeMounts', [])
    if not any(row.get('name') == VOLUME for row in volumes):
        volumes.append(deepcopy(volume))
    if not any(row.get('name') == VOLUME for row in mounts):
        mounts.append(deepcopy(mount))
    security = container.setdefault('securityContext', {})
    runner.require(security.get('allowPrivilegeEscalation', False) is False,
                   'Isolated worker cannot allow privilege escalation')
    security['allowPrivilegeEscalation'] = False
    validate_isolated_pod(pod, environment)
    return pod


def validate_isolated_pod(pod, environment):
    """Check the prepared template and independently observed live Pod spec."""
    if 'isolated_storage' not in environment:
        return None
    namespace, host_namespace = _layout(environment)
    from .suite_scratch import validate_scratch_pod
    validate_scratch_pod(pod, environment['execution_profile'])
    container = pod['containers'][0]
    security = container.get('securityContext', {})
    runner.require(security.get('privileged', False) is False
                   and security.get('allowPrivilegeEscalation') is False
                   and set(security.get('capabilities', {}).get('add', [])) <= {'IPC_LOCK'}
                   and not container.get('volumeDevices') and not pod.get('hostPID') and not pod.get('hostIPC'),
                   'Isolated worker may not bypass read-only mounts')
    scratch = environment['execution_profile']['scratch']
    volumes = {row['name']: row for row in pod['volumes']}
    expected = {'vepfs': VEPFS, 'user-home': HOST_HOME, VOLUME: host_namespace}
    for name, path in expected.items():
        runner.require(volumes.get(name) == {'name': name, 'hostPath': {'path': str(path), 'type': 'Directory'}},
                       'Isolated hostPath differs from the exact input/namespace contract: ' + name)
    runner.require(volumes.get('cache-volume') == {'name': 'cache-volume', 'emptyDir': {'medium': 'Memory'}},
                   'Only the original memory-backed shm volume is allowed')
    expected_mounts = {
        ('vepfs', str(VEPFS)): True, ('user-home', str(HOST_HOME)): True,
        ('user-home', str(USER_HOME)): True, (VOLUME, str(namespace)): False,
        (scratch['volume_name'], scratch['mount_path']): False, ('cache-volume', '/dev/shm'): False}
    found = set()
    protected = (VEPFS, USER_HOME, _path(scratch['mount_path']), PurePosixPath('/dev/shm'))
    mounted_names = set()
    for row in container['volumeMounts']:
        path = _path(row.get('mountPath'))
        name, key = row.get('name'), (row.get('name'), str(path))
        runner.require(key not in found and set(row) <= {
            'name', 'mountPath', 'readOnly', 'subPath', 'subPathExpr', 'mountPropagation'}
            and row.get('subPath', '') == '' and row.get('subPathExpr', '') == ''
            and row.get('mountPropagation', 'None') == 'None', 'Ambiguous isolated volume mount')
        found.add(key); mounted_names.add(name)
        if key in expected_mounts:
            runner.require(row.get('readOnly', False) is expected_mounts[key],
                           'Legacy mounts must be read-only; only the namespace/scratch/shm may be writable')
        else:
            volume = volumes.get(name, {})
            runner.require(name not in expected and name not in {'cache-volume', scratch['volume_name']}
                           and row.get('readOnly') is True
                           and len(set(volume) - {'name'}) == 1
                           and set(volume) - {'name'} <= {'projected', 'secret', 'configMap'}
                           and not any(path.is_relative_to(p) or p.is_relative_to(path) for p in protected),
                           'Unreviewed writable mount or input alias in isolated Pod')
    runner.require(set(expected_mounts) <= found and mounted_names == set(volumes),
                   'Isolated Pod mounts must include every exact input and writable namespace volume')
    return {'namespace_root': str(namespace), 'host_namespace': str(host_namespace),
            'read_only_mounts': [str(VEPFS), str(HOST_HOME), str(USER_HOME)],
            'writable_home_mounts': [str(namespace)]}


def inspect_isolated_mounts(environment, mounts):
    """Verify kernel mount flags and namespace backing after live Pod validation."""
    if 'isolated_storage' not in environment:
        return None
    namespace, _ = _layout(environment)
    exact = {}
    for path, mode in ((VEPFS, 'ro'), (HOST_HOME, 'ro'), (USER_HOME, 'ro'), (namespace, 'rw')):
        rows = [row for row in mounts if row['mount_path'] == str(path)]
        runner.require(len(rows) == 1 and mode in rows[0]['options'].split(','),
                       'Kernel did not enforce the exact isolated mount mode: ' + str(path))
        exact[str(path)] = rows[0]
    home, overlay = exact[str(USER_HOME)], exact[str(namespace)]
    runner.require(overlay['device'] == home['device']
                   and _path(overlay['root']) == _path(home['root']) / namespace.relative_to(USER_HOME),
                   'Writable namespace does not map to the held home subtree')
    for row in mounts:
        path = _path(row['mount_path'])
        if path == namespace:
            continue
        if path.is_relative_to(namespace):
            raise ValueError('Unexpected nested mount inside the isolated namespace')
        if path.is_relative_to(VEPFS) or path.is_relative_to(USER_HOME):
            runner.require('ro' in row['options'].split(','), 'Writable legacy mount or alias bypasses isolation')
    return {'status': 'ISOLATED_INPUT_MOUNTS_VERIFIED', 'mounts': exact,
            'parent_ledger_writable': False, 'legacy_cache_writable': False}
