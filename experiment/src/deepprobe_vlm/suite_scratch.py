"""Declared per-Pod disk scratch; shared project storage remains a separate domain."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import re
import stat

from . import suite_runner as runner


GIB = 1024 ** 3
LIMIT_BYTES = 100 * GIB
VOLUME_NAME = 'deepprobe-scratch'
DEFAULT_SCRATCH = {'schema_version': 1, 'volume_name': VOLUME_NAME,
    'mount_path': '/deepprobe-scratch', 'limit_bytes': LIMIT_BYTES,
    'free_guard_bytes': GIB, 'safety_bytes': GIB, 'shared_home_reserve_bytes': 1_000_000_000}


def validate_scratch(value):
    runner.require(isinstance(value, dict) and set(value) == set(DEFAULT_SCRATCH),
                   'Scratch requires the complete explicit disk-emptyDir policy')
    for name in set(DEFAULT_SCRATCH) - {'mount_path'}:
        runner.require(type(value[name]) is type(DEFAULT_SCRATCH[name]) and value[name] == DEFAULT_SCRATCH[name],
                       'Unsupported scratch policy field: ' + name)
    raw = value['mount_path']
    runner.require(isinstance(raw, str) and raw and '\\' not in raw and not any(c.isspace() for c in raw),
                   'Scratch mount must be an absolute physical directory')
    path = runner.safe(raw)
    runner.require(str(path) == raw and str(path) != '/' and path.parts[1] not in {'proc', 'sys', 'dev'},
                   'Scratch mount must be a canonical non-system directory')
    return value


def _policy(profile):
    runner.require(profile.get('schema_version') == 3, 'Node-local scratch requires execution profile v3')
    return validate_scratch(profile['scratch'])


def worker_run_root(profile, worker_id):
    from .suite_fleet import worker_identity
    worker_identity(profile, worker_id)
    return runner.safe(_policy(profile)['mount_path']) / 'runs' / worker_id


def _quantity_bytes(value):
    # Kubernetes may canonicalize an admitted quantity to bytes or a binary unit.
    text = str(value)
    match = re.fullmatch(r'([0-9]+)(Ki|Mi|Gi|Ti)?', text)
    runner.require(match is not None and type(value) in {str, int}, 'Invalid scratch storage quantity')
    return int(match[1]) * {'Ki': 1024, 'Mi': 1024 ** 2, 'Gi': GIB, 'Ti': 1024 ** 4, None: 1}[match[2]]


def validate_scratch_pod(pod, profile):
    """Validate both prepared Job templates and the actual live Pod spec."""
    config = _policy(profile)
    volumes = pod.get('volumes', [])
    runner.require(isinstance(volumes, list) and all(isinstance(v, dict) and isinstance(v.get('name'), str) for v in volumes)
                   and len({v['name'] for v in volumes}) == len(volumes), 'Invalid or duplicate Pod volumes')
    selected = [v for v in volumes if v['name'] == config['volume_name']]
    runner.require(len(selected) == 1 and set(selected[0]) == {'name', 'emptyDir'},
                   'Scratch must be the declared emptyDir, never a hostPath or PVC')
    empty = selected[0]['emptyDir']
    runner.require(isinstance(empty, dict) and {'sizeLimit'} <= set(empty) <= {'sizeLimit', 'medium'}
                   and empty.get('medium', '') == '' and _quantity_bytes(empty['sizeLimit']) == config['limit_bytes'],
                   'Scratch requires a bounded 100Gi disk emptyDir, not memory')
    containers = pod.get('containers', [])
    runner.require(len(containers) == 1 and not pod.get('initContainers') and not pod.get('ephemeralContainers'),
                   'Scratch requires exactly one execution container')
    container = containers[0]
    mounts = container.get('volumeMounts', [])
    runner.require(isinstance(mounts, list) and all(isinstance(m, dict) for m in mounts), 'Invalid volume mounts')
    selected_mounts = [m for m in mounts if m.get('name') == config['volume_name']]
    runner.require(len(selected_mounts) == 1 and set(selected_mounts[0]) <= {
                       'name', 'mountPath', 'readOnly', 'subPath', 'subPathExpr', 'mountPropagation'}
                   and selected_mounts[0].get('mountPath') == config['mount_path']
                   and selected_mounts[0].get('readOnly', False) is False
                   and selected_mounts[0].get('subPath', '') == ''
                   and selected_mounts[0].get('subPathExpr', '') == ''
                   and selected_mounts[0].get('mountPropagation', 'None') == 'None',
                   'Scratch must mount its complete writable volume without subPath or propagation')
    mount = Path(config['mount_path'])
    for item in mounts:
        if item is selected_mounts[0]:
            continue
        # Other mounts belong to the Pod namespace, not the preparation host.
        # For example /var/run is commonly a harmless symlink on that host.
        raw = item.get('mountPath')
        runner.require(isinstance(raw, str) and raw and '\\' not in raw, 'Invalid Pod mount path')
        other = Path(raw)
        runner.require(other.is_absolute() and '..' not in other.parts and str(other) == raw,
                       'Pod mount paths must be canonical absolute paths')
        runner.require(not other.is_relative_to(mount) and not mount.is_relative_to(other),
                       'Another Pod mount overlaps node-local scratch')
    resources = container.get('resources', {})
    runner.require(all(_quantity_bytes(resources.get(kind, {}).get('ephemeral-storage', '')) == config['limit_bytes']
                       for kind in ('requests', 'limits')), 'Scratch requires explicit 100Gi ephemeral-storage request and limit')
    security = pod.get('securityContext', {})
    runner.require(type(security.get('runAsUser')) is int and security['runAsUser'] > 0
                   and type(security.get('runAsGroup')) is int and security['runAsGroup'] > 0
                   and security.get('runAsNonRoot') is True
                   and security.get('fsGroup') == security['runAsGroup'],
                   'Scratch requires the declared non-root UID/GID and volume fsGroup')
    runner.require('fsGroupChangePolicy' not in security,
                   'The supported PyTorchJob CRD does not accept fsGroupChangePolicy')
    overrides = container.get('securityContext', {})
    runner.require(all(key not in overrides or overrides[key] == security[key]
                       for key in ('runAsUser', 'runAsGroup', 'runAsNonRoot')),
                   'Container security context overrides the scratch identity')
    return {'volume_name': config['volume_name'], 'mount_path': config['mount_path'],
            'limit_bytes': config['limit_bytes'], 'medium': 'disk',
            'uid': security['runAsUser'], 'gid': security['runAsGroup']}


def configure_scratch_pod(pod, profile, machine_identity):
    """Configure only the declared scratch fields in a copied worker template."""
    config = _policy(profile)
    runner.require(set(machine_identity) == {'uid', 'gid'} and
                   all(type(v) is int and v > 0 for v in machine_identity.values()), 'Explicit scratch machine identity required')
    volume = {'name': config['volume_name'], 'emptyDir': {'sizeLimit': '100Gi'}}
    volumes = pod.setdefault('volumes', [])
    existing = [v for v in volumes if v.get('name') == config['volume_name']]
    runner.require(len(existing) <= 1, 'Duplicate scratch volume in worker template')
    if not existing:
        volumes.append(deepcopy(volume))
    runner.require(len(pod.get('containers', [])) == 1, 'Exactly one scratch execution container required')
    container = pod['containers'][0]
    mount = {'name': config['volume_name'], 'mountPath': config['mount_path']}
    mounts = container.setdefault('volumeMounts', [])
    existing = [m for m in mounts if m.get('name') == config['volume_name']]
    runner.require(len(existing) <= 1, 'Duplicate scratch mount in worker template')
    if not existing:
        mounts.append(mount)
    for kind in ('requests', 'limits'):
        resources = container.setdefault('resources', {}).setdefault(kind, {})
        runner.require('ephemeral-storage' not in resources or _quantity_bytes(resources['ephemeral-storage']) == config['limit_bytes'],
                       'Template ephemeral-storage conflicts with the declared scratch limit')
        resources['ephemeral-storage'] = '100Gi'
    security = pod.setdefault('securityContext', {})
    # The cluster's PyTorchJob CRD predates fsGroupChangePolicy. It does not
    # affect emptyDir volumes; retain fsGroup for the private disk permissions.
    security.update(runAsUser=machine_identity['uid'], runAsGroup=machine_identity['gid'], runAsNonRoot=True,
                    fsGroup=machine_identity['gid'])
    validate_scratch_pod(pod, profile)
    return pod


def _mounts():
    raw = Path('/proc/self/mountinfo').read_text()
    def decode(value):
        return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), value)
    result = []
    for line in raw.splitlines():
        before, after = line.split(' - ', 1)
        left, right = before.split(), after.split()
        runner.require(len(left) >= 6 and len(right) >= 3, 'Malformed live mountinfo')
        result.append({'mount_id': left[0], 'parent_id': left[1], 'device': left[2],
            'root': decode(left[3]), 'mount_path': decode(left[4]), 'options': left[5],
            'filesystem': right[0], 'source': decode(right[1]), 'super_options': right[2]})
    return result


def _pod_for_live(environment, live):
    from .suite_cluster import list_workloads
    from .suite_fleet import verify_job_identity
    from .suite_actions import terminal_job
    profile = environment['execution_profile']
    document = list_workloads(environment['tools'],
        project_name_prefixes=environment['allocation_identity']['project_name_prefixes'])
    runner.require(not document.get('metadata', {}).get('continue'), 'Incomplete scratch workload observation')
    jobs = [item for item in document['items'] if item.get('kind') == 'PyTorchJob'
            and item.get('metadata', {}).get('uid') == live['job_uid']]
    pods = [item for item in document['items'] if item.get('kind') == 'Pod'
            and item.get('metadata', {}).get('uid') == live['pod_uid']]
    runner.require(len(jobs) == len(pods) == 1, 'Exact scratch allocation is no longer observable')
    runner.require(isinstance(live.get('binding_sha256'), str) and runner.SHA.fullmatch(live['binding_sha256']),
                   'Scratch observation requires the independently bound fleet SHA')
    verify_job_identity(jobs[0], profile, live['worker_id'], live['binding_sha256'])
    pod = pods[0]
    runner.require(not terminal_job(jobs[0]) and not jobs[0]['metadata'].get('deletionTimestamp')
                   and not pod['metadata'].get('deletionTimestamp')
                   and pod.get('status', {}).get('phase') == 'Running' and pod['spec'].get('nodeName') == live['node']
                   and pod['metadata'].get('namespace') == 'default'
                   and any(r.get('uid') == live['job_uid'] and r.get('kind') == 'PyTorchJob'
                           for r in pod['metadata'].get('ownerReferences', [])), 'Scratch Pod identity or owner changed')
    validate_scratch_pod(pod['spec'], profile)
    if 'isolated_storage' in environment:
        from .suite_isolated_pod import validate_isolated_pod
        validate_isolated_pod(pod['spec'], environment)
    return pod


def _directory_info(path):
    return runner.safe(path).lstat()


def inspect_scratch(environment, live):
    """Read-only live Pod/mount/capacity gate; never create local directories."""
    from .suite_fleet import validate_execution_profile, worker_identity
    profile = validate_execution_profile(environment['execution_profile'], environment['suite_id'])
    config = _policy(profile)
    worker_identity(profile, live.get('worker_id'))
    runner.require(all(isinstance(live.get(k), str) and live[k] for k in ('worker_id', 'job_uid', 'pod_uid', 'node')),
                   'Exact worker, Job, Pod and node identity required for scratch')
    machine = environment['machine_identity']
    runner.require(os.getuid() == os.geteuid() == machine['uid'] > 0
                   and os.getgid() == os.getegid() == machine['gid'] > 0, 'Scratch must run as its declared non-root UID/GID')
    mount = runner.safe(config['mount_path'])
    runner.require(environment['run_root'] == str(mount / 'runs'), 'Environment run_root differs from scratch declaration')
    for root in environment['storage']['roots']:
        shared = runner.safe(root)
        runner.require(not mount.is_relative_to(shared) and not shared.is_relative_to(mount),
                       'Node-local scratch must not overlap accounted shared home roots')
    pod = _pod_for_live(environment, live)
    contract = validate_scratch_pod(pod['spec'], profile)
    isolation = None
    if 'isolated_storage' in environment:
        from .suite_isolated_pod import inspect_isolated_mounts
        isolation = inspect_isolated_mounts(environment, _mounts())
    runner.require((contract['uid'], contract['gid']) == (machine['uid'], machine['gid']), 'Live scratch Pod UID/GID differs')
    matches = [row for row in _mounts() if Path(row['mount_path']).is_relative_to(mount)]
    runner.require(len(matches) == 1 and matches[0]['mount_path'] == str(mount),
                   'Declared scratch must be an actual distinct mount without nested mounts')
    observed = matches[0]
    suffix = ('pods', live['pod_uid'], 'volumes', 'kubernetes.io~empty-dir', config['volume_name'])
    runner.require(tuple(Path(observed['root']).parts[-5:]) == suffix
                   and observed['filesystem'] in {'ext4', 'xfs', 'btrfs', 'zfs'}
                   and 'rw' in observed['options'].split(','), 'Live scratch mount is not this Pod\'s writable disk emptyDir')
    info = _directory_info(mount)
    runner.require(stat.S_ISDIR(info.st_mode) and info.st_uid in {0, machine['uid']}
                   and info.st_gid == machine['gid'] and os.access(mount, os.W_OK | os.X_OK)
                   and observed['device'] == f'{os.major(info.st_dev)}:{os.minor(info.st_dev)}',
                   'Scratch filesystem identity or write access differs')
    usage = os.statvfs(mount)
    available = usage.f_bavail * usage.f_frsize
    runner.require(available >= config['limit_bytes'] + config['free_guard_bytes'],
                   'Node-local scratch filesystem lacks its declared peak plus free-space guard')
    after = _directory_info(mount)
    runner.require((info.st_dev, info.st_ino, info.st_uid, info.st_gid) ==
                   (after.st_dev, after.st_ino, after.st_uid, after.st_gid)
                   and [row for row in _mounts() if Path(row['mount_path']).is_relative_to(mount)] == matches,
                   'Scratch mount changed during startup inspection')
    return {'schema_version': 1, 'status': 'SUITE_SCRATCH_STARTUP_VERIFIED', 'observed_utc': runner.utc(),
            **{k: live[k] for k in ('worker_id', 'job_uid', 'pod_uid', 'node', 'binding_sha256')},
            'policy_sha256': runner.digest(config), 'mount': observed,
            'directory': {'path': str(mount), 'device': info.st_dev, 'inode': info.st_ino,
                          'uid': info.st_uid, 'gid': info.st_gid},
            'worker_run_root': str(worker_run_root(profile, live['worker_id'])),
            'limit_bytes': config['limit_bytes'], 'available_bytes': available,
            'free_guard_bytes': config['free_guard_bytes'], 'safety_bytes': config['safety_bytes'],
            'shared_home_reserve_bytes': config['shared_home_reserve_bytes'],
            'pod_spec_sha256': runner.digest(pod['spec']), 'local_directories_created': False,
            **({'isolated_input_mounts': isolation} if isolation is not None else {}),
            'interrupted_local_checkpoint_survives_pod_loss': False,
            'recovery_scope': 'only independently verified remote checkpoints'}
