"""Prepared/live isolation contracts; synthetic mounts do not certify a node."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from deepprobe_vlm import suite_cluster, suite_fleet_launch, suite_launch
from deepprobe_vlm import suite_isolated_pod as isolated
from deepprobe_vlm import suite_runner as runner, suite_scratch as scratch
from test_suite_scratch import profile, pod_spec


def legacy_inputs(spec):
    spec['volumes'].extend([
        {'name': 'vepfs', 'hostPath': {'path': '/mnt/vepfs', 'type': 'Directory'}},
        {'name': 'user-home', 'hostPath': {'path': '/mnt/vepfs/base2/jinhaopeng', 'type': 'Directory'}},
        {'name': 'cache-volume', 'emptyDir': {'medium': 'Memory'}}])
    spec['containers'][0]['volumeMounts'].extend([
        {'name': 'vepfs', 'mountPath': '/mnt/vepfs'},
        {'name': 'user-home', 'mountPath': '/mnt/vepfs/base2/jinhaopeng'},
        {'name': 'user-home', 'mountPath': '/home/jinhaopeng'},
        {'name': 'cache-volume', 'mountPath': '/dev/shm'}])
    return spec


@pytest.fixture
def prepared(profile):
    profile['workers'] = profile['workers'][:1]
    spec = scratch.configure_scratch_pod(pod_spec(profile), profile, {'uid': 1000, 'gid': 1000})
    legacy_inputs(spec)
    env = {'execution_profile': profile,
           'isolated_storage': {'namespace_root': '/home/jinhaopeng/.cache/Paper/new-isolated-attempt'}}
    before = deepcopy(spec)
    isolated.configure_isolated_pod(spec, env)
    return env, spec, before


def test_only_held_namespace_is_writable_home_and_preparation_is_idempotent(prepared):
    env, spec, before = prepared
    receipt = isolated.validate_isolated_pod(spec, env)
    assert receipt['host_namespace'] == '/mnt/vepfs/base2/jinhaopeng/.cache/Paper/new-isolated-attempt'
    assert receipt['writable_home_mounts'] == [env['isolated_storage']['namespace_root']]
    assert len(receipt['read_only_mounts']) == 3
    assert len(spec['volumes']) == len(before['volumes']) + 1
    assert spec['volumes'][:-1] == before['volumes']
    same = deepcopy(spec)
    assert isolated.configure_isolated_pod(spec, env) == same
    plain = deepcopy(before)
    assert isolated.configure_isolated_pod(plain, {}) == before
    assert isolated.validate_isolated_pod(plain, {}) is None


@pytest.mark.parametrize('damage', ['old_home_rw', 'old_alias_rw', 'vepfs_rw', 'wrong_host_subtree',
    'missing_overlay', 'ro_overlay', 'duplicate_alias', 'extra_hostpath', 'nested_mount', 'subpath',
    'propagation', 'privileged', 'sys_admin', 'escalation', 'second_worker', 'unmounted_volume'])
def test_prepared_and_actual_pod_reject_isolation_bypass(prepared, damage):
    env, spec, _ = prepared
    c = spec['containers'][0]; mounts = c['volumeMounts']
    by_path = {m['mountPath']: m for m in mounts}
    ns = env['isolated_storage']['namespace_root']
    if damage == 'old_home_rw': by_path['/home/jinhaopeng']['readOnly'] = False
    elif damage == 'old_alias_rw': by_path['/mnt/vepfs/base2/jinhaopeng']['readOnly'] = False
    elif damage == 'vepfs_rw': by_path['/mnt/vepfs']['readOnly'] = False
    elif damage == 'wrong_host_subtree': spec['volumes'][-1]['hostPath']['path'] += '-different'
    elif damage == 'missing_overlay': mounts.remove(by_path[ns])
    elif damage == 'ro_overlay': by_path[ns]['readOnly'] = True
    elif damage == 'duplicate_alias': mounts.append({'name': 'user-home', 'mountPath': '/unsafe-alias'})
    elif damage == 'extra_hostpath':
        spec['volumes'].append({'name': 'other', 'hostPath': {'path': '/home'}})
        mounts.append({'name': 'other', 'mountPath': '/other', 'readOnly': True})
    elif damage == 'nested_mount':
        spec['volumes'].append({'name': 'other', 'projected': {'sources': []}})
        mounts.append({'name': 'other', 'mountPath': ns + '/hidden', 'readOnly': True})
    elif damage == 'subpath': by_path[ns]['subPath'] = 'other'
    elif damage == 'propagation': by_path[ns]['mountPropagation'] = 'Bidirectional'
    elif damage == 'privileged': c['securityContext']['privileged'] = True
    elif damage == 'sys_admin': c['securityContext']['capabilities'] = {'add': ['SYS_ADMIN']}
    elif damage == 'escalation': c['securityContext']['allowPrivilegeEscalation'] = True
    elif damage == 'second_worker': env['execution_profile']['workers'].append({'worker_id': 'worker-02'})
    else: spec['volumes'].append({'name': 'unused', 'emptyDir': {}})
    with pytest.raises(ValueError):
        isolated.validate_isolated_pod(spec, env)


def test_injected_read_only_service_account_projection_is_allowed(prepared):
    env, spec, _ = prepared
    spec['volumes'].append({'name': 'kube-api-access', 'projected': {'sources': [], 'defaultMode': 420}})
    spec['containers'][0]['volumeMounts'].append({'name': 'kube-api-access', 'readOnly': True,
        'mountPath': '/var/run/secrets/kubernetes.io/serviceaccount'})
    assert isolated.validate_isolated_pod(spec, env)['writable_home_mounts'] == [env['isolated_storage']['namespace_root']]


def kernel_mounts(env):
    ns = env['isolated_storage']['namespace_root']
    return [{'mount_path': path, 'options': mode + ',relatime', 'root': root,
             'device': '0:456', 'filesystem': 'fuse.aliyun-alinas-efc'} for path, root, mode in [
        ('/mnt/vepfs', '/', 'ro'), ('/mnt/vepfs/base2/jinhaopeng', '/base2/jinhaopeng', 'ro'),
        ('/home/jinhaopeng', '/base2/jinhaopeng', 'ro'),
        (ns, '/base2/jinhaopeng/.cache/Paper/new-isolated-attempt', 'rw')]]


def test_actual_kernel_flags_bind_held_subtree_and_read_only_aliases(prepared):
    env, _, _ = prepared
    mounts = kernel_mounts(env)
    result = isolated.inspect_isolated_mounts(env, mounts)
    assert result['status'] == 'ISOLATED_INPUT_MOUNTS_VERIFIED'
    assert result['parent_ledger_writable'] is result['legacy_cache_writable'] is False
    assert isolated.inspect_isolated_mounts({}, []) is None


@pytest.mark.parametrize('damage', ['home_rw', 'vepfs_rw', 'alias_rw', 'namespace_ro',
    'duplicate', 'wrong_subtree', 'wrong_device', 'hidden_rw_alias', 'namespace_nested'])
def test_kernel_mount_readback_rejects_manifest_only_isolation(prepared, damage):
    env, _, _ = prepared
    mounts = kernel_mounts(env)
    if damage == 'home_rw': mounts[2]['options'] = 'rw'
    elif damage == 'vepfs_rw': mounts[0]['options'] = 'rw'
    elif damage == 'alias_rw': mounts[1]['options'] = 'rw'
    elif damage == 'namespace_ro': mounts[3]['options'] = 'ro'
    elif damage == 'duplicate': mounts.append(deepcopy(mounts[2]))
    elif damage == 'wrong_subtree': mounts[3]['root'] += '-other'
    elif damage == 'wrong_device': mounts[3]['device'] = '0:999'
    else:
        mounts.append({**mounts[3], 'mount_path': ('/mnt/vepfs/hidden' if damage == 'hidden_rw_alias'
            else env['isolated_storage']['namespace_root'] + '/nested')})
    with pytest.raises(ValueError):
        isolated.inspect_isolated_mounts(env, mounts)


def test_renderer_and_final_job_validator_apply_isolation(prepared, tmp_path):
    env, _, before = prepared
    env.update(suite_id=env['execution_profile']['fleet_id'], machine_identity={'uid': 1000, 'gid': 1000},
        experiment_root=str(tmp_path.resolve()), runtime_environment={}, allocation_identity={})
    template = {'apiVersion': 'kubeflow.org/v1', 'kind': 'PyTorchJob', 'spec': {
        'pytorchReplicaSpecs': {'Master': {'replicas': 1, 'restartPolicy': 'Never',
                                         'template': {'spec': before}}}}}
    plan, binding = {'path': '/frozen/PLAN.json', 'sha256': 'a' * 64}, {'path': '/frozen/BINDING.json', 'sha256': 'b' * 64}
    with patch.object(suite_fleet_launch, 'runner_argv', return_value=['/usr/bin/python', '-B', 'fixture']):
        bundle = suite_fleet_launch.prepare_worker_jobs(template, env, plan, binding, '/held/state', tmp_path.resolve() / 'jobs')
    job = runner.read(runner.read(bundle['path'])['jobs'][0]['job']['path'])
    spec = job['spec']['pytorchReplicaSpecs']['Master']['template']['spec']
    assert isolated.validate_isolated_pod(spec, env)['writable_home_mounts'] == [env['isolated_storage']['namespace_root']]
    argv = spec['containers'][0]['command']
    assert suite_launch.validate_job(job, env, argv, worker_id='worker-01', binding_sha256='b' * 64) == job
    next(m for m in spec['containers'][0]['volumeMounts'] if m['mountPath'] == '/home/jinhaopeng')['readOnly'] = False
    with pytest.raises(ValueError, match='read-only'):
        suite_launch.validate_job(job, env, argv, worker_id='worker-01', binding_sha256='b' * 64)


def test_kubectl_cache_is_explicit_and_legacy_invocations_remain_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv('DP_SUITE_KUBECTL_CACHE', raising=False)
    assert suite_cluster._cache_args() == []
    target = tmp_path.resolve() / 'new-runtime-cache' / 'kubectl'
    monkeypatch.setenv('DP_SUITE_KUBECTL_CACHE', str(target))
    assert suite_cluster._cache_args() == ['--cache-dir', str(target)]
    assert target.is_dir()
    escaped = tmp_path.resolve() / 'symlink-cache'; escaped.symlink_to(target)
    monkeypatch.setenv('DP_SUITE_KUBECTL_CACHE', str(escaped))
    with pytest.raises(ValueError, match='symlink'):
        suite_cluster._cache_args()
