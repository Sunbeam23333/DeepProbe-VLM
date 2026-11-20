"""Synthetic Pod/mount fixtures; no node, GPU, quota or real mount is certified."""
from copy import deepcopy
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from deepprobe_vlm import suite_actions, suite_cluster, suite_fleet, suite_fleet_launch
from deepprobe_vlm import suite_runner as runner, suite_scratch as scratch


@pytest.fixture
def profile(tmp_path):
    return {'schema_version': 3, 'fleet_id': 'synthetic-scratch', 'priority': 'low',
        'gpu_product': 'NVIDIA-B300',
        'node_selector': {'sslaunch-pool': 'stream', 'aliyun.accelerator/nvidia_name': 'NVIDIA-L20D'},
        'workers': [{'worker_id': f'worker-0{i}', 'job_name': f'jinhaopeng-dpvlm-scratch-test-{i}',
                     'namespace': 'default'} for i in (1, 2)],
        'scratch': {**scratch.DEFAULT_SCRATCH, 'mount_path': str(tmp_path.resolve() / 'scratch')}}


def pod_spec(profile):
    return {'restartPolicy': 'Never', 'nodeSelector': deepcopy(profile['node_selector']),
        'priorityClassName': profile['priority'], 'containers': [{'name': 'worker', 'image': 'synthetic/fixture',
        'resources': {'requests': {'nvidia.com/gpu': 8}, 'limits': {'nvidia.com/gpu': 8}}}]}


def job_document(profile, spec, binding='b' * 64):
    worker = profile['workers'][0]
    return {'apiVersion': 'kubeflow.org/v1', 'kind': 'PyTorchJob',
        'metadata': {'name': worker['job_name'], 'namespace': 'default', 'uid': 'job-01',
            'labels': suite_fleet.job_labels(profile, worker['worker_id'], binding),
            'annotations': {'deepprobe/binding-sha256': binding, 'deepprobe/profile-sha256': runner.digest(profile)}},
        'spec': {'pytorchReplicaSpecs': {'Master': {'replicas': 1, 'restartPolicy': 'Never',
                                                'template': {'spec': deepcopy(spec)}}}}}


def test_schema3_policy_is_explicit_and_schema2_remains_unchanged(profile):
    assert suite_fleet.validate_execution_profile(profile) == profile
    legacy = {k: deepcopy(v) for k, v in profile.items() if k != 'scratch'}; legacy['schema_version'] = 2
    assert suite_fleet.validate_execution_profile(legacy) == legacy
    for broken in ({**profile, 'unexpected': True}, {**legacy, 'scratch': profile['scratch']},
                   {**profile, 'scratch': {**profile['scratch'], 'limit_bytes': 150 * scratch.GIB}},
                   {**profile, 'scratch': {**profile['scratch'], 'safety_bytes': 0}}):
        with pytest.raises(ValueError):
            suite_fleet.validate_execution_profile(broken)
    assert scratch.worker_run_root(profile, 'worker-01') != scratch.worker_run_root(profile, 'worker-02')
    with pytest.raises(ValueError, match='not declared'):
        scratch.worker_run_root(profile, 'worker-03')


@pytest.mark.parametrize('change', ['memory', 'host_path', 'no_limit', 'no_request', 'sub_path',
                                   'read_only', 'nested_mount', 'wrong_fsgroup', 'container_root'])
def test_actual_pod_contract_rejects_unsafe_storage(profile, change):
    spec = scratch.configure_scratch_pod(pod_spec(profile), profile, {'uid': 1000, 'gid': 1000})
    container = spec['containers'][0]
    if change == 'memory': spec['volumes'][0]['emptyDir']['medium'] = 'Memory'
    elif change == 'host_path': spec['volumes'][0] = {'name': scratch.VOLUME_NAME, 'hostPath': {'path': '/home'}}
    elif change == 'no_limit': del spec['volumes'][0]['emptyDir']['sizeLimit']
    elif change == 'no_request': del container['resources']['requests']['ephemeral-storage']
    elif change == 'sub_path': container['volumeMounts'][0]['subPath'] = 'runs'
    elif change == 'read_only': container['volumeMounts'][0]['readOnly'] = True
    elif change == 'nested_mount': container['volumeMounts'].append({'name': 'other', 'mountPath': profile['scratch']['mount_path'] + '/runs'})
    elif change == 'wrong_fsgroup': spec['securityContext']['fsGroup'] = 42
    else: container['securityContext'] = {'runAsUser': 0}
    with pytest.raises(ValueError):
        suite_fleet.allocation_contract(job_document(profile, spec), profile)


def test_equivalent_server_storage_quantities_and_empty_defaults_are_accepted(profile):
    spec = scratch.configure_scratch_pod(pod_spec(profile), profile, {'uid': 1000, 'gid': 1000})
    spec['volumes'][0]['emptyDir'].update(sizeLimit=str(scratch.LIMIT_BYTES), medium='')
    spec['containers'][0]['volumeMounts'][0].update(readOnly=False, subPath='', subPathExpr='', mountPropagation='None')
    spec['containers'][0]['resources']['limits']['ephemeral-storage'] = '102400Mi'
    suite_fleet.allocation_contract(job_document(profile, spec), profile)
    scratch.configure_scratch_pod(spec, profile, {'uid': 1000, 'gid': 1000})


def test_prepared_worker_jobs_really_include_disk_scratch_and_ephemeral_resources(profile, tmp_path):
    shared = tmp_path.resolve() / 'shared'; shared.mkdir()
    env = {'suite_id': profile['fleet_id'], 'execution_profile': profile, 'machine_identity': {'uid': 1000, 'gid': 1000},
        'python': sys.executable, 'experiment_root': str(shared), 'runtime_environment': {},
        'allocation_identity': {'job_name': profile['workers'][0]['job_name'], 'namespace': 'default'}}
    path = shared / 'input.json'; path.write_text('{}'); pin = runner.artifact(path)
    template = job_document(profile, pod_spec(profile)); template['metadata'] = {}
    spec = template['spec']['pytorchReplicaSpecs']['Master']['template']['spec']
    spec['volumes'] = [{'name': 'kube-api-access-test', 'projected': {'sources': []}}]
    spec['containers'][0]['volumeMounts'] = [{'name': 'kube-api-access-test',
        'mountPath': '/var/run/secrets/kubernetes.io/serviceaccount', 'readOnly': True}]
    bundle = suite_fleet_launch.prepare_worker_jobs(template, env, pin, pin, shared / 'state', shared / 'jobs')
    rows = runner.read(bundle['path'])['jobs']
    assert len(rows) == 2
    for row in rows:
        job = runner.read(row['job']['path'])
        spec = job['spec']['pytorchReplicaSpecs']['Master']['template']['spec']
        proof = scratch.validate_scratch_pod(spec, profile)
        assert proof['medium'] == 'disk' and proof['limit_bytes'] == 100 * scratch.GIB
        assert spec['securityContext']['fsGroup'] == 1000
    assert not Path(profile['scratch']['mount_path']).exists()


@pytest.fixture
def observed(profile, tmp_path, monkeypatch):
    mount = Path(profile['scratch']['mount_path']); mount.mkdir()
    shared = tmp_path.resolve() / 'shared'; shared.mkdir()
    machine = {'uid': 1000, 'gid': 1000}
    for name in ('getuid', 'geteuid', 'getgid', 'getegid'):
        monkeypatch.setattr(scratch.os, name, lambda: 1000)
    info = mount.lstat()
    metadata = SimpleNamespace(st_mode=stat.S_IFDIR | 0o770, st_uid=0, st_gid=1000,
                               st_dev=info.st_dev, st_ino=info.st_ino)
    monkeypatch.setattr(scratch, '_directory_info', lambda _: metadata)
    monkeypatch.setattr(scratch.os, 'statvfs', lambda _: SimpleNamespace(f_bavail=3 * 1024 ** 4, f_frsize=1))
    spec = scratch.configure_scratch_pod(pod_spec(profile), profile, machine)
    job = job_document(profile, spec)
    pod = {'kind': 'Pod', 'metadata': {'name': 'synthetic-pod', 'namespace': 'default', 'uid': 'pod-01',
            'ownerReferences': [{'uid': 'job-01', 'kind': 'PyTorchJob'}]},
           'spec': {**deepcopy(spec), 'nodeName': 'node-01'}, 'status': {'phase': 'Running'}}
    document = {'items': [job, pod]}
    monkeypatch.setattr(suite_cluster, 'list_workloads', lambda *_a, **_k: deepcopy(document))
    mounts = [{'mount_id': '123', 'parent_id': '1', 'device': f'{os.major(info.st_dev)}:{os.minor(info.st_dev)}',
        'root': '/var/lib/kubelet/pods/pod-01/volumes/kubernetes.io~empty-dir/' + scratch.VOLUME_NAME,
        'mount_path': str(mount), 'options': 'rw,relatime', 'filesystem': 'ext4',
        'source': '/dev/synthetic', 'super_options': 'rw'}]
    monkeypatch.setattr(scratch, '_mounts', lambda: deepcopy(mounts))
    env = {'suite_id': profile['fleet_id'], 'execution_profile': profile, 'machine_identity': machine,
        'run_root': str(mount / 'runs'), 'storage': {'roots': [str(shared)]}, 'tools': {},
        'allocation_identity': {'project_name_prefixes': ['jinhaopeng-dpvlm-']}}
    live = {'worker_id': 'worker-01', 'job_uid': 'job-01', 'pod_uid': 'pod-01', 'node': 'node-01', 'binding_sha256': 'b' * 64}
    return env, live, document, mounts


def test_inspection_binds_real_mount_allocation_and_capacity_without_creating_directories(observed):
    env, live, _, _ = observed
    proof = scratch.inspect_scratch(env, live)
    assert proof['status'] == 'SUITE_SCRATCH_STARTUP_VERIFIED'
    assert proof['pod_uid'] == live['pod_uid'] and proof['binding_sha256'] == live['binding_sha256']
    assert proof['available_bytes'] == 3 * 1024 ** 4 and not proof['local_directories_created']
    assert not Path(env['run_root']).exists()
    assert not proof['interrupted_local_checkpoint_survives_pod_loss']


def test_actual_service_account_mount_is_independent_of_host_symlinks(observed, tmp_path):
    env, live, document, _ = observed
    # The service-account projection is injected into the actual Pod. Its
    # namespace may differ from the submitter or even use /var/run -> /run.
    spec = document['items'][1]['spec']
    spec['volumes'].append({'name': 'kube-api-access-test', 'projected': {'sources': []}})
    spec['containers'][0]['volumeMounts'].append({'name': 'kube-api-access-test',
        'mountPath': '/var/run/secrets/kubernetes.io/serviceaccount', 'readOnly': True})
    assert scratch.inspect_scratch(env, live)['status'] == 'SUITE_SCRATCH_STARTUP_VERIFIED'
    alias = tmp_path / 'host-only-alias'; alias.symlink_to(tmp_path / 'missing-on-this-host')
    spec['containers'][0]['volumeMounts'][-1]['mountPath'] = str(alias / 'serviceaccount')
    assert scratch.inspect_scratch(env, live)['status'] == 'SUITE_SCRATCH_STARTUP_VERIFIED'


@pytest.mark.parametrize('change', ['self_reported_binding', 'terminal_job', 'wrong_pod', 'wrong_owner', 'wrong_node',
                                   'memory', 'network', 'unknown_fs', 'other_pod_mount', 'nested_real_mount', 'home_overlap'])
def test_inspection_rejects_identity_or_mount_substitution(observed, change):
    env, live, document, mounts = observed
    if change == 'self_reported_binding':
        document['items'][0]['metadata']['annotations']['deepprobe/binding-sha256'] = 'c' * 64
        document['items'][0]['metadata']['labels'].update(suite_fleet.job_labels(env['execution_profile'], 'worker-01', 'c' * 64))
    elif change == 'terminal_job': document['items'][0]['status'] = {'conditions': [{'type': 'Succeeded', 'status': 'True'}]}
    elif change == 'wrong_pod': document['items'][1]['metadata']['uid'] = 'replacement-pod'
    elif change == 'wrong_owner': document['items'][1]['metadata']['ownerReferences'][0]['uid'] = 'another-job'
    elif change == 'wrong_node': document['items'][1]['spec']['nodeName'] = 'node-02'
    elif change == 'memory': mounts[0]['filesystem'] = 'tmpfs'
    elif change == 'network': mounts[0]['filesystem'] = 'nfs4'
    elif change == 'unknown_fs': mounts[0]['filesystem'] = 'fuse.unknown'
    elif change == 'other_pod_mount': mounts[0]['root'] = mounts[0]['root'].replace('pod-01', 'pod-02')
    elif change == 'nested_real_mount': mounts.append({**mounts[0], 'mount_id': '124', 'mount_path': env['run_root']})
    else: env['storage']['roots'] = [env['execution_profile']['scratch']['mount_path']]
    with pytest.raises(ValueError):
        scratch.inspect_scratch(env, live)
    assert not Path(env['run_root']).exists()


def test_inspection_rejects_insufficient_space_and_changed_mount(observed, monkeypatch):
    env, live, _, mounts = observed
    monkeypatch.setattr(scratch.os, 'statvfs', lambda _: SimpleNamespace(f_bavail=scratch.LIMIT_BYTES, f_frsize=1))
    with pytest.raises(ValueError, match='free-space guard'):
        scratch.inspect_scratch(env, live)
    monkeypatch.setattr(scratch.os, 'statvfs', lambda _: SimpleNamespace(f_bavail=3 * 1024 ** 4, f_frsize=1))
    with mock.patch.object(scratch, '_mounts', side_effect=[deepcopy(mounts), []]):
        with pytest.raises(ValueError, match='changed during'):
            scratch.inspect_scratch(env, live)


@pytest.fixture
def startup_worker(observed, tmp_path, monkeypatch):
    env, live, _, _ = observed
    worker = object.__new__(suite_actions.Actions)
    worker.env = {**env, 'source_files': [], 'python': sys.executable}
    worker.plan = {}; worker.root = tmp_path.resolve(); worker.source_sha = 'a' * 64
    worker.live = mock.Mock(return_value=live); worker.budget = mock.Mock()
    proof = scratch.inspect_scratch(env, live)
    inspect = mock.Mock(return_value=proof); monkeypatch.setattr(scratch, 'inspect_scratch', inspect)
    monkeypatch.setattr('deepprobe_vlm.suite_bindings.validate_environment', lambda *_: None)
    nvidia = mock.Mock(return_value=subprocess.CompletedProcess([], 0, '\n'.join(f'gpu-{i}' for i in range(8)), ''))
    monkeypatch.setattr(suite_actions.subprocess, 'run', nvidia)
    def command(argv, *_):
        if '--verify-control-group' in argv:
            value = {'status': 'SUITE_EXPORT_CONTROL_GROUP_READY', 'world_size': 8,
                     'control_backend': 'gloo', 'control_timeout_seconds': 21600}
        else:
            value = {'status': 'gpu_gate_passed', 'world_size': 8, 'distributed_all_reduce': 'pass',
                     'gpus': [{'expected_gpu': 'B300', 'expected_gpu_matches': True}] * 8}
        Path(argv[argv.index('--output') + 1]).write_text(json.dumps(value))
    worker.command = mock.Mock(side_effect=command)
    return worker, proof, inspect, nvidia


def test_actions_startup_checks_scratch_before_gpu_work_and_retains_proof(startup_worker, observed, tmp_path):
    worker, proof, inspect, nvidia = startup_worker
    _, live, _, _ = observed
    output = tmp_path / 'STARTUP.json'; worker.startup(output)
    report = runner.read(output)
    assert report['scratch'] == proof
    runner.verify_pin(report['checkpoint_export_control_group'])
    assert [call.args[2] for call in worker.command.call_args_list] == ['hardware', 'checkpoint-export-control']
    command = worker.command.call_args_list[1].args[0]
    assert command[:4] == [sys.executable, '-B', '-m', 'torch.distributed.run']
    assert '--nproc_per_node=8' in command and '--nnodes=1' in command and '--max_restarts=0' in command
    assert command[-5:-2] == ['-m', 'deepprobe_vlm.suite_checkpoint_export', '--verify-control-group']
    inspect.assert_called_once_with(worker.env, live)
    inspect.side_effect = ValueError('synthetic mount rejection'); nvidia.reset_mock()
    with pytest.raises(ValueError, match='mount rejection'):
        worker.startup(tmp_path / 'DENIED.json')
    nvidia.assert_not_called()
    assert not (tmp_path / 'DENIED.json').exists()


@pytest.mark.parametrize('change', [{'status': 'not_ready'}, {'world_size': 1},
                                  {'control_backend': 'nccl'}, {'control_timeout_seconds': 1800}])
def test_private_startup_does_not_accept_invalid_export_control_group(startup_worker, tmp_path, change):
    worker, _, _, _ = startup_worker
    original = worker.command.side_effect
    def command(argv, *args):
        original(argv, *args)
        if '--verify-control-group' in argv:
            path = Path(argv[argv.index('--output') + 1])
            path.write_text(json.dumps({**runner.read(path), **change}))
    worker.command.side_effect = command
    output = tmp_path / 'DENIED.json'
    with pytest.raises(ValueError, match='export control group gate failed'):
        worker.startup(output)
    assert not output.exists()


def test_schema2_startup_does_not_launch_private_export_control_group(startup_worker, tmp_path):
    worker, _, inspect, _ = startup_worker
    worker.env['execution_profile'] = {k: v for k, v in worker.env['execution_profile'].items() if k != 'scratch'}
    worker.env['execution_profile']['schema_version'] = 2
    output = tmp_path / 'STARTUP.json'; worker.startup(output)
    inspect.assert_not_called()
    assert worker.command.call_count == 1
    assert 'checkpoint_export_control_group' not in runner.read(output)
