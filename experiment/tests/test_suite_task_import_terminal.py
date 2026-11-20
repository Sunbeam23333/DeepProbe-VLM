"""Terminal-import evidence checks; no cluster, OSS, or GPU is contacted."""
from copy import deepcopy
import json

import pytest

from deepprobe_vlm import suite_fleet as fleet, suite_runner as runner
from deepprobe_vlm.suite_task_import_terminal import verify_terminal


class TerminalFixture:
    def __init__(self, root):
        self.root = root.resolve()
        self.state = self.root / 'old-state'
        self.profile = {
            'schema_version': 2, 'fleet_id': 'old-fleet', 'priority': 'low',
            'gpu_product': 'NVIDIA-B300',
            'node_selector': {'sslaunch-pool': 'stream', 'aliyun.accelerator/nvidia_name': 'NVIDIA-L20D'},
            'workers': [{'worker_id': 'worker-0' + str(i), 'namespace': 'default',
                         'job_name': 'jinhaopeng-dpvlm-old-worker-' + str(i)} for i in (1, 2)],
        }
        self.binding = {'suite_id': 'old-fleet', 'execution_profile': self.profile, 'source_files': []}
        self.identity = {
            'schema_version': 2, 'suite_id': 'old-fleet', 'plan_sha256': 'a' * 64,
            'binding_sha256': runner.digest(self.binding),
            'source_identity_sha256': runner.source_identity(self.binding),
            'execution_profile_sha256': runner.digest(self.profile),
        }
        self.registrations, self.closures, self.jobs, self.pods = [], [], [], []
        for worker in self.profile['workers']:
            wid = worker['worker_id']
            job_uid, pod_uid, node = 'job-' + wid, 'pod-' + wid, 'node-' + wid
            self.registrations.append({'worker_id': wid, 'job_uid': job_uid, 'pod_uid': pod_uid,
                                       'node': node, 'utc': '2026-11-21T00:00:00Z'})
            evidence = self.write(self.root / ('process-' + wid + '.txt'), 'no active writers or uploads\n')
            self.closures.append({
                'schema_version': 1, 'status': 'SUITE_WORKER_PROCESSES_QUIESCENT',
                'worker_id': wid, 'job_uid': job_uid, 'pod_uid': pod_uid,
                'controller_exited': True, 'active_writers': [], 'active_uploads': [],
                'evidence': [evidence],
            })
            pod_spec = {'restartPolicy': 'Never', 'nodeSelector': self.profile['node_selector'],
                        'priorityClassName': 'low', 'containers': [{'name': 'worker', 'resources': {
                            'limits': {'nvidia.com/gpu': 8}, 'requests': {'nvidia.com/gpu': 8}}}]}
            self.jobs.append({
                'apiVersion': 'kubeflow.org/v1', 'kind': 'PyTorchJob',
                'metadata': {'namespace': 'default', 'name': worker['job_name'], 'uid': job_uid,
                             'labels': fleet.job_labels(self.profile, wid, self.identity['binding_sha256']),
                             'annotations': {'deepprobe/binding-sha256': self.identity['binding_sha256'],
                                             'deepprobe/profile-sha256': runner.digest(self.profile)}},
                'spec': {'pytorchReplicaSpecs': {'Master': {'replicas': 1, 'restartPolicy': 'Never',
                                                          'template': {'spec': deepcopy(pod_spec)}}}},
                'status': {'conditions': [{'type': 'Succeeded', 'status': 'True'}],
                           'replicaStatuses': {'Master': {'active': 0}}},
            })
            self.pods.append({
                'apiVersion': 'v1', 'kind': 'Pod',
                'metadata': {'namespace': 'default', 'name': worker['job_name'] + '-master-0',
                             'uid': pod_uid, 'ownerReferences': [{
                                 'apiVersion': 'kubeflow.org/v1', 'kind': 'PyTorchJob',
                                 'name': worker['job_name'], 'uid': job_uid, 'controller': True}]},
                'spec': {**pod_spec, 'nodeName': node}, 'status': {'phase': 'Succeeded'},
            })
        self.census = {'metadata': {'continue': '', 'remainingItemCount': 0}, 'items': self.jobs + self.pods}
        self.proof = {
            'schema_version': 1, 'status': 'SUITE_FLEET_TERMINAL_QUIESCENT',
            'source_identity': deepcopy(self.identity), 'complete_project_workload_snapshot': True,
            'automatic_requeue_disabled': True, 'active_writers': [], 'active_uploads': [],
        }
        self.request = {'source_state_root': str(self.state)}

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value, sort_keys=True))
        return runner.artifact(path)

    def publish(self):
        self.proof['worker_registrations'] = [self.write(
            self.state / 'workers' / declared['worker_id'] / 'JOB_OBSERVED.json', row)
            for declared, row in zip(self.profile['workers'], self.registrations)]
        self.proof['process_closures'] = [self.write(self.root / ('closure-' + str(i) + '.json'), row)
                                          for i, row in enumerate(self.closures)]
        self.proof['workloads'] = self.write(self.root / 'WORKLOADS.json', self.census)
        self.repin_proof()
        return self

    def repin_proof(self):
        self.request['terminal_proof'] = self.write(self.root / 'TERMINAL.json', self.proof)

    def verify(self):
        return verify_terminal(self.request, self.binding, self.identity)


@pytest.fixture
def case(tmp_path):
    return TerminalFixture(tmp_path)


def test_accepts_all_original_terminal_workers_and_raw_process_evidence(case):
    case.jobs[1]['status']['conditions'][0]['type'] = 'Failed'
    case.pods[1]['status']['phase'] = 'Failed'
    result = case.publish().verify()
    assert result['worker_ids'] == ['worker-01', 'worker-02']
    assert result['absent_original_pod_workers'] == []
    assert result['terminal_proof'] == case.request['terminal_proof']


@pytest.mark.parametrize('removed', [1, 2])
def test_accepts_original_pod_absence_under_complete_census(case, removed):
    case.census['items'] = case.jobs + case.pods[removed:]
    assert len(case.publish().verify()['absent_original_pod_workers']) == removed


def test_other_fleet_activity_is_not_old_fleet_quiescence(case):
    unrelated = deepcopy(case.jobs[0])
    unrelated['metadata'] = {'namespace': 'default', 'name': 'jinhaopeng-dpvlm-other',
                             'uid': 'other-job', 'labels': {'deepprobe/fleet': 'other-fleet'}}
    unrelated['status'] = {'conditions': [{'type': 'Running', 'status': 'True'}]}
    case.census['items'].append(unrelated)
    case.publish().verify()


@pytest.mark.parametrize('field,value', [
    ('automatic_requeue_disabled', False), ('automatic_requeue_disabled', 1),
    ('complete_project_workload_snapshot', False), ('active_writers', ['pid']),
    ('active_uploads', ['upload']), ('schema_version', True), ('status', 'TERMINAL'),
])
def test_rejects_unproven_global_quiescence(case, field, value):
    case.proof[field] = value
    with pytest.raises(ValueError):
        case.publish().verify()


@pytest.mark.parametrize('field,value', [
    ('controller_exited', False), ('controller_exited', 1), ('active_writers', ['pid']),
    ('active_uploads', ['upload']), ('evidence', []), ('job_uid', 'recreated-job'),
    ('pod_uid', 'replacement-pod'), ('status', 'SUITE_WORKER_STARTED'),
])
def test_rejects_wrong_or_live_worker_closure(case, field, value):
    case.closures[0][field] = value
    with pytest.raises(ValueError):
        case.publish().verify()


@pytest.mark.parametrize('change', [
    'missing-job', 'running-job', 'contradictory-job', 'deleting-job', 'active-replica',
    'replica-restart', 'pod-template-restart', 'replacement-job', 'renamed-job',
    'wrong-namespace', 'wrong-binding-label', 'wrong-profile-annotation',
    'wrong-binding-annotation', 'wrong-worker-label', 'wrong-fleet-label',
    'running-pod', 'deleting-pod', 'pod-restart', 'wrong-node', 'wrong-owner-uid',
    'wrong-owner-name', 'original-uid-outside-prefix', 'original-uid-wrong-kind',
    'duplicate-uid', 'duplicate-identity', 'paginated', 'remaining-items',
])
def test_rejects_incomplete_changed_or_active_cluster_identity(case, change):
    job, pod = case.jobs[0], case.pods[0]
    if change == 'missing-job':
        case.census['items'].remove(job)
    elif change == 'running-job':
        job['status']['conditions'] = [{'type': 'Running', 'status': 'True'}]
    elif change == 'contradictory-job':
        job['status']['conditions'].append({'type': 'Failed', 'status': 'True'})
    elif change == 'deleting-job':
        job['metadata']['deletionTimestamp'] = '2026-11-21T00:01:00Z'
    elif change == 'active-replica':
        job['status']['replicaStatuses']['Master']['active'] = 1
    elif change == 'replica-restart':
        job['spec']['pytorchReplicaSpecs']['Master']['restartPolicy'] = 'OnFailure'
    elif change == 'pod-template-restart':
        job['spec']['pytorchReplicaSpecs']['Master']['template']['spec']['restartPolicy'] = 'OnFailure'
    elif change == 'replacement-job':
        job['metadata']['uid'] = 'new-job'
    elif change == 'renamed-job':
        job['metadata']['name'] = 'unrelated-renamed'
        job['metadata'].pop('labels')
        job['metadata'].pop('annotations')
    elif change == 'wrong-namespace':
        job['metadata']['namespace'] = 'other'
    elif change.endswith('-label'):
        field = {'wrong-binding-label': 'deepprobe/binding', 'wrong-worker-label': 'deepprobe/worker',
                 'wrong-fleet-label': 'deepprobe/fleet'}[change]
        job['metadata']['labels'][field] = 'wrong'
    elif change.endswith('-annotation'):
        field = 'deepprobe/profile-sha256' if change == 'wrong-profile-annotation' else 'deepprobe/binding-sha256'
        job['metadata']['annotations'][field] = '0' * 64
    elif change == 'running-pod':
        pod['status']['phase'] = 'Running'
    elif change == 'deleting-pod':
        pod['metadata']['deletionTimestamp'] = '2026-11-21T00:01:00Z'
    elif change == 'pod-restart':
        pod['spec']['restartPolicy'] = 'OnFailure'
    elif change == 'wrong-node':
        pod['spec']['nodeName'] = 'another-node'
    elif change == 'wrong-owner-uid':
        pod['metadata']['ownerReferences'][0]['uid'] = 'new-job'
    elif change == 'wrong-owner-name':
        pod['metadata']['ownerReferences'][0]['name'] = 'unrelated'
    elif change == 'original-uid-outside-prefix':
        pod['metadata'] = {'namespace': 'default', 'name': 'arbitrary', 'uid': pod['metadata']['uid']}
    elif change == 'original-uid-wrong-kind':
        pod['metadata'] = {'namespace': 'default', 'name': 'arbitrary', 'uid': pod['metadata']['uid']}
        pod['kind'] = 'Deployment'
    elif change == 'duplicate-uid':
        duplicate = deepcopy(pod)
        duplicate['metadata']['name'] += '-copy'
        case.census['items'].append(duplicate)
    elif change == 'duplicate-identity':
        duplicate = deepcopy(pod)
        duplicate['metadata']['uid'] += '-copy'
        case.census['items'].append(duplicate)
    elif change == 'paginated':
        case.census['metadata']['continue'] = 'next-page'
    elif change == 'remaining-items':
        case.census['metadata']['remainingItemCount'] = 1
    with pytest.raises(ValueError):
        case.publish().verify()


@pytest.mark.parametrize('terminal', [True, False])
def test_rejects_any_undeclared_old_fleet_job_even_terminal(case, terminal):
    extra = deepcopy(case.jobs[0])
    extra['metadata']['name'], extra['metadata']['uid'] = 'arbitrary-extra', 'extra-job'
    if not terminal:
        extra['status']['conditions'] = [{'type': 'Running', 'status': 'True'}]
    case.census['items'].append(extra)
    with pytest.raises(ValueError, match='undeclared Job'):
        case.publish().verify()


@pytest.mark.parametrize('keep_original', [True, False])
def test_rejects_arbitrarily_named_unlabelled_replacement_pod(case, keep_original):
    replacement = deepcopy(case.pods[0])
    replacement['metadata']['name'], replacement['metadata']['uid'] = 'arbitrary-copy', 'replacement-pod'
    if not keep_original:
        case.census['items'].remove(case.pods[0])
    case.census['items'].append(replacement)
    with pytest.raises(ValueError, match='replacement Pod'):
        case.publish().verify()


@pytest.mark.parametrize('change', [
    'missing-registration', 'duplicate-registration', 'registration-copy-path', 'extra-registration',
    'missing-closure', 'duplicate-closure', 'incomplete-pin', 'wrong-source-identity',
    'duplicate-registration-job-uid', 'duplicate-registration-pod-uid', 'evidence-tampered',
    'proof-tampered', 'duplicate-evidence',
])
def test_rejects_missing_changed_or_duplicate_source_evidence(case, change):
    if change == 'duplicate-registration-job-uid':
        case.registrations[1]['job_uid'] = case.registrations[0]['job_uid']
    elif change == 'duplicate-registration-pod-uid':
        case.registrations[1]['pod_uid'] = case.registrations[0]['pod_uid']
    elif change == 'duplicate-evidence':
        case.closures[0]['evidence'] *= 2
    case.publish()
    if change == 'missing-registration':
        case.proof['worker_registrations'].pop()
    elif change == 'duplicate-registration':
        case.proof['worker_registrations'][1] = case.proof['worker_registrations'][0]
    elif change == 'registration-copy-path':
        case.proof['worker_registrations'][0] = case.write(case.root / 'REG_COPY.json', case.registrations[0])
    elif change == 'extra-registration':
        case.write(case.state / 'workers' / 'worker-03' / 'JOB_OBSERVED.json', case.registrations[0])
    elif change == 'missing-closure':
        case.proof['process_closures'].pop()
    elif change == 'duplicate-closure':
        case.proof['process_closures'][1] = case.proof['process_closures'][0]
    elif change == 'incomplete-pin':
        case.proof['workloads'].pop('bytes')
    elif change == 'wrong-source-identity':
        case.proof['source_identity']['binding_sha256'] = '0' * 64
    elif change == 'evidence-tampered':
        (case.root / 'process-worker-01.txt').write_text('xx active writers or uploads\n')
    case.repin_proof()
    if change == 'proof-tampered':
        path = case.root / 'TERMINAL.json'
        path.write_text(path.read_text().replace('QUIESCENT', 'QVIESCENT'))
    with pytest.raises(ValueError):
        case.verify()
