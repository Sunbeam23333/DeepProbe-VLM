"""Pinned execution profiles and durable shared-state scheduling for suite workers.

Scientific tasks remain in the original plan. A profile only selects one to
four single-node allocations. Claims never expire: a lost worker preserves its
claim until an operator supplies separately verified recovery evidence.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import re
import socket
import uuid

from . import suite_runner as runner

require, safe, read = runner.require, runner.safe, runner.read


def validate_execution_profile(profile, suite_id=None):
    require(isinstance(profile, dict) and set(profile) == {
        'schema_version', 'fleet_id', 'priority', 'gpu_product', 'node_selector', 'workers'}, 'Invalid execution profile fields')
    require(type(profile['schema_version']) is int and profile['schema_version'] == 2, 'Unsupported execution profile')
    require(isinstance(profile['fleet_id'], str) and runner.ID.fullmatch(profile['fleet_id']), 'Invalid fleet identity')
    require(suite_id is None or profile['fleet_id'] == suite_id, 'Execution profile belongs to another suite')
    require(profile['priority'] in {'normal', 'low'}, 'Only reviewed normal or low priority is supported')
    require(profile['gpu_product'] == 'NVIDIA-B300', 'This execution profile requires NVIDIA B300')
    require(profile['node_selector'] == {'sslaunch-pool': 'stream', 'aliyun.accelerator/nvidia_name': 'NVIDIA-L20D'},
            'Execution profile must bind the verified stream B300 scheduler selector')
    workers = profile['workers']
    require(isinstance(workers, list) and 1 <= len(workers) <= 4, 'A fleet requires one to four workers')
    for worker in workers:
        require(isinstance(worker, dict) and set(worker) == {'worker_id', 'job_name', 'namespace'}, 'Invalid worker identity')
        require(isinstance(worker['worker_id'], str) and re.fullmatch(r'worker-0[1-4]', worker['worker_id']), 'Invalid worker id')
        require(isinstance(worker['job_name'], str) and re.fullmatch(r'jinhaopeng-dpvlm-[a-z0-9][-a-z0-9.]{0,220}', worker['job_name']), 'Invalid exact worker Job name')
        require(worker['namespace'] == 'default', 'Unexpected worker namespace')
    for key in ('worker_id', 'job_name'):
        require(len({w[key] for w in workers}) == len(workers), 'Duplicate fleet worker identity')
    return profile


def profile_allocation(profile):
    validate_execution_profile(profile)
    return {**runner.ALLOCATION, 'priority': profile['priority']}


def worker_identity(profile, worker_id):
    validate_execution_profile(profile)
    rows = [row for row in profile['workers'] if row['worker_id'] == worker_id]
    require(len(rows) == 1, 'Worker is not declared in the frozen profile')
    return rows[0]


def allocation_contract(job, profile):
    """Validate actual scheduler resources; exact fleet labels bind live jobs."""
    validate_execution_profile(profile)
    require(job.get('apiVersion') == 'kubeflow.org/v1' and job.get('kind') == 'PyTorchJob', 'Expected PyTorchJob')
    replicas = job['spec']['pytorchReplicaSpecs']
    require(set(replicas) == {'Master'}, 'Each worker must be exactly one Master node')
    master = replicas['Master']
    pod = master['template']['spec']
    require(master.get('replicas') == 1 and master.get('restartPolicy') == 'Never'
            and pod.get('restartPolicy', 'Never') == 'Never', 'One node with restart Never is required')
    selector = pod.get('nodeSelector', {})
    require(all(selector.get(key) == value for key, value in profile['node_selector'].items())
            and set(selector) == set(profile['node_selector'])
            and pod.get('priorityClassName') == profile['priority'] and not pod.get('nodeName'),
            'Wrong stream/B300/priority scheduler contract')
    require(not pod.get('initContainers') and not pod.get('ephemeralContainers')
            and len(pod.get('containers', [])) == 1, 'Unexpected auxiliary worker containers')
    resources = pod['containers'][0].get('resources', {})
    require(str(resources.get('limits', {}).get('nvidia.com/gpu')) == '8'
            and str(resources.get('requests', {}).get('nvidia.com/gpu')) == '8', 'Each worker requires eight GPUs')
    return profile_allocation(profile)


def job_labels(profile, worker_id, binding_sha256):
    worker_identity(profile, worker_id)
    require(isinstance(binding_sha256, str) and runner.SHA.fullmatch(binding_sha256), 'Invalid binding identity')
    # A Kubernetes label is limited to 63 characters; full hashes live in annotations.
    return {'deepprobe/fleet': runner.digest(profile)[:32], 'deepprobe/worker': worker_id,
            'deepprobe/binding': binding_sha256[:32]}


def verify_job_identity(job, profile, worker_id, binding_sha256):
    expected = worker_identity(profile, worker_id)
    metadata = job.get('metadata', {})
    require(metadata.get('name') == expected['job_name'] and metadata.get('namespace') == expected['namespace'], 'Wrong exact fleet Job')
    require(all(metadata.get('labels', {}).get(k) == v for k, v in job_labels(profile, worker_id, binding_sha256).items()),
            'Live Job lacks the frozen fleet/binding identity labels')
    require(metadata.get('annotations', {}).get('deepprobe/binding-sha256') == binding_sha256
            and metadata.get('annotations', {}).get('deepprobe/profile-sha256') == runner.digest(profile),
            'Live Job full identity hashes differ')
    allocation_contract(job, profile)
    return expected


def audit_fleet_jobs(profile, document, state_root, current_worker_id=None, *, binding_sha256=None,
                     project_name_prefixes=('jinhaopeng-dpvlm-', 'deepprobe-'), require_current=True):
    """A full listing may contain only exact, SHA-bound jobs from this fleet.

    All active Pods need a verified fleet Job UID owner. An unregistered/unknown
    UID is never exempted merely because its name resembles another worker.
    """
    from .suite_actions import terminal_job
    validate_execution_profile(profile)
    state = safe(state_root)
    identity = read(state / 'SUITE_IDENTITY.json') if (state / 'SUITE_IDENTITY.json').exists() else None
    binding_sha256 = binding_sha256 or (identity or {}).get('binding_sha256')
    require(isinstance(binding_sha256, str) and runner.SHA.fullmatch(binding_sha256), 'Fleet binding identity is required')
    require(isinstance(document, dict) and isinstance(document.get('items'), list)
            and not document.get('metadata', {}).get('continue'), 'Complete Job/Pod listing required')
    require(any('jinhaopeng-dpvlm-'.startswith(p) for p in project_name_prefixes), 'Complete project scope required')
    exact = {(w['namespace'], w['job_name']): w for w in profile['workers']}
    verified, nodes, current = {}, {}, None
    project_items = [item for item in document['items'] if any(
        item.get('metadata', {}).get('name', '').startswith(p) for p in project_name_prefixes)]
    for item in project_items:
        kind, meta = item.get('kind'), item.get('metadata', {})
        require(kind in {'PyTorchJob', 'Pod'}, 'Unexpected project workload kind')
        if kind != 'PyTorchJob':
            continue
        key = (meta.get('namespace'), meta.get('name'))
        if key not in exact:
            require(terminal_job(item), 'Another project Job is active')
            continue
        worker = exact[key]
        verify_job_identity(item, profile, worker['worker_id'], binding_sha256)
        uid = meta.get('uid')
        require(isinstance(uid, str) and uid and uid not in verified, 'Job UID missing or duplicated')
        path = state / 'workers' / worker['worker_id'] / 'JOB_OBSERVED.json'
        if path.exists():
            require(read(path)['job_uid'] == uid, 'Fleet Job UID changed; automatic adoption is forbidden')
        verified[uid] = {'worker_id': worker['worker_id'], 'job': item, 'pods': []}
    for item in project_items:
        if item.get('kind') != 'Pod':
            continue
        meta = item['metadata']
        inactive = item.get('status', {}).get('phase') in {'Succeeded', 'Failed'} and not meta.get('deletionTimestamp')
        owners = [ref.get('uid') for ref in meta.get('ownerReferences', []) if ref.get('uid') in verified]
        if not owners:
            require(inactive, 'Another project Pod or orphan is active')
            continue
        require(len(owners) == 1, 'Ambiguous Pod owner identity')
        if not inactive:
            require(not terminal_job(verified[owners[0]]['job']), 'Terminal fleet Job has a live Pod')
            verified[owners[0]]['pods'].append(item)
    for uid, row in verified.items():
        require(len(row['pods']) <= 1, 'Fleet worker has multiple active Pods')
        for pod in row['pods']:
            node = pod.get('spec', {}).get('nodeName')
            if node:
                require(node not in nodes, 'Two eight-GPU fleet workers share a node')
                nodes[node] = uid
        if row['worker_id'] == current_worker_id:
            require(not terminal_job(row['job']) and len(row['pods']) == 1
                    and row['pods'][0].get('status', {}).get('phase') == 'Running', 'Current worker requires one Running owned Pod')
            pod = row['pods'][0]
            current = {'job_uid': uid, 'pod_uid': pod['metadata']['uid'], 'node': pod['spec']['nodeName'],
                       'worker_id': current_worker_id, 'other_active_project_jobs': [],
                       'authorized_fleet_job_uids': sorted(verified)}
    require(not require_current or current is not None, 'Current exact fleet worker is not live')
    return current if current is not None else {'status': 'ONLY_FROZEN_FLEET_WORKLOADS', 'job_uids': sorted(verified)}


@contextmanager
def project_transaction(storage):
    root = safe(storage['roots'][0]) / 'locks'
    root.mkdir(exist_ok=True)
    with safe(root / 'experiment-suite.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield root
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class FleetState:
    """Short shared-filesystem transactions; no worker holds a lock while training."""
    def __init__(self, plan, binding, state_root, *, plan_sha256, binding_sha256):
        self.plan, self.binding, self.state = plan, binding, safe(state_root)
        self.profile = validate_execution_profile(binding['execution_profile'], binding['suite_id'])
        self.plan_sha, self.binding_sha = plan_sha256, binding_sha256
        self.identity = {'schema_version': 2, 'suite_id': binding['suite_id'], 'plan_sha256': plan_sha256,
                         'binding_sha256': binding_sha256, 'source_identity_sha256': runner.source_identity(binding),
                         'execution_profile_sha256': runner.digest(self.profile)}

    def initialize(self):
        with project_transaction(self.binding['storage']) as locks:
            lease_dir = locks / 'fleet-leases'
            lease_dir.mkdir(exist_ok=True)
            lease_path = lease_dir / (self.profile['fleet_id'] + '.json')
            for path in lease_dir.glob('*.json'):
                lease = read(path)
                if path == lease_path:
                    require(lease['identity'] == self.identity and lease['state_root'] == str(self.state), 'Global fleet lease differs')
                else:
                    done = safe(lease['state_root']) / 'COMPLETE.json'
                    require(done.is_file() and read(done).get('identity') == lease['identity'], 'Another fleet has an unfinished global project lease')
            if not lease_path.exists():
                runner.write_once(lease_path, {'identity': self.identity, 'state_root': str(self.state), 'created_utc': runner.utc(), 'automatic_expiry': False})
            self.state.mkdir(parents=True, exist_ok=True)
            path = self.state / 'SUITE_IDENTITY.json'
            if path.exists():
                require(read(path) == self.identity, 'Suite state belongs to another exact plan/binding/profile')
            else:
                require(not any(self.state.iterdir()), 'Unidentified existing suite state')
                runner.write_once(path, self.identity)
            for name in ('tasks', 'workers', 'invocations'):
                (self.state / name).mkdir(exist_ok=True)

    def join(self, worker_id):
        worker_identity(self.profile, worker_id)
        with project_transaction(self.binding['storage']):
            root = self.state / 'workers' / worker_id
            root.mkdir(exist_ok=True)
            require(not (root / 'WORKER_CLAIM.json').exists(), 'Worker already claimed; no duplicate or automatic restart')
            claim = {'worker_id': worker_id, 'claim_id': uuid.uuid4().hex, 'pid': os.getpid(),
                     'linux_process_start': runner._process_start(os.getpid()), 'hostname': socket.gethostname(),
                     'identity': self.identity, 'utc': runner.utc(), 'automatic_expiry': False}
            runner.write_once(root / 'WORKER_CLAIM.json', claim)
            invocation = runner._new_numbered(self.state / 'invocations', 'invocation-')
            return claim, invocation

    def register(self, worker_id, startup):
        with project_transaction(self.binding['storage']):
            path = self.state / 'workers' / worker_id / 'JOB_OBSERVED.json'
            runner.write_once(path, {'worker_id': worker_id, 'job_uid': startup['job_uid'],
                                    'pod_uid': startup['pod_uid'], 'node': startup['node'], 'utc': runner.utc()})

    def workers_ready(self):
        """Require every allocation to observe the same shared state before work.

        A private per-node directory can never pass this rendezvous, even when
        its absolute path happens to match the coordinator's reviewed path.
        """
        with project_transaction(self.binding['storage']):
            require(not list((self.state / 'invocations').glob('*/STOPPED.json')), 'A worker stopped before shared-state rendezvous')
            workers = []
            for worker in self.profile['workers']:
                root = self.state / 'workers' / worker['worker_id']
                if not (root / 'WORKER_CLAIM.json').exists() or not (root / 'JOB_OBSERVED.json').exists():
                    return False
                claim, job = read(root / 'WORKER_CLAIM.json'), read(root / 'JOB_OBSERVED.json')
                require(claim['identity'] == self.identity and claim['worker_id'] == job['worker_id'] == worker['worker_id'],
                        'Shared worker registration differs from this fleet')
                workers.append({'worker_id': worker['worker_id'], 'claim_id': claim['claim_id'], **job})
            require(len({w['job_uid'] for w in workers}) == len(workers)
                    and len({w['node'] for w in workers}) == len(workers), 'Fleet workers must have distinct Jobs and nodes')
            path = self.state / 'FLEET_RENDEZVOUS.json'
            value = {'identity': self.identity, 'workers': workers, 'shared_state_observed': True}
            if path.exists():
                require(read(path) == value, 'Fleet registration changed after shared-state rendezvous')
            else:
                runner.write_once(path, value)
            return True

    def completions(self):
        completed = {}
        for task in self.plan['tasks']:
            if (self.state / 'tasks' / task['id'] / 'COMPLETED.json').exists():
                value = runner.completion_for(self.state, task['id'])
                require(value['plan_sha256'] == self.plan_sha and value['binding_sha256'] == self.binding_sha,
                        'Accepted task belongs to another exact plan/binding')
                completed[task['id']] = value
        return completed

    def claim(self, worker_id, job_uid, reserve):
        """reserve(task, owner) atomically admits a shared budget or raises StorageError."""
        from .suite_storage import StorageError
        with project_transaction(self.binding['storage']):
            failures = list((self.state / 'tasks').glob('*/FAILED.json'))
            require(not failures, 'Fleet stopped after a failed task; explicit recovery required')
            completed = self.completions()
            if len(completed) == len(self.plan['tasks']):
                return {'status': 'complete', 'completed': completed}
            pending = [t for t in self.plan['tasks'] if t['id'] not in completed]
            active = [t for t in pending if (self.state / 'tasks' / t['id'] / 'CLAIM.json').exists()]
            ready = [t for t in pending if t not in active and set(t['dependencies']) <= completed.keys()]
            # Free accepted checkpoints promptly; keep original scientific order within each class.
            order = {'cleanup': 0, 'archive': 1, 'evaluate': 2, 'latency': 2, 'compare_latency': 2}
            ready.sort(key=lambda t: order.get(t['kind'], 3))
            blocked = []
            for task in ready:
                owner = {'fleet_id': self.profile['fleet_id'], 'worker_id': worker_id, 'task_id': task['id'],
                         'claim_id': uuid.uuid4().hex, 'job_uid': job_uid}
                try:
                    token = reserve(task, owner)
                except StorageError as exc:
                    blocked.append({'task_id': task['id'], 'error': str(exc)})
                    continue
                root = self.state / 'tasks' / task['id']
                root.mkdir(exist_ok=True)
                require(not (root / 'attempts').exists(), 'An unfinished task cannot restart without verified recovery')
                attempt = runner._new_numbered(root / 'attempts', 'attempt-')
                claim = {'owner': owner, 'task_id': task['id'], 'task_sha256': runner.digest(task),
                         'attempt': str(attempt), 'storage_token': token, 'utc': runner.utc(), 'automatic_expiry': False}
                runner.write_once(root / 'CLAIM.json', claim)
                runner._journal(self.state, {'event': 'TASK_CLAIMED', 'task_id': task['id'], 'owner': owner})
                return {'status': 'claimed', 'task': task, 'claim': claim, 'completed': completed}
            require(active, 'No runnable task; dependency or storage admission failed: ' + str(blocked))
            return {'status': 'wait', 'active_task_ids': [t['id'] for t in active], 'storage_blocked': blocked}

    def finish(self, task, claim, completion):
        with project_transaction(self.binding['storage']):
            root = self.state / 'tasks' / task['id']
            require(read(root / 'CLAIM.json') == claim, 'Task claim changed before acceptance')
            runner.write_once(root / 'COMPLETED.json', completion)
            runner._journal(self.state, {'event': 'TASK_ACCEPTED', 'task_id': task['id'], 'worker_id': claim['owner']['worker_id']})

    def fail(self, task, claim, error):
        with project_transaction(self.binding['storage']):
            runner.write_once(self.state / 'tasks' / task['id'] / 'FAILED.json', {
                'task_id': task['id'], 'claim_id': claim['owner']['claim_id'], 'error_type': type(error).__name__,
                'error': str(error), 'utc': runner.utc(), 'automatic_retry': False})

    def complete(self, worker_id):
        with project_transaction(self.binding['storage']):
            completed = self.completions()
            require(len(completed) == len(self.plan['tasks']), 'Cannot complete an unfinished fleet')
            result = {'status': 'EXPERIMENT_SUITE_COMPLETE', 'identity': self.identity, 'utc': runner.utc(),
                      'completed_tasks': len(completed), 'completion_inventory': [
                          runner.artifact(self.state / 'tasks' / t['id'] / 'COMPLETED.json') for t in self.plan['tasks']],
                      'model_quality_or_speedup_claim': False}
            if not (self.state / 'COMPLETE.json').exists():
                runner.write_once(self.state / 'COMPLETE.json', result)
            return read(self.state / 'COMPLETE.json')
