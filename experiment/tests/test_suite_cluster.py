"""Complete census/UID/read-only retry boundaries, without a cluster or GPU."""
from contextlib import contextmanager
from copy import deepcopy
import os
import sys
import time
import urllib.parse
import json
import subprocess

import pytest

from deepprobe_vlm import suite_cluster as cluster


TOOLS = {'kubectl': '/usr/bin/kubectl', 'kubeconfig': '/fixture/config'}
PREFIXES = ('jinhaopeng-dpvlm-', 'deepprobe-')


def object_(kind, name, uid, *, owners=(), labels=None):
    return {'apiVersion': 'v1' if kind == 'Pod' else 'kubeflow.org/v1', 'kind': kind,
            'metadata': {'namespace': 'default', 'name': name, 'uid': uid, 'resourceVersion': '17',
                         'labels': labels or {}, 'ownerReferences': [{'uid': v} for v in owners]},
            'spec': {'source': 'not part of the census'}, 'status': {'phase': 'Pending'}}


def census(rows, kind, *, continuation='', remaining='', list_version='17'):
    header = {'resourceVersion': list_version, 'continue': continuation}
    if remaining != '':
        header['remainingItemCount'] = remaining
    return json.dumps({'apiVersion': 'meta.k8s.io/v1', 'kind': 'PartialObjectMetadataList',
        'metadata': header, 'items': [{'apiVersion': 'meta.k8s.io/v1', 'kind': 'PartialObjectMetadata',
                                     'metadata': deepcopy(row['metadata'])} for row in rows]})


class Fake:
    def __init__(self, rows=()):
        self.rows, self.calls, self.sleeps = list(rows), [], []
        self.failures, self.census_override, self.exact_override = [], {}, {}
        self.proxy_closed = False

    @contextmanager
    def connection(self, tools, deadline, clock):
        try:
            yield self.metadata_get
        finally:
            self.proxy_closed = True

    def metadata_get(self, path, timeout):
        self.calls.append((['metadata', path], {'timeout': timeout}))
        if self.failures:
            error = self.failures.pop(0)
            if 'Forbidden' in error:
                raise cluster.ClusterQueryError('Metadata API unavailable: HTTP 403')
            if error:
                raise cluster._TransientQueryError(error)
        resource = urllib.parse.urlsplit(path).path.rsplit('/', 1)[-1]
        kind = {'pytorchjobs': 'PyTorchJob', 'pods': 'Pod'}[resource]
        rows = [r for r in self.rows if r['kind'] == kind]
        override = self.census_override.get(kind)
        return override.pop(0) if isinstance(override, list) else override if override is not None else census(rows, kind)

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if self.failures:
            error = self.failures.pop(0)
            if error == 'timeout':
                raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
            if error:
                return subprocess.CompletedProcess(argv, 1, '', error)
        resource = argv[argv.index('get') + 1]
        kind = {'pytorchjobs': 'PyTorchJob', 'pods': 'Pod'}[resource]
        rows = [r for r in self.rows if r['kind'] == kind]
        name = argv[argv.index('get') + 2]
        row = self.exact_override.get(name, next(r for r in rows if r['metadata']['name'] == name))
        return subprocess.CompletedProcess(argv, 0, json.dumps(row), '')

    def query(self, **kwargs):
        return cluster.list_workloads(TOOLS, project_name_prefixes=PREFIXES, command=self,
            sleep=self.sleeps.append, metadata_connection=self.connection, **kwargs)


def test_transient_timeout_retries_get_and_retains_complete_scope():
    job = object_('PyTorchJob', 'jinhaopeng-dpvlm-test', 'job-1')
    pod = object_('Pod', 'unprefixed-worker', 'pod-1', owners=['job-1'])
    unrelated = object_('Pod', 'someone-else', 'pod-2')
    f = Fake([job, pod, unrelated]); f.failures = ['timeout']
    result = f.query()
    expected = {'complete': True, 'all_namespaces': True,
        'resource_versions': {'pytorchjobs': '17', 'pods': '17'}, 'objects': 3,
        'exact_project_objects': 2, 'get_calls': 5, 'transport_retries': 1}
    assert {key: result['census'][key] for key in expected} == expected
    assert result['census']['census_bytes'] > 0 and result['census']['exact_object_bytes'] > 0
    assert result['items'][:2] == [job, pod]
    assert 'spec' not in result['items'][2]
    assert f.sleeps == [1]
    assert all('create' not in argv and 'pytorchjobs,pods' not in argv for argv, _ in f.calls)
    assert all('limit=100' in argv[1] for argv, _ in f.calls if argv[0] == 'metadata')
    assert f.proxy_closed


def test_persistent_timeout_never_returns_empty_or_cached_success():
    f = Fake(); f.failures = ['timeout'] * 3
    with pytest.raises(cluster.ClusterQueryError, match='admission remains closed'):
        f.query()
    assert len(f.calls) == 3 and f.sleeps == [1, 2]
    assert all(argv[0] == 'metadata' for argv, _ in f.calls)


def test_permission_denial_is_not_retried_or_interpreted_as_empty():
    f = Fake(); f.failures = ['Error from server (Forbidden): forbidden']
    with pytest.raises(cluster.ClusterQueryError):
        f.query()
    assert len(f.calls) == 1 and not f.sleeps


@pytest.mark.parametrize('header', [census([], 'Pod', remaining=1), '', '{}',
    json.dumps({'apiVersion': 'v1', 'kind': 'PodList', 'metadata': {}, 'items': []})])
def test_incomplete_or_unrecognized_list_is_rejected(header):
    f = Fake(); f.census_override['Pod'] = header
    with pytest.raises(cluster.ClusterQueryError):
        f.query()
    assert len(f.calls) == 2


def test_missing_or_duplicate_uid_is_rejected_before_any_exact_read():
    row = object_('Pod', 'jinhaopeng-dpvlm-orphan', 'pod-1')
    for rows in ([row, row], [{**row, 'metadata': {**row['metadata'], 'uid': ''}}]):
        f = Fake(rows)
        with pytest.raises(cluster.ClusterQueryError):
            f.query()
        assert len(f.calls) == 2


def test_uid_change_during_exact_read_is_rejected():
    row = object_('Pod', 'jinhaopeng-dpvlm-orphan', 'pod-1')
    f = Fake([row]); new = deepcopy(row); new['metadata']['uid'] = 'pod-2'
    f.exact_override[row['metadata']['name']] = new
    with pytest.raises(cluster.ClusterQueryError, match='UID changed'):
        f.query()


def test_scope_includes_prefixless_identity_and_bidirectional_owners():
    job = object_('PyTorchJob', 'legacy-name', 'job-1')
    pod = object_('Pod', 'legacy-pod', 'pod-1', owners=['job-1'], labels={'deepprobe/worker': 'worker-01'})
    sibling = object_('Pod', 'another-pod', 'pod-2', owners=['job-1'])
    f = Fake([job, pod, sibling])
    result = f.query()
    assert result['census']['exact_project_objects'] == 3
    assert result['items'] == [job, pod, sibling]


def test_unknown_project_and_orphan_objects_remain_for_admission_audit():
    rows = [object_('PyTorchJob', 'jinhaopeng-dpvlm-other-pending', 'job-1'),
            object_('Pod', 'jinhaopeng-dpvlm-orphan', 'pod-1')]
    result = Fake(rows).query()
    assert cluster.project_workloads(result['items'], PREFIXES) == rows
    from deepprobe_vlm.suite_actions import terminal_job
    assert not terminal_job(result['items'][0])
    assert result['items'][1]['status']['phase'] == 'Pending'


@pytest.mark.parametrize('kind', ['PyTorchJob', 'Pod'])
def test_real_admission_rejects_unexpected_project_workload(tmp_path, kind):
    row = object_(kind, 'jinhaopeng-dpvlm-unexpected', 'foreign-uid')
    profile = {'schema_version': 2, 'fleet_id': 'fixture', 'priority': 'low', 'gpu_product': 'NVIDIA-B300',
        'node_selector': {'sslaunch-pool': 'stream', 'aliyun.accelerator/nvidia_name': 'NVIDIA-L20D'},
        'workers': [{'worker_id': 'worker-01', 'job_name': 'jinhaopeng-dpvlm-expected', 'namespace': 'default'}]}
    from deepprobe_vlm.suite_fleet import audit_fleet_jobs
    with pytest.raises(ValueError, match='Another project'):
        audit_fleet_jobs(profile, Fake([row]).query(), tmp_path.resolve(), require_current=False,
                         binding_sha256='a' * 64, project_name_prefixes=PREFIXES)


def test_prefixless_orphan_is_selected_by_missing_owners_project_name():
    row = object_('Pod', 'random-controller-pod', 'orphan-uid')
    row['metadata']['ownerReferences'] = [{'kind': 'PyTorchJob', 'name': 'jinhaopeng-dpvlm-gone', 'uid': 'gone-uid'}]
    result = Fake([row]).query()
    assert result['census']['exact_project_objects'] == 1
    assert cluster.project_workloads(result['items'], PREFIXES) == [row]


def test_scope_label_cannot_disappear_between_census_and_exact_read():
    row = object_('Pod', 'random-pod', 'pod-1', labels={'deepprobe/worker': 'worker-01'})
    f = Fake([row]); changed = deepcopy(row); changed['metadata']['labels'] = {}
    f.exact_override['random-pod'] = changed
    with pytest.raises(cluster.ClusterQueryError, match='identity changed'):
        f.query()


def test_shared_missing_owner_keeps_prefixless_active_sibling_in_scope():
    first = object_('Pod', 'jinhaopeng-dpvlm-terminal', 'pod-1', owners=['deleted-job'])
    first['status']['phase'] = 'Succeeded'
    sibling = object_('Pod', 'random-active-sibling', 'pod-2', owners=['deleted-job'])
    result = Fake([first, sibling]).query()
    assert result['census']['exact_project_objects'] == 2
    assert cluster.project_workloads(result['items'], PREFIXES) == [first, sibling]


def test_exact_read_transport_failure_retries_without_second_census_or_create():
    row = object_('Pod', 'jinhaopeng-dpvlm-orphan', 'pod-1')
    f = Fake([row]); f.failures = ['', '', 'connection reset by peer']
    assert f.query()['items'] == [row]
    assert len(f.calls) == 4 and sum(argv[0] == 'metadata' for argv, _ in f.calls) == 2


def test_total_deadline_stops_before_any_next_request():
    f = Fake(); times = iter([0, 0, 121, 121])
    f.failures = ['timeout']
    with pytest.raises(cluster.ClusterQueryError, match='deadline exceeded'):
        f.query(clock=lambda: next(times))
    assert len(f.calls) == 1


def test_empty_server_metadata_resource_version_is_rejected():
    f = Fake(); f.census_override['Pod'] = census([], 'Pod', list_version='')
    with pytest.raises(cluster.ClusterQueryError, match='resourceVersion'):
        f.query()


def test_server_partial_metadata_has_no_full_spec_fallback():
    assert cluster.METADATA_ACCEPT == 'application/json;as=PartialObjectMetadataList;g=meta.k8s.io;v=v1'
    f = Fake(); f.census_override['Pod'] = json.dumps({'kind': 'PodList', 'apiVersion': 'v1',
        'metadata': {'resourceVersion': '17'}, 'items': []})
    with pytest.raises(cluster.ClusterQueryError, match='full-spec fallback'):
        f.query()
    assert f.proxy_closed


def test_fleet_live_uses_complete_census_then_exact_uid_reads(tmp_path, monkeypatch):
    from deepprobe_vlm import suite_fleet, suite_runner
    profile = {'schema_version': 2, 'fleet_id': 'fixture', 'priority': 'low', 'gpu_product': 'NVIDIA-B300',
        'node_selector': {'sslaunch-pool': 'stream', 'aliyun.accelerator/nvidia_name': 'NVIDIA-L20D'},
        'workers': [{'worker_id': 'worker-01', 'job_name': 'jinhaopeng-dpvlm-expected', 'namespace': 'default'}]}
    job = object_('PyTorchJob', 'jinhaopeng-dpvlm-expected', 'job-1')
    job['metadata']['labels'] = suite_fleet.job_labels(profile, 'worker-01', 'a' * 64)
    job['metadata']['annotations'] = {'deepprobe/binding-sha256': 'a' * 64,
                                     'deepprobe/profile-sha256': suite_runner.digest(profile)}
    job['spec'] = {'pytorchReplicaSpecs': {'Master': {'replicas': 1, 'restartPolicy': 'Never',
        'template': {'spec': {'nodeSelector': profile['node_selector'], 'priorityClassName': 'low',
            'containers': [{'resources': {'limits': {'nvidia.com/gpu': 8},
                                         'requests': {'nvidia.com/gpu': 8}}}]}}}}}
    pod = object_('Pod', 'random-owned-pod', 'pod-1', owners=['job-1'])
    pod['spec'] = {'nodeName': 'node-1'}; pod['status'] = {'phase': 'Running'}
    state = tmp_path.resolve(); (state / 'SUITE_IDENTITY.json').write_text(json.dumps({'binding_sha256': 'a' * 64}))
    f = Fake([job, pod]); monkeypatch.setattr(cluster.subprocess, 'run', f)
    monkeypatch.setattr(cluster, '_metadata_connection', f.connection)
    suite_runner._fleet_live({'tools': TOOLS, 'allocation_identity': {'project_name_prefixes': PREFIXES}},
                             {'execution_profile': profile}, state, 'worker-01')
    assert len(f.calls) == 4
    assert all('pytorchjobs,pods' not in argv for argv, _ in f.calls)


def test_all_pages_including_late_unexpected_project_object_are_read():
    row = object_('Pod', 'jinhaopeng-dpvlm-late-orphan', 'late-uid')
    f = Fake([row]); f.census_override['Pod'] = [census([], 'Pod', continuation='token+/='), census([row], 'Pod')]
    result = f.query()
    assert result['items'] == [row] and result['census']['pages']['pods'] == 2
    assert any('continue=token%2B%2F%3D' in argv[1] for argv, _ in f.calls if argv[0] == 'metadata')


@pytest.mark.parametrize('second', [census([], 'Pod', continuation='again'), census([], 'Pod', list_version='18')])
def test_repeated_page_token_or_changed_snapshot_is_rejected(second):
    f = Fake(); f.census_override['Pod'] = [census([], 'Pod', continuation='again'), second]
    with pytest.raises(cluster.ClusterQueryError):
        f.query()
    assert f.proxy_closed


def test_real_loopback_proxy_forwards_accept_and_is_cleaned_on_success_and_failure(tmp_path, monkeypatch):
    script = tmp_path / 'fake-kubectl'
    script.write_text('#!' + sys.executable + '\n' + """
import http.server,json
class Handler(http.server.BaseHTTPRequestHandler):
 def do_GET(self):
  value={'accept':self.headers.get('Accept'),'path':self.path}
  payload=json.dumps(value).encode();self.send_response(200);self.end_headers();self.wfile.write(payload)
 def log_message(self,*args):pass
server=http.server.HTTPServer(('127.0.0.1',0),Handler)
print('Starting to serve on 127.0.0.1:'+str(server.server_port),flush=True)
server.serve_forever()
""")
    script.chmod(0o700)
    children = []; original = cluster.subprocess.Popen
    def record(*args, **kwargs):
        process = original(*args, **kwargs); children.append(process); return process
    monkeypatch.setattr(cluster.subprocess, 'Popen', record)
    for fail in (False, True):
        try:
            with cluster._metadata_connection({'kubectl': str(script), 'kubeconfig': '/fixture'},
                    time.monotonic() + 10, time.monotonic) as get:
                observed = json.loads(get('/api/v1/pods?limit=100', 3))
                assert observed == {'accept': cluster.METADATA_ACCEPT, 'path': '/api/v1/pods?limit=100'}
                if fail:
                    raise ValueError('test body failure')
        except ValueError:
            assert fail
        assert children[-1].poll() is not None
        with pytest.raises(ProcessLookupError):
            os.killpg(children[-1].pid, 0)
