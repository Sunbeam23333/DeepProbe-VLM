"""Fleet admission and ambiguous submission handling against fake scheduler I/O."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from deepprobe_vlm import suite_bindings, suite_fleet, suite_fleet_launch, suite_runner as runner
from test_suite_launch import SuiteLaunchTest, FakeStore


class FakeFleetKubectl:
    def __init__(self):
        self.jobs, self.calls, self.creates = {}, [], 0
        self.timeout_worker = None
        self.created_before_timeout = False
        self.extra = []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        if 'create' in argv:
            job = json.loads(Path(argv[argv.index('-f') + 1]).read_text())
            if '--dry-run=server' in argv:
                return subprocess.CompletedProcess(argv, 0, json.dumps(job), '')
            self.creates += 1
            name = job['metadata']['name']; job['metadata']['uid'] = 'uid-' + name
            if name != self.timeout_worker or self.created_before_timeout:
                self.jobs[name] = job
            if name == self.timeout_worker:
                raise subprocess.TimeoutExpired(argv, 30)
            return subprocess.CompletedProcess(argv, 0, json.dumps(job), '')
        if 'pytorchjobs,pods' in argv:
            return subprocess.CompletedProcess(argv, 0, json.dumps({'items': list(self.jobs.values()) + self.extra}), '')
        if 'pytorchjob' in argv:
            name = argv[argv.index('pytorchjob') + 1]
            return subprocess.CompletedProcess(argv, 0, json.dumps(self.jobs[name]) if name in self.jobs else '', '')
        raise AssertionError(argv)


class FleetLaunchTest(unittest.TestCase):
    def setUp(self):
        f = self.f = SuiteLaunchTest(); f.setUp(); self.addCleanup(f.doCleanups)
        profile = {'schema_version': 2, 'fleet_id': f.env['suite_id'], 'priority': 'normal', 'gpu_product': 'NVIDIA-B300',
            'node_selector': {'sslaunch-pool': 'stream', 'aliyun.accelerator/nvidia_name': 'NVIDIA-L20D'},
            'workers': [{'worker_id': 'worker-0' + str(i), 'job_name': 'jinhaopeng-dpvlm-launch-fixture-' + str(i), 'namespace': 'default'} for i in (1, 2)]}
        f.env['execution_profile'] = profile
        f.env['machine_identity'] = {'uid': 1000, 'gid': 1000}
        f.env['storage'] = {'schema_version': 2, 'roots': [str(f.project), str(f.staging)], 'ledger_root': str(f.project / 'ledger'),
            'home_limit_bytes': 150_000_000_000, 'oss_limit_bytes': 150_000_000_000,
            'oss_prefixes': ['oss://ss-devbox-stream/project/serial-suite/'],
            'home_safety_bytes': 1024, 'oss_safety_bytes': 1024, 'free_guard_bytes': 1024}
        f.env_pin = f.write('FLEET_ENV.json', f.env)
        f.binding = suite_bindings.build_binding(f.plan, f.env, plan_path=f.plan_pin['path'], plan_sha256=f.plan_pin['sha256'],
            environment_path=f.env_pin['path'], environment_sha256=f.env_pin['sha256'], config_root=f.experiment)
        f.binding_pin = f.write('FLEET_BINDING.json', f.binding)
        self.bundle_pin = suite_fleet_launch.prepare_worker_jobs(f.job, f.env, f.plan_pin, f.binding_pin, f.state, f.project / 'worker-jobs')
        f.preflight.update(binding_sha256=f.binding_pin['sha256'], environment_sha256=f.env_pin['sha256'])
        f.preflight_pin = f.write('FLEET_CPU.json', f.preflight)
        self.kube = FakeFleetKubectl()
        self.capacity = patch.object(runner, 'capacity', return_value={'synthetic_fixture': True})
        self.capacity.start(); self.addCleanup(self.capacity.stop)

    def launcher(self):
        f = self.f
        return suite_fleet_launch.FleetLauncher(f.plan_pin, f.binding_pin, f.env_pin, self.bundle_pin, f.preflight_pin,
            state_root=f.state, output=f.output, command=self.kube, store_factory=lambda prefix, directory: FakeStore({}))

    def test_prepare_and_server_dry_run_create_no_jobs(self):
        value = self.launcher().run('prepare')
        self.assertEqual(value['status'], 'FLEET_LAUNCH_PREPARED_NOT_SUBMITTED')
        self.assertFalse(value['gpu_submitted']); self.assertEqual(self.kube.creates, 0)
        value = self.launcher().run('dry-run')
        self.assertTrue(value['server_dry_run']); self.assertEqual(self.kube.creates, 0)
        self.assertEqual(sum('--dry-run=server' in call for call in self.kube.calls), 2)
        self.assertFalse(self.f.state.exists())

    def test_job_preparation_reuses_only_identical_immutable_documents(self):
        f = self.f
        again = suite_fleet_launch.prepare_worker_jobs(f.job, f.env, f.plan_pin, f.binding_pin, f.state, f.project / 'worker-jobs')
        self.assertEqual(again, self.bundle_pin)
        changed = deepcopy(f.job)
        changed['spec']['pytorchReplicaSpecs']['Master']['template']['spec']['containers'][0]['image'] = 'changed/image'
        with self.assertRaisesRegex(ValueError, 'prepared Job differs'):
            suite_fleet_launch.prepare_worker_jobs(changed, f.env, f.plan_pin, f.binding_pin, f.state, f.project / 'worker-jobs')

    def test_two_workers_submit_once_and_timeout_with_exact_job_reconciles(self):
        self.kube.timeout_worker = self.f.env['execution_profile']['workers'][1]['job_name']
        self.kube.created_before_timeout = True
        value = self.launcher().run('submit')
        self.assertEqual(value['status'], 'EXACT_FLEET_JOBS_OBSERVED')
        self.assertEqual(self.kube.creates, 2)
        value = self.launcher().run('submit')
        self.assertEqual(value['status'], 'EXACT_FLEET_JOBS_OBSERVED'); self.assertEqual(self.kube.creates, 2)

    def test_unobserved_timeout_stops_fleet_and_never_creates_again(self):
        self.kube.timeout_worker = self.f.env['execution_profile']['workers'][0]['job_name']
        first = self.launcher().run('submit')
        self.assertEqual(first['status'], 'SUBMISSION_OUTCOME_UNKNOWN'); self.assertEqual(self.kube.creates, 1)
        self.assertEqual(first['workers'][1]['status'], 'NOT_SUBMITTED')
        self.launcher().run('submit')
        self.assertEqual(self.kube.creates, 1)

    def test_explicit_continuation_only_creates_previously_unattempted_workers(self):
        name = self.f.env['execution_profile']['workers'][0]['job_name']
        self.kube.timeout_worker = name
        self.launcher().run('submit')
        with self.assertRaisesRegex(ValueError, 'still unknown'):
            self.launcher().run('continue-unsubmitted')
        self.assertEqual(self.kube.creates, 1)
        # The original server request becomes observable later; no repeat create.
        row = runner.read(self.bundle_pin['path'])['jobs'][0]
        job = runner.read(row['job']['path']); job['metadata']['uid'] = 'uid-' + name
        self.kube.jobs[name] = job
        value = self.launcher().run('continue-unsubmitted')
        self.assertEqual(value['status'], 'EXACT_FLEET_JOBS_OBSERVED')
        self.assertEqual(self.kube.creates, 2)
        self.launcher().run('continue-unsubmitted')
        self.assertEqual(self.kube.creates, 2)

    def test_explicit_continuation_rejects_a_terminal_previously_attempted_worker(self):
        name = self.f.env['execution_profile']['workers'][0]['job_name']
        self.kube.timeout_worker = name
        self.launcher().run('submit')
        row = runner.read(self.bundle_pin['path'])['jobs'][0]
        job = runner.read(row['job']['path']); job['metadata']['uid'] = 'uid-' + name
        job['status'] = {'conditions': [{'type': 'Failed', 'status': 'True'}]}
        self.kube.jobs[name] = job
        with self.assertRaisesRegex(ValueError, 'not active'):
            self.launcher().run('continue-unsubmitted')
        self.assertEqual(self.kube.creates, 1)
        self.assertEqual(self.launcher().run('reconcile')['status'], 'FLEET_WORKER_TERMINATED')

    def test_generated_jobs_bind_actual_scheduler_alias_user_and_independent_gangs(self):
        names = []
        for row in runner.read(self.bundle_pin['path'])['jobs']:
            job = runner.read(row['job']['path'])
            master = job['spec']['pytorchReplicaSpecs']['Master']
            spec = master['template']['spec']; labels = master['template']['metadata']['labels']
            self.assertEqual(spec['nodeSelector'], {'sslaunch-pool': 'stream', 'aliyun.accelerator/nvidia_name': 'NVIDIA-L20D'})
            self.assertEqual(spec['securityContext']['runAsUser'], 1000)
            self.assertEqual(labels['pod-group.scheduling.sigs.k8s.io/name'], job['metadata']['name'])
            self.assertEqual(labels['pod-group.scheduling.sigs.k8s.io/min-available'], '1')
            names.append(labels['pod-group.scheduling.sigs.k8s.io/name'])
        self.assertEqual(len(set(names)), 2)

    def test_uid_change_is_not_adopted_on_reconcile(self):
        self.launcher().run('submit')
        next(iter(self.kube.jobs.values()))['metadata']['uid'] = 'replacement-job'
        value = self.launcher().run('reconcile')
        self.assertEqual(value['status'], 'SUBMISSION_OUTCOME_UNKNOWN')
        self.assertEqual(self.kube.creates, 2)

    def test_another_active_job_blocks_all_submission(self):
        self.kube.extra = [{'kind': 'PyTorchJob', 'metadata': {'name': 'jinhaopeng-dpvlm-unrelated', 'namespace': 'default', 'uid': 'other'}, 'status': {}}]
        with self.assertRaisesRegex(ValueError, 'Another project Job'):
            self.launcher().run('submit')
        self.assertEqual(self.kube.creates, 0)

    def test_live_audit_rejects_orphan_wrong_hash_and_shared_node(self):
        self.launcher().run('submit')
        items = list(self.kube.jobs.values())
        for i, job in enumerate(list(items)):
            items.append({'kind': 'Pod', 'metadata': {'name': job['metadata']['name'] + '-master-0', 'uid': 'pod-' + str(i),
                'ownerReferences': [{'uid': job['metadata']['uid']}]}, 'spec': {'nodeName': 'node-' + str(i)}, 'status': {'phase': 'Running'}})
        profile = self.f.env['execution_profile']
        def audit(values):
            return suite_fleet.audit_fleet_jobs(profile, {'items': values}, self.f.state, 'worker-01', binding_sha256=self.f.binding_pin['sha256'])
        self.assertEqual(audit(items)['worker_id'], 'worker-01')
        bad = deepcopy(items); bad[0]['metadata']['annotations']['deepprobe/binding-sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'hashes differ'): audit(bad)
        bad = deepcopy(items); bad[-1]['metadata']['ownerReferences'] = [{'uid': 'orphan'}]
        with self.assertRaisesRegex(ValueError, 'orphan'): audit(bad)
        bad = deepcopy(items); bad[-1]['spec']['nodeName'] = bad[-2]['spec']['nodeName']
        with self.assertRaisesRegex(ValueError, 'share a node'): audit(bad)


if __name__ == '__main__': unittest.main()
