"""One-shot scheduler admission tested against a fake kubectl and empty bucket."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from deepprobe_vlm import experiment_suite, suite_bindings, suite_launch, suite_relocation, suite_runner as runner


class FakeStore:
    def __init__(self, objects):
        self.objects = objects

    def list_prefix(self, prefix):
        return [{"uri": key, "bytes": len(value)} for key, value in sorted(self.objects.items()) if key.startswith(prefix)]

    def iter_object(self, uri):
        yield self.objects[uri]


class FakeKubectl:
    def __init__(self, job):
        self.job = deepcopy(job)
        self.items = []
        self.calls = []
        self.creates = 0
        self.created = None
        self.timeout_create = False
        self.timeout_but_created = False

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        if 'create' in argv:
            if '--dry-run=server' in argv:
                return subprocess.CompletedProcess(argv, 0, json.dumps(self.job), '')
            self.creates += 1
            if not self.timeout_create or self.timeout_but_created:
                self.created = deepcopy(self.job)
                self.created['metadata']['uid'] = 'fake-actual-job-uid'
            if self.timeout_create:
                raise subprocess.TimeoutExpired(argv, 60)
            return subprocess.CompletedProcess(argv, 0, json.dumps(self.created), '')
        if 'pytorchjobs,pods' in argv:
            return subprocess.CompletedProcess(argv, 0, json.dumps({'items': self.items}), '')
        if 'pytorchjob' in argv:
            return subprocess.CompletedProcess(argv, 0, json.dumps(self.created) if self.created else '', '')
        raise AssertionError(argv)


class SuiteLaunchTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.project, self.staging = self.base / 'project', self.base / 'staging'
        self.project.mkdir(); self.staging.mkdir()
        self.experiment = self.project / 'experiment'
        self.experiment.mkdir()
        original = Path(__file__).resolve().parents[1]
        shutil.copytree(original / 'configs/train/suite', self.experiment / 'configs/train/suite')
        for name in ('src/deepprobe_vlm/__init__.py', 'src/deepprobe_vlm/suite_runner.py',
                     'src/deepprobe_vlm/suite_actions.py', 'scripts/preflight.py',
                     'scripts/operations/score_tempcompass_mc.py', 'scripts/operations/run_p1_stage_v9.py'):
            path = self.experiment / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# synthetic frozen fixture source\n')
        self.plan = experiment_suite.default_plan()
        self.plan_pin = self.write('PLAN.json', self.plan)
        self.env = {'schema_version': 1, 'suite_id': 'tiny-launch-suite', 'experiment_root': str(self.experiment),
                    'python': sys.executable, 'run_root': str(self.project / 'runs'),
                    'archive_root': 'oss://ss-devbox-stream/project/serial-suite',
                    'source_files': [runner.artifact(p) for p in sorted(self.experiment.rglob('*.py'))],
                    'runtime_environment': {}, 'history': {},
                    'storage': {'roots': [str(self.project), str(self.staging)], 'limit_bytes': 99_000_000_000,
                                'other_reserve_bytes': 0, 'free_guard_bytes': 1_000_000_000},
                    'task_reserve_bytes': {k: 0 for k in suite_bindings.RESERVE_KEYS},
                    'assets': {k: {} for k in ('training', 'benchmarks', 'backbones', 'development')},
                    'tools': {'kubectl': '/usr/bin/kubectl', 'kubeconfig': str(self.project / 'kubeconfig'),
                              'ossutil': '/usr/bin/ossutil', 'oss_config': str(self.project / 'ossconfig')},
                    'allocation_identity': {'job_name': 'jinhaopeng-dpvlm-suite-fixture-n1', 'namespace': 'default',
                                            'project_name_prefixes': ['jinhaopeng-dpvlm-', 'deepprobe-']}}
        self.env_pin = self.write('ENVIRONMENT.json', self.env)
        self.binding = suite_bindings.build_binding(self.plan, self.env, plan_path=self.plan_pin['path'],
            plan_sha256=self.plan_pin['sha256'], environment_path=self.env_pin['path'],
            environment_sha256=self.env_pin['sha256'], config_root=self.experiment)
        self.binding_pin = self.write('BINDING.json', self.binding)
        self.state, self.output = self.project / 'state', self.project / 'launch'
        argv = suite_launch.runner_argv(self.env, self.plan_pin, self.binding_pin, self.state)
        self.job = {'apiVersion': 'kubeflow.org/v1', 'kind': 'PyTorchJob', 'metadata': {
            'name': self.env['allocation_identity']['job_name'], 'namespace': 'default'},
            'spec': {'pytorchReplicaSpecs': {'Master': {'replicas': 1, 'restartPolicy': 'Never', 'template': {'spec': {
                'nodeSelector': {'sslaunch-pool': 'stream'}, 'priorityClassName': 'low', 'restartPolicy': 'Never',
                'containers': [{'name': 'pytorch', 'image': 'fixture/runtime@sha256:' + 'a' * 64,
                    'workingDir': str(self.experiment), 'command': argv, 'args': [],
                    'resources': {'limits': {'nvidia.com/gpu': 8}, 'requests': {'nvidia.com/gpu': 8}},
                    'env': [{'name': key, 'value': value} for key, value in
                            {**runner.OFFLINE, 'PYTHONPATH': str(self.experiment / 'src')}.items()]}]}}}}}}
        self.job_pin = self.write('JOB.yaml', self.job)  # JSON is also YAML; no custom loader behavior.
        proof = self.write('CPU-proof.json', {'synthetic_fixture': True})
        self.preflight = {'status': 'SUITE_CPU_ADMITTED', 'failures': [], 'generated_utc': runner.utc(),
            'source_identity_sha256': runner.source_identity(self.binding), 'outputs': [proof],
            **{key + '_sha256': value['sha256'] for key, value in
               [('plan', self.plan_pin), ('binding', self.binding_pin), ('environment', self.env_pin)]}}
        self.preflight_pin = self.write('CPU.json', self.preflight)
        self.kube = FakeKubectl(self.job)
        self.objects = {}
        self.relocation_pin = None

    def write(self, name, value):
        path = self.project / name
        path.write_text(json.dumps(value))
        return runner.artifact(path)

    def launcher(self, *, resume=False):
        return suite_launch.SuiteLauncher(self.plan_pin, self.binding_pin, self.env_pin, self.job_pin,
            self.preflight_pin, state_root=self.state, output=self.output, resume=resume, command=self.kube,
            store_factory=lambda prefix, directory: FakeStore(self.objects), relocation_pin=self.relocation_pin)

    def prepare_relocation(self):
        self.state.mkdir()
        (self.state / 'tasks').mkdir()
        self.old_binding_pin, self.old_env_pin = self.binding_pin, self.env_pin
        self.old_env = deepcopy(self.env)
        runner.write_once(self.state / 'SUITE_IDENTITY.json', {
            'suite_id': self.env['suite_id'], 'plan_sha256': self.plan_pin['sha256'],
            'binding_sha256': self.binding_pin['sha256'], 'source_identity_sha256': runner.source_identity(self.binding)})
        self.env = deepcopy(self.env)
        self.env['allocation_identity']['job_name'] = 'jinhaopeng-dpvlm-suite-fixture-next-n1'
        self.env_pin = self.write('ENVIRONMENT-next.json', self.env)
        self.binding = suite_bindings.build_binding(self.plan, self.env, plan_path=self.plan_pin['path'],
            plan_sha256=self.plan_pin['sha256'], environment_path=self.env_pin['path'],
            environment_sha256=self.env_pin['sha256'], config_root=self.experiment)
        self.binding_pin = self.write('BINDING-next.json', self.binding)
        receipt = suite_relocation.relocation(self.plan_pin, self.old_binding_pin, self.old_env_pin,
            self.binding_pin, self.env_pin, state_root=self.state)
        self.relocation_pin = self.write('RELOCATION.json', receipt)
        self.job = deepcopy(self.job)
        self.job['metadata']['name'] = self.env['allocation_identity']['job_name']
        self.job['spec']['pytorchReplicaSpecs']['Master']['template']['spec']['containers'][0]['command'] = (
            suite_launch.runner_argv(self.env, self.plan_pin, self.binding_pin, self.state,
                                     resume=True, relocation_pin=self.relocation_pin))
        self.job_pin = self.write('JOB-next.yaml', self.job)
        self.preflight.update(binding_sha256=self.binding_pin['sha256'], environment_sha256=self.env_pin['sha256'])
        self.preflight_pin = self.write('CPU-next.json', self.preflight)
        self.kube = FakeKubectl(self.job)

    def test_new_allocation_reuses_only_equivalent_bound_state_and_keeps_original_receipts(self):
        self.prepare_relocation()
        task = self.state / 'tasks/gpu_gate'
        task.mkdir()
        proof = self.write('old-gate.json', {'synthetic': True})
        completed = {'status': 'TASK_ACCEPTED', 'task_id': 'gpu_gate', 'receipts': [proof],
                     'plan_sha256': self.plan_pin['sha256'], 'binding_sha256': self.old_binding_pin['sha256']}
        runner.write_once(task / 'COMPLETED.json', completed)
        original_bytes = (task / 'COMPLETED.json').read_bytes()
        result = self.launcher(resume=True).run('submit')
        self.assertEqual(result['status'], 'EXACT_SUITE_JOB_OBSERVED')
        self.assertEqual(self.kube.creates, 1)
        self.assertFalse((self.state / 'relocations').exists())  # Launcher does not rewrite runner state.
        accepted = suite_relocation.resume_bindings(self.state,
            plan_sha256=self.plan_pin['sha256'], binding_sha256=self.binding_pin['sha256'],
            suite_id=self.env['suite_id'], source_sha256=runner.source_identity(self.binding),
            relocation_pin=self.relocation_pin, apply=True)
        self.assertEqual(accepted, {self.old_binding_pin['sha256'], self.binding_pin['sha256']})
        self.assertEqual((task / 'COMPLETED.json').read_bytes(), original_bytes)
        self.assertEqual(runner.read(self.state / 'SUITE_IDENTITY.json')['binding_sha256'], self.old_binding_pin['sha256'])
        again = suite_relocation.resume_bindings(self.state, plan_sha256=self.plan_pin['sha256'],
            binding_sha256=self.binding_pin['sha256'], suite_id=self.env['suite_id'],
            source_sha256=runner.source_identity(self.binding), apply=True)
        self.assertEqual(again, accepted)
        self.assertEqual(len(list((self.state / 'relocations').iterdir())), 1)

    def test_relocation_rejects_asset_runtime_storage_or_command_changes(self):
        self.prepare_relocation()
        for mutate in (lambda e: e['assets']['training'].update(unreviewed='changed'),
                       lambda e: e['runtime_environment'].update(OMP_NUM_THREADS='2'),
                       lambda e: e['storage'].update(other_reserve_bytes=1),
                       lambda e: e['allocation_identity'].update(namespace='other')):
            changed = deepcopy(self.env)
            mutate(changed)
            env_pin = self.write('forbidden-environment.json', changed)
            with self.assertRaises(ValueError):
                suite_relocation.relocation(self.plan_pin, self.old_binding_pin, self.old_env_pin,
                    self.binding_pin, env_pin, state_root=self.state)
        changed_binding = deepcopy(self.binding)
        changed_binding['tasks']['gpu_gate']['argv'].append('--unreviewed')
        bind_pin = self.write('forbidden-binding.json', changed_binding)
        with self.assertRaisesRegex(ValueError, 'complete generated'):
            suite_relocation.relocation(self.plan_pin, self.old_binding_pin, self.old_env_pin,
                bind_pin, self.env_pin, state_root=self.state)

    def test_relocation_does_not_allow_active_prior_job_or_unproved_new_binding(self):
        self.prepare_relocation()
        self.kube.items = [{'kind': 'PyTorchJob', 'metadata': {
            'name': self.old_env['allocation_identity']['job_name'], 'namespace': 'default', 'uid': 'old'},
            'status': {'conditions': [{'type': 'Running', 'status': 'True'}]}}]
        with self.assertRaisesRegex(ValueError, 'Another project'):
            self.launcher(resume=True).run('submit')
        self.assertEqual(self.kube.creates, 0)
        with self.assertRaisesRegex(ValueError, 'explicit audited'):
            suite_relocation.resume_bindings(self.state, plan_sha256=self.plan_pin['sha256'],
                binding_sha256=self.binding_pin['sha256'], suite_id=self.env['suite_id'],
                source_sha256=runner.source_identity(self.binding))

    def test_prepare_and_server_dry_run_never_submit_or_create_runner_state(self):
        result = self.launcher().run()
        self.assertFalse(result['gpu_submitted'])
        self.assertFalse(self.state.exists())
        value = runner.read(result['receipt']['path'])
        self.assertEqual(len(value['archive_prefixes']), 52)
        self.assertTrue(all(row['status'] == 'EMPTY_EXACT_PREFIX' for row in value['archive_prefixes']))
        result = self.launcher().run('dry-run')
        self.assertTrue(result['server_dry_run'])
        self.assertEqual(self.kube.creates, 0)

    def test_submit_exactly_once_then_reconcile_never_creates_again(self):
        first = self.launcher().run('submit')
        self.assertEqual(first['status'], 'EXACT_SUITE_JOB_OBSERVED')
        self.assertEqual(first['job_uid'], 'fake-actual-job-uid')
        second = self.launcher().run('submit')
        self.assertEqual(second['status'], first['status'])
        self.assertEqual(self.kube.creates, 1)
        self.assertTrue((self.output / 'SUBMISSION_INTENT.json').is_file())
        self.assertFalse(self.state.exists())

    def test_exact_status_reconciliation_remains_available_while_runner_holds_project_lock(self):
        self.launcher().run('submit')
        with suite_launch.project_lock(self.env):
            result = self.launcher().run('reconcile')
        self.assertEqual(result['status'], 'EXACT_SUITE_JOB_OBSERVED')
        self.assertEqual(self.kube.creates, 1)

    def test_unknown_client_outcome_and_empty_exact_lookup_never_blindly_retries(self):
        self.kube.timeout_create = True
        first = self.launcher().run('submit')
        self.assertEqual(first['status'], 'SUBMISSION_OUTCOME_UNKNOWN')
        second = self.launcher().run('submit')
        self.assertEqual(second['status'], 'SUBMISSION_OUTCOME_UNKNOWN')
        self.assertFalse(second['automatic_resubmit'])
        self.assertEqual(self.kube.creates, 1)

    def test_client_timeout_after_creation_is_reconciled_by_exact_job(self):
        self.kube.timeout_create = self.kube.timeout_but_created = True
        result = self.launcher().run('submit')
        self.assertEqual(result['status'], 'EXACT_SUITE_JOB_OBSERVED')
        self.assertEqual(self.kube.creates, 1)

    def test_project_pending_job_blocks_admission_but_unrelated_job_does_not(self):
        pending = {'kind': 'PyTorchJob', 'metadata': {'name': 'jinhaopeng-dpvlm-existing',
                    'namespace': 'default', 'uid': 'other'}, 'status': {'conditions': []}}
        self.kube.items = [pending]
        with self.assertRaisesRegex(ValueError, 'Another project'):
            self.launcher().run('submit')
        self.assertEqual(self.kube.creates, 0)
        self.kube.items[0]['metadata']['name'] = 'another-researcher-training'
        self.launcher().run('prepare')
        self.assertEqual(self.kube.creates, 0)

    def test_nonempty_new_archive_prefix_blocks_submission(self):
        stage = next(task['id'] for task in self.plan['tasks'] if task['kind'] == 'train')
        key = self.env['archive_root'] + '/runs/' + self.env['suite_id'] + '--' + stage + '/surprise'
        self.objects[key] = b'existing'
        with self.assertRaisesRegex(ValueError, 'not empty'):
            self.launcher().run('submit')
        self.assertEqual(self.kube.creates, 0)

    def test_changed_low_contract_or_runner_argv_is_rejected(self):
        original = deepcopy(self.job)
        cases = [lambda pod: pod.update(priorityClassName='normal'),
                 lambda pod: pod['containers'][0]['command'].append('--unknown'),
                 lambda pod: pod['containers'][0].update(command=['bash', '-c', 'echo run']),
                 lambda pod: pod['containers'][0]['env'].append({'name': 'HF_HUB_OFFLINE', 'value': '0'}),
                 lambda pod: pod.update(initContainers=[{'name': 'surprise'}])]
        for mutate in cases:
            self.job = deepcopy(original)
            mutate(self.job['spec']['pytorchReplicaSpecs']['Master']['template']['spec'])
            self.job_pin = self.write('JOB.yaml', self.job)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.launcher().run('submit')
        self.assertEqual(self.kube.creates, 0)

    def test_stale_preflight_and_changed_sources_fail_closed(self):
        self.preflight['generated_utc'] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        self.preflight_pin = self.write('CPU.json', self.preflight)
        with self.assertRaisesRegex(ValueError, '30 minutes'):
            self.launcher().run()
        self.preflight['generated_utc'] = runner.utc()
        self.preflight_pin = self.write('CPU.json', self.preflight)
        Path(self.env['source_files'][0]['path']).write_text('changed source')
        with self.assertRaisesRegex(ValueError, 'pinned artifact changed'):
            self.launcher().run('submit')
        self.assertEqual(self.kube.creates, 0)


if __name__ == '__main__':
    unittest.main()
