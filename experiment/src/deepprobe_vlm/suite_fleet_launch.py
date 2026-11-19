"""Prepare or explicitly launch a frozen one-to-four-worker B300 suite fleet.

Default mode only validates and emits preparation evidence. --dry-run uses
server dry-run. Only --submit creates Jobs, once per exact reviewed identity;
an uncertain client outcome always reconciles instead of resubmitting.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

from . import suite_runner as runner
from .suite_launch import SuiteLauncher, save, load_yaml, runner_argv, validate_job, project_lock, _subset, complete_sources
from .suite_actions import terminal_job
from .suite_fleet import (validate_execution_profile, job_labels, allocation_contract,
                          verify_job_identity, audit_fleet_jobs)

require, safe, read = runner.require, runner.safe, runner.read


def prepare_worker_jobs(template, environment, plan_pin, binding_pin, state_root, output):
    """Materialize exact worker YAML from a reviewed pod template; no scheduler calls."""
    profile = validate_execution_profile(environment['execution_profile'], environment['suite_id'])
    output = safe(output)
    output.mkdir(parents=True, exist_ok=True)
    def retained(path, value):
        if path.exists():
            require(read(path) == value, 'Existing prepared Job differs; use a fresh reviewed output directory')
            return runner.artifact(path)
        return save(path, value)
    jobs = []
    for worker in profile['workers']:
        job = deepcopy(template)
        metadata = job.setdefault('metadata', {})
        require(not any(key in metadata for key in ('uid', 'resourceVersion', 'generateName')) and 'status' not in job,
                'Template must not contain server state')
        metadata.update(name=worker['job_name'], namespace=worker['namespace'])
        metadata.setdefault('labels', {}).update(job_labels(profile, worker['worker_id'], binding_pin['sha256']))
        metadata.setdefault('annotations', {}).update({'deepprobe/binding-sha256': binding_pin['sha256'],
                                                       'deepprobe/profile-sha256': runner.digest(profile)})
        replicas = job['spec']['pytorchReplicaSpecs']
        require(set(replicas) == {'Master'}, 'Template must contain only one Master replica')
        replicas['Master']['replicas'] = 1
        replicas['Master']['restartPolicy'] = 'Never'
        spec = replicas['Master']['template']['spec']
        require(not spec.get('nodeName'), 'A fleet template cannot pin an unreviewed node')
        require(set(spec.get('nodeSelector', {})) <= {'sslaunch-pool', 'aliyun.accelerator/nvidia_name', 'nvidia.com/gpu.product'},
                'Template contains an unreviewed additional node selector')
        spec['nodeSelector'] = deepcopy(profile['node_selector'])
        pod_meta = replicas['Master']['template'].setdefault('metadata', {})
        labels = pod_meta.setdefault('labels', {})
        labels['pod-group.scheduling.sigs.k8s.io/name'] = worker['job_name']
        labels['pod-group.scheduling.sigs.k8s.io/min-available'] = '1'
        for key in ('training.kubeflow.org/job-name', 'pytorch-job-name', 'job-name'):
            if key in labels:
                labels[key] = worker['job_name']
        spec['priorityClassName'], spec['restartPolicy'] = profile['priority'], 'Never'
        machine = environment.get('machine_identity', {})
        require(set(machine) == {'uid', 'gid'} and all(type(machine[k]) is int and machine[k] > 0 for k in machine),
                'Exact non-root machine uid/gid must be discovered before preparing worker Jobs')
        spec.setdefault('securityContext', {}).update(runAsUser=machine['uid'], runAsGroup=machine['gid'], runAsNonRoot=True)
        require(len(spec['containers']) == 1, 'Template needs exactly one execution container')
        container = spec['containers'][0]
        container.setdefault('resources', {}).setdefault('requests', {})['nvidia.com/gpu'] = 8
        container['resources'].setdefault('limits', {})['nvidia.com/gpu'] = 8
        argv = runner_argv(environment, plan_pin, binding_pin, state_root, worker_id=worker['worker_id'])
        container['command'], container['args'] = argv, []
        container['workingDir'] = environment['experiment_root']
        container['env'] = [{'name': key, 'value': value} for key, value in {
            **environment['runtime_environment'], **runner.OFFLINE,
            'PYTHONPATH': str(safe(environment['experiment_root']) / 'src')}.items()]
        validate_job(job, environment, argv, worker_id=worker['worker_id'], binding_sha256=binding_pin['sha256'])
        jobs.append({'worker_id': worker['worker_id'], 'job': retained(output / (worker['worker_id'] + '.yaml'), job)})
    bundle = {'schema_version': 1, 'execution_profile_sha256': runner.digest(profile), 'jobs': jobs}
    return retained(output / 'FLEET_JOBS.json', bundle)


class FleetLauncher(SuiteLauncher):
    def __init__(self, plan_pin, binding_pin, environment_pin, jobs_pin, preflight_pin, **kwargs):
        runner.verify_pin(jobs_pin)
        bundle = read(jobs_pin['path'])
        require(set(bundle) == {'schema_version', 'execution_profile_sha256', 'jobs'} and bundle['schema_version'] == 1,
                'Invalid worker Job bundle')
        env = read(environment_pin['path'])
        self.profile = validate_execution_profile(env['execution_profile'], env['suite_id'])
        require(bundle['execution_profile_sha256'] == runner.digest(self.profile), 'Job bundle belongs to another profile')
        expected = [worker['worker_id'] for worker in self.profile['workers']]
        require(isinstance(bundle['jobs'], list) and [row['worker_id'] for row in bundle['jobs']] == expected
                and all(set(row) == {'worker_id', 'job'} for row in bundle['jobs']), 'One exact Job per worker is required')
        self.job_rows = bundle['jobs']
        self.worker_id = expected[0]
        for row in self.job_rows:
            runner.verify_pin(row['job'])
        super().__init__(plan_pin, binding_pin, environment_pin, self.job_rows[0]['job'], preflight_pin, **kwargs)
        self.pins['jobs'] = jobs_pin
        require(not self.resume and not self.relocation_pin, 'Fleet recovery needs new separately audited claims, never automatic relocation')

    def identity(self):
        return {**super().identity(), 'execution_profile_sha256': runner.digest(self.profile),
                'worker_jobs': self.job_rows, 'fleet_jobs_sha256': self.pins['jobs']['sha256']}

    def validate(self):
        capacity = super().validate()
        for row in self.job_rows:
            argv = runner_argv(self.env, self.pins['plan'], self.pins['binding'], self.state, worker_id=row['worker_id'])
            validate_job(load_yaml(row['job']['path']), self.env, argv,
                         worker_id=row['worker_id'], binding_sha256=self.pins['binding']['sha256'])
        return capacity

    def cluster(self):
        result = self.kubectl(['get', 'pytorchjobs,pods', '-A', '-o', 'json'])
        require(result.returncode == 0, 'Fresh complete cluster query failed')
        document = json.loads(result.stdout)
        report = audit_fleet_jobs(self.profile, document, self.state, require_current=False,
            binding_sha256=self.pins['binding']['sha256'],
            project_name_prefixes=self.env['allocation_identity']['project_name_prefixes'])
        exact = {(w['namespace'], w['job_name']): w['worker_id'] for w in self.profile['workers']}
        for item in document['items']:
            meta = item.get('metadata', {})
            worker_id = exact.get((meta.get('namespace'), meta.get('name'))) if item.get('kind') == 'PyTorchJob' else None
            if worker_id:
                require(not terminal_job(item) and not meta.get('deletionTimestamp'),
                        'An attempted fleet worker is no longer active; do not allocate more nodes')
                root = self.output / 'workers' / worker_id
                require((root / 'SUBMISSION_INTENT.json').exists(), 'Exact worker Job exists without this durable submission intent')
                if (root / 'JOB_UID.json').exists():
                    require(read(root / 'JOB_UID.json')['job_uid'] == meta['uid'], 'Exact worker Job UID changed')
        return report

    def dry_run(self, rows=None):
        values = []
        for row in (self.job_rows if rows is None else rows):
            job = load_yaml(row['job']['path'])
            result = self.kubectl(['create', '--dry-run=server', '-f', row['job']['path'], '-o', 'json'])
            require(result.returncode == 0, 'Worker server dry-run failed: ' + row['worker_id'])
            observed = json.loads(result.stdout)
            require(_subset(job['spec'], observed.get('spec')), 'Server dry-run changed the reviewed worker spec')
            verify_job_identity(observed, self.profile, row['worker_id'], self.pins['binding']['sha256'])
            values.append({'worker_id': row['worker_id'], 'response_document_sha256': runner.digest(observed)})
        return {'status': 'FLEET_SERVER_DRY_RUN_ACCEPTED', 'workers': values}

    def reconcile_worker(self, row):
        root = self.output / 'workers' / row['worker_id']
        intent_path = root / 'SUBMISSION_INTENT.json'
        if not intent_path.exists():
            return {'worker_id': row['worker_id'], 'status': 'NOT_SUBMITTED', 'automatic_resubmit': False}
        intent = read(intent_path)
        require(intent.get('identity') == self.identity() and intent.get('worker_id') == row['worker_id'],
                'Worker submission intent differs from frozen fleet')
        worker = next(w for w in self.profile['workers'] if w['worker_id'] == row['worker_id'])
        try:
            result = self.kubectl(['get', 'pytorchjob', worker['job_name'], '-n', worker['namespace'], '--ignore-not-found=true', '-o', 'json'])
            require(result.returncode == 0 and result.stdout.strip(), 'Exact worker Job not observable')
            job = json.loads(result.stdout)
            verify_job_identity(job, self.profile, row['worker_id'], self.pins['binding']['sha256'])
            require(_subset(load_yaml(row['job']['path'])['spec'], job['spec']), 'Observed worker execution spec differs')
            uid = job['metadata']['uid']
            require(isinstance(uid, str) and uid, 'Actual Job UID missing')
            if (root / 'JOB_UID.json').exists():
                require(read(root / 'JOB_UID.json')['job_uid'] == uid, 'Exact worker Job UID changed')
            else:
                save(root / 'JOB_UID.json', {'worker_id': row['worker_id'], 'job_uid': uid, 'identity': self.identity()})
            value = {'worker_id': row['worker_id'], 'status': 'EXACT_SUITE_JOB_OBSERVED', 'job_uid': uid,
                     'job_name': worker['job_name'], 'automatic_resubmit': False,
                     'job_terminal': terminal_job(job), 'job_terminating': bool(job['metadata'].get('deletionTimestamp'))}
        except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
            value = {'worker_id': row['worker_id'], 'status': 'SUBMISSION_OUTCOME_UNKNOWN',
                     'error_type': type(exc).__name__, 'reason': str(exc), 'automatic_resubmit': False}
        save(root / 'reconciliations' / (uuid.uuid4().hex + '.json'), {**value, 'utc': runner.utc()})
        return value

    def reconcile_fleet(self, intent):
        require(intent.get('identity') == self.identity(), 'Fleet submission intent changed')
        workers = [self.reconcile_worker(row) for row in self.job_rows]
        status = ('FLEET_WORKER_TERMINATED' if any(w.get('job_terminal') or w.get('job_terminating') for w in workers)
                  else 'EXACT_FLEET_JOBS_OBSERVED' if all(w['status'] == 'EXACT_SUITE_JOB_OBSERVED' for w in workers)
                  else 'SUBMISSION_OUTCOME_UNKNOWN')
        return {'status': status, 'workers': workers, 'automatic_resubmit': False, 'identity': self.identity()}

    def _submit_rows(self, rows, intent):
        for row in rows:
            root = self.output / 'workers' / row['worker_id']
            require(not (root / 'SUBMISSION_INTENT.json').exists(), 'Never recreate an attempted worker Job')
            self.cluster()
            for value in [*self.pins.values(), row['job']]:
                runner.verify_pin(value)
            complete_sources(self.env)
            save(root / 'SUBMISSION_INTENT.json', {**intent, 'worker_id': row['worker_id']})
            try:
                result = self.kubectl(['create', '-f', row['job']['path'], '-o', 'json'])
                client = {'returncode': result.returncode, 'stdout_sha256': hashlib.sha256(result.stdout.encode()).hexdigest(),
                          'stderr_sha256': hashlib.sha256(result.stderr.encode()).hexdigest()}
            except (OSError, subprocess.TimeoutExpired) as exc:
                client = {'returncode': None, 'error_type': type(exc).__name__}
            save(root / 'SUBMISSION_CLIENT_RESULT.json', {**client, 'automatic_resubmit': False})
            observed = self.reconcile_worker(row)
            if observed['status'] != 'EXACT_SUITE_JOB_OBSERVED' or observed.get('job_terminal') or observed.get('job_terminating'):
                break
        return self.reconcile_fleet(intent)

    def continue_unsubmitted(self, intent):
        """Explicitly submit only never-attempted workers after fresh admission.

        An attempted but still-unobservable Job blocks continuation. No age or
        absent lookup is treated as permission to repeat its create request.
        """
        with project_lock(self.env):
            current = self.reconcile_fleet(intent)
            require(all(w['status'] in {'NOT_SUBMITTED', 'EXACT_SUITE_JOB_OBSERVED'} for w in current['workers']),
                    'An attempted worker outcome is still unknown; reconcile before continuing')
            require(not any(w.get('job_terminal') or w.get('job_terminating') for w in current['workers']),
                    'An attempted worker is not active; use explicit failed-worker recovery instead of allocating more nodes')
            missing = {w['worker_id'] for w in current['workers'] if w['status'] == 'NOT_SUBMITTED'}
            if not missing:
                return current
            self.allow_existing_fleet_state = True
            capacity = self.validate()  # includes a fresh, <=30-minute CPU admission
            cluster, prefixes = self.cluster(), self.archive_prefixes()
            rows = [row for row in self.job_rows if row['worker_id'] in missing]
            dry_run = self.dry_run(rows)
            save(self.output / 'continuations' / (uuid.uuid4().hex + '.json'), {
                'status': 'UNATTEMPTED_WORKERS_READMITTED', 'identity': self.identity(), 'utc': runner.utc(),
                'worker_ids': sorted(missing), 'capacity': capacity, 'cluster': cluster, 'archive_prefixes': prefixes,
                'server_dry_run': dry_run, 'preflight': self.pins['preflight'], 'automatic_resubmit': False})
            return self._submit_rows(rows, intent)

    def run(self, mode='prepare'):
        require(mode in {'prepare', 'dry-run', 'submit', 'reconcile', 'continue-unsubmitted'}, 'Invalid fleet launch mode')
        self.output.mkdir(parents=True, exist_ok=True)
        intent_path = self.output / 'FLEET_SUBMISSION_INTENT.json'
        if intent_path.exists():
            if mode == 'continue-unsubmitted':
                return self.continue_unsubmitted(read(intent_path))
            return self.reconcile_fleet(read(intent_path))
        with project_lock(self.env):
            if intent_path.exists():
                return self.reconcile_fleet(read(intent_path))
            require(mode not in {'reconcile', 'continue-unsubmitted'}, 'No fleet submission intent exists')
            capacity = self.validate()
            cluster, prefixes = self.cluster(), self.archive_prefixes()
            prepared = {'status': 'FLEET_LAUNCH_PREPARED_NOT_SUBMITTED', 'utc': runner.utc(), 'identity': self.identity(),
                        'inputs': self.pins, 'capacity': capacity, 'cluster': cluster, 'archive_prefixes': prefixes,
                        'gpu_submitted': False, 'worker_count': len(self.job_rows)}
            if mode in {'dry-run', 'submit'}:
                prepared['server_dry_run'] = self.dry_run()
            receipt = save(self.output / 'preparations' / (uuid.uuid4().hex + '.json'), prepared)
            if mode != 'submit':
                return {'status': prepared['status'], 'gpu_submitted': False, 'server_dry_run': mode == 'dry-run', 'receipt': receipt}
            intent = {'status': 'FLEET_ONE_SHOT_SUBMISSION_INTENT', 'utc': runner.utc(), 'identity': self.identity(),
                      'preparation': receipt, 'automatic_resubmit': False}
            save(intent_path, intent)
            return self._submit_rows(self.job_rows, intent)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('plan', 'binding', 'environment', 'jobs', 'preflight'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--state-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument('--dry-run', action='store_true')
    choices.add_argument('--submit', action='store_true')
    choices.add_argument('--reconcile', action='store_true')
    choices.add_argument('--continue-unsubmitted', action='store_true')
    args = parser.parse_args(argv)
    pins = [{'path': str(getattr(args, key)), 'sha256': getattr(args, key + '_sha256')}
            for key in ('plan', 'binding', 'environment', 'jobs', 'preflight')]
    mode = 'continue-unsubmitted' if args.continue_unsubmitted else 'submit' if args.submit else 'dry-run' if args.dry_run else 'reconcile' if args.reconcile else 'prepare'
    result = FleetLauncher(*pins, state_root=args.state_root, output=args.output).run(mode)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2 if result['status'] in {'SUBMISSION_OUTCOME_UNKNOWN', 'FLEET_WORKER_TERMINATED'} else 0


if __name__ == '__main__':
    raise SystemExit(main())
