"""Prepare, server-dry-run, or explicitly submit one frozen stream/low suite.

The input YAML is the operator-reviewed result of scheduler dry-run preparation.
This module does not discover free nodes, choose another pool, requeue a job,
delete checkpoints, or stop another workload. A durable submission intent allows
exact-job reconciliation after an ambiguous client result, never resubmission.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

from . import checkpoint_retention as retention
from . import suite_runner as runner
from .suite_actions import allocation_contract, terminal_job
from .suite_bindings import validate_environment
from .experiment_suite import validate_plan

require, safe, read, pin = runner.require, runner.safe, runner.read, runner.artifact


def save(path, value):
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    runner.write_once(path, value)
    return pin(path)


def load_yaml(path):
    import yaml
    documents = list(yaml.safe_load_all(safe(path).read_text()))
    require(len(documents) == 1 and isinstance(documents[0], dict), 'Exactly one reviewed job document is required')
    return documents[0]


def runner_argv(environment, plan_pin, binding_pin, state_root, *, resume=False, relocation_pin=None):
    argv = [environment['python'], '-B', '-m', 'deepprobe_vlm.suite_runner',
            '--plan', plan_pin['path'], '--plan-sha256', plan_pin['sha256'],
            '--binding', binding_pin['path'], '--binding-sha256', binding_pin['sha256'],
            '--state-root', str(safe(state_root)), '--execute']
    require(not relocation_pin or resume, 'Relocation requires explicit resume')
    argv += ['--resume'] if resume else []
    if relocation_pin:
        argv += ['--relocation', relocation_pin['path'], '--relocation-sha256', relocation_pin['sha256']]
    return argv


def validate_job(job, environment, argv):
    require(job.get('apiVersion') == 'kubeflow.org/v1' and job.get('kind') == 'PyTorchJob',
            'Reviewed YAML must be one PyTorchJob')
    metadata = job.get('metadata', {})
    identity = environment['allocation_identity']
    require(metadata.get('name') == identity['job_name'] and metadata.get('namespace') == identity['namespace']
            and not metadata.get('generateName') and not metadata.get('uid')
            and not metadata.get('resourceVersion') and 'status' not in job,
            'Reviewed YAML needs the exact fresh name/namespace and no server state')
    allocation_contract(job)
    replica = job['spec']['pytorchReplicaSpecs']['Master']
    spec = replica['template']['spec']
    container = spec['containers'][0]
    require(container.get('command', []) + container.get('args', []) == argv,
            'Container entry must be the exact frozen direct suite runner argv')
    require(container.get('workingDir') == environment['experiment_root'], 'Wrong frozen source working directory')
    require(isinstance(container.get('image'), str) and container['image'], 'Reviewed runtime image is required')
    require(not container.get('lifecycle') and not container.get('stdin') and not container.get('tty')
            and not container.get('envFrom') and not spec.get('ephemeralContainers')
            and not spec.get('hostPID') and not spec.get('hostIPC'), 'Unexpected auxiliary execution configuration')
    values = container.get('env', [])
    require(isinstance(values, list) and all(isinstance(x, dict) and set(x) == {'name', 'value'}
            and isinstance(x['name'], str) and isinstance(x['value'], str) for x in values)
            and len({x['name'] for x in values}) == len(values), 'Only explicit unique frozen environment values are allowed')
    env = {x['name']: x['value'] for x in values}
    expected = {**environment['runtime_environment'], **runner.OFFLINE,
                'PYTHONPATH': str(safe(environment['experiment_root']) / 'src')}
    require(all(env.get(key) == value for key, value in expected.items()), 'Offline/import runtime environment differs')
    require(spec.get('restartPolicy', 'Never') == 'Never', 'Pod must not restart automatically')
    return job


def complete_sources(environment):
    """All package code and dynamic operation imports must be frozen, not only the CLI."""
    root = safe(environment['experiment_root'])
    package = root / 'src/deepprobe_vlm'
    require(package.is_dir(), 'Frozen package source is missing')
    expected = {str(path) for path in package.rglob('*.py')}
    expected.update(str(root / name) for name in (
        'scripts/preflight.py', 'scripts/operations/score_tempcompass_mc.py',
        'scripts/operations/run_p1_stage_v9.py'))
    rows = environment['source_files']
    require(expected <= {x['path'] for x in rows}, 'Source inventory omits imported package or operation code')
    for row in rows:
        runner.verify_pin(row)
    return runner.source_identity({'source_files': rows})


def _subset(expected, observed):
    if isinstance(expected, dict):
        return isinstance(observed, dict) and all(key in observed and _subset(value, observed[key])
                                                 for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(observed, list) and len(expected) == len(observed) and all(
            _subset(a, b) for a, b in zip(expected, observed))
    return type(expected) is type(observed) and expected == observed


@contextmanager
def project_lock(environment):
    directory = safe(environment['storage']['roots'][0]) / 'locks'
    directory.mkdir(exist_ok=True)
    with safe(directory / 'experiment-suite.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('A project suite is active or another submission owns the project lock') from exc
        yield


class SuiteLauncher:
    def __init__(self, plan_pin, binding_pin, environment_pin, job_pin, preflight_pin, *, state_root,
                 output, resume=False, command=subprocess.run, store_factory=None, relocation_pin=None):
        self.pins = {'plan': plan_pin, 'binding': binding_pin, 'environment': environment_pin,
                     'job': job_pin, 'preflight': preflight_pin}
        if relocation_pin:
            self.pins['relocation'] = relocation_pin
        for item in self.pins.values():
            runner.verify_pin(item)
        self.plan, self.binding, self.env = (read(self.pins[k]['path']) for k in ('plan', 'binding', 'environment'))
        self.job = load_yaml(job_pin['path'])
        self.state, self.output = safe(state_root), safe(output)
        self.resume, self.command, self.relocation_pin = resume, command, relocation_pin
        require(type(resume) is bool, 'Resume must be explicit Boolean')
        require(not relocation_pin or resume, 'Relocation requires explicit resume')
        if store_factory is None:
            from .checkpoint_store import OssutilStore
            store_factory = lambda prefix, directory: OssutilStore(self.env['tools']['ossutil'],
                self.env['tools']['oss_config'], allowed_prefix=prefix, state_dir=directory)
        self.store_factory = store_factory

    def identity(self):
        return {'suite_id': self.env['suite_id'], 'plan_sha256': self.pins['plan']['sha256'],
                'binding_sha256': self.pins['binding']['sha256'],
                'environment_sha256': self.pins['environment']['sha256'],
                'job_yaml_sha256': self.pins['job']['sha256'], 'state_root': str(self.state), 'resume': self.resume,
                'relocation_sha256': self.relocation_pin['sha256'] if self.relocation_pin else None}

    def kubectl(self, args, *, timeout=60):
        tools = self.env['tools']
        return self.command([tools['kubectl'], '--kubeconfig', tools['kubeconfig'],
                             '--request-timeout=45s', *args], capture_output=True, text=True, timeout=timeout)

    def validate(self):
        validate_plan(self.plan, config_root=self.env['experiment_root'])
        validate_environment(self.env, self.plan)
        runner.validate_binding(self.plan, self.binding, plan_sha256=self.pins['plan']['sha256'])
        require(self.binding['suite_id'] == self.env['suite_id']
                and self.binding['source_files'] == self.env['source_files']
                and self.binding['storage'] == self.env['storage'], 'Machine environment/binding differs')
        from .suite_bindings import build_binding
        expected = build_binding(self.plan, self.env, plan_path=self.pins['plan']['path'],
            plan_sha256=self.pins['plan']['sha256'], environment_path=self.pins['environment']['path'],
            environment_sha256=self.pins['environment']['sha256'], config_root=self.env['experiment_root'])
        require(self.binding == expected, 'Binding is not the complete generated suite command set')
        source = complete_sources(self.env)
        roots = [safe(p) for p in self.env['storage']['roots']]
        require(all(any(path != root and path.is_relative_to(root) for root in roots)
                    for path in (self.state, self.output)), 'State/launch evidence must stay in accounted storage')
        require(self.state != self.output and not self.output.is_relative_to(self.state),
                'Launch receipts cannot occupy the fresh runner state directory')
        if self.resume:
            from .suite_relocation import resume_bindings
            accepted = resume_bindings(self.state, plan_sha256=self.pins['plan']['sha256'],
                binding_sha256=self.pins['binding']['sha256'], suite_id=self.env['suite_id'],
                source_sha256=source, relocation_pin=self.relocation_pin)
            for path in (self.state / 'tasks').glob('*/COMPLETED.json'):
                completed = runner.completion_for(self.state, path.parent.name)
                require(completed.get('plan_sha256') == self.pins['plan']['sha256']
                        and completed.get('binding_sha256') in accepted, 'Resume task journal differs')
        else:
            require(not self.state.exists(), 'Existing suite state requires explicit resume')
        validate_job(self.job, self.env, runner_argv(self.env, self.pins['plan'], self.pins['binding'], self.state,
                                                    resume=self.resume, relocation_pin=self.relocation_pin))
        preflight = read(self.pins['preflight']['path'])
        require(preflight.get('status') == 'SUITE_CPU_ADMITTED' and preflight.get('failures') == []
                and preflight.get('source_identity_sha256') == source
                and all(preflight.get(key + '_sha256') == self.pins[key]['sha256']
                        for key in ('plan', 'binding', 'environment')), 'CPU admission does not attest these exact inputs')
        generated = datetime.fromisoformat(preflight['generated_utc'].replace('Z', '+00:00'))
        require(generated.tzinfo is not None and 0 <= (datetime.now(timezone.utc) - generated).total_seconds() <= 1800,
                'CPU admission must be generated within the last 30 minutes')
        require(isinstance(preflight.get('outputs'), list) and preflight['outputs'], 'CPU admission needs retained evidence')
        for item in preflight['outputs']:
            runner.verify_pin(item)
        return runner.capacity(self.binding, self.binding['startup'].get('reserve_bytes', 0))

    def cluster(self):
        result = self.kubectl(['get', 'pytorchjobs,pods', '-A', '-o', 'json'])
        require(result.returncode == 0, 'Fresh complete Job/Pod listing failed')
        document = json.loads(result.stdout)
        require(isinstance(document.get('items'), list) and not document.get('metadata', {}).get('continue'),
                'Cluster listing is incomplete')
        prefixes = self.env['allocation_identity']['project_name_prefixes']
        require(any('jinhaopeng-dpvlm-'.startswith(prefix) for prefix in prefixes),
                'Project audit must include the full DeepProbe job-name prefix')
        active, observed = [], []
        exact = self.env['allocation_identity']
        for item in document['items']:
            kind, metadata = item.get('kind'), item.get('metadata', {})
            name, namespace = metadata.get('name', ''), metadata.get('namespace')
            if kind == 'PyTorchJob' and name == exact['job_name'] and namespace == exact['namespace']:
                raise ValueError('Exact Job already exists; use its original launch receipt to reconcile, never recreate')
            if not any(name.startswith(prefix) for prefix in prefixes):
                continue
            require(kind in {'PyTorchJob', 'Pod'}, 'Unexpected project workload kind')
            inactive = terminal_job(item) if kind == 'PyTorchJob' else (
                item.get('status', {}).get('phase') in {'Succeeded', 'Failed'}
                and not metadata.get('deletionTimestamp'))
            observed.append({'kind': kind, 'name': name, 'namespace': namespace, 'uid': metadata.get('uid'),
                             'inactive': inactive})
            if not inactive:
                active.append(observed[-1])
        require(not active, 'Another project Job or Pod is active, including Pending/terminating workloads')
        return {'status': 'NO_ACTIVE_PROJECT_WORKLOADS', 'generated_utc': runner.utc(), 'observed': observed}

    def archive_prefixes(self):
        results = []
        for task in self.plan['tasks']:
            if not (task['kind'] == 'train' or task['kind'] == 'archive' and task['archive_scope'] == 'results'):
                continue
            run_id = self.env['suite_id'] + '--' + task['id']
            prefix = self.env['archive_root'].rstrip('/') + '/runs/' + run_id + '/'
            store = self.store_factory(prefix, self.output / 'oss-audit' / task['id'])
            listing = retention._listing(store, prefix)
            expected = {}
            if listing:
                require(self.resume, 'Fresh suite archive prefix is not empty: ' + prefix)
                if task['kind'] == 'train':
                    manifest = retention.validate_manifest(read(Path(self.env['run_root']) / run_id / '.retention/MANIFEST.json'))
                    owner = manifest['owner']
                    completed = runner.completion_for(self.state, task['id'])
                    accepted = read(completed['receipts'][0]['path'])
                    require(owner['suite_id'] == self.env['suite_id'] and owner['run_id'] == run_id
                            and owner['source_sha256'] == runner.source_identity(self.binding)
                            and manifest['archive_prefix'] == prefix and accepted.get('owner') == owner
                            and accepted.get('training_complete') == manifest['training_completion'],
                            'Resume archive manifest identity differs from accepted training')
                    retention.check_owner(owner)
                    retention.verify_training_completion(manifest)
                    expected = retention._objects(manifest)
                else:
                    manifest_path = self.state / 'artifacts' / task['id'] / 'RESULT_MANIFEST.json'
                    manifest = read(manifest_path)
                    require(manifest.get('schema_version') == 1 and isinstance(manifest.get('files'), list),
                            'Unknown result archive manifest')
                    declared_rows = []
                    for result_task in task['result_task_ids']:
                        completed = runner.completion_for(self.state, result_task)
                        accepted = read(completed['receipts'][0]['path'])
                        require(accepted.get('task_id') == result_task and accepted.get('status') == 'SUITE_TASK_ACCEPTED',
                                'Result archive source was not accepted')
                        declared_rows.extend({**row, 'object': prefix + result_task + '/' + str(index) + '-' + Path(row['path']).name}
                                             for index, row in enumerate(accepted['outputs']))
                    require(manifest['files'] == declared_rows, 'Resume result manifest differs from all declared accepted outputs')
                    for row in manifest['files']:
                        runner.verify_pin({key: row[key] for key in ('path', 'bytes', 'sha256')})
                        require(row['object'].startswith(prefix), 'Result archive escapes its exact prefix')
                        require(row['object'] not in expected, 'Duplicate result archive object')
                        expected[row['object']] = row
                    expected[prefix + 'RESULT_MANIFEST.json'] = pin(manifest_path)
                require(set(listing) <= set(expected), 'Unrecognized objects in resume archive prefix')
                for uri, value in listing.items():
                    wanted = {key: expected[uri][key] for key in ('bytes', 'sha256')}
                    require(value['bytes'] == wanted['bytes'] and retention._object_digest(store, uri) == wanted,
                            'Existing resume archive object differs by full GET/SHA')
                require(listing == retention._listing(store, prefix), 'Resume archive changed during audit')
            results.append({'prefix': prefix, 'objects': len(listing), 'bytes': sum(row['bytes'] for row in listing.values()),
                            'status': 'KNOWN_RESUME_OBJECTS_FULL_GET_VERIFIED' if listing else 'EMPTY_EXACT_PREFIX'})
        return results

    def dry_run(self):
        result = self.kubectl(['create', '--dry-run=server', '-f', self.pins['job']['path'], '-o', 'json'])
        require(result.returncode == 0, 'Reviewed YAML failed server dry-run; nothing submitted')
        observed = json.loads(result.stdout)
        require(_subset(self.job['spec'], observed.get('spec')) and
                observed.get('metadata', {}).get('name') == self.env['allocation_identity']['job_name'],
                'Server dry-run changed the reviewed execution contract')
        allocation_contract(observed)
        return {'status': 'SERVER_DRY_RUN_ACCEPTED', 'response_document_sha256': runner.digest(observed)}

    def reconcile(self, intent):
        require(intent.get('identity') == self.identity(), 'Existing submission intent belongs to different inputs')
        identity = self.env['allocation_identity']
        try:
            result = self.kubectl(['get', 'pytorchjob', identity['job_name'], '-n', identity['namespace'],
                                   '--ignore-not-found=true', '-o', 'json'])
            require(result.returncode == 0, 'Exact Job query failed')
            observed = json.loads(result.stdout) if result.stdout.strip() else None
            if observed is None:
                value = {'status': 'SUBMISSION_OUTCOME_UNKNOWN', 'exact_job_observed': False,
                         'automatic_resubmit': False, 'reason': 'No exact Job currently visible after durable submission intent'}
            else:
                require(observed.get('metadata', {}).get('name') == identity['job_name']
                        and observed.get('metadata', {}).get('namespace') == identity['namespace']
                        and _subset(self.job['spec'], observed.get('spec')), 'Exact Job belongs to a different execution contract')
                allocation_contract(observed)
                value = {'status': 'EXACT_SUITE_JOB_OBSERVED', 'exact_job_observed': True,
                         'job_name': identity['job_name'], 'namespace': identity['namespace'],
                         'job_uid': observed['metadata']['uid'], 'automatic_resubmit': False}
        except (OSError, ValueError, subprocess.TimeoutExpired, KeyError) as error:
            value = {'status': 'SUBMISSION_OUTCOME_UNKNOWN', 'exact_job_observed': False,
                     'automatic_resubmit': False, 'error_type': type(error).__name__, 'reason': str(error)}
        receipt = save(self.output / 'reconciliations' / (uuid.uuid4().hex + '.json'),
                       {**value, 'generated_utc': runner.utc(), 'identity': self.identity()})
        return {**value, 'receipt': receipt}

    def run(self, mode='prepare'):
        require(mode in {'prepare', 'dry-run', 'submit', 'reconcile'}, 'Invalid launch mode')
        self.output.mkdir(parents=True, exist_ok=True)
        intent_path = self.output / 'SUBMISSION_INTENT.json'
        if intent_path.exists():
            # Read-only status reconciliation must remain available while the
            # running suite owns its long-lived project execution lock.
            return self.reconcile(read(intent_path))
        with project_lock(self.env):
            if intent_path.exists():
                # An earlier process may have died at any point after intent.
                # Never issue a second create, even when exact lookup is empty.
                return self.reconcile(read(intent_path))
            require(mode != 'reconcile', 'No submission intent exists to reconcile')
            capacity = self.validate()
            cluster = self.cluster()
            prefixes = self.archive_prefixes()
            prepared = {'status': 'SUITE_LAUNCH_PREPARED_NOT_SUBMITTED', 'generated_utc': runner.utc(),
                        'identity': self.identity(), 'inputs': self.pins, 'capacity': capacity,
                        'cluster': cluster, 'archive_prefixes': prefixes, 'gpu_submitted': False}
            if mode in {'dry-run', 'submit'}:
                prepared['server_dry_run'] = self.dry_run()
            prepared_pin = save(self.output / 'preparations' / (uuid.uuid4().hex + '.json'), prepared)
            if mode != 'submit':
                return {'status': prepared['status'], 'server_dry_run': mode == 'dry-run',
                        'gpu_submitted': False, 'receipt': prepared_pin}
            # Keep the fresh no-duplicate query adjacent to the one create.
            self.cluster()
            for value in self.pins.values():
                runner.verify_pin(value)
            complete_sources(self.env)
            intent = {'schema_version': 1, 'status': 'ONE_SHOT_SUBMISSION_INTENT', 'generated_utc': runner.utc(),
                      'identity': self.identity(), 'preparation': prepared_pin, 'automatic_resubmit': False}
            save(intent_path, intent)
            try:
                result = self.kubectl(['create', '-f', self.pins['job']['path'], '-o', 'json'])
                outcome = {'returncode': result.returncode,
                           'stdout_sha256': hashlib.sha256(result.stdout.encode()).hexdigest(),
                           'stderr_sha256': hashlib.sha256(result.stderr.encode()).hexdigest()}
            except (OSError, subprocess.TimeoutExpired) as error:
                outcome = {'error_type': type(error).__name__, 'returncode': None}
            save(self.output / 'SUBMISSION_CLIENT_RESULT.json', {**outcome, 'automatic_resubmit': False})
            return self.reconcile(intent)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('plan', 'binding', 'environment', 'job', 'preflight'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--state-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--relocation', type=Path)
    parser.add_argument('--relocation-sha256')
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--dry-run', action='store_true')
    choice.add_argument('--submit', action='store_true')
    choice.add_argument('--reconcile', action='store_true')
    args = parser.parse_args(argv)
    pins = [{ 'path': str(getattr(args, key)), 'sha256': getattr(args, key + '_sha256')}
            for key in ('plan', 'binding', 'environment', 'job', 'preflight')]
    mode = 'submit' if args.submit else 'dry-run' if args.dry_run else 'reconcile' if args.reconcile else 'prepare'
    require(bool(args.relocation) == bool(args.relocation_sha256) and (not args.relocation or args.resume),
            '--relocation and --relocation-sha256 require explicit --resume')
    relocation_pin = {'path': str(args.relocation), 'sha256': args.relocation_sha256} if args.relocation else None
    result = SuiteLauncher(*pins, state_root=args.state_root, output=args.output, resume=args.resume,
                           relocation_pin=relocation_pin).run(mode)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['status'] != 'SUBMISSION_OUTCOME_UNKNOWN' else 2


if __name__ == '__main__':
    raise SystemExit(main())
