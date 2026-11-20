"""Synthetic closed-chain trust-boundary fixtures; no real job or metric claims.

The plan, action wrappers, retention transactions and scheduler are real code.
Tiny model bytes and external audit attestations simulate already-reviewed old
work only, and are never production acceptance evidence.
"""
from copy import deepcopy
import json
from pathlib import Path
from unittest import mock

import pytest

from deepprobe_vlm import checkpoint_retention as retention
from deepprobe_vlm import experiment_suite, suite_actions as actions, suite_bindings as bindings
from deepprobe_vlm import suite_fleet as fleet, suite_runner as runner, suite_scratch as scratch
from deepprobe_vlm import suite_task_import as imports
from test_suite_actions_retention import MemoryStore


class ClosedFixture:
    def __init__(self, root):
        self.root = root.resolve(); self.state = self.root / 'old-state'
        self.target = self.root / 'new-state'; self.runs = self.root / 'old-runs'; self.runs.mkdir()
        self.plan = experiment_suite.default_plan(); self.tasks = {t['id']: t for t in self.plan['tasks']}
        self.chain = 'train__ouro_r1_s29__alignment'
        self.group, self.selected = imports._group(self.plan, self.chain)
        self.proof = self.pin('synthetic-proof.json', {'scope': 'SYNTHETIC_TEST_ONLY'})
        self.source = self.pin('source.json', {'executor': 'synthetic-old'})
        self.frozen = self.pin('model-source.json', {'model_training_source': 'distinct-from-executor'})
        self.profile = {'schema_version': 2, 'fleet_id': 'old-fixture', 'priority': 'low',
            'gpu_product': 'NVIDIA-B300', 'node_selector': {'sslaunch-pool': 'stream',
            'aliyun.accelerator/nvidia_name': 'NVIDIA-L20D'}, 'workers': [{'worker_id': 'worker-01',
            'job_name': 'jinhaopeng-dpvlm-old-fixture', 'namespace': 'default'}]}
        self.old_binding = {'suite_id': 'old-fixture', 'source_files': [self.source],
                            'execution_profile': self.profile}
        self.plan_pin = self.pin('PLAN.json', self.plan)
        self.binding_pin = self.pin('OLD_BINDING.json', self.old_binding)
        self.identity = self.make_identity(self.old_binding, self.binding_pin['sha256'])
        self.identity_pin = self.pin('old-state/SUITE_IDENTITY.json', self.identity)
        self.archive_root = 'oss://ss-devbox-stream/synthetic-import'
        self.env = {'suite_id': 'old-fixture', 'experiment_root': str(Path(__file__).resolve().parents[1]),
            'run_root': str(self.runs), 'source_files': [self.source], 'archive_root': self.archive_root,
            'storage': {'roots': [str(self.runs), str(self.state)], 'limit_bytes': 99_000_000_000,
                        'other_reserve_bytes': 0, 'free_guard_bytes': 0}}
        self.store = MemoryStore(); self.worker = actions.Actions(self.plan, self.env, self.state)
        self.worker.store = lambda owner: (self.store, self.prefix(owner['run_id']))
        self.results, self.histories, self.task_pins = {}, {}, {}
        self.publish('gpu_gate', {'outputs': [self.proof]})
        for task in self.selected.values():
            if task['kind'] == 'train': self.train(task)
            elif task['kind'] == 'evaluate': self.evaluate(task)
            elif task['kind'] == 'archive': self.archive(task)
            else: self.cleanup(task)
        target_profile = {**deepcopy(self.profile), 'schema_version': 3, 'fleet_id': 'new-fixture',
            'scratch': {**scratch.DEFAULT_SCRATCH, 'mount_path': str(self.root / 'scratch')}}
        target_profile['workers'][0]['job_name'] = 'jinhaopeng-dpvlm-new-fixture'
        storage = {'schema_version': 2, 'roots': [str(self.root)], 'ledger_root': str(self.root / 'ledger'),
            'oss_prefixes': [self.archive_root + '/'], 'home_limit_bytes': 150_000_000_000,
            'oss_limit_bytes': None, 'home_safety_bytes': 1000, 'oss_safety_bytes': 0, 'free_guard_bytes': 0}
        self.target_binding = {'suite_id': 'new-fixture', 'source_files': [self.source],
                               'execution_profile': target_profile, 'storage': storage}
        self.target_identity = self.make_identity(self.target_binding, runner.digest(self.target_binding))
        self.target_env = {'execution_profile': target_profile, 'storage': storage}
        self.scheduler = fleet.FleetState(self.plan, self.target_binding, self.target,
            plan_sha256=self.plan_pin['sha256'], binding_sha256=self.target_identity['binding_sha256'])
        self.scheduler.initialize()
        self.terminal()
        self.request = {'schema_version': 1, 'chain_id': self.chain, 'source_state_root': str(self.state),
            'source_plan': self.plan_pin, 'source_binding': self.binding_pin, 'source_identity': self.identity_pin,
            'terminal_proof': self.terminal_pin, 'tasks': self.task_pins, 'histories': self.histories}
        self.refresh_review()

    def pin(self, name, value):
        path = self.root / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value)); return runner.artifact(path)

    def prefix(self, run_id): return self.archive_root + '/runs/' + run_id + '/'

    def make_identity(self, binding, binding_sha):
        return {'schema_version': 2, 'suite_id': binding['suite_id'], 'plan_sha256': self.plan_pin['sha256'],
            'binding_sha256': binding_sha, 'source_identity_sha256': runner.source_identity(binding),
            'execution_profile_sha256': runner.digest(binding['execution_profile'])}

    def publish(self, tid, result):
        task = self.tasks[tid]; result = {'schema_version': 1, 'status': 'SUITE_TASK_ACCEPTED',
            'task_id': tid, 'task_sha256': runner.digest(task), **result}
        action = self.pin('old-state/actions/' + tid + '.json', result)
        deps = [runner.artifact(self.state / 'tasks' / dep / 'COMPLETED.json') for dep in task['dependencies']]
        attempt = self.state / 'tasks' / tid / 'attempts/attempt-0001'
        claim = {'task_id': tid, 'task_sha256': runner.digest(task), 'attempt': str(attempt), 'owner': {
            'fleet_id': 'old-fixture', 'task_id': tid, 'claim_id': 'claim-' + tid,
            'worker_id': 'worker-01', 'job_uid': 'job-01'}}
        self.pin('old-state/tasks/' + tid + '/CLAIM.json', claim)
        intent = self.pin(str(attempt.relative_to(self.root) / 'TASK_INTENT.json'), {
            'task_id': tid, 'kind': task['kind'], 'plan_sha256': self.plan_pin['sha256'],
            'binding_sha256': self.binding_pin['sha256'], 'dependencies': deps, 'claim': claim})
        self.pin(str(attempt.relative_to(self.root) / 'COMMAND_EXIT.json'),
                 {'returncode': 0, 'group_cleanup_verified': True})
        completion = self.pin('old-state/tasks/' + tid + '/COMPLETED.json', {
            'status': 'TASK_ACCEPTED', 'task_id': tid, 'kind': task['kind'],
            'plan_sha256': self.plan_pin['sha256'], 'binding_sha256': self.binding_pin['sha256'],
            'receipts': [action], 'attempt': str(attempt), 'mode': 'execute',
            'dependencies': deps, 'task_intent_sha256': intent['sha256'], 'claim_id': claim['owner']['claim_id']})
        self.results[tid] = result
        if tid != 'gpu_gate': self.task_pins[tid] = {'completion': completion, 'receipt': action}

    def train(self, task):
        tid = task['id']; owner = self.worker.owner(tid); final = Path(owner['run_root']) / 'training/final'
        for name in ('deepprobe_config.json', 'training_provenance.json', 'projector.safetensors',
            'language_model/config.json', 'language_model/model.safetensors', 'vision_model/config.json',
            'vision_model/model.safetensors', 'tokenizer/tokenizer_config.json', 'image_processor/preprocessor_config.json'):
            self.pin(str((final / name).relative_to(self.root)), {'synthetic': tid, 'name': name})
        files = actions.inventory(final)
        complete = self.pin('old-state/training/' + tid + '.json', {'schema_version': 1,
            'status': 'SUITE_TRAINING_COMPLETE', 'run_id': owner['run_id'],
            'source_sha256': self.identity['source_identity_sha256'], 'optimizer_steps': 718,
            'target_optimizer_steps': 718, 'exit_code': 0, 'outputs': [self.proof],
            'output_inventory_sha256': runner.digest([self.proof]),
            'final_checkpoint_inventory_sha256': runner.digest([{**r, 'path': 'training/final/' + r['path']} for r in files])})
        config = runner.artifact(Path(self.env['experiment_root']) / task['config']['path'])
        initializer = task['initializer'].get('checkpoint_task_id')
        entry = {'mode': 'accepted', 'config': config, 'frozen_source_manifest': self.frozen,
            'source_root': str(self.root), 'checkpoint_step': 718,
            'checkpoint': {'path': str(final), 'files': files, 'oss_prefix': self.prefix(owner['run_id']) + 'training/final/'}}
        facts = {'source_root': entry['source_root'], 'config_sha256': config['sha256'],
            'frozen_source_manifest_sha256': self.frozen['sha256'], 'checkpoint_inventory_sha256': runner.digest(files),
            'checkpoint_oss_prefix': entry['checkpoint']['oss_prefix'], 'optimizer_step': 718,
            'archive_verified': True, 'formal_training_complete': True}
        if initializer: facts['initializer_checkpoint_inventory_sha256'] = runner.digest(self.results[initializer]['checkpoint_files'])
        self.history(task, entry, facts)
        consumer = [self.worker.consume(initializer, task, [complete])] if initializer else []
        self.publish(tid, {'owner': owner, 'checkpoint': str(final), 'checkpoint_files': files,
            'training_complete': complete, 'outputs': [complete], 'consumer_receipts': consumer})

    def history(self, task, entry, facts):
        review = self.pin('reviews/' + task['id'] + '.json', {
            'status': 'SUITE_HISTORY_IMPORT_REVIEW_ACCEPTED', 'approved': True, 'production_evidence': True,
            'task_id': task['id'], 'task_sha256': runner.digest(task), 'entry_sha256': runner.digest(entry),
            'evidence': [self.proof], 'verified_facts': facts})
        self.histories[task['id']] = bindings.import_history_entry(self.plan, task['id'], entry, review,
            self.root / 'histories' / (task['id'] + '.json'))

    def evaluate(self, task):
        qa = 2697 if task['benchmark'] == 'videomme899' else 1580
        result = self.pin('evaluations/' + task['id'] + '.json', {'status': 'SUITE_EVALUATION_COMPLETE',
            'suite_protocol': task['protocol'], 'task_sha256': runner.digest(task), 'qa': qa, 'failed_units': 0,
            'rows': [{'sample_id': str(i), 'correct': i % 2, 'question_identity_sha256': runner.digest(i)} for i in range(qa)]})
        checksum = runner.digest(self.results[task['checkpoint_task_id']]['checkpoint_files'])
        self.history(task, {'mode': 'accepted', 'evaluation': result, 'checkpoint_inventory_sha256': checksum},
            {'evaluation_sha256': result['sha256'], 'checkpoint_inventory_sha256': checksum,
             'protocol': task['protocol'], 'qa': qa, 'failed_units': 0})
        self.publish(task['id'], {'evaluation': result, 'outputs': [result],
            'consumer_receipts': [self.worker.consume(task['checkpoint_task_id'], task, [result])]})

    def archive(self, task):
        if task['archive_scope'] == 'results':
            self.publish(task['id'], self.worker.archive_results(task)); return
        trained = self.results[task['checkpoint_task_id']]; owner = trained['owner']
        guard = lambda: {'run_id': owner['run_id'], 'source_sha256': owner['source_sha256'], 'active_writers': []}
        manifest = retention.prepare_manifest(owner, archive_prefix=self.prefix(owner['run_id']),
            required_consumers=self.worker.requirements(task['checkpoint_task_id']),
            training_completion=trained['training_complete'], guard=guard)
        verified = retention.archive_and_verify(manifest, self.store, guard=guard)
        manifest_pin = runner.artifact(Path(owner['run_root']) / '.retention/MANIFEST.json')
        self.publish(task['id'], {'owner': owner, 'manifest': manifest_pin, 'outputs': [manifest_pin, verified['receipt']]})

    def cleanup(self, task):
        archive = self.results[task['checkpoint_task_id'].replace('train__', 'archive__', 1)]
        manifest = runner.read(archive['manifest']['path']); owner = manifest['owner']
        consumers = {tid: self.results[tid]['consumer_receipts'][0] for tid in task['required_consumers']}
        plan = retention.plan_cleanup(manifest, consumers)
        deleted = retention.cleanup(plan, self.store, execute=True, guard=lambda: {
            'run_id': owner['run_id'], 'source_sha256': owner['source_sha256'], 'active_writers': []})
        output = self.pin('deletions/' + task['id'] + '.json', deleted)
        self.publish(task['id'], {'outputs': [output]})

    def terminal(self):
        from test_suite_scratch import job_document, pod_spec
        job = job_document(self.profile, pod_spec(self.profile), self.binding_pin['sha256'])
        job['status'] = {'conditions': [{'type': 'Succeeded', 'status': 'True'}]}
        reg = self.pin('old-state/workers/worker-01/JOB_OBSERVED.json', {'worker_id': 'worker-01',
            'job_uid': job['metadata']['uid'], 'pod_uid': 'old-pod-01', 'node': 'synthetic-node', 'utc': 'synthetic'})
        closure = self.pin('process-closure.json', {'schema_version': 1,
            'status': 'SUITE_WORKER_PROCESSES_QUIESCENT', 'worker_id': 'worker-01',
            'job_uid': job['metadata']['uid'], 'pod_uid': 'old-pod-01', 'controller_exited': True,
            'active_writers': [], 'active_uploads': [], 'evidence': [self.proof]})
        self.workloads = self.pin('old-workloads.json', {'items': [job]})
        self.terminal_pin = self.pin('terminal-proof.json', {'schema_version': 1,
            'status': 'SUITE_FLEET_TERMINAL_QUIESCENT', 'source_identity': self.identity,
            'complete_project_workload_snapshot': True, 'automatic_requeue_disabled': True,
            'workloads': self.workloads, 'worker_registrations': [reg], 'process_closures': [closure],
            'active_writers': [], 'active_uploads': []})

    def refresh_review(self):
        evidence = [self.request[k] for k in ('source_plan', 'source_binding', 'source_identity', 'terminal_proof')]
        evidence += [pin for value in self.request['tasks'].values() for pin in value.values()]
        self.review = self.pin('external-import-review.json', {'schema_version': 1,
            'status': 'SUITE_TASK_IMPORT_REVIEW_ACCEPTED', 'approved': True, 'production_evidence': True,
            'request_sha256': runner.digest(self.request), 'target_identity': self.target_identity,
            'chain_id': self.chain, 'task_ids': sorted(self.selected), 'closed_checkpoint_chain': True,
            'new_run_materialized': False, 'new_deletion_authorized': False, 'evidence': evidence})

    def verify(self): return imports.verify_closed_group(self.plan, self.target_identity, self.target_env, self.request, self.review)
    def publish_import(self): return imports.import_closed_group(self.plan, self.target_identity, self.target_env, self.request, self.review, self.target)


@pytest.fixture
def closed(tmp_path): return ClosedFixture(tmp_path)


def test_real_plan_closed_chain_import_keeps_gate_and_no_fake_journals(closed):
    assert len(closed.verify()['task_ids']) == 9
    closed.publish_import()
    first = runner.completion_for(closed.target, closed.chain)
    assert first['status'] == imports.STATUS and not first['new_deletion_authorized']
    assert first['plan_sha256'] == closed.target_identity['plan_sha256']
    assert first['binding_sha256'] == closed.target_identity['binding_sha256']
    assert not list((closed.target / 'tasks').glob('*/COMPLETED.json'))
    assert not list((closed.target / 'tasks').glob('*/CLAIM.json'))
    with mock.patch.object(runner, 'machine_environment', return_value=closed.target_env):
        assert set(closed.scheduler.completions()) == set(closed.selected)
        claim = closed.scheduler.claim('worker-01', 'new-job', lambda task, owner: 'synthetic-token')
    assert claim['task']['id'] == 'gpu_gate'
    assert not (closed.target / 'tasks' / closed.chain).exists()
    assert not (closed.target / 'AFFINITY.json').exists()
    assert not (closed.root / 'scratch').exists()


def test_partial_group_rejected_before_publication(closed):
    del closed.request['tasks']['cleanup__ouro_r1_s29__sft']; closed.refresh_review()
    with pytest.raises(ValueError, match='Every task'): closed.publish_import()
    assert not (closed.target / 'task-imports').exists()


def test_tampered_history_evidence_rejected(closed):
    Path(closed.proof['path']).write_text('tampered')
    with pytest.raises(ValueError): closed.publish_import()


def test_unfinished_attempt_prevents_import(closed):
    attempt = closed.target / 'tasks' / closed.chain / 'attempts/attempt-0001'
    attempt.mkdir(parents=True); (attempt / 'unfinished.txt').write_text('preserve')
    with pytest.raises(ValueError, match='claim or execution'): closed.publish_import()
    assert (attempt / 'unfinished.txt').read_text() == 'preserve'


def test_nonterminal_exact_job_rejected_even_with_new_review(closed):
    census = runner.read(closed.workloads['path']); census['items'][0]['status'] = {}
    workloads = closed.pin('old-workloads-running.json', census)
    terminal = runner.read(closed.terminal_pin['path']); terminal['workloads'] = workloads
    closed.request['terminal_proof'] = closed.pin('terminal-running.json', terminal); closed.refresh_review()
    with pytest.raises(ValueError, match='not terminal'): closed.publish_import()


def test_old_owner_receipt_does_not_authorize_new_run_cleanup(closed):
    closed.publish_import()
    imported = imports.get_task_import(closed.target, 'cleanup__ouro_r1_s29__sft')
    old_delete = runner.read(runner.read(imported['receipts'][0]['path'])['outputs'][0]['path'])
    new_runs = closed.root / 'new-runs'; new_runs.mkdir()
    new_owner = retention.claim_owned_run(new_runs, 'new-owned-run', 'new-fixture', 'a' * 64)
    old_manifest = runner.read(closed.results['archive__ouro_r1_s29__sft']['manifest']['path'])
    old_manifest['owner'] = new_owner
    with pytest.raises(ValueError): retention.validate_manifest(old_manifest)
    assert old_delete['suite_id'] == 'old-fixture'
    assert imported['historical_facts_only'] and not imported['new_run_materialized']


def test_repointed_typed_receipt_cannot_cross_task_boundary(closed):
    closed.publish_import()
    path = closed.target / 'task-imports' / (closed.chain + '.json')
    record = runner.read(path)
    record['original_receipt'] = closed.task_pins['train__ouro_r1_s29__sft']['receipt']
    path.write_text(json.dumps(record))
    ready_path = Path(record['group']['path']).parent / 'READY.json'; ready = runner.read(ready_path)
    ready['records'] = [runner.artifact(path) if pin['path'] == str(path) else pin for pin in ready['records']]
    ready_path.write_text(json.dumps(ready))
    with pytest.raises(ValueError, match='exact reviewed'): imports.get_task_import(closed.target, closed.chain)


def test_partial_publication_cannot_silently_fall_back_to_execution(closed):
    closed.publish_import()
    (closed.target / 'task-imports' / (closed.chain + '.json')).unlink()
    with pytest.raises(ValueError, match='lacks a required typed'): imports.get_task_import(closed.target, closed.chain)


@pytest.mark.parametrize('missing', ['all', 'half'])
def test_interrupted_publication_blocks_fleet_claim_without_new_journal(closed, missing):
    closed.publish_import()
    files = sorted((closed.target / 'task-imports').glob('*.json'))
    for path in files if missing == 'all' else files[:len(files) // 2]: path.unlink()
    with pytest.raises(ValueError): imports.verify_import_publications(closed.target)
    with mock.patch.object(runner, 'machine_environment', return_value=closed.target_env):
        with pytest.raises(ValueError):
            closed.scheduler.claim('worker-01', 'new-job', lambda task, owner: 'synthetic-token')
    assert not list((closed.target / 'tasks').glob('*/CLAIM.json'))


@pytest.mark.parametrize('change', ['record-path', 'record-task', 'group-inventory', 'claim-conflict'])
def test_typed_publication_cannot_be_rewritten_with_fresh_local_pins(closed, change):
    closed.publish_import()
    path = closed.target / 'task-imports' / (closed.chain + '.json')
    record = runner.read(path); group_path = Path(record['group']['path'])
    ready_path = group_path.parent / 'READY.json'; ready = runner.read(ready_path)
    if change == 'claim-conflict':
        claim = closed.target / 'tasks' / closed.chain / 'CLAIM.json'
        claim.parent.mkdir(parents=True); claim.write_text('{}')
    elif change == 'group-inventory':
        group = runner.read(group_path); group['task_ids'] = [closed.chain]
        group_path.write_text(json.dumps(group)); record['group'] = runner.artifact(group_path)
        path.write_text(json.dumps(record)); ready['group'] = record['group']; ready['records'] = [runner.artifact(path)]
    elif change == 'record-path':
        other = closed.root / (closed.chain + '.json'); other.write_text(path.read_text())
        ready['records'] = [runner.artifact(other) if pin['path'] == str(path) else pin for pin in ready['records']]
    else:
        record['task_sha256'] = 'f' * 64; path.write_text(json.dumps(record))
        ready['records'] = [runner.artifact(path) if pin['path'] == str(path) else pin for pin in ready['records']]
    ready_path.write_text(json.dumps(ready))
    with pytest.raises(ValueError): imports.get_task_import(closed.target, closed.chain)


def test_missing_or_false_review_never_imports(closed):
    review = runner.read(closed.review['path']); review['production_evidence'] = False
    closed.review = closed.pin('nonproduction-review.json', review)
    with pytest.raises(ValueError, match='independent closed-chain review'): closed.publish_import()
    assert not (closed.target / 'task-imports').exists()


def test_repeated_same_import_is_idempotent(closed):
    first = closed.publish_import()
    assert closed.publish_import() == first
    assert len(list((closed.target / 'task-imports').glob('*.json'))) == 9


@pytest.mark.parametrize('change', ['missing', 'claim-id', 'worker-job', 'intent-claim'])
def test_original_durable_claim_must_bind_accepted_execution(closed, change):
    claim_path = closed.state / 'tasks' / closed.chain / 'CLAIM.json'
    claim = runner.read(claim_path)
    if change == 'missing': claim_path.unlink()
    else:
        if change == 'claim-id': claim['owner']['claim_id'] = 'different'
        elif change == 'worker-job': claim['owner']['job_uid'] = 'another-job'
        else: claim['extra_unbound_field'] = True
        claim_path.write_text(json.dumps(claim))
    with pytest.raises((ValueError, FileNotFoundError)): closed.publish_import()


def test_latency_groups_are_explicitly_unsupported(closed):
    with pytest.raises(ValueError, match='does not support'): imports._group(closed.plan, 'train__ouro_r1_s17__alignment')
