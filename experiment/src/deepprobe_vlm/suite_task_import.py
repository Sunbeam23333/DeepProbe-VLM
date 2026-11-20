"""Import complete historical checkpoint chains without fabricating executions.

Only train/evaluate/archive/cleanup groups are supported. Groups containing
latency work are rejected until their complete paired evidence has a dedicated
import verifier. Gates and aggregates always execute in the new suite.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from . import checkpoint_retention as retention, suite_bindings as bindings, suite_runner as runner
from .suite_task_import_terminal import verify_terminal as _verify_terminal


STATUS = 'SUITE_ACCEPTED_TASK_IMPORTED'
KINDS = frozenset({'train', 'evaluate', 'archive', 'cleanup'})


def equal(value, expected, message):
    runner.require(isinstance(value, dict) and all(type(value.get(k)) is type(v) and value[k] == v
                                                  for k, v in expected.items()), message)


def read_pin(pin):
    bindings._pin_shape(pin)
    runner.verify_pin(pin)
    return runner.read(pin['path'])


def _outputs(value):
    outputs = value.get('outputs')
    runner.require(isinstance(outputs, list) and outputs and len({p['path'] for p in outputs}) == len(outputs),
                   'Historical task requires complete unique output pins')
    if 'output_inventory_sha256' in value:
        runner.require(value['output_inventory_sha256'] == runner.digest(outputs), 'Historical output inventory differs')
    for pin in outputs:
        bindings._pin_shape(pin); runner.verify_pin(pin)
        if Path(pin['path']).suffix == '.json':
            nested = runner.read(pin['path'])
            if isinstance(nested, dict) and 'outputs' in nested:
                _outputs(nested)
    return outputs


def _scope(environment):
    return {'execution_profile': deepcopy(environment['execution_profile']),
            'storage': {k: deepcopy(environment['storage'][k]) for k in ('roots', 'oss_prefixes')}}


def _group(plan, chain_id):
    from .suite_chains import training_components
    tasks = {task['id']: task for task in plan['tasks']}
    matches = [group for group in training_components(plan) if group['chain_id'] == chain_id]
    runner.require(len(matches) == 1, 'Task import requires one declared checkpoint chain')
    group = matches[0]
    models = {tasks[tid]['model_id'] for tid in group['training_task_ids']}
    selected = {tid: task for tid, task in tasks.items() if task.get('model_id') in models
                and task['kind'] not in {'gate', 'aggregate'}}
    unsupported = sorted({task['kind'] for task in selected.values()} - KINDS)
    runner.require(not unsupported, 'Closed-group import does not support these kinds: ' + str(unsupported))
    return group, selected


def _old_tasks(plan, request, selected, source_plan, source_binding, identity):
    runner.require(set(request['tasks']) == set(selected), 'Every task in the checkpoint chain must be explicitly imported')
    state, tasks = runner.safe(request['source_state_root']), {task['id']: task for task in plan['tasks']}
    results, completions = {}, {}
    for tid, task in selected.items():
        supplied = request['tasks'][tid]
        runner.require(set(supplied) == {'completion', 'receipt'}, 'Each import needs exact original completion and action pins')
        runner.require(supplied['completion']['path'] == str(state / 'tasks' / tid / 'COMPLETED.json'),
                       'Import completion is outside its original task state')
        complete = read_pin(supplied['completion']); result = read_pin(supplied['receipt'])
        equal(complete, {'status': 'TASK_ACCEPTED', 'task_id': tid, 'kind': task['kind'],
            'plan_sha256': source_plan['sha256'], 'binding_sha256': source_binding['sha256'],
            'receipts': [supplied['receipt']]}, 'Original completion identity differs')
        equal(result, {'schema_version': 1, 'status': 'SUITE_TASK_ACCEPTED', 'task_id': tid,
                       'task_sha256': runner.digest(task)}, 'Original action does not accept this exact task')
        runner.require(not (state / 'tasks' / tid / 'FAILED.json').exists(), 'Original task has contradictory failure evidence')
        attempt = runner.safe(complete['attempt'])
        runner.require(attempt.is_relative_to(state / 'tasks' / tid / 'attempts'), 'Original attempt path differs')
        claim = runner.read(state / 'tasks' / tid / 'CLAIM.json')
        equal(claim, {'task_id': tid, 'task_sha256': runner.digest(task), 'attempt': str(attempt)},
              'Original claim belongs to another task or attempt')
        owner = claim['owner']
        equal(owner, {'fleet_id': identity['suite_id'], 'task_id': tid, 'claim_id': complete['claim_id']},
              'Original claim identity differs from accepted completion')
        runner.require(isinstance(owner.get('worker_id'), str) and runner.ID.fullmatch(owner['worker_id']),
                       'Original claim has an unsafe worker identity')
        registration = runner.read(state / 'workers' / owner['worker_id'] / 'JOB_OBSERVED.json')
        equal(registration, {'worker_id': owner['worker_id'], 'job_uid': owner['job_uid']},
              'Original task claim belongs to another worker Job')
        intent = runner.read(attempt / 'TASK_INTENT.json')
        runner.require(runner.sha(attempt / 'TASK_INTENT.json') == complete['task_intent_sha256'], 'Original task intent changed')
        equal(intent, {'task_id': tid, 'kind': task['kind'], 'plan_sha256': source_plan['sha256'],
            'binding_sha256': source_binding['sha256'], 'dependencies': complete['dependencies']}, 'Original intent identity differs')
        if 'claim' in intent:
            runner.require(intent['claim'] == claim, 'Original task intent and durable claim differ')
        deps = {}
        for pin in complete['dependencies']:
            value = read_pin(pin); dep = value['task_id']
            runner.require(dep in tasks and dep not in deps and pin['path'] == str(state / 'tasks' / dep / 'COMPLETED.json'),
                           'Original dependency provenance differs')
            equal(value, {'status': 'TASK_ACCEPTED', 'plan_sha256': source_plan['sha256'],
                          'binding_sha256': source_binding['sha256']}, 'Original dependency belongs to another suite')
            deps[dep] = pin
        runner.require(set(deps) == set(task['dependencies']), 'Original dependency closure is incomplete')
        if complete.get('mode') == 'execute':
            exited = runner.read(attempt / 'COMMAND_EXIT.json')
            equal(exited, {'returncode': 0, 'group_cleanup_verified': True}, 'Original task child closure is missing')
        else:
            runner.require(complete.get('mode') == 'reuse', 'Unknown original completion mode')
        _outputs(result)
        for pin in result.get('consumer_receipts', []):
            _outputs(read_pin(pin))
        results[tid], completions[tid] = result, complete
    return results, completions


def _required_consumers(plan, tid):
    from .experiment_suite import checkpoint_consumers
    tasks = {task['id']: task for task in plan['tasks']}
    required = []
    for cid in checkpoint_consumers(plan, tid):
        task = tasks[cid]; kind = {'train': 'training', 'evaluate': 'evaluation', 'latency': 'latency'}[task['kind']]
        units = task['expected_optimizer_steps'] if kind == 'training' else (
            416 if kind == 'latency' else 2697 if task['benchmark'] == 'videomme899' else 1580)
        required.append({'id': cid, 'kind': kind, 'checkpoint': 'training/final',
                         'protocol_sha256': runner.digest(task), 'expected_units': units})
    # prepare_manifest normalizes the action's plan-order requirements.
    return sorted(required, key=lambda row: row['id'])


def _typed_evidence(plan, request, selected, results, identity, environment):
    histories = request['histories']
    expected_history = {tid for tid, task in selected.items() if task['kind'] in {'train', 'evaluate'}}
    runner.require(set(histories) == expected_history, 'Imported training/evaluation need their independently accepted histories')
    for tid in expected_history:
        history, task, result = histories[tid], selected[tid], results[tid]
        runner.require(history['mode'] == 'accepted', 'A partial resume is not a completed task import')
        bindings.verify_history_binding(plan, task, history)
        if task['kind'] == 'train':
            runner.require(history['checkpoint']['files'] == result['checkpoint_files'], 'Imported final differs from accepted history')
            complete = read_pin(result['training_complete'])
            equal(complete, {'schema_version': 1, 'status': 'SUITE_TRAINING_COMPLETE',
                'run_id': result['owner']['run_id'], 'source_sha256': identity['source_identity_sha256'],
                'optimizer_steps': task['expected_optimizer_steps'], 'target_optimizer_steps': task['expected_optimizer_steps'],
                'exit_code': 0, 'final_checkpoint_inventory_sha256': runner.digest([
                    {**row, 'path': 'training/final/' + row['path']} for row in result['checkpoint_files']])},
                'Imported training completion differs from its original final')
            equal(result['owner'], {'suite_id': identity['suite_id'],
                'run_id': identity['suite_id'] + '--' + tid,
                'source_sha256': identity['source_identity_sha256']}, 'Old run executor owner differs')
            runner.require(result['checkpoint'] == str(runner.safe(result['owner']['run_root']) / 'training/final'),
                           'Old final path differs from its original owner')
            initializer = task['initializer'].get('checkpoint_task_id')
            if initializer:
                acceptance = read_pin(history['acceptance'])
                runner.require(acceptance.get('initializer_checkpoint_inventory_sha256')
                    == runner.digest(results[initializer]['checkpoint_files']),
                    'Historical training does not bind its declared matching initializer')
            _outputs(complete)
        else:
            runner.require(history['evaluation'] == result['evaluation'], 'Imported evaluation differs from accepted history')
            parent = results[task['checkpoint_task_id']]
            runner.require(history['checkpoint_inventory_sha256'] == runner.digest(parent['checkpoint_files']),
                           'Imported evaluation uses another final checkpoint')
            evaluation = read_pin(result['evaluation'])
            for name in ('provenance', 'data_acceptance'):
                if name in evaluation:
                    bindings._pin_shape(evaluation[name]); runner.verify_pin(evaluation[name])
    manifests = {}
    for tid, task in selected.items():
        if task['kind'] != 'archive' or task['archive_scope'] != 'checkpoint':
            continue
        result, train = results[tid], results[task['checkpoint_task_id']]
        manifest = retention.validate_manifest(read_pin(result['manifest']))
        runner.require(manifest['owner'] == result['owner'] == train['owner']
                       and manifest['training_completion'] == train['training_complete']
                       and manifest['required_consumers'] == _required_consumers(plan, task['checkpoint_task_id']),
                       'Historical archive owner/training/consumer closure differs')
        runner.require(any(manifest['archive_prefix'].startswith(prefix) for prefix in environment['storage']['oss_prefixes']),
                       'Historical archive escapes approved OSS prefixes')
        runner.require(manifest['checkpoints']['training/final'] == runner.digest([
            {**row, 'path': 'training/final/' + row['path']} for row in train['checkpoint_files']]),
            'Archived final differs from original training inventory')
        retention.verify_training_completion(manifest)
        proofs = [read_pin(pin) for pin in result['outputs'] if Path(pin['path']).suffix == '.json']
        expected = retention._objects(manifest)
        runner.require(any(value.get('status') == 'FULL_ARCHIVE_GET_SHA256_VERIFIED'
            and value.get('manifest_sha256') == runner.digest(manifest)
            and value.get('archive_prefix') == manifest['archive_prefix']
            and value.get('objects') == len(expected) and value.get('bytes') == sum(row['bytes'] for row in expected.values())
            and value.get('all_objects_full_get_sha256') is True for value in proofs),
            'Historical archive has no exact complete full-GET acceptance')
        manifests[task['checkpoint_task_id']] = manifest
    runner.require(set(manifests) == {tid for tid, task in selected.items() if task['kind'] == 'train'},
                   'Imported training requires every original checkpoint archive')
    for tid, task in selected.items():
        if task['kind'] == 'cleanup':
            _cleanup_evidence(task, results[tid], manifests[task['checkpoint_task_id']], results)
        elif task['kind'] == 'archive' and task['archive_scope'] == 'results':
            _result_archive_evidence(task, results[tid], results, identity, environment)


def _cleanup_evidence(task, result, manifest, results):
    consumers = {}
    for cid in task['required_consumers']:
        matches = [pin for pin in results[cid].get('consumer_receipts', [])
                   if read_pin(pin).get('consumer_id') == cid and read_pin(pin).get('checkpoint_inventory_sha256')
                   == manifest['checkpoints']['training/final']]
        runner.require(len(matches) == 1, 'Historical cleanup lacks an exact original consumer')
        consumers[cid] = matches[0]
    retention.verify_consumers(manifest, consumers)
    targets = [row for row in manifest['files'] if retention.cleanable(row['path'])]
    plan = {'schema_version': 1, 'status': 'CLEANUP_PLAN_REQUIRES_FRESH_ARCHIVE_VERIFICATION',
        'manifest': manifest, 'manifest_sha256': runner.digest(manifest), 'checkpoints': sorted(manifest['checkpoints']),
        'consumer_receipts': consumers, 'targets': targets, 'reclaim_bytes': sum(row['bytes'] for row in targets)}
    candidates = []
    for pin in result['outputs']:
        value = read_pin(pin)
        if value.get('status') == 'OWNED_CHECKPOINT_WEIGHTS_REMOVED':
            runner.require('receipt' in value, 'Historical deletion lacks its original transaction receipt')
            original = read_pin(value['receipt'])
            runner.require(original == {k: v for k, v in value.items() if k != 'receipt'}, 'Historical deletion wrapper changed')
            candidates.append((value['receipt'], original))
    runner.require(len(candidates) == 1, 'Historical cleanup requires one complete original deletion transaction')
    pin, deletion = candidates[0]
    equal(deletion, {'schema_version': 1, 'status': 'OWNED_CHECKPOINT_WEIGHTS_REMOVED',
        'suite_id': manifest['owner']['suite_id'], 'run_id': manifest['owner']['run_id'],
        'plan_sha256': runner.digest(plan), 'manifest_sha256': runner.digest(manifest), 'files': targets,
        'reclaimed_target_bytes': plan['reclaim_bytes'], 'non_weight_files_preserved': True, 'consumer_receipts': consumers},
        'Historical cleanup does not prove this old owner and complete consumer set')
    expected = retention._objects(manifest)
    equal(deletion['archive_verification'], {'status': 'FULL_ARCHIVE_GET_SHA256_VERIFIED',
        'manifest_sha256': runner.digest(manifest), 'archive_prefix': manifest['archive_prefix'],
        'objects': len(expected), 'bytes': sum(row['bytes'] for row in expected.values()),
        'all_objects_full_get_sha256': True}, 'Historical deletion lacks exact full-GET verification')
    transaction = runner.safe(manifest['owner']['run_root']) / '.retention/cleanup' / runner.digest(plan)
    runner.require(pin['path'] == str(transaction / 'DELETION.json'), 'Historical deletion receipt path differs')
    runner.require(runner.read(transaction / 'PLAN.json') == plan, 'Historical cleanup plan is unavailable or changed')
    for row in targets:
        runner.require(not (runner.safe(manifest['owner']['run_root']) / row['path']).exists(), 'Historically deleted weight reappeared')
        runner.require(runner.read(transaction / (runner.digest(row) + '.done.json')) == row, 'Historical deletion completion differs')
        runner.require(retention._recorded_deletion(runner.safe(manifest['owner']['run_root']), row, manifest),
                       'Historical deletion has no matching durable intent')


def _result_archive_evidence(task, result, results, identity, environment):
    runner.require(result.get('result_tasks') == task['result_task_ids'], 'Historical result archive task set differs')
    runner.require(len(result['outputs']) == 1, 'Historical result archive receipt is ambiguous')
    archive = read_pin(result['outputs'][0])
    equal(archive, {'status': 'SUITE_RESULTS_ARCHIVED_FULL_GET_VERIFIED'}, 'Historical results were not fully archived')
    manifest = read_pin(archive['manifest'])
    rows = manifest.get('files')
    runner.require(manifest.get('schema_version') == 1 and isinstance(rows, list) and rows,
                   'Historical result archive has no exact file inventory')
    suffix = '/runs/' + identity['suite_id'] + '--' + task['id'] + '/'
    first = rows[0].get('object', '')
    runner.require(first.count(suffix) == 1, 'Historical result archive has another owner prefix')
    prefix = first.split(suffix)[0] + suffix
    retention.check_prefix(prefix, identity['suite_id'] + '--' + task['id'])
    runner.require(any(prefix.startswith(root) for root in environment['storage']['oss_prefixes']),
                   'Historical result archive escapes approved OSS prefixes')
    expected = [{**pin, 'object': prefix + tid + '/' + str(index) + '-' + Path(pin['path']).name}
                for tid in task['result_task_ids'] for index, pin in enumerate(results[tid]['outputs'])]
    runner.require(rows == expected and archive['files'] == [*expected, {**archive['manifest'],
                   'object': prefix + 'RESULT_MANIFEST.json'}],
                   'Historical result archive does not cover all exact task outputs and manifest')
    for row in expected:
        runner.verify_pin({key: row[key] for key in ('path', 'bytes', 'sha256')})

def verify_closed_group(plan, target_identity, environment, request, review_pin):
    from .suite_fleet import validate_execution_profile
    runner.require(isinstance(request, dict) and set(request) == {'schema_version', 'chain_id', 'source_state_root',
        'source_plan', 'source_binding', 'source_identity', 'terminal_proof', 'tasks', 'histories'},
        'Closed-group import request fields differ')
    runner.require(request.get('schema_version') == 1 and environment['execution_profile'].get('schema_version') == 3,
                   'Typed task import requires an explicit schema-3 target')
    validate_execution_profile(environment['execution_profile'], target_identity['suite_id'])
    runner.require(target_identity['execution_profile_sha256'] == runner.digest(environment['execution_profile']),
                   'Target execution profile differs from its identity')
    group, selected = _group(plan, request['chain_id'])
    old_plan, binding, identity = (read_pin(request[name]) for name in ('source_plan', 'source_binding', 'source_identity'))
    runner.require(old_plan == plan and request['source_plan']['sha256'] == target_identity['plan_sha256'],
                   'Typed task import must preserve the exact original scientific plan bytes')
    equal(identity, {'schema_version': 2, 'suite_id': binding['suite_id'],
        'plan_sha256': request['source_plan']['sha256'], 'binding_sha256': request['source_binding']['sha256'],
        'source_identity_sha256': runner.source_identity(binding),
        'execution_profile_sha256': runner.digest(binding['execution_profile'])}, 'Original fleet identity differs')
    runner.require(identity != target_identity and request['source_identity']['path'] == str(
        runner.safe(request['source_state_root']) / 'SUITE_IDENTITY.json'), 'Original fleet state identity is not separately bound')
    for pin in binding['source_files']:
        runner.verify_pin(pin)
    review = read_pin(review_pin)
    equal(review, {'schema_version': 1, 'status': 'SUITE_TASK_IMPORT_REVIEW_ACCEPTED', 'approved': True,
        'production_evidence': True, 'request_sha256': runner.digest(request), 'target_identity': target_identity,
        'chain_id': request['chain_id'], 'task_ids': sorted(selected),
        'closed_checkpoint_chain': True, 'new_run_materialized': False, 'new_deletion_authorized': False},
        'Typed task import lacks its exact independent closed-chain review')
    required = [request[name] for name in ('source_plan', 'source_binding', 'source_identity', 'terminal_proof')]
    required += [row[key] for row in request['tasks'].values() for key in ('completion', 'receipt')]
    runner.require(all(pin in review.get('evidence', []) for pin in required), 'Independent review omits original task evidence')
    for pin in review['evidence']:
        bindings._pin_shape(pin); runner.verify_pin(pin)
    _verify_terminal(request, binding, identity)
    results, completions = _old_tasks(plan, request, selected, request['source_plan'], request['source_binding'], identity)
    _typed_evidence(plan, request, selected, results, identity, environment)
    return {'chain_id': group['chain_id'], 'task_ids': sorted(selected), 'cleanup_task_ids': group['cleanup_task_ids'],
            'source_identity': identity, 'results': results, 'completions': completions,
            'historical_facts_only': True, 'new_run_materialized': False, 'new_deletion_authorized': False}


def import_closed_group(plan, target_identity, environment, request, review_pin, state_root):
    from .suite_actions import save
    from .suite_fleet import project_transaction
    verified = verify_closed_group(plan, target_identity, environment, request, review_pin)
    state = runner.safe(state_root)
    with project_transaction(environment['storage']):
        runner.require(runner.read(state / 'SUITE_IDENTITY.json') == target_identity, 'Target suite identity changed')
        for tid in verified['task_ids']:
            root = state / 'tasks' / tid
            runner.require(not any((root / name).exists() for name in ('CLAIM.json', 'COMPLETED.json', 'FAILED.json', 'attempts')),
                           'Cannot import over any target task claim or execution')
            existing = state / 'task-imports' / (tid + '.json')
            if existing.exists():
                runner.require(runner.read(existing)['group']['path'] == str(
                    state / 'task-import-groups' / runner.digest(request) / 'GROUP.json'),
                    'Task already belongs to another typed import group')
        verify_closed_group(plan, target_identity, environment, request, review_pin)
        group_root = state / 'task-import-groups' / runner.digest(request)
        document = {'schema_version': 1, 'status': 'SUITE_CLOSED_GROUP_IMPORT', 'target_identity': target_identity,
            'request': request, 'review': review_pin, 'environment_scope': _scope(environment),
            'task_ids': verified['task_ids'], 'cleanup_task_ids': verified['cleanup_task_ids']}
        group_pin = save(group_root / 'GROUP.json', document)
        records = []
        for tid in verified['task_ids']:
            records.append(save(state / 'task-imports' / (tid + '.json'), {'schema_version': 1, 'status': STATUS,
                'task_id': tid, 'task_sha256': runner.digest(next(t for t in plan['tasks'] if t['id'] == tid)),
                'target_identity': target_identity, 'group': group_pin,
                'original_completion': request['tasks'][tid]['completion'], 'original_receipt': request['tasks'][tid]['receipt'],
                'historical_facts_only': True, 'new_run_materialized': False, 'new_deletion_authorized': False}))
        return save(group_root / 'READY.json', {'schema_version': 1, 'status': 'SUITE_CLOSED_GROUP_IMPORT_READY',
                    'group': group_pin, 'records': records})


def get_task_import(state_root, task_id, *, plan=None, target_identity=None, environment=None, verify_inputs=True):
    """Resolve typed history without calling completion_for or creating a journal."""
    runner.require(isinstance(task_id, str) and runner.ID.fullmatch(task_id), 'Unsafe imported task id')
    state = runner.safe(state_root); path = state / 'task-imports' / (task_id + '.json')
    if not path.exists():
        for pending in (state / 'task-import-groups').glob('*/GROUP.json'):
            runner.require(task_id not in runner.read(pending).get('task_ids', []),
                           'Closed-group publication lacks a required typed task record')
        return None
    runner.require(not any((state / 'tasks' / task_id / name).exists()
        for name in ('CLAIM.json', 'COMPLETED.json', 'FAILED.json', 'attempts')),
        'Task has both execution and import evidence')
    record_pin = runner.artifact(path); record = read_pin(record_pin); group = read_pin(record['group'])
    target = target_identity or runner.read(state / 'SUITE_IDENTITY.json')
    runner.require(runner.read(state / 'SUITE_IDENTITY.json') == target, 'Target suite identity changed')
    equal(record, {'schema_version': 1, 'status': STATUS, 'task_id': task_id, 'target_identity': target}, 'Typed import identity differs')
    equal(group, {'schema_version': 1, 'status': 'SUITE_CLOSED_GROUP_IMPORT', 'target_identity': target},
          'Typed group target changed')
    runner.require(record['group']['path'] == str(state / 'task-import-groups'
        / runner.digest(group['request']) / 'GROUP.json'), 'Typed group is outside its exact target state path')
    ready = runner.read(Path(record['group']['path']).parent / 'READY.json')
    equal(ready, {'schema_version': 1, 'status': 'SUITE_CLOSED_GROUP_IMPORT_READY', 'group': record['group']},
          'Closed-group import publication is incomplete')
    expected_paths = {str(state / 'task-imports' / (tid + '.json')): tid for tid in group['task_ids']}
    runner.require(len(ready['records']) == len(expected_paths)
                   and {pin['path'] for pin in ready['records']} == set(expected_paths)
                   and record_pin in ready['records'], 'Closed-group import record inventory differs')
    old_plan = read_pin(group['request']['source_plan'])
    tasks = {task['id']: task for task in old_plan['tasks']}
    declared, selected = _group(old_plan, group['request']['chain_id'])
    runner.require(group['task_ids'] == sorted(selected)
        and group['cleanup_task_ids'] == declared['cleanup_task_ids'], 'Typed group closed-chain inventory changed')
    for pin in ready['records']:
        tid = expected_paths[pin['path']]; item = read_pin(pin)
        equal(item, {'schema_version': 1, 'status': STATUS, 'task_id': tid,
            'task_sha256': runner.digest(tasks[tid]), 'target_identity': target, 'group': record['group'],
            'original_completion': group['request']['tasks'][tid]['completion'],
            'original_receipt': group['request']['tasks'][tid]['receipt'],
            'historical_facts_only': True, 'new_run_materialized': False, 'new_deletion_authorized': False},
            'Typed task record is not bound to its exact reviewed group/task/original evidence')
    if environment is not None:
        runner.require(_scope(environment) == group['environment_scope'], 'Typed import target environment scope changed')
    runner.require(plan is None or plan == old_plan, 'Typed import scientific plan changed')
    if verify_inputs:
        verify_closed_group(old_plan, target, environment or group['environment_scope'], group['request'], group['review'])
    original = read_pin(record['original_completion'])
    runner.require(original['receipts'] == [record['original_receipt']], 'Typed original action receipt changed')
    return {'schema_version': 1, 'status': STATUS, 'task_id': task_id, 'kind': original['kind'],
            'plan_sha256': target['plan_sha256'], 'binding_sha256': target['binding_sha256'],
            'receipts': [record['original_receipt']], 'original_completion': original,
            'source_identity': read_pin(group['request']['source_identity']), 'target_identity': target,
            'import_record': record_pin, 'closed_chain_id': group['request']['chain_id'],
            'historical_facts_only': True, 'new_run_materialized': False, 'new_deletion_authorized': False}


def verify_import_publications(state_root):
    """Fail closed on GROUP-only/partial publication before any scheduler claim."""
    state = runner.safe(state_root)
    for path in sorted((state / 'task-import-groups').glob('*/GROUP.json')):
        document = runner.read(path)
        task_ids = document.get('task_ids')
        runner.require(isinstance(task_ids, list) and task_ids, 'Closed-group publication lacks its task set')
        for tid in task_ids:
            value = get_task_import(state, tid, verify_inputs=False)
            runner.require(value is not None, 'Closed-group publication lacks a required typed task record')
    return {'status': 'SUITE_TASK_IMPORT_PUBLICATIONS_COMPLETE'}
