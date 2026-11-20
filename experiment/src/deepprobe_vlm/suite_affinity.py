"""Durable checkpoint-chain affinity for explicitly private worker disks.

Callers hold ``project_transaction`` through selection, reservation and claim.
An assignment never expires and cannot be stolen after node loss. Completed
cleanup barriers free a worker slot, not the historical assignment record.
"""
from __future__ import annotations

from . import suite_runner as runner
from .suite_chains import training_components


def chain_tasks(plan):
    groups = training_components(plan)
    runner.require(groups, 'Private scratch requires a scientific checkpoint DAG')
    tasks = {task['id']: task for task in plan['tasks']}
    model_chains = {}
    for group in groups:
        for tid in group['training_task_ids']:
            model = tasks[tid]['model_id']
            runner.require(model not in model_chains or model_chains[model] == group['chain_id'],
                           'One model cannot span independent private checkpoint chains')
            model_chains[model] = group['chain_id']
    mapping = {}
    for task in plan['tasks']:
        if task['kind'] in {'gate', 'aggregate'}:
            continue
        runner.require(task.get('model_id') in model_chains, 'Checkpoint consumer has no declared chain')
        chain = model_chains[task['model_id']]
        checkpoint = task.get('checkpoint_task_id') or task.get('initializer', {}).get('checkpoint_task_id')
        if checkpoint:
            runner.require(model_chains[tasks[checkpoint]['model_id']] == chain,
                           'Private checkpoint consumer crosses chain affinity')
        mapping[task['id']] = chain
    return groups, mapping


def read_assignments(state, identity, plan, profile):
    groups, mapping = chain_tasks(plan)
    known = {group['chain_id'] for group in groups}
    workers = {row['worker_id'] for row in profile['workers']}
    directory = runner.safe(state) / 'chain-affinity'
    assignments = {}
    if directory.exists():
        for path in sorted(directory.iterdir()):
            runner.require(path.is_file() and path.suffix == '.json', 'Unexpected chain affinity entry')
            value = runner.read(path)
            chain = value.get('chain_id')
            runner.require(set(value) == {'schema_version', 'identity', 'chain_id', 'worker_id', 'job_uid', 'utc'}
                           and value['schema_version'] == 1 and value['identity'] == identity
                           and chain in known and path.name == chain + '.json'
                           and value['worker_id'] in workers and isinstance(value['job_uid'], str)
                           and value['job_uid'] and chain not in assignments,
                           'Private scratch affinity differs from the frozen fleet identity')
            assignments[chain] = value
    return groups, mapping, assignments


def window(state, identity, plan, profile, completed, started, active, worker_id, job_uid):
    """One open component per worker, with its exact archive/cleanup barriers."""
    groups, mapping, assigned = read_assignments(state, identity, plan, profile)
    imported = ({tid for tid, value in completed.items() if value.get('status') == 'SUITE_ACCEPTED_TASK_IMPORTED'}
                if isinstance(completed, dict) else set())
    completed, started = set(completed), set(started)
    pending = {group['chain_id']: group for group in groups
               if not set(group['cleanup_task_ids']) <= completed}
    for tid in started | completed:
        if tid in mapping:
            if tid in imported and tid not in started:
                runner.require(mapping[tid] not in pending,
                               'Imported private task must belong to a fully closed cleanup chain')
                continue
            runner.require(mapping[tid] in assigned, 'Started private task lacks durable chain affinity')
    worker_open = {}
    for chain, value in assigned.items():
        if chain not in pending:
            continue
        owner = value['worker_id']
        runner.require(owner not in worker_open, 'Worker owns multiple resident private chains')
        worker_open[owner] = chain
        if owner == worker_id:
            runner.require(value['job_uid'] == job_uid, 'New Job cannot adopt a lost private scratch chain')
    for task in active:
        if task['id'] in mapping:
            runner.require(mapping[task['id']] in assigned, 'Active task lacks private scratch ownership')
    archives = {task['checkpoint_task_id']: task['id'] for task in plan['tasks']
                if task['kind'] == 'archive' and task.get('archive_scope') == 'checkpoint'}
    train_ids = {tid for group in groups for tid in group['training_task_ids']}
    runner.require(set(archives) == train_ids, 'Every private training stage needs its archive barrier')
    return {'groups': pending, 'mapping': mapping, 'assignments': assigned,
            'worker_open': worker_open, 'archives': archives, 'completed': completed,
            'active': active, 'worker_id': worker_id, 'job_uid': job_uid}


def eligible(task, view):
    chain = view['mapping'].get(task['id'])
    if chain is None:
        return True
    assigned = view['assignments'].get(chain)
    if assigned:
        if assigned['worker_id'] != view['worker_id'] or assigned['job_uid'] != view['job_uid']:
            return False
    elif task['kind'] != 'train' or view['worker_id'] in view['worker_open']:
        return False
    if task['kind'] == 'train':
        group = view['groups'][chain]
        if any(t['kind'] == 'train' and view['mapping'].get(t['id']) == chain for t in view['active']):
            return False
        if any(tid in view['completed'] and view['archives'][tid] not in view['completed']
               for tid in group['training_task_ids']):
            return False
    return True


def assign(state, identity, task, view):
    """After budget admission, before CLAIM; crash leaves a conservative slot."""
    chain = view['mapping'].get(task['id'])
    if chain is None or chain in view['assignments']:
        return
    runner.require(eligible(task, view) and task['kind'] == 'train', 'Cannot claim this private chain')
    directory = runner.safe(state) / 'chain-affinity'
    directory.mkdir(exist_ok=True)
    runner.write_once(directory / (chain + '.json'), {'schema_version': 1, 'identity': identity,
        'chain_id': chain, 'worker_id': view['worker_id'], 'job_uid': view['job_uid'], 'utc': runner.utc()})
