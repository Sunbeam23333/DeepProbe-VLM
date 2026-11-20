"""Bound resident checkpoint chains without changing the scientific task DAG."""
from __future__ import annotations

from . import suite_runner as runner


def training_components(plan):
    """Connect shared initializers, including cross-model frozen-vision consumers."""
    tasks = [task for task in plan['tasks'] if task['kind'] == 'train']
    if not any('initializer' in task or 'stage' in task for task in tasks):
        # Tiny executor fixtures are generic command graphs, not scientific
        # checkpoint plans. Production validate_plan requires these fields.
        return []
    runner.require(all('initializer' in task and task.get('stage') in {'alignment', 'sft'} for task in tasks),
                   'A checkpoint chain requires complete training declarations')
    by_id = {task['id']: task for task in tasks}
    parents = {tid: tid for tid in by_id}
    def root(tid):
        while parents[tid] != tid:
            tid = parents[tid]
        return tid
    for task in tasks:
        initializer = task['initializer'].get('checkpoint_task_id')
        if initializer is not None:
            runner.require(initializer in by_id and initializer in task['dependencies'],
                           'Training initializer must be an exact training dependency')
            parents[root(task['id'])] = root(initializer)
    groups = {}
    for task in tasks:
        groups.setdefault(root(task['id']), []).append(task['id'])
    result = []
    for members in groups.values():
        cleanups = [task for task in plan['tasks'] if task['kind'] == 'cleanup'
                    and task.get('checkpoint_task_id') in members]
        runner.require(len(cleanups) == len(members) and
                       {task['checkpoint_task_id'] for task in cleanups} == set(members),
                       'Every checkpoint chain stage needs its declared cleanup consumer barrier')
        result.append({'chain_id': members[0], 'training_task_ids': members,
                       'cleanup_task_ids': [task['id'] for task in cleanups]})
    return result


def training_window(plan, completed, started, active):
    """Derive the durable chain slot while the caller holds project_transaction."""
    groups = training_components(plan)
    if not groups:
        return None
    completed, started = set(completed), set(started)
    unfinished = [group for group in groups if not set(group['cleanup_task_ids']) <= completed]
    opened = [group for group in unfinished if set(group['training_task_ids']) & (started | completed)]
    runner.require(len(opened) <= 1, 'Multiple resident training chains require explicit capacity recovery')
    selected = opened[0] if opened else (unfinished[0] if unfinished else None)
    active_train = [task['id'] for task in active if task['kind'] == 'train']
    runner.require(len(active_train) <= 1, 'Multiple active training claims violate checkpoint-chain backpressure')
    training_ids = {task['id'] for task in plan['tasks'] if task['kind'] == 'train'}
    archives = {task['checkpoint_task_id']: task['id'] for task in plan['tasks']
                if task['kind'] == 'archive' and task.get('checkpoint_task_id') in training_ids}
    runner.require(set(archives) == training_ids, 'Every training stage needs its exact intermediate archive barrier')
    unarchived = [tid for tid in (selected['training_task_ids'] if selected else [])
                  if tid in completed and archives.get(tid) not in completed]
    return {'chain_id': selected['chain_id'] if selected else None,
            'training_task_ids': selected['training_task_ids'] if selected else [],
            'cleanup_task_ids': selected['cleanup_task_ids'] if selected else [],
            'active_training_task_ids': active_train,
            'completed_training_awaiting_archive': unarchived,
            'new_chain_waits_for_all_declared_cleanup_receipts': True}


def component_peak_plans(plan, binding, environment, estimates):
    """Conservative serial-chain peaks, retaining every prior final in a group.

    Intermediates are released only by the mandatory completed archive barrier.
    Final cleanup is never credited within a component. Concurrent consumers
    are bounded by the remaining worker count and their upfront reservations.
    """
    by_id = {task['id']: task for task in plan['tasks']}
    workers = len(environment.get('execution_profile', {}).get('workers', [None]))
    results = []
    for group in training_components(plan):
        resident, stages = 0, []
        model_ids = {by_id[tid]['model_id'] for tid in group['training_task_ids']}
        consumer_reserves = sorted((binding['tasks'][task['id']]['reserve_bytes'] for task in plan['tasks']
            if task.get('model_id') in model_ids and task['kind'] not in {'train', 'aggregate'}), reverse=True)
        parallel = sum(consumer_reserves[:max(0, workers - 1)])
        for tid in group['training_task_ids']:
            task, estimate = by_id[tid], estimates[tid]
            history = environment.get('history', {}).get(tid, {})
            restore = sum(row['bytes'] for row in history.get('checkpoint', {}).get('files', []))
            reserve = binding['tasks'][tid]['reserve_bytes']
            runner.require(type(estimate.get('model_saved_bytes')) is int and estimate['model_saved_bytes'] > 0,
                           'Measured serialized model bound is required for every checkpoint chain')
            if history.get('mode') == 'accepted':
                runner.require(reserve >= restore, 'Accepted initializer restore needs upfront reservation')
                additional, final = reserve, restore
            else:
                runner.require(reserve >= estimate['required_additional_bytes'], 'Stage reserve understates measured peak')
                additional = restore + reserve
                final = estimate['model_saved_bytes'] + 128 * 1024 * 1024
            stages.append({'task_id': tid, 'resident_prior_final_bytes': resident,
                           'historical_restore_bytes': restore, 'bound_stage_reserve_bytes': reserve,
                           'peak_additional_bytes': resident + additional + parallel,
                           'retained_final_bound_bytes': final})
            resident += final
        consumer_peak = resident + sum(consumer_reserves[:workers])
        peak = max([row['peak_additional_bytes'] for row in stages] + [consumer_peak])
        results.append({**group, 'stages': stages, 'all_final_resident_bound_bytes': resident,
                        'parallel_consumer_reserve_bytes': parallel, 'consumer_phase_peak_bytes': consumer_peak,
                        'required_additional_bytes': peak, 'prior_final_cleanup_credited': False,
                        'intermediate_archive_barrier_required': True, 'maximum_concurrent_training': 1})
    return results
