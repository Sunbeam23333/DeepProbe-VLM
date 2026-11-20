"""Measured-reserve arithmetic for shared-initializer checkpoint lifetimes."""
from copy import deepcopy

import pytest

from deepprobe_vlm.suite_chains import component_peak_plans, training_components, training_window


def fixture():
    train = [
        {'id': 'alignment', 'kind': 'train', 'model_id': 'main', 'stage': 'alignment',
         'dependencies': [], 'initializer': {'kind': 'backbones'}},
        {'id': 'sft', 'kind': 'train', 'model_id': 'main', 'stage': 'sft',
         'dependencies': ['alignment'], 'initializer': {'checkpoint_task_id': 'alignment'}},
        {'id': 'frozen', 'kind': 'train', 'model_id': 'frozen', 'stage': 'sft',
         'dependencies': ['alignment'], 'initializer': {'checkpoint_task_id': 'alignment'}}]
    plan = {'tasks': train + [
        {'id': 'archive_' + t['id'], 'kind': 'archive', 'model_id': t['model_id'],
         'checkpoint_task_id': t['id']} for t in train] + [
        {'id': 'cleanup_' + t['id'], 'kind': 'cleanup', 'model_id': t['model_id'],
         'checkpoint_task_id': t['id']} for t in train] + [
        {'id': 'evaluate', 'kind': 'evaluate', 'model_id': 'main'},
        {'id': 'evaluate_frozen', 'kind': 'evaluate', 'model_id': 'frozen'}]}
    mib = 1024 * 1024
    binding = {'tasks': {t['id']: {'reserve_bytes': mib} for t in plan['tasks']}}
    for tid, reserve in [('alignment', 8_000_000_000), ('sft', 48_000_000_000),
                         ('frozen', 48_000_000_000), ('evaluate', 11_000_000_000),
                         ('evaluate_frozen', 11_000_000_000)]:
        binding['tasks'][tid]['reserve_bytes'] = reserve
    env = {'execution_profile': {'workers': [{}]}, 'history': {
        'alignment': {'mode': 'accepted', 'checkpoint': {'files': [{'bytes': 7_500_000_000}]}},
        'sft': {'mode': 'resume', 'checkpoint': {'files': [{'bytes': 22_500_000_000}]}}}}
    estimates = {tid: {'model_saved_bytes': 7_500_000_000, 'required_additional_bytes': 48_000_000_000}
                 for tid in ('alignment', 'sft', 'frozen')}
    return plan, binding, env, estimates


def test_shared_frozen_consumer_and_native_resume_peak_are_costed_without_final_cleanup_credit():
    plan, binding, env, estimates = fixture()
    result = component_peak_plans(plan, binding, env, estimates)
    assert len(result) == 1
    row = result[0]
    assert row['training_task_ids'] == ['alignment', 'sft', 'frozen']
    assert row['stages'][1]['peak_additional_bytes'] == 7_500_000_000 + 22_500_000_000 + 48_000_000_000
    assert row['stages'][2]['resident_prior_final_bytes'] == 15_000_000_000 + 128 * 1024 * 1024
    assert row['all_final_resident_bound_bytes'] == 22_500_000_000 + 256 * 1024 * 1024
    assert row['consumer_phase_peak_bytes'] == row['all_final_resident_bound_bytes'] + 11_000_000_000
    assert row['required_additional_bytes'] == 78_000_000_000
    assert row['intermediate_archive_barrier_required'] and not row['prior_final_cleanup_credited']


def test_parallel_workers_add_real_consumer_reservations_to_the_training_peak():
    plan, binding, env, estimates = fixture()
    env['execution_profile']['workers'] *= 2
    result = component_peak_plans(plan, binding, env, estimates)[0]
    assert result['parallel_consumer_reserve_bytes'] == 11_000_000_000
    assert result['required_additional_bytes'] == 89_000_000_000


def test_a_completed_stage_still_blocks_next_training_until_archive_releases_intermediates():
    plan, _, _, _ = fixture()
    window = training_window(plan, {'alignment'}, {'alignment'}, [])
    assert window['completed_training_awaiting_archive'] == ['alignment']
    assert training_window(plan, {'alignment', 'archive_alignment'}, {'alignment'}, [])['completed_training_awaiting_archive'] == []


@pytest.mark.parametrize('damage', ['missing_cleanup', 'unknown_initializer', 'missing_saved_bound', 'underreserved_restore'])
def test_incomplete_capacity_proof_never_admits_a_new_chain(damage):
    plan, binding, env, estimates = fixture()
    if damage == 'missing_cleanup':
        plan['tasks'] = [t for t in plan['tasks'] if t['id'] != 'cleanup_frozen']
    elif damage == 'unknown_initializer':
        plan['tasks'][1]['initializer']['checkpoint_task_id'] = 'unbound'
    elif damage == 'missing_saved_bound':
        del estimates['sft']['model_saved_bytes']
    else:
        binding['tasks']['alignment']['reserve_bytes'] = 1
    with pytest.raises(ValueError):
        component_peak_plans(plan, binding, env, estimates)
