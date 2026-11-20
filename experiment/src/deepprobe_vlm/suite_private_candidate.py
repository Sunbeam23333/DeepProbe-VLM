"""Derive a new private-disk candidate from explicit reviewed migration inputs.

This command validates local pins and emits a candidate; it never retires an
old fleet, imports COMPLETED journals, contacts a bucket or submits a Job.
The ordinary suite_workflow CPU admission and suite_fleet_launch admission
remain mandatory, using fresh storage/job/prefix observations after handoff.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from . import suite_runner as runner


def replace_f16_declarations(candidate, pin):
    """Replace exactly two declared future caches; preserve every F4/F8 byte."""
    runner.verify_pin(pin)
    replacements = runner.read(pin['path'])
    keys = {'tempcompass_mc:f16', 'videomme899:f16'}
    runner.require(isinstance(replacements, dict) and set(replacements) == keys,
                   'F16 replacement must contain exactly the two reviewed benchmark declarations')
    declarations = candidate['environment'].get('benchmark_preparations', {})
    runner.require(keys <= set(declarations), 'Existing consumer must already declare both F16 protocols')
    for key, value in replacements.items():
        contract = value.get('contract', {})
        request = contract.get('request', {})
        runner.require(set(value) == {'contract', 'contract_sha256', 'ready_path'}
                       and value['contract_sha256'] == runner.digest(contract)
                       and request.get('frames') == 16 and request.get('benchmark') == key.split(':')[0],
                       'F16 declaration pin does not bind its exact future benchmark contract')
        declarations[key] = deepcopy(value)
    requests = candidate.get('benchmark_preparations', [])
    replaced = set()
    for index, request in enumerate(requests):
        key = str(request.get('benchmark')) + ':f' + str(request.get('frames'))
        if key in keys:
            requests[index] = deepcopy(replacements[key]['contract']['request'])
            replaced.add(key)
    runner.require(replaced == keys, 'Both F16 producer requests must be replaced with their pinned declarations')
    candidate.setdefault('input_indexes', {})['f16_declarations'] = deepcopy(pin)


def derive_candidate(base, plan, changes, *, verify=True):
    from .suite_bindings import validate_environment, verify_history_binding
    from .suite_fleet import validate_execution_profile
    required = {'suite_id', 'experiment_root', 'source_files', 'frozen_source_manifest',
                'history', 'state_root', 'archive_root', 'execution_profile'}
    runner.require(isinstance(changes, dict) and required <= set(changes) <= required | {'f16_declarations', 'legacy_private_resume'},
                   'Private migration requires complete explicit source/history/state/profile replacements')
    runner.require(base.get('schema_version') == 1 and isinstance(base.get('environment'), dict),
                   'A pinned previous workflow candidate is required')
    candidate = deepcopy(base)
    environment = candidate['environment']
    previous = deepcopy(environment)
    profile = validate_execution_profile(changes['execution_profile'], changes['suite_id'])
    runner.require(profile['schema_version'] == 3, 'A private migration requires execution profile schema 3')
    state = runner.safe(changes['state_root'])
    runner.require(str(state) != base.get('state_root') and not state.exists(),
                   'Migration must use an absent new state directory, never an old fleet journal')
    runner.require(any(state != runner.safe(root) and state.is_relative_to(runner.safe(root))
                       for root in environment['storage']['roots']), 'New fleet state must remain inside accounted shared home')
    runner.require(changes['archive_root'].rstrip('/') != environment['archive_root'].rstrip('/'),
                   'Migration requires a new exact archive namespace')
    same_private_source = (previous.get('execution_profile', {}).get('schema_version') == 3
        and previous['assets']['training'].get('frozen_source_manifest', {}).get('sha256')
            == changes['frozen_source_manifest'].get('sha256')
        and previous.get('source_files') == changes['source_files'])
    runner.require(changes['experiment_root'] != environment['experiment_root'] or same_private_source,
                   'Migration requires a new frozen source tree or exact same-source private recovery, never a hot edit')
    environment.update({key: deepcopy(changes[key]) for key in
                        ('suite_id', 'experiment_root', 'source_files', 'history', 'archive_root', 'execution_profile')})
    environment['assets']['training']['frozen_source_manifest'] = deepcopy(changes['frozen_source_manifest'])
    environment['runtime_environment']['PYTHONPATH'] = str(runner.safe(changes['experiment_root']) / 'src')
    environment['run_root'] = str(runner.safe(profile['scratch']['mount_path']) / 'runs')
    environment['allocation_identity'].update({key: profile['workers'][0][key] for key in ('job_name', 'namespace')})
    environment.pop('legacy_private_resume', None)
    if 'legacy_private_resume' in changes:
        environment['legacy_private_resume'] = deepcopy(changes['legacy_private_resume'])
    candidate['state_root'] = str(state)
    # Preserve measured fresh/consumer reserves while replacing old native
    # resume reservations with the exact accepted final restore footprint.
    overrides = environment.setdefault('task_reserve_by_id', {})
    tasks = {task['id']: task for task in plan['tasks']}
    for task_id, entry in environment['history'].items():
        if task_id in tasks and tasks[task_id]['kind'] == 'train' and entry.get('mode') == 'accepted':
            overrides[task_id] = sum(row['bytes'] for row in entry['checkpoint']['files'])
    # Old import recipes must not overwrite the newly supplied accepted
    # history again when the ordinary workflow is prepared.
    candidate.pop('history_imports', None)
    candidate.get('input_indexes', {}).pop('history_imports', None)
    candidate.pop('capacity_recovery_plan', None)
    if 'f16_declarations' in changes:
        replace_f16_declarations(candidate, changes['f16_declarations'])
    for task_id, value in environment['history'].items():
        if value.get('mode') == 'resume':
            runner.require(value.get('frozen_source_manifest', {}).get('sha256')
                           == changes['frozen_source_manifest'].get('sha256')
                           or task_id == 'train__ouro_r4_s17__sft' and environment.get('legacy_private_resume'),
                           'Finish and archive legacy-source resume work before a private-disk migration')
    runner.require(set(environment['history']) <= set(tasks), 'Undeclared migration history task')
    for task_id, old in previous.get('history', {}).items():
        runner.require(task_id in environment['history'], 'Migration cannot drop an existing historical input')
        new = environment['history'][task_id]
        if old.get('mode') == 'accepted':
            runner.require(old == new, 'Previously accepted history must retain its original byte-pinned evidence')
        else:
            runner.require(new.get('mode') == 'accepted' or same_private_source
                           or task_id == 'train__ouro_r4_s17__sft' and environment.get('legacy_private_resume'),
                           'Legacy historical resume must become independently accepted before migration')
    validate_environment(environment, plan)
    if verify:
        from .experiment_suite import validate_plan
        from .suite_launch import complete_sources
        validate_plan(plan, config_root=environment['experiment_root'])
        complete_sources(environment)
        runner.verify_pin(changes['frozen_source_manifest'])
        for task_id, entry in environment['history'].items():
            verify_history_binding(plan, tasks[task_id], entry)
            if entry['mode'] == 'resume':
                from .suite_checkpoint_resume import verify_private_resume
                verify_private_resume(tasks[task_id], entry, environment)
    return candidate


def main(argv=None):
    import argparse
    from .suite_workflow import save
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'plan', 'changes', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('base', 'plan', 'changes'):
        parser.add_argument('--' + name + '-sha256', required=True)
    args = parser.parse_args(argv)
    for name in ('base', 'plan', 'changes'):
        runner.verify_pin({'path': str(getattr(args, name)), 'sha256': getattr(args, name + '_sha256')})
    result = derive_candidate(runner.read(args.base), runner.read(args.plan), runner.read(args.changes))
    output = runner.safe(args.output)
    runner.require(not output.exists(), 'Candidate output must be new')
    candidate = save(output / 'CANDIDATE.json', result)
    save(output / 'DERIVATION.json', {'schema_version': 1, 'status': 'PRIVATE_CANDIDATE_DERIVED_NOT_ADMITTED',
        'candidate': candidate, 'inputs': [runner.artifact(getattr(args, name)) for name in ('base', 'plan', 'changes')],
        'plan_sha256': args.plan_sha256, 'plan_modified': False, 'old_fleet_modified': False,
        'gpu_submitted': False, 'remote_operations': False, 'typed_completed_task_imported': False,
        'fresh_cpu_and_scheduler_admission_required': True})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
