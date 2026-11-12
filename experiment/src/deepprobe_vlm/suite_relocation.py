"""Audit the only permitted resume migration: a new exact scheduler job name.

Scientific plans, source bytes, model/data/history pins, storage roots, suite
state and commands remain fixed. Only the environment document location/hash
and its allocation_identity.job_name change in regenerated task bindings.
Original task completion receipts and the initial suite identity are immutable.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re

from . import suite_runner as runner

require, read, safe = runner.require, runner.read, runner.safe


def immutable_environment(value):
    normalized = deepcopy(value)
    normalized['allocation_identity'].pop('job_name')
    return normalized


def relocation(plan_pin, previous_binding_pin, previous_environment_pin, binding_pin, environment_pin, *, state_root):
    """Produce a deterministic, reviewable equivalence receipt without executing."""
    from .experiment_suite import validate_plan
    from .suite_bindings import validate_environment, build_binding
    pins = [runner.verify_pin(value) for value in
            (plan_pin, previous_binding_pin, previous_environment_pin, binding_pin, environment_pin)]
    plan_pin, previous_binding_pin, previous_environment_pin, binding_pin, environment_pin = pins
    plan = read(plan_pin['path'])
    old_env, new_env = read(previous_environment_pin['path']), read(environment_pin['path'])
    validate_plan(plan, config_root=new_env['experiment_root'])
    for environment in (old_env, new_env):
        validate_environment(environment, plan)
    require(immutable_environment(old_env) == immutable_environment(new_env),
            'Relocation may change only allocation_identity.job_name; scientific/runtime/data/storage changes are forbidden')
    require(old_env['allocation_identity']['job_name'] != new_env['allocation_identity']['job_name'],
            'Relocation requires a distinct new exact job name')
    bindings = []
    for env, env_pin, bind_pin in ((old_env, previous_environment_pin, previous_binding_pin),
                                    (new_env, environment_pin, binding_pin)):
        observed = read(bind_pin['path'])
        expected = build_binding(plan, env, plan_path=plan_pin['path'], plan_sha256=plan_pin['sha256'],
            environment_path=env_pin['path'], environment_sha256=env_pin['sha256'], config_root=env['experiment_root'])
        require(observed == expected, 'Both relocation bindings must equal their complete generated suite commands')
        bindings.append(observed)
    require(previous_binding_pin['sha256'] != binding_pin['sha256'], 'Relocation binding must change')
    require(runner.source_identity(bindings[0]) == runner.source_identity(bindings[1]), 'Relocation source identity differs')
    for item in new_env['source_files']:
        runner.verify_pin(item)
    return {'schema_version': 1, 'status': 'SUITE_ALLOCATION_RELOCATION_VERIFIED',
            'suite_id': new_env['suite_id'], 'state_root': str(safe(state_root)), 'plan': plan_pin,
            'previous_binding': previous_binding_pin, 'previous_environment': previous_environment_pin,
            'binding': binding_pin, 'environment': environment_pin,
            'source_identity_sha256': runner.source_identity(bindings[1]),
            'immutable_environment_sha256': runner.digest(immutable_environment(new_env)),
            'previous_job': old_env['allocation_identity'], 'job': new_env['allocation_identity'],
            'allowed_changes': ['allocation_identity.job_name', 'environment_document_pin_in_generated_commands'],
            'old_task_receipts_modified': False, 'scheduler_submitted': False}


def verify_relocation(pin):
    pinned = runner.verify_pin(pin)
    value = read(pinned['path'])
    require(isinstance(value, dict), 'Invalid relocation receipt')
    expected = relocation(value['plan'], value['previous_binding'], value['previous_environment'],
                          value['binding'], value['environment'], state_root=value['state_root'])
    require(value == expected, 'Relocation receipt changed or is not an exact equivalence proof')
    return value


def resume_bindings(state_root, *, plan_sha256, binding_sha256, suite_id, source_sha256,
                    relocation_pin=None, apply=False):
    """Validate the append-only allocation chain; caller holds the project lock."""
    state = safe(state_root)
    identity = read(state / 'SUITE_IDENTITY.json')
    require(identity.get('plan_sha256') == plan_sha256 and identity.get('suite_id') == suite_id
            and identity.get('source_identity_sha256') == source_sha256, 'Resume scientific/source identity changed')
    accepted = [identity['binding_sha256']]
    directory = state / 'relocations'
    records = sorted(directory.iterdir()) if directory.exists() else []
    require(all(path.is_file() and re.fullmatch(r'relocation-[0-9]{6}\.json', path.name) for path in records),
            'Unexpected relocation journal entry')
    for index, path in enumerate(records, 1):
        require(path.name == f'relocation-{index:06d}.json', 'Allocation relocation journal is not contiguous')
        record = read(path)
        value = verify_relocation(record['receipt'])
        require(value['state_root'] == str(state) and value['suite_id'] == suite_id
                and value['plan']['sha256'] == plan_sha256 and value['source_identity_sha256'] == source_sha256
                and value['previous_binding']['sha256'] == accepted[-1]
                and value['binding']['sha256'] not in accepted, 'Allocation relocation chain differs or loops')
        require(record == {'schema_version': 1, 'status': 'SUITE_RELOCATION_APPLIED',
                           'receipt': runner.verify_pin(record['receipt']),
                           'previous_binding_sha256': accepted[-1], 'binding_sha256': value['binding']['sha256']},
                'Applied relocation journal changed')
        accepted.append(value['binding']['sha256'])
    if binding_sha256 == accepted[-1]:
        if relocation_pin:
            value = verify_relocation(relocation_pin)
            require(len(accepted) > 1 and value['binding']['sha256'] == accepted[-1]
                    and value['previous_binding']['sha256'] == accepted[-2]
                    and value['state_root'] == str(state), 'Supplied relocation does not describe the current allocation')
        return set(accepted)
    require(relocation_pin, 'resume plan/binding identity changed; a new binding requires an explicit audited allocation relocation receipt')
    value = verify_relocation(relocation_pin)
    require(value['state_root'] == str(state) and value['suite_id'] == suite_id
            and value['plan']['sha256'] == plan_sha256 and value['source_identity_sha256'] == source_sha256
            and value['previous_binding']['sha256'] == accepted[-1]
            and value['binding']['sha256'] == binding_sha256 and binding_sha256 not in accepted,
            'Relocation does not continue this exact suite allocation chain')
    if apply:
        directory.mkdir(exist_ok=True)
        runner.write_once(directory / f'relocation-{len(records) + 1:06d}.json', {
            'schema_version': 1, 'status': 'SUITE_RELOCATION_APPLIED',
            'receipt': runner.verify_pin(relocation_pin), 'previous_binding_sha256': accepted[-1],
            'binding_sha256': binding_sha256})
    return set([*accepted, binding_sha256])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ('plan', 'previous-binding', 'previous-environment', 'binding', 'environment'):
        parser.add_argument('--' + key, type=Path, required=True)
        parser.add_argument('--' + key + '-sha256', required=True)
    parser.add_argument('--state-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    pins = [{'path': str(getattr(args, key)), 'sha256': getattr(args, key + '_sha256')}
            for key in ('plan', 'previous_binding', 'previous_environment', 'binding', 'environment')]
    value = relocation(*pins, state_root=args.state_root)
    # Check chain compatibility now, without changing the live runner journal.
    resume_bindings(args.state_root, plan_sha256=value['plan']['sha256'],
        binding_sha256=value['previous_binding']['sha256'], suite_id=value['suite_id'],
        source_sha256=value['source_identity_sha256'])
    temporary = safe(args.output)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    runner.write_once(temporary, value)
    print(json.dumps({'status': value['status'], 'receipt': runner.artifact(temporary),
                      'scheduler_submitted': False, 'state_modified': False}, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
