from copy import deepcopy
import json
from pathlib import Path

import pytest

from deepprobe_vlm.suite_private_candidate import derive_candidate, replace_f16_declarations
from deepprobe_vlm import suite_runner as runner
from test_suite_private_execution import profile


@pytest.fixture
def inputs():
    import test_suite_bindings as fixtures
    case = fixtures.SuiteBindingsTest(); case.setUp()
    try:
        env = deepcopy(case.environment)
        env['storage'] = {'schema_version': 2, 'roots': [str(case.a), str(case.b)],
            'ledger_root': str(case.a / 'ledger'), 'home_limit_bytes': 150_000_000_000,
            'oss_limit_bytes': None, 'oss_prefixes': ['oss://ss-devbox-stream/DeepProbe-VLM/'],
            'home_safety_bytes': 1_000_000, 'oss_safety_bytes': 0, 'free_guard_bytes': 0}
        env['task_reserve_by_id'] = {'train__ouro_r4_s29__sft': 54_000_000_000}
        base = {'schema_version': 1, 'environment': env, 'state_root': str(case.a / 'old-state'),
                'history_imports': {'old': 'must-not-replay'}, 'input_indexes': {'history_imports': 'old-index'},
                'capacity_recovery_plan': {'old': 'must-not-reexecute'}}
        changes = {'suite_id': 'synthetic-private', 'experiment_root': str(case.a / 'source-v7'),
            'source_files': env['source_files'], 'frozen_source_manifest': runner.artifact(case.source),
            'history': {}, 'state_root': str(case.a / 'new-state'),
            'archive_root': 'oss://ss-devbox-stream/DeepProbe-VLM/new-v7', 'execution_profile': profile(case.root)}
        yield case, base, changes
    finally:
        case.doCleanups()


def test_derivation_preserves_plan_old_candidate_and_measured_reserves(inputs):
    case, base, changes = inputs
    old, plan_sha = deepcopy(base), runner.digest(case.plan)
    result = derive_candidate(base, case.plan, changes, verify=False)
    assert base == old and runner.digest(case.plan) == plan_sha
    assert result['environment']['run_root'] == changes['execution_profile']['scratch']['mount_path'] + '/runs'
    assert result['environment']['task_reserve_by_id'] == old['environment']['task_reserve_by_id']
    assert 'history_imports' not in result and 'capacity_recovery_plan' not in result
    assert 'history_imports' not in result['input_indexes']
    assert not case.a.joinpath('new-state').exists()


@pytest.mark.parametrize('field', ['state_root', 'archive_root', 'experiment_root'])
def test_derivation_rejects_reusing_old_mutable_namespace(inputs, field):
    case, base, changes = inputs
    changes[field] = base[field] if field == 'state_root' else base['environment'][field]
    with pytest.raises(ValueError, match='new state|new exact archive|new frozen source'):
        derive_candidate(base, case.plan, changes, verify=False)


def test_unfinished_historical_resume_cannot_silently_switch_source(inputs):
    case, base, changes = inputs
    changes['history']['train__ouro_r4_s17__sft'] = {'mode': 'resume'}
    with pytest.raises(ValueError, match='Finish and archive'):
        derive_candidate(base, case.plan, changes, verify=False)


def test_f16_pin_replaces_only_future_f16_contracts_and_never_creates_ready(tmp_path):
    tmp_path = tmp_path.resolve()
    old, new = {}, {}
    for benchmark in ('videomme899', 'tempcompass_mc'):
        for frames in (4, 8, 16):
            key = f'{benchmark}:f{frames}'
            contract = {'request': {'benchmark': benchmark, 'frames': frames, 'producer': 'old'}}
            old[key] = {'contract': contract, 'contract_sha256': runner.digest(contract),
                        'ready_path': str(tmp_path / (key + '-old-READY.json'))}
        contract = {'request': {'benchmark': benchmark, 'frames': 16, 'producer': 'new'}}
        new[f'{benchmark}:f16'] = {'contract': contract, 'contract_sha256': runner.digest(contract),
                                 'ready_path': str(tmp_path / (benchmark + '-new-READY.json'))}
    candidate = {'environment': {'benchmark_preparations': deepcopy(old)},
                 'benchmark_preparations': [deepcopy(value['contract']['request']) for value in old.values()]}
    declaration = tmp_path / 'F16.json'; declaration.write_text(json.dumps(new))
    replace_f16_declarations(candidate, runner.artifact(declaration))
    for key, value in candidate['environment']['benchmark_preparations'].items():
        assert value == (new[key] if key.endswith(':f16') else old[key])
        assert not Path(value['ready_path']).exists()
