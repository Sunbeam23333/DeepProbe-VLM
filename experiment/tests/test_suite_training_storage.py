"""Exercise the real action's offline snapshot gate with local HF file aliases."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from deepprobe_vlm import declared_training, suite_actions as actions, suite_runner as runner
from test_suite_preflight import backbone


@pytest.fixture
def storage_action(backbone, tmp_path, monkeypatch):
    assets, env, plan = backbone
    home = tmp_path.resolve() / 'allowed-home'
    physical = home / 'assets'; physical.mkdir(parents=True)
    # Map only the existing fixed personal-home trust root into this isolated
    # test directory; tree_identity's real traversal/hash/alias logic runs.
    monkeypatch.setattr(declared_training, 'Path', lambda value: home if str(value) == '/home/jinhaopeng' else Path(value))
    for index, pin in enumerate(assets['files']):
        alias = Path(pin['path'])
        target = physical / str(index); target.write_bytes(alias.read_bytes())
        alias.unlink(); alias.symlink_to(target)
    task = plan['tasks'][0]; task['stage'] = 'sft'
    env.update(run_root=str(tmp_path.resolve() / 'runs'), source_files=[],
               assets={'backbones': {'ouro_r1': assets}})
    worker = actions.Actions(plan, env, tmp_path.resolve() / 'state')
    estimate = Mock(return_value={'required_additional_bytes': 4096})
    monkeypatch.setattr(actions, 'load_operation', lambda *args: SimpleNamespace(estimate_stage_storage=estimate))
    worker.budget = Mock(return_value={'synthetic': 'capacity-only'})
    return worker, task, assets, estimate, tmp_path.resolve() / 'future-output'


def test_real_training_storage_accepts_fully_pinned_hf_file_aliases(storage_action):
    worker, task, assets, estimate, output = storage_action
    with pytest.raises(ValueError, match='symlink path'):
        runner.verify_pin(assets['files'][0])  # General source/receipt rules stay strict.
    result = worker.training_storage(task, output)
    assert result['backbone_validation']['all_file_sha256_verified'] is True
    assert result['owned_stage_resident_bytes'] == 0
    estimate.assert_called_once()
    worker.budget.assert_called_once_with(4096)


@pytest.mark.parametrize('change', ['payload', 'outside_home', 'missing_pin', 'wrong_revision', 'directory_alias'])
def test_real_action_rejects_snapshot_drift_before_estimation_or_reservation(storage_action, change, tmp_path):
    worker, task, assets, estimate, output = storage_action
    alias = Path(assets['files'][0]['path'])
    if change == 'payload':
        alias.resolve().write_bytes(b'changed-pinned-payload')
    elif change == 'outside_home':
        outside = tmp_path.resolve() / 'outside'; outside.write_bytes(alias.read_bytes())
        alias.unlink(); alias.symlink_to(outside)
    elif change == 'missing_pin':
        assets['files'].pop()
    elif change == 'wrong_revision':
        worker.env['runtime_environment']['HF_HUB_CACHE'] += '-different'
    else:
        directory = Path(assets['language_root']) / 'unexpected-directory'
        directory.symlink_to(tmp_path.resolve(), target_is_directory=True)
    with pytest.raises(ValueError):
        worker.training_storage(task, output)
    estimate.assert_not_called()
    worker.budget.assert_not_called()
