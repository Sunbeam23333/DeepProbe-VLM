"""Real temporary-directory regressions; no production admission or GPU work."""
from copy import deepcopy
import json
import os
from pathlib import Path
from unittest import mock

import pytest

from deepprobe_vlm import checkpoint_retention as retention
from deepprobe_vlm import suite_run_root as paths


@pytest.fixture
def environment(tmp_path):
    root = tmp_path.resolve()
    return {'suite_id': 'synthetic-suite', 'run_root': str(root / 'runs/suite'),
        'storage': {'schema_version': 2, 'roots': [str(root)], 'ledger_root': str(root / 'ledger'),
            'home_limit_bytes': 150000000000, 'oss_limit_bytes': None,
            'home_safety_bytes': 1000000, 'oss_safety_bytes': 0, 'free_guard_bytes': 0,
            'oss_prefixes': ['oss://ss-devbox-stream/synthetic/project/']},
        'tools': {'ossutil': '/bin/true', 'oss_config': str(root / 'unused-config')}}


def test_missing_root_inspection_and_owner_remain_read_only(environment):
    path = Path(environment['run_root'])
    with pytest.raises(ValueError, match='run root is missing'):
        paths.inspect_run_root(environment)
    with pytest.raises(ValueError, match='runs root must already exist'):
        retention.claim_owned_run(path, 'train', 'synthetic-suite', 'a' * 64)
    assert not path.parent.exists()
    assert not Path(environment['storage']['ledger_root']).exists()


def test_preparation_crosses_owner_boundary_without_claiming_existing_runs(environment):
    root = Path(environment['run_root'])
    prepared = paths.prepare_run_root(environment)
    assert prepared['created_directories'] == [str(root.parent), str(root)]
    assert prepared['storage_reservation_released'] and not prepared['run_ownership_claimed']
    assert list(root.iterdir()) == []
    owner = retention.claim_owned_run(root, 'train', 'synthetic-suite', 'a' * 64)
    old = root / 'train' / 'preserve.safetensors'; old.write_bytes(b'existing checkpoint')
    marker = root / 'train' / retention.OWNER; marker_bytes = marker.read_bytes()
    with mock.patch('deepprobe_vlm.suite_storage.budget_for', side_effect=AssertionError('existing root must be read-only')):
        again = paths.prepare_run_root(environment)
    assert again['status'] == 'SUITE_RUN_ROOT_CHECKED' and not again['created_directories']
    assert old.read_bytes() == b'existing checkpoint' and marker.read_bytes() == marker_bytes
    assert retention.check_owner(owner) == root / 'train'
    unowned = root / 'foreign'; unowned.mkdir(); (unowned / 'data').write_bytes(b'old data')
    with pytest.raises(ValueError, match='nonempty unowned'):
        retention.claim_owned_run(root, 'foreign', 'synthetic-suite', 'a' * 64)
    ledger = json.loads((Path(environment['storage']['ledger_root']) / 'LEDGER.json').read_bytes())
    assert len(ledger['reservations']) == 1
    assert all(row['state'] == 'RELEASED' for row in ledger['reservations'].values())


@pytest.mark.parametrize('kind', ['file', 'target_symlink', 'ancestor_symlink', 'escape', 'wrong_uid', 'unwritable'])
def test_unsafe_locations_do_not_start_preparation(environment, tmp_path, kind):
    env = deepcopy(environment); root = Path(env['run_root']); sibling = tmp_path / 'untouched'
    sibling.mkdir(); sentinel = sibling / 'payload'; sentinel.write_bytes(b'keep')
    if kind == 'file':
        root.parent.mkdir(); root.write_bytes(b'not a directory')
    elif kind == 'target_symlink':
        root.parent.mkdir(); root.symlink_to(sibling, target_is_directory=True)
    elif kind == 'ancestor_symlink':
        root.parent.symlink_to(sibling, target_is_directory=True)
    elif kind == 'escape':
        env['run_root'] = str(tmp_path.parent / 'outside-suite')
    elif kind == 'wrong_uid':
        env['machine_identity'] = {'uid': os.getuid() + 1, 'gid': os.getgid()}
    else:
        root.mkdir(parents=True)
    with mock.patch('deepprobe_vlm.suite_storage.budget_for', side_effect=AssertionError('unsafe root must not initialize ledger')):
        if kind == 'unwritable':
            with mock.patch.object(paths.os, 'access', return_value=False), pytest.raises(ValueError, match='writable'):
                paths.prepare_run_root(env)
        else:
            with pytest.raises((ValueError, OSError)):
                paths.prepare_run_root(env)
    assert sentinel.read_bytes() == b'keep'
    assert not Path(env['storage']['ledger_root']).exists()


def test_real_capacity_denial_prevents_run_root_creation(environment):
    environment['storage']['home_limit_bytes'] = paths.METADATA_RESERVE_BYTES
    with pytest.raises(ValueError, match='capacity gated'):
        paths.prepare_run_root(environment)
    assert not Path(environment['run_root']).parent.exists()
    ledger = json.loads((Path(environment['storage']['ledger_root']) / 'LEDGER.json').read_bytes())
    assert ledger['reservations'] == {}


@pytest.mark.parametrize('layout', ['shared_missing_parent', 'ledger_under_run_root'])
def test_overlapping_ledger_initialization_is_rejected_before_any_budget_write(environment, tmp_path, layout):
    from deepprobe_vlm import suite_workflow as workflow
    env = deepcopy(environment)
    if layout == 'shared_missing_parent':
        env['run_root'] = str(tmp_path / 'new/runs')
        env['storage']['ledger_root'] = str(tmp_path / 'new/ledger')
    else:
        env['storage']['ledger_root'] = env['run_root'] + '/ledger'
    with mock.patch('deepprobe_vlm.suite_storage.budget_for', side_effect=AssertionError('must not initialize ledger')):
        with pytest.raises(ValueError, match='[Ll]edger'):
            paths.prepare_run_root(env)
        candidate = {'schema_version': 1, 'environment': env, 'state_root': str(tmp_path / 'state')}
        with mock.patch.object(workflow, 'workflow_report', return_value={'synthetic': True}):
            result = workflow.prepare_workflow({}, candidate, tmp_path / 'control', prepare_assets=True)
    assert result['status'] == 'SUITE_WORKFLOW_RUN_ROOT_GATED'
    assert not Path(env['run_root']).exists() and not Path(env['storage']['ledger_root']).exists()


@pytest.mark.parametrize('relationship', ['inside', 'ancestor'])
def test_plan_report_output_cannot_create_declared_run_root(environment, relationship):
    from deepprobe_vlm import suite_workflow as workflow
    root = Path(environment['run_root'])
    candidate = {'schema_version': 1, 'environment': environment, 'state_root': str(root.parent / 'state')}
    output = root / 'control' if relationship == 'inside' else root.parent
    with mock.patch.object(workflow, 'workflow_report', return_value={'synthetic': True}):
        with pytest.raises(ValueError, match='reports must be disjoint'):
            workflow.prepare_workflow({}, candidate, output)
    assert not root.parent.exists()


def test_concurrent_directory_appearance_is_not_adopted(environment, monkeypatch):
    root = Path(environment['run_root']); mkdir = os.mkdir
    def racing_mkdir(path, mode=0o777, *, dir_fd=None):
        if path == 'runs' and dir_fd is not None:
            mkdir(path, mode, dir_fd=dir_fd)
            (root.parent / 'foreign').write_bytes(b'concurrent data')
        return mkdir(path, mode, dir_fd=dir_fd)
    monkeypatch.setattr(paths.os, 'mkdir', racing_mkdir)
    with pytest.raises(FileExistsError):
        paths.prepare_run_root(environment)
    assert (root.parent / 'foreign').read_bytes() == b'concurrent data' and not root.exists()
