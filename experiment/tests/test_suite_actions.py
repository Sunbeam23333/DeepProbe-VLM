import copy
import json
import pytest
from deepprobe_vlm import suite_actions as a
from deepprobe_vlm import suite_results as results


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return a.pin(path)


def test_actual_allocation_rejects_workers_priority_and_gpu_count():
    spec = {'nodeSelector': {'sslaunch-pool': 'stream'}, 'priorityClassName': 'low',
            'containers': [{'resources': {'limits': {'nvidia.com/gpu': 8}}}]}
    job = {'spec': {'pytorchReplicaSpecs': {'Master': {'replicas': 1, 'restartPolicy': 'Never', 'template': {'spec': spec}}}}}
    assert a.allocation_contract(job)['gpus'] == 8
    for change in ('priority', 'worker', 'restart', 'gpu'):
        bad = copy.deepcopy(job);master = bad['spec']['pytorchReplicaSpecs']['Master']
        if change == 'priority': master['template']['spec']['priorityClassName'] = 'normal'
        if change == 'worker': bad['spec']['pytorchReplicaSpecs']['Worker'] = master
        if change == 'restart': master['restartPolicy'] = 'OnFailure'
        if change == 'gpu': master['template']['spec']['containers'][0]['resources']['limits']['nvidia.com/gpu'] = 7
        with pytest.raises(ValueError): a.allocation_contract(bad)


def test_inventory_detects_extra_files_tamper_and_symlink(tmp_path):
    (tmp_path / 'model.safetensors').write_bytes(b'weight');rows = a.inventory(tmp_path)
    a.verify_inventory(tmp_path, rows)
    (tmp_path / 'extra').write_bytes(b'not declared')
    with pytest.raises(ValueError): a.verify_inventory(tmp_path, rows)
    (tmp_path / 'extra').unlink();(tmp_path / 'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError): a.verify_inventory(tmp_path, rows)
    (tmp_path / 'alias').symlink_to(tmp_path / 'model.safetensors')
    with pytest.raises(ValueError): a.inventory(tmp_path)


def test_formal_acceptance_requires_exact_step_media_and_complete_recovery(tmp_path):
    out = tmp_path / 'training'; out.mkdir()
    expected = {'status': 'formal_training_complete', 'stage': 'sft', 'world_size': 8,
        'optimizer_step': 718, 'train_manifest_sha256': 'a'*64, 'eval_manifest_sha256': 'b'*64,
        'formal_ready_sha256': 'c'*64, 'frozen_source_manifest_sha256': 'd'*64,
        'media_before': {'digest': 'e'*64}, 'media_after': {'digest': 'e'*64}}
    for name, obj in (('trainer_state.json', {'global_step': 718}), ('run_manifest.json', {'source': 'known'}),
                      ('train_results.json', {'train_loss': 1.2}), ('eval_results.json', {'eval_loss': 1.3})):
        p = write(out / name, obj)
        if name in ('trainer_state.json', 'run_manifest.json'): expected[name.replace('.json', '_sha256')] = p['sha256']
    write(out / 'FORMAL_TRAIN_COMPLETE.json', expected)
    write(out / 'final/deepprobe_config.json', {'num_frames': 8});(out / 'final/model.safetensors').write_bytes(b'model')
    cp = out / 'checkpoint-718';cp.mkdir();write(cp / 'trainer_state.json', {'global_step': 718, 'max_steps': 718})
    for name in ['optimizer.pt', 'scheduler.pt'] + [f'rng_state_{rank}.pth' for rank in range(8)]:
        (cp / name).write_bytes(b'state')
    task = {'stage': 'sft', 'expected_optimizer_steps': 718}
    data = {'train_manifest': {'sha256': 'a'*64}, 'eval_manifest': {'sha256': 'b'*64}, 'formal_ready': {'sha256': 'c'*64}}
    assert len(a.validate_training_output(out, task, data, 'd'*64)[0]) == 5
    (cp / 'rng_state_7.pth').unlink()
    with pytest.raises(ValueError, match='RNG'): a.validate_training_output(out, task, data, 'd'*64)
    expected['optimizer_step'] = 717;write(out / 'FORMAL_TRAIN_COMPLETE.json', expected)
    with pytest.raises(ValueError, match='Formal completion'): a.validate_training_output(out, task, data, 'd'*64)


def test_shape_ablation_refuses_f8_cache_before_model_load(tmp_path):
    records = [{'sample_id':str(i), 'source_video_id':str(i//3), 'question':'Which?', 'answer':'A',
                'choices':['yes','no'], 'split':'test', 'image_paths':[f'{k}.png' for k in range(8)]} for i in range(2697)]
    manifest = tmp_path / 'manifest.jsonl';manifest.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
    for k in range(8): (tmp_path / f'{k}.png').write_bytes(b'frame')
    data = {'manifest':a.pin(manifest), 'acceptance':write(tmp_path / 'ready.json', {}), 'media_root':str(tmp_path)}
    with pytest.raises(ValueError, match='Frame cache shape'):
        a.validate_benchmark(data, {'benchmark':'videomme899', 'frames':16})


def test_source_components_union_and_paired_coverage():
    rows = [dict(sample_id='a', source_video_id='clip1', underlying_source_video_ids=['1','2'], correct=1),
            dict(sample_id='b', source_video_id='clip2', underlying_source_video_ids=['2','3'], correct=0),
            dict(sample_id='c', source_video_id='clip3', underlying_source_video_ids=['4'], correct=1)]
    assert results.components(rows) == {'a':'1','b':'1','c':'4'}
    result = results.bootstrap(rows, repeats=100)
    assert result['source_components'] == 2 and result['estimate'] == pytest.approx(2/3)
    assert [r['delta'] for r in results.paired(rows, rows)] == [0,0,0]
    with pytest.raises(ValueError, match='coverage'):results.paired(rows[:1], rows)
    changed = copy.deepcopy(rows);changed[0]['underlying_source_video_ids']=['5']
    with pytest.raises(ValueError, match='identity'):results.paired(changed, rows)


def test_aggregate_never_drops_missing_experiment():
    with pytest.raises(ValueError, match='All declared'):
        results.aggregate({'tasks':[{'id':'needed', 'kind':'evaluate'}]}, {})


def test_finite_metrics_reject_nan_and_inf():
    for v in (float('nan'),float('inf'),-float('inf')):
        with pytest.raises(ValueError): a.finite_metrics({'train':{'loss':v}})


def test_complete_matrix_aggregation_keeps_seed_variance_separate():
    from deepprobe_vlm.experiment_suite import default_plan
    plan = default_plan()
    evaluations = {}
    for task in plan['tasks']:
        if task['kind'] != 'evaluate': continue
        size = 2697 if task['benchmark'] == 'videomme899' else 1580
        rows = [{'sample_id':str(i), 'source_video_id':str(i//3), 'correct':1,
                 'question_identity_sha256':str(i), 'answer':'A', 'split':'test', 'metric':task['benchmark']}
                for i in range(size)]
        if task['model_id'] == 'ouro_r1_s17': rows[0]['correct'] = 0
        evaluations[task['id']] = {'status':'SUITE_EVALUATION_COMPLETE', 'task_sha256':a.digest(task),
                                  'failed_units':0, 'suite_protocol':task['protocol'], 'qa':size, 'rows':rows}
    result = results.aggregate(plan, evaluations, repeats=100)
    assert len(result['models']) == 18
    assert result['training_seed_statistics']['ouro_r1']['videomme899']['mean_accuracy'] == pytest.approx(1-1/(2697*3))
    assert result['training_seed_statistics']['ouro_r1']['videomme899']['sample_std_across_training_seeds'] > 0
    assert result['paired_accuracy_deltas']['ouro_r2_s17']['benchmarks']['videomme899']['estimate'] == pytest.approx(1/2697)


def test_cross_allocation_latency_is_remeasured_as_a_pair(tmp_path, monkeypatch):
    from deepprobe_vlm import compare_latency
    task = {'id':'comparison', 'kind':'compare_latency', 'dependencies':['ref','new'],
            'reference_task_id':'ref', 'candidate_task_id':'new'}
    plan = {'tasks':[task, {'id':'ref'}, {'id':'new'}]}
    env = {'experiment_root':str(tmp_path), 'run_root':str(tmp_path), 'source_files':[]}
    actions = a.Actions(plan, env, tmp_path)
    monkeypatch.setattr(actions, 'live', lambda: {})
    monkeypatch.setattr(actions, 'completed', lambda t: {'measurement_root':t})
    monkeypatch.setattr(compare_latency, 'HARDWARE_AND_SOFTWARE', ('gpu_uuid',))
    monkeypatch.setattr(compare_latency, 'read_run', lambda p: (None, {'gpu_uuid':p}, None, None))
    called = []
    def latency(task, *, emit_consumer):
        assert emit_consumer is False
        called.append(task['id'])
        return {'measurement_root':'paired-'+task['id'], 'outputs':[write(tmp_path/(task['id']+'.json'),{})]}
    monkeypatch.setattr(actions, 'latency', latency)
    def compare(ref, new):
        assert (ref,new) == ('paired-ref','paired-new')
        return {'status':'TOKEN_PARITY_PASSED'}
    monkeypatch.setattr(compare_latency, 'compare', compare)
    actions.run('comparison', tmp_path/'accepted.json')
    assert called == ['ref','new']
    assert a.read(tmp_path/'accepted.json')['cross_allocation_pair_remeasured'] is True


def test_latest_recovery_preserves_progress_and_torn_saves(tmp_path):
    for step in (300,450):
        cp=tmp_path/f'checkpoint-{step}';cp.mkdir()
        write(cp/'trainer_state.json',{'global_step':step,'max_steps':718})
        for name in ['optimizer.pt','scheduler.pt','model.safetensors']+[f'rng_state_{i}.pth' for i in range(8)]:
            (cp/name).write_bytes(b'state')
    torn=tmp_path/'checkpoint-500';torn.mkdir();(torn/'optimizer.pt').write_bytes(b'partial')
    assert a.latest_complete_checkpoint(tmp_path).name=='checkpoint-450'
    assert (torn/'optimizer.pt').read_bytes()==b'partial'
    assert a.latest_complete_checkpoint(tmp_path/'missing') is None
