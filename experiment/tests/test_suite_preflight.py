"""Synthetic local evidence exercises CPU gates without a real bucket or GPU."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from deepprobe_vlm import checkpoint_retention as retain
from deepprobe_vlm import formal_integrity as integrity
from deepprobe_vlm import suite_preflight as p
from deepprobe_vlm import suite_runner as runner
from deepprobe_vlm import suite_media as media


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')
    return retain.file_pin(path)


@pytest.fixture
def formal(tmp_path):
    import test_formal_integrity as fixtures
    f = fixtures.FormalIntegrityTests().fixture(tmp_path)
    t = {'train_manifest': retain.file_pin(f['data'] / 'train.jsonl'),
         'eval_manifest': retain.file_pin(f['data'] / 'validation.jsonl'),
         'formal_ready': retain.file_pin(f['formal_ready']),
         'frozen_source_manifest': retain.file_pin(f['source_manifest']),
         'data_dir': str(f['data']), 'media_root': str(f['media'])}
    f['training'] = t
    f['declared'] = {'train_manifest_sha256': t['train_manifest']['sha256'],
        'validation_manifest_sha256': t['eval_manifest']['sha256'],
        'train_examples': 1, 'validation_examples': 1, 'total_qa': 2, 'source_clips': 2}
    return f


def test_native_formal_api_does_not_replace_current_media_hashing(formal):
    assert p._formal_gate(formal['training'])['policy_id'] == 'deepprobe-formal-benchmark-source-v1'
    assert p._training_gate(formal['training'], formal['declared'])['all_current_media_sha256_verified']
    formal['train_media'].write_bytes(b'changed!')
    assert p._formal_gate(formal['training'])
    with pytest.raises(ValueError, match='SHA-256'): p._training_gate(formal['training'], formal['declared'])


def test_training_declared_hash_and_clip_count(formal):
    changed = deepcopy(formal['declared']); changed['train_manifest_sha256'] = 'a'*64
    with pytest.raises(ValueError, match='scientific suite'): p._training_gate(formal['training'], changed)
    changed = deepcopy(formal['declared']); changed['source_clips'] = 3
    with pytest.raises(ValueError, match='source-clip coverage'): p._training_gate(formal['training'], changed)


def test_complete_live_source_tree_required(formal):
    root = formal['source']; pins = [retain.file_pin(root/name) for name in integrity.source_inventory(root)]
    env = {'experiment_root': str(root), 'source_files': pins}
    assert p._source_gate(env, formal['training'], {'source_files': pins})['file_count'] == len(pins)
    with pytest.raises(ValueError, match='omits'):
        p._source_gate({**env, 'source_files': pins[1:]}, formal['training'], {'source_files': pins[1:]})
    (root/'src/deepprobe_vlm/train.py').write_text('changed')
    with pytest.raises(ValueError, match='source tree differs'): p._source_gate(env, formal['training'])


@pytest.fixture
def backbone(tmp_path):
    root, hub = tmp_path/'experiment', tmp_path/'hub'
    model = {'language_model': 'Test/Language', 'language_revision': 'a'*40,
             'vision_model': 'Test/Vision', 'vision_revision': 'b'*40}
    write(root/'recipe.json', {'model':model})
    assets = {'files': []}
    for label,key in (('language','Language'),('vision','Vision')):
        d = hub/('models--Test--'+key)/'snapshots'/model[label+'_revision']
        write(d/'config.json', {'hidden_size':4}); (d/'model.safetensors').write_bytes(b'weight')
        write(d/('tokenizer.json' if label=='language' else 'preprocessor_config.json'), {'fixture':True})
        assets[label+'_root']=str(d);assets['files'].extend(retain.file_pin(x) for x in sorted(d.iterdir()))
    env={'experiment_root':str(root),'runtime_environment':{'HF_HUB_CACHE':str(hub)}}
    plan={'models':[{'id':'tiny','family':'ouro_r1'}],
          'tasks':[{'id':'train','kind':'train','model_id':'tiny','config':{'path':'recipe.json'}}]}
    return assets,env,plan


def test_backbone_inventory_and_offline_cache_binding(backbone):
    assets,env,plan=backbone
    assert p._backbone_gate('ouro_r1',assets,env,plan)['all_file_sha256_verified']
    with pytest.raises(ValueError,match='empty'): p._backbone_gate('ouro_r1',{**assets,'files':[]},env,plan)
    with pytest.raises(ValueError,match='complete current snapshots'):
        p._backbone_gate('ouro_r1',{**assets,'files':assets['files'][:-1]},env,plan)
    wrong=deepcopy(env);wrong['runtime_environment']['HF_HUB_CACHE']+='-other'
    with pytest.raises(ValueError,match='actual offline Hub revision'):p._backbone_gate('ouro_r1',assets,wrong,plan)
    wrong=deepcopy(env);wrong['runtime_environment']['TRANSFORMERS_CACHE']='/different'
    with pytest.raises(ValueError,match='cache locations differ'):p._backbone_gate('ouro_r1',assets,wrong,plan)


def test_backbone_tokenizer_and_index_shards_required(backbone):
    assets,env,plan=backbone;d=Path(assets['language_root'])
    write(d/'model.safetensors.index.json',{'weight_map':{'weight':'missing.safetensors'}})
    with pytest.raises(ValueError,match='missing shard'):p._backbone_gate('ouro_r1',assets,env,plan)
    (d/'model.safetensors.index.json').unlink();(d/'tokenizer.json').unlink()
    with pytest.raises(ValueError,match='tokenizer'):p._backbone_gate('ouro_r1',assets,env,plan)


@pytest.fixture
def cached(tmp_path,monkeypatch):
    root=tmp_path/'frames';root.mkdir();cache={'manifest':write(tmp_path/'manifest.json',{'fixture':True})}
    frame=root/'one.png';frame.write_bytes(b'pinned PNG fixture')
    item={**retain.file_pin(frame),'path':'one.png'};files=[item]
    write(root/'.suite-media-owner.json',{'schema_version':1,'kind':'suite_benchmark_media',
        'cache_binding_sha256':retain.digest(cache),'media_root':str(root),'max_bytes':1000,'png_bytes':item['bytes']})
    metadata=root.parent/(root.name+'.acceptance')
    ready=write(metadata/'SUITE_BENCHMARK_READY.json',{'status':'SUITE_BENCHMARK_READY',
        'cache_binding_sha256':retain.digest(cache),'png_bytes':item['bytes'],'source_count':1,
        'all_png_full_oss_get_and_rgb_verified':True,'evidence':[cache['manifest']]})
    fragment={'manifest':cache['manifest'],'media_root':str(root),'media_files':files,'acceptance':ready}
    write(metadata/'ENVIRONMENT_FRAGMENT.json',fragment)
    monkeypatch.setattr(media,'_bound_cache_inventory',lambda _:([{}],{'sources':[{}]},files,4,{},['manifest']))
    def verify(data,_task):
        for row in data['media_files']:retain.check_pin({**row,'path':str(Path(data['media_root'])/row['path'])})
        return [{}]
    monkeypatch.setattr(p,'validate_benchmark',verify)
    return {'cache':cache,'media_root':str(root),'max_bytes':1000},root,metadata


def test_default_cache_checks_correct_location_without_network_or_writes(cached,monkeypatch):
    data,root,metadata=cached;forbid=Mock(side_effect=AssertionError('must not access bucket or hydrate'))
    monkeypatch.setattr(media,'hydrate_frame_cache',forbid);monkeypatch.setattr(media,'OSSStore',forbid)
    before={str(x):x.read_bytes() for x in root.parent.rglob('*') if x.is_file()}
    result=p._benchmark_gate(data,'tempcompass_mc',4,{})
    assert result['acceptance']['path']==str(metadata/'SUITE_BENCHMARK_READY.json')
    assert not result['network_or_hydration_requested'];forbid.assert_not_called()
    assert before=={str(x):x.read_bytes() for x in root.parent.rglob('*') if x.is_file()}
    (root/'one.png').write_bytes(b'changed PNG fixture')
    with pytest.raises(ValueError,match='pinned file changed'):p._benchmark_gate(data,'tempcompass_mc',4,{})


def test_missing_ready_never_downloads_implicitly(cached,monkeypatch):
    data,_,metadata=cached;(metadata/'SUITE_BENCHMARK_READY.json').unlink()
    hydrate=Mock(side_effect=AssertionError('must not hydrate'));monkeypatch.setattr(media,'hydrate_frame_cache',hydrate)
    with pytest.raises(FileNotFoundError):p._benchmark_gate(data,'tempcompass_mc',4,{})
    hydrate.assert_not_called()


def test_explicit_prepare_checks_capacity_before_creating_client(cached,monkeypatch):
    data,_,_=cached;calls=[]
    monkeypatch.setattr(runner,'capacity',lambda *_:calls.append('capacity'))
    monkeypatch.setattr(media,'OSSStore',lambda *a,**k:calls.append('client'))
    monkeypatch.setattr(media,'hydrate_frame_cache',lambda *a,**k:calls.append('hydrate'))
    env={'storage':{},'tools':{'oss_config':'/installed'}}
    p._benchmark_gate(data,'tempcompass_mc',4,env,prepare_media=True)
    assert calls==['capacity','client','hydrate'];calls.clear()
    monkeypatch.setattr(runner,'capacity',Mock(side_effect=ValueError('capacity blocked')))
    with pytest.raises(ValueError,match='capacity blocked'):p._benchmark_gate(data,'tempcompass_mc',4,env,prepare_media=True)
    assert not calls


def test_initial_capacity_combines_restore_and_first_training_peak(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b';a.mkdir();b.mkdir()
    plan={'tasks':[{'id':'gate','kind':'gate'},{'id':'alignment','kind':'train','stage':'alignment'},
        {'id':'sft','kind':'train','stage':'sft','historical_hint':{'checkpoint_step':600}}]}
    env={'suite_id':'suite','run_root':str(a/'runs'),'task_reserve_bytes':{'gate':1,'train_sft':200},'history':{
        'alignment':{'mode':'accepted','checkpoint':{'files':[{'path':'model','bytes':10,'sha256':'a'*64}]}},
        'sft':{'mode':'resume','checkpoint':{'files':[{'path':'model','bytes':20,'sha256':'b'*64}]}}}}
    binding={'startup':{'reserve_bytes':1},'tasks':{'sft':{'reserve_bytes':200}},'storage':{'roots':[str(a),str(b)],'limit_bytes':225,
        'other_reserve_bytes':0,'free_guard_bytes':0}};estimates={'sft':{'required_additional_bytes':200}}
    assert runner.capacity(binding,200)['projected_bytes']==200
    assert runner.capacity(binding,30)['projected_bytes']==30
    with pytest.raises(ValueError,match='projected storage'):p._initial_storage_gate(plan,binding,env,estimates)
    binding['storage']['limit_bytes']=231
    result=p._initial_storage_gate(plan,binding,env,estimates)
    assert result['task_reserve_bytes']==230 and not result['future_cleanup_credited']


def test_invalid_foundation_returns_gated_and_creates_nothing(tmp_path):
    result=p.inspect({},{},{},plan_sha256='a'*64,binding_sha256='b'*64,environment_sha256='c'*64,
        state_root=tmp_path/'untouched',prepare_media=True)
    assert result['status']=='SUITE_CPU_GATED' and len(result['failures'])==3
    assert not result['gpu_used'] and not (tmp_path/'untouched').exists();json.dumps(result)


def test_insufficient_training_reserve_gates_entire_suite_before_hydration(tmp_path,monkeypatch):
    task={'id':'train','kind':'train','stage':'alignment','model_id':'tiny',
        'config':{'path':'recipe.json'},'historical_hint':{'status':'planned'}}
    plan={'tasks':[task],'models':[{'id':'tiny','family':'ouro_r1'}],'training_data':{}}
    source=write(tmp_path/'source.json',{'fixture':True});write(tmp_path/'recipe.json',{'model':{},'training':{}})
    env={'experiment_root':str(tmp_path),'source_files':[source],'history':{},'task_reserve_bytes':{'train_alignment':1},
        'assets':{'training':{k:source for k in ('train_manifest','eval_manifest','formal_ready','frozen_source_manifest')},
                  'backbones':{'ouro_r1':{'language_root':'/lm','vision_root':'/vision'}},'benchmarks':{}}}
    binding={'source_files':[source],'tasks':{'train':{'reserve_bytes':1}}}
    for name in ('validate_plan','validate_environment'):monkeypatch.setattr(p,name,lambda *a,**k:True)
    monkeypatch.setattr(runner,'validate_binding',lambda *a,**k:True)
    for name in ('_bound_environment','_formal_gate','_training_gate','_source_gate','_backbone_gate','_development_gate','_initial_storage_gate'):
        monkeypatch.setattr(p,name,lambda *a,**k:{'fixture':True})
    monkeypatch.setattr(p,'load_operation',lambda *_:SimpleNamespace(estimate_stage_storage=lambda *_:{'required_additional_bytes':200}))
    hydrate=Mock(side_effect=AssertionError('upstream failures forbid hydration'));monkeypatch.setattr(p,'_benchmark_gate',hydrate)
    result=p.inspect(plan,binding,env,plan_sha256='a'*64,binding_sha256='b'*64,environment_sha256='c'*64,
        state_root=tmp_path/'state',prepare_media=True)
    assert result['status']=='SUITE_CPU_GATED' and any('understates' in f['error'] for f in result['failures'])
    hydrate.assert_not_called();json.dumps(result)


@pytest.fixture
def history(formal):
    root=formal['source'];config=json.loads(formal['config'].read_text());recipe=write(root/'configs/recipe.json',config)
    frozen=write(formal['source_manifest'],{'schema_version':1,'purpose':'DEEPPROBE_FROZEN_P1_SOURCE',
        'files':integrity.source_inventory(root)});training=formal['training'];cp=root.parent/'historical'/'checkpoint-600'
    write(cp/'deepprobe_config.json',config['model'])
    contract={'stage':'sft','config_sha256':recipe['sha256'],'frozen_source_manifest_sha256':frozen['sha256'],
        'train_manifest_sha256':training['train_manifest']['sha256'],'eval_manifest_sha256':training['eval_manifest']['sha256'],
        'formal_ready_sha256':training['formal_ready']['sha256'],'formal_training':True,'global_batch_size':128,'world_size':8}
    write(cp/'training_provenance.json',{'training_contract':contract});write(cp/'trainer_state.json',{'global_step':600,'max_steps':718})
    for name in ('projector.safetensors','language_model/model.safetensors','vision_model/model.safetensors',
        'language_model/config.json','vision_model/config.json','tokenizer/tokenizer_config.json',
        'image_processor/preprocessor_config.json','optimizer.pt','scheduler.pt',*(f'rng_state_{i}.pth' for i in range(8))):
        write(cp/name,{'synthetic':True})
    files=p.inventory(cp);evidence=write(root.parent/'independent.json',{'synthetic':True})
    audit={'status':'SUITE_HISTORY_ACCEPTED','task_id':'sft','mode':'resume','config_sha256':recipe['sha256'],
        'frozen_source_manifest_sha256':frozen['sha256'],'checkpoint_inventory_sha256':runner.digest(files),
        'evidence':[evidence],'optimizer_step':600,'exact_resume_accepted':True}
    h={'mode':'resume','acceptance':write(root.parent/'acceptance.json',audit),'config':recipe,
        'frozen_source_manifest':frozen,'source_root':str(root),'checkpoint':{'path':str(cp),'files':files}}
    task={'id':'sft','stage':'sft','config':{'path':'configs/recipe.json'},'expected_optimizer_steps':718,
        'historical_hint':{'checkpoint_step':600}}
    return task,{'experiment_root':str(root),'history':{'sft':h},'assets':{'training':training}},h,audit


def test_history_validates_actual_source_and_complete_recovery_state(history):
    task,env,h,_=history
    assert p._history_gate(task,env)['all_current_checkpoint_payloads_sha256_verified']
    Path(h['checkpoint']['path'],'rng_state_7.pth').unlink()
    with pytest.raises((ValueError, FileNotFoundError)):p._history_gate(task,env)
    (Path(h['source_root'])/'src/deepprobe_vlm/train.py').write_text('changed')
    with pytest.raises(ValueError,match='source tree differs'):p._history_gate(task,env)


def test_remote_history_requires_archive_audit_and_metadata_pins_without_downloading(history,monkeypatch):
    task,env,h,audit=history;actual=Path(h['checkpoint']['path'])
    prefix='oss://ss-devbox-stream/project/runs/old/training/checkpoint-600/'
    h['checkpoint'].update(path=str(actual.parent/'not-hydrated'),oss_prefix=prefix)
    h['checkpoint_metadata']={n:retain.file_pin(actual/n) for n in ('deepprobe_config.json','training_provenance.json','trainer_state.json')}
    archive=write(actual.parent/'archive.json',{'all_objects_full_get_sha256_verified':True,
        'checkpoint_oss_prefix':prefix,'checkpoint_inventory_sha256':runner.digest(h['checkpoint']['files'])})
    h['archive_acceptance']=archive;audit.update(checkpoint_oss_prefix=prefix,evidence=audit['evidence']+[archive])
    h['acceptance']=write(Path(h['acceptance']['path']),audit)
    monkeypatch.setattr(p,'verify_inventory',Mock(side_effect=AssertionError('no remote payload hydration')))
    result=p._history_gate(task,env)
    assert result['fresh_remote_payload_verification_deferred_to_restore']
    assert not result['all_current_checkpoint_payloads_sha256_verified'] and not Path(h['checkpoint']['path']).exists()
    h['checkpoint_metadata']['trainer_state.json']['sha256']='0'*64
    with pytest.raises(ValueError,match='archived file inventory'):p._history_gate(task,env)
    h['checkpoint_metadata']['trainer_state.json']=retain.file_pin(actual/'trainer_state.json');h.pop('archive_acceptance')
    with pytest.raises(ValueError,match='archive acceptance is not pinned'):p._history_gate(task,env)


def test_development_gate_requires_source_description_and_actual_selected_media(tmp_path):
    root=tmp_path/'media';root.mkdir();(root/'one.png').write_bytes(b'frame')
    records=[{'sample_id':str(i),'source_video_id':str(i),'split':'validation','dataset':'llava',
              'question':'What?','answer':'yes','image_paths':['one.png']} for i in range(32)]
    manifest=tmp_path/'dev.jsonl';manifest.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
    manifest_pin=retain.file_pin(manifest)
    meta={'scope':'development','manifest_sha256':manifest_pin['sha256'],
          'quality_score_based_selection':False,'source_description':'Synthetic development fixture'}
    source=write(tmp_path/'source-metadata.json',meta)
    env={'assets':{'development':{'manifest':manifest_pin,'source_metadata':source,'media_root':str(root)}}}
    assert p._development_gate(env)['sample_count']==32
    meta['source_description']='';env['assets']['development']['source_metadata']=write(Path(source['path']),meta)
    with pytest.raises(ValueError,match='Unbound latency'):p._development_gate(env)
    meta['source_description']='fixture';env['assets']['development']['source_metadata']=write(Path(source['path']),meta)
    (root/'one.png').unlink()
    with pytest.raises((ValueError,FileNotFoundError)):p._development_gate(env)
