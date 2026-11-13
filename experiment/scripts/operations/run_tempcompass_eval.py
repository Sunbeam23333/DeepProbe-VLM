#!/usr/bin/env python3
"""Strict TempCompass runner and explicit raw-output to official-scorer bridge.

Never submits scheduler jobs or writes OSS. CPU preflight is the default unless
--execute is explicitly supplied on an already admitted one-node eight-GPU job.
The frozen reference inference engine remains unchanged; its scores are retained
as diagnostic raw artifacts and never reported as the official TempCompass score.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

OPS=Path(__file__).resolve().parent
EXPERIMENT=OPS.parents[1]
PACKAGE=EXPERIMENT.parent
sys.path.insert(0,str(OPS))
sys.path.insert(0,str(EXPERIMENT/'src'))
from deepprobe_vlm.distributed_launch import single_node_torchrun, single_node_torchrun_environment
from model_registry import model_sequence, verify_registered_checkpoint, validate_model_config, canonical_sha
from prepare_tempcompass_reviewed_data import (PROTOCOL,COUNTS,PINS,REMOTE_ROOT,require,canonical,digest,sha,safe,
                                              json_once,verify_accepted_data)
SCORER_SHA='f35b543e76400236110964fbe6bb4c2f70d485fb501f7751b5e6ce7544d3f290'


def read(path):return json.loads(safe(path).read_text())


def relative_path(value):
    path=Path(value)
    require(not path.is_absolute() and '..' not in path.parts,'unsafe_relative_path')
    return path


def pinned(item,base=None):
    path=(Path(base)/relative_path(item['path'])) if base is not None else safe(item['path'])
    require(path.is_file() and sha(path)==item['sha256'] and
            ('bytes' not in item or path.stat().st_size==item['bytes']),'pinned_artifact_changed')
    return safe(path)


def tree_files(root):
    root=safe(root); require(root.is_dir(),'missing_tree')
    result={}
    for path in root.rglob('*'):
        require(not path.is_symlink(),'symlink_in_frozen_tree')
        if path.is_file():result[str(path.relative_to(root))]=path
    return result


def validate_plan(plan):
    require(plan.get('schema_version')==1 and plan.get('protocol')==PROTOCOL and
            plan.get('evaluation_scope')=='tempcompass_only' and plan.get('counts')==COUNTS,'plan_protocol_or_counts_changed')
    model_sequence(plan)
    require(plan.get('num_gpus')==8 and plan.get('num_nodes')==1 and plan.get('max_new_tokens')==16 and
            plan.get('num_frames')==8 and plan.get('pool_grid')==4 and plan.get('audio') is False and
            plan.get('subtitles') is False and plan.get('include_answer') is False and plan.get('decoding')=='greedy_argmax',
            'inference_contract_changed')
    require(plan.get('media_root')==str(REMOTE_ROOT) and plan.get('manifest_sha256')==PINS['manifest'],'data_identity_changed')
    require(plan.get('reference_engine_protocol')=='reference' and plan.get('official_scorer_sha256')==SCORER_SHA,
            'engine_or_scorer_changed')
    require(plan.get('storage_limit_bytes')==99000000000 and plan.get('working_reserve_bytes')==1000000000 and
            plan.get('restart_policy')=='Never' and plan.get('priority')=='low','admission_bounds_changed')
    require(isinstance(plan.get('run_id'),str) and plan['run_id'].startswith('dp-benchmark-tempcompass-') and
            Path(plan['run_id']).name==plan['run_id'] and '..' not in plan['run_id'] and
            plan['run_root']=='/home/jinhaopeng/.cache/Paper/DeepProbe-VLM/evaluations/'+plan['run_id'],'run_identity_changed')
    return plan


def verify_sources(plan,package=PACKAGE):
    supplied={}
    for item in plan['source_files']:
        require(item['path'] not in supplied,'duplicate_source_pin')
        supplied[item['path']]=pinned(item,package)
    actual={str(p.relative_to(package)) for p in (Path(package)/'experiment/src').rglob('*.py')}
    actual|={f'experiment/scripts/operations/{name}' for name in
             ('prepare_tempcompass_reviewed_data.py','run_tempcompass_eval.py','score_tempcompass_mc.py','model_registry.py')}
    require(set(supplied)==actual,'complete_executable_source_set_required')
    require(sha(supplied['experiment/scripts/operations/score_tempcompass_mc.py'])==SCORER_SHA,'official_scorer_changed')
    return digest(canonical(plan['source_files']))


def verify_checkpoint(binding):
    return verify_registered_checkpoint(binding)


def check_training_sources(checkpoint,records,accepted_training_sources):
    provenance=read(Path(checkpoint)/'training_provenance.json')
    for name in ('training_sources','tuning_sources'):
        require(isinstance(provenance.get(name),list) and all(isinstance(s,str) and s for s in provenance[name]),'training_provenance_invalid')
    observed=set(provenance['training_sources'])|set(provenance['tuning_sources'])
    require(observed <= set(accepted_training_sources),'checkpoint_sources_not_covered_by_content_scan')
    benchmark={s for r in records for s in r['underlying_source_video_ids']}|{r['source_video_id'] for r in records}
    require(not observed & benchmark,'checkpoint_training_or_tuning_overlap')
    return {'training_provenance_sha256':sha(Path(checkpoint)/'training_provenance.json'),
            'training_source_count':len(set(provenance['training_sources'])),'tuning_source_count':len(set(provenance['tuning_sources'])),
            'all_recorded_sources_covered_by_complete_content_scan':True,'declared_source_overlap':[]}


def prompt_budget(records,tokenizer,max_text_tokens):
    sys.path.insert(0,str(EXPERIMENT/'src'))
    from deepprobe_vlm.data import MultimodalCollator
    collator=MultimodalCollator(tokenizer,num_visual_tokens=128,max_text_tokens=max_text_tokens,include_answer=False)
    lengths=[]
    for row in records:
        prefix,suffix,answer=collator.tokenize_record(row)
        require(answer==[],'gold_tokens_in_prompt')
        lengths.append(len(prefix)+len(suffix))
    require(len(lengths)==1580 and max(lengths)<=max_text_tokens,'prompt_budget_failed')
    return {'num_examples':1580,'max_prompt_text_tokens':max(lengths),'max_text_tokens':max_text_tokens,
            'visual_tokens':128,'max_generated_tokens':16,'num_over_budget':0,'include_answer':False,'truncation':False,
            'ordered_prompt_lengths_sha256':digest(canonical(lengths))}


def preflight(plan,plan_sha,package=PACKAGE):
    validate_plan(plan); source_digest=verify_sources(plan,package)
    ready_path=pinned(plan['accepted_data'],package)
    data=verify_accepted_data(ready_path,plan['accepted_data']['sha256'],plan['media_root'])
    require(data['manifest_sha256']==plan['manifest_sha256'],'accepted_manifest_changed')
    # HuggingFace CPU reads must stay local, including tokenizer assets.
    os.environ.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'})
    from transformers import AutoTokenizer
    models={}
    for name in model_sequence(plan):
        item=verify_checkpoint(plan['model_bindings'][name])
        item['source_check']=check_training_sources(item['checkpoint'],data['records'],data['training_sources'])
        tokenizer=AutoTokenizer.from_pretrained(Path(item['checkpoint'])/'tokenizer',trust_remote_code=False,local_files_only=True)
        item['token_budget']=prompt_budget(data['records'],tokenizer,item['config']['max_text_tokens'])
        item.update({'model':name,'plan_sha256':plan_sha,'manifest_sha256':plan['manifest_sha256'],
                     'accepted_data_sha256':plan['accepted_data']['sha256'],'source_files_sha256':source_digest,
                     'protocol':PROTOCOL,'reference_engine_protocol':'reference'})
        models[name]=item
    # Reuse the already pinned capacity routine; this call never writes output.
    release=Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/tempcompass-cpu-release-20261021-v2')
    helper_path=release/'experiment/scripts/operations/complete_videomme_from_oss.py'
    require(sha(helper_path)=='444187e3155686f8f82344ccc1e798d595b2258472f6afee6b9f3c8bea087c34','capacity_helper_changed')
    spec=importlib.util.spec_from_file_location('tempcompass_capacity',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    roots=[Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM'),Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1')]
    current=sum(helper.tree_bytes(root) for root in roots)
    require(current+plan['working_reserve_bytes']<plan['storage_limit_bytes'],'evaluation_capacity_failed')
    return {'status':'TEMP_COMPASS_CPU_PREFLIGHT_PASSED','protocol':PROTOCOL,'plan_sha256':plan_sha,
            'accepted_data_sha256':plan['accepted_data']['sha256'],'manifest_path':data['manifest_path'],
            'counts':COUNTS,'models':models,'source_files_sha256':source_digest,'num_gpus':8,'num_nodes':1,
            'model_sequence':model_sequence(plan),'current_bytes':current,'projected_bytes':current+plan['working_reserve_bytes'],
            'gpu_submitted':False,'gpu_admission_granted':False}


def assigned_records(records,rank,world=8):
    require(world==8 and 0<=rank<world,'invalid_shard')
    owners={s:i%world for i,s in enumerate(dict.fromkeys(r['source_video_id'] for r in records))}
    return [r for r in records if owners[r['source_video_id']]==rank]


def bridge_predictions(records,raw_rows):
    require(len(records)==1580 and len({r['sample_id'] for r in records})==1580,'bridge_manifest_incomplete')
    by_id={r['sample_id']:r for r in raw_rows}
    require(len(by_id)==len(raw_rows)==1580 and set(by_id)=={r['sample_id'] for r in records},'raw_prediction_coverage_changed')
    bridged=[]
    for record in records:
        raw=by_id[record['sample_id']]
        require(raw.get('record_sha256')==digest(canonical(record)) and
                all(raw.get(k)==record[k] for k in ('sample_id','source_video_id','answer','split')),'raw_record_binding_changed')
        require('prediction' in raw and (raw['prediction'] is None or isinstance(raw['prediction'],str)),'raw_prediction_must_be_text_or_null')
        bridged.append({'sample_id':record['sample_id'],'manifest_row_sha256':raw['record_sha256'],
                        'prediction':raw['prediction'],'raw_prediction_row_sha256':digest(canonical(raw))})
    return bridged


def shard_file(root,rank):return Path(root)/'raw'/f'predictions.rank-{rank:05d}-of-00008.jsonl'


def merge_and_score(model_root,records,provenance_path,plan,plan_sha):
    model_root=safe(model_root); provenance=read(provenance_path); raw_rows=[]; shards=[]
    for rank in range(8):
        path=shard_file(model_root,rank); protocol_path=path.with_suffix('.protocol.json')
        protocol=read(protocol_path); summary=read(path.with_suffix('.summary.json'))
        scoped=read(path.with_suffix('.tempcompass.json')); assigned=assigned_records(records,rank)
        expected={'protocol':'reference','max_new_tokens':16,'cache':True,'decoding':'greedy_argmax',
                  'audio':False,'subtitles':False,'manifest_sha256':PINS['manifest'],'checkpoint':provenance['checkpoint'],
                  'checkpoint_config':provenance['config'],'run_provenance_sha256':sha(provenance_path),
                  'shard_index':rank,'num_shards':8,'num_examples':len(assigned),
                  'ordered_sample_ids_sha256':digest(canonical([r['sample_id'] for r in assigned]))}
        require(all(protocol.get(k)==v for k,v in expected.items()),'raw_engine_protocol_changed')
        require(scoped.get('protocol')==PROTOCOL and scoped.get('plan_sha256')==plan_sha and
                scoped.get('run_provenance_sha256')==sha(provenance_path) and scoped.get('rank')==rank and scoped.get('world_size')==8,
                'scoped_engine_protocol_changed')
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        require(len(rows)==len(assigned) and summary['predictions_sha256']==sha(path) and summary['num_examples']==len(assigned),
                'raw_shard_incomplete')
        for row,record in zip(rows,assigned):
            require(row['sample_id']==record['sample_id'] and row['record_sha256']==digest(canonical(record)) and
                    row['protocol_sha256']==digest(canonical(protocol)) and row['shard_index']==rank and row['num_shards']==8 and
                    row.get('cache') is True and row.get('num_loops')==provenance['config']['num_loops'] and
                    type(row.get('generated_tokens')) is int and 0<=row['generated_tokens']<=16,'raw_shard_row_changed')
        raw_rows.extend(rows);shards.append({'rank':rank,'path':str(path),'sha256':sha(path),
            'protocol_sha256':sha(protocol_path),'scoped_protocol_sha256':sha(path.with_suffix('.tempcompass.json')),
            'summary_sha256':sha(path.with_suffix('.summary.json')),'count':len(rows)})
    bridged=bridge_predictions(records,raw_rows)
    by_id={r['sample_id']:r for r in raw_rows}; ordered=[by_id[r['sample_id']] for r in records]
    result_root=model_root/'official';result_root.mkdir(exist_ok=False)
    def lines(name,rows):
        with (result_root/name).open('xb') as f:
            for row in rows:f.write(canonical(row)+b'\n')
            f.flush();os.fsync(f.fileno())
    lines('raw_predictions_merged.jsonl',ordered)
    lines('bridged_predictions.jsonl',bridged)
    scorer_path=OPS/'score_tempcompass_mc.py';require(sha(scorer_path)==SCORER_SHA,'scorer_changed')
    spec=importlib.util.spec_from_file_location('tempcompass_official_scorer',scorer_path)
    scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
    scored,summary=scorer.score(records,bridged);lines('scored_predictions.jsonl',scored)
    summary.update({'manifest_sha256':PINS['manifest'],'plan_sha256':plan_sha,
        'raw_predictions_sha256':sha(result_root/'raw_predictions_merged.jsonl'),
        'bridged_predictions_sha256':sha(result_root/'bridged_predictions.jsonl'),
        'model_provenance_admission_performed':True,'model_provenance_sha256':sha(provenance_path),
        'accepted_data_sha256':plan['accepted_data']['sha256'],'quality_claim_scope':'fixed TempCompass MC uniform8 protocol; sampled-frame training overlap gate passed with two bound visual reviews'})
    json_once(result_root/'OFFICIAL_SCORES.json',summary)
    json_once(result_root/'BRIDGE_RECEIPT.json',{'schema_version':1,'status':'COMPLETE_RAW_PREDICTIONS_EXPLICITLY_BRIDGED',
        'manifest_sha256':PINS['manifest'],'count':1580,'raw_text_preserved':True,'original_raw_shards_modified':False,
        'source_record_sha256_validated':True,'destination_manifest_row_sha256_equal':True,
        'raw_predictions_sha256':sha(result_root/'raw_predictions_merged.jsonl'),
        'bridged_predictions_sha256':sha(result_root/'bridged_predictions.jsonl'),'shards':shards,'scorer_sha256':SCORER_SHA})
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args(argv);require(sha(a.plan)==a.plan_sha256,'plan_sha_changed');plan=read(a.plan)
    result=preflight(plan,a.plan_sha256,PACKAGE)
    if not a.execute:print(json.dumps(result,indent=2,sort_keys=True));return 0
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count()==8,'exactly_eight_visible_gpus_required')
    # Execution has no scheduler/OSS API and cannot grant its own scheduling admission.
    require(os.environ.get('TEMP_COMPASS_ADMITTED_PLAN_SHA256')==a.plan_sha256,'parent_gpu_admission_required')
    root=safe(plan['run_root']);require(not root.exists(),'fresh_run_required_no_automatic_resume')
    lock_path=Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM/locks/p1-stage.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        root.mkdir();json_once(root/'PLAN.json',plan);json_once(root/'CPU_PREFLIGHT.json',result)
        data=verify_accepted_data(pinned(plan['accepted_data'],PACKAGE),plan['accepted_data']['sha256'])
        environment=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
            PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONPATH=str(EXPERIMENT/'src'),PYTHONUNBUFFERED='1',
            PET_LOCAL_ADDR='127.0.0.1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        scores={}
        for name in plan['model_sequence']:
            model_root=root/name;(model_root/'raw').mkdir(parents=True)
            provenance=model_root/'MODEL_PROVENANCE.json';json_once(provenance,result['models'][name])
            command=single_node_torchrun(sys.executable)+[
                '-m','deepprobe_vlm.evaluate_tempcompass','--plan',str(a.plan),'--plan-sha256',a.plan_sha256,
                '--model',name,'--provenance',str(provenance)]
            with (model_root/'inference.log').open('xb') as log:
                completed=subprocess.run(command,cwd=EXPERIMENT,env=single_node_torchrun_environment(environment),
                                         stdout=log,stderr=subprocess.STDOUT)
            json_once(model_root/'EXIT.json',{'exit_code':completed.returncode,'argv':command,'automatic_retry':False})
            if completed.returncode:return completed.returncode
            scores[name]=merge_and_score(model_root,data['records'],provenance,plan,a.plan_sha256)
        json_once(root/'EVALUATION_COMPLETE.json',{'status':'TEMP_COMPASS_DECLARED_MODELS_OFFICIALLY_SCORED','plan_sha256':a.plan_sha256,
            'counts':COUNTS,'num_models':len(scores),'raw_predictions':len(data['records'])*len(scores),'models':{name:{'overall':score['overall'],
                'official_scores_sha256':sha(root/name/'official/OFFICIAL_SCORES.json')} for name,score in scores.items()},
            'finished_utc':datetime.now(timezone.utc).isoformat(),'oss_uploaded':False})
    return 0


if __name__=='__main__':raise SystemExit(main())
