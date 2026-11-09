import pytest
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unified_evaluation_fixtures import catalog

REPO=Path(__file__).resolve().parents[2]
OPS=REPO/'experiment/scripts/operations'
sys.path.insert(0,str(OPS))
import run_tempcompass_eval as runner
import score_tempcompass_mc as scorer


class Tokenizer:
    eos_token_id=0
    pad_token_id=0
    bos_token_id=1
    def __init__(self):self.text=[]
    def encode(self,text,add_special_tokens=False):
        self.text.append(text)
        return list(range(len(text)))


@pytest.mark.private_evidence
class TempCompassEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=REPO/'experiment/docs/operations/evidence/tempcompass-cpu-preparation-20261021/execution-v2/collected/output/manifest.jsonl'
        cls.records=[json.loads(s) for s in p.read_text().splitlines()]

    def raw(self,prediction='A'):
        return [{'sample_id':r['sample_id'],'source_video_id':r['source_video_id'],'answer':r['answer'],'split':r['split'],
                 'record_sha256':runner.digest(runner.canonical(r)),'prediction':prediction,'diagnostic_correct':123}
                for r in self.records]

    def test_complete_bridge_preserves_raw_text_and_all_rows(self):
        text='  A. a raw response\n</s> 原样  '
        raw=self.raw(text);before=copy.deepcopy(raw)
        bridged=runner.bridge_predictions(self.records,raw)
        self.assertEqual(raw,before)
        self.assertEqual(len(bridged),1580)
        self.assertEqual({r['prediction'] for r in bridged},{text})
        for source,dest in zip(raw,bridged):
            self.assertEqual(dest['manifest_row_sha256'],source['record_sha256'])
            self.assertEqual(dest['raw_prediction_row_sha256'],runner.digest(runner.canonical(source)))

    def test_bridge_rejects_missing_duplicate_extra_or_foreign_samples(self):
        raw=self.raw()
        cases=[raw[:-1],raw+[raw[0]],raw[:-1]+[raw[0]]]
        foreign=copy.deepcopy(raw);foreign[0]['sample_id']='foreign';cases.append(foreign)
        for changed in cases:
            with self.subTest(length=len(changed)),self.assertRaises(ValueError):runner.bridge_predictions(self.records,changed)

    def test_bridge_rejects_wrong_record_hash_source_gold_or_type(self):
        for key,value in [('record_sha256','0'*64),('source_video_id','wrong'),('answer','Z'),('prediction',False)]:
            raw=self.raw();raw[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):runner.bridge_predictions(self.records,raw)

    def test_bridge_does_not_replace_null_failures_or_official_invalid_text(self):
        for prediction in (None,'Answer: A'):
            bridged=runner.bridge_predictions(self.records,self.raw(prediction))
            scored,summary=scorer.score(self.records,bridged)
            self.assertEqual(summary['overall']['count'],1580)
            self.assertEqual(summary['overall']['correct'],0)
            self.assertEqual(summary['overall']['rule_match_count'],0)
            self.assertTrue(all(r['prediction']==prediction for r in scored))

    def test_shards_preserve_entire_source_components_and_all_clips(self):
        shards=[runner.assigned_records(self.records,i) for i in range(8)]
        ids=[r['sample_id'] for shard in shards for r in shard]
        self.assertEqual(len(ids),1580);self.assertEqual(len(set(ids)),1580)
        owners={}
        for rank,shard in enumerate(shards):
            for row in shard:owners.setdefault(row['source_video_id'],set()).add(rank)
        self.assertEqual(len(owners),188)
        self.assertTrue(all(len(x)==1 for x in owners.values()))
        self.assertEqual(len({r['video_id'] for s in shards for r in s}),410)

    def test_prompt_check_is_answer_blind_and_never_truncates(self):
        tokenizer=Tokenizer()
        a=runner.prompt_budget(self.records,tokenizer,4096)
        changed=copy.deepcopy(self.records)
        for row in changed:row['answer']='A_GOLD_SENTINEL_SHOULD_NOT_APPEAR'
        second=Tokenizer();b=runner.prompt_budget(changed,second,4096)
        self.assertEqual(a,b);self.assertEqual(tokenizer.text,second.text)
        self.assertFalse(any('A_GOLD_SENTINEL' in text for text in second.text))
        with self.assertRaises(ValueError):runner.prompt_budget(self.records,Tokenizer(),1)

    def test_no_video_mme_metric_is_used_for_official_score(self):
        raw=self.raw('Answer: A')
        for row in raw:row.update(correct=1,metric='strict_mcq_accuracy')
        bridged=runner.bridge_predictions(self.records,raw)
        _,summary=scorer.score(self.records,bridged)
        self.assertEqual(summary['overall']['correct'],0)
        self.assertTrue(summary['official_avg_is_qa_micro_accuracy'])

    def synthetic_shards(self,root):
        root=Path(root).resolve();(root/'raw').mkdir()
        provenance={'checkpoint':'/synthetic-model','config':{'num_loops':1},'manifest_sha256':runner.PINS['manifest']}
        proof=root/'MODEL_PROVENANCE.json';runner.json_once(proof,provenance)
        plan={'accepted_data':{'sha256':'1'*64}};plan_sha='2'*64
        for rank in range(8):
            path=runner.shard_file(root,rank);assigned=runner.assigned_records(self.records,rank)
            protocol={'protocol':'reference','max_new_tokens':16,'cache':True,'decoding':'greedy_argmax',
                'audio':False,'subtitles':False,'manifest_sha256':runner.PINS['manifest'],'checkpoint':provenance['checkpoint'],
                'checkpoint_config':provenance['config'],'run_provenance_sha256':runner.sha(proof),
                'shard_index':rank,'num_shards':8,'num_examples':len(assigned),
                'ordered_sample_ids_sha256':runner.digest(runner.canonical([r['sample_id'] for r in assigned]))}
            runner.json_once(path.with_suffix('.protocol.json'),protocol)
            runner.json_once(path.with_suffix('.tempcompass.json'),{'protocol':runner.PROTOCOL,'plan_sha256':plan_sha,
                'run_provenance_sha256':runner.sha(proof),'rank':rank,'world_size':8})
            with path.open('x') as stream:
                for record in assigned:
                    row={'sample_id':record['sample_id'],'source_video_id':record['source_video_id'],
                        'answer':record['answer'],'split':record['split'],'record_sha256':runner.digest(runner.canonical(record)),
                        'protocol_sha256':runner.digest(runner.canonical(protocol)),'shard_index':rank,'num_shards':8,
                        'cache':True,'num_loops':1,'generated_tokens':1,'prediction':record['answer']}
                    stream.write(json.dumps(row)+'\n')
            runner.json_once(path.with_suffix('.summary.json'),{'predictions_sha256':runner.sha(path),'num_examples':len(assigned)})
        return proof,plan,plan_sha

    def test_complete_shard_merge_bridges_scores_and_preserves_originals(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            proof,plan,plan_sha=self.synthetic_shards(root)
            raw={p.name:p.read_bytes() for p in (Path(root)/'raw').iterdir()}
            summary=runner.merge_and_score(root,self.records,proof,plan,plan_sha)
            self.assertEqual(summary['overall']['count'],1580)
            self.assertEqual(summary['overall']['correct'],1580)
            self.assertEqual(raw,{p.name:p.read_bytes() for p in (Path(root)/'raw').iterdir()})
            receipt=json.loads((Path(root)/'official/BRIDGE_RECEIPT.json').read_text())
            self.assertTrue(receipt['raw_text_preserved'])
            self.assertEqual(len(receipt['shards']),8)
            with self.assertRaises(FileExistsError):runner.merge_and_score(root,self.records,proof,plan,plan_sha)

    def test_shard_protocol_or_completion_tampering_blocks_merge(self):
        for target in ('protocol','completion','row'):
            with self.subTest(target=target),tempfile.TemporaryDirectory() as directory:
                root=Path(directory).resolve()
                proof,plan,plan_sha=self.synthetic_shards(root)
                path=runner.shard_file(root,0)
                if target=='protocol':
                    p=path.with_suffix('.protocol.json');data=json.loads(p.read_text());data['max_new_tokens']=32;p.write_text(json.dumps(data))
                elif target=='completion':
                    p=path.with_suffix('.summary.json');data=json.loads(p.read_text());data['num_examples']-=1;p.write_text(json.dumps(data))
                else:
                    rows=[json.loads(s) for s in path.read_text().splitlines()];rows[0]['generated_tokens']=17
                    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                    p=path.with_suffix('.summary.json');data=json.loads(p.read_text());data['predictions_sha256']=runner.sha(path);p.write_text(json.dumps(data))
                with self.assertRaises(ValueError):runner.merge_and_score(root,self.records,proof,plan,plan_sha)

    def test_original_model_and_generation_sources_equal_frozen_videomme(self):
        plan=json.loads((REPO/'experiment/docs/operations/evidence/experiment-execution-20261021/videomme-a3-preparation/release/EVALUATION_PLAN.json').read_text())
        for relative in ('data.py','modeling.py','evaluate.py','manifest.py'):
            ref=next(x for x in plan['source_files'] if x['path'].endswith('/src/deepprobe_vlm/'+relative))
            self.assertEqual(runner.sha(REPO/'experiment/src/deepprobe_vlm'/relative),ref['sha256'])





class TempCompassPlanTests(unittest.TestCase):
    def test_video_mme_or_weaker_inference_contract_cannot_enter(self):
        models = catalog()['model_bindings']
        selected = {name: models[name] for name in ('r1', 'r2')}
        for index, (name, binding) in enumerate(selected.items(), 1):
            binding.update(checkpoint={'path': '/synthetic/' + name, 'run_id': 'synthetic-' + name},
                           training_graph={}, training_acceptance={'sha256': str(index) * 64})
        plan={'schema_version':1,'protocol':runner.PROTOCOL,'evaluation_scope':'tempcompass_only','counts':runner.COUNTS,
              'model_sequence':['r1','r2'],'model_bindings':selected,
              'benchmark_score_based_model_selection':False,'num_gpus':8,'num_nodes':1,
              'max_new_tokens':16,'num_frames':8,'pool_grid':4,'audio':False,'subtitles':False,'include_answer':False,
              'decoding':'greedy_argmax','media_root':str(runner.REMOTE_ROOT),'manifest_sha256':runner.PINS['manifest'],
              'reference_engine_protocol':'reference','official_scorer_sha256':runner.SCORER_SHA,
              'storage_limit_bytes':99000000000,'working_reserve_bytes':1000000000,'restart_policy':'Never','priority':'low',
              'run_id':'dp-benchmark-tempcompass-r1-r2-s17-a1',
              'run_root':'/home/jinhaopeng/.cache/Paper/DeepProbe-VLM/evaluations/dp-benchmark-tempcompass-r1-r2-s17-a1'}
        runner.validate_plan(plan)
        for key,value in [('protocol','videomme-899-no-subtitles-no-audio-v1'),('include_answer',True),('max_new_tokens',64),
                          ('num_gpus',1),('model_sequence',['r1','undeclared']),('priority','normal'),('media_root','/other')]:
            changed=copy.deepcopy(plan);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):runner.validate_plan(changed)


if __name__=='__main__':unittest.main()
