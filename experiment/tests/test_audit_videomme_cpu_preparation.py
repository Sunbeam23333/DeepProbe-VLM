from pathlib import Path
import hashlib
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / 'scripts/operations/audit_videomme_cpu_preparation.py'
spec = importlib.util.spec_from_file_location('cpu_independent_gate', MODULE)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class EvidenceGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name).resolve()

    def tearDown(self): self.tmp.cleanup()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value)); return path

    def test_pin_rejects_changed_and_symlink_bytes(self):
        p = self.root/'proof.json'; p.write_text('a'); item=m.pin(p)
        self.assertEqual(m.verify(item),p);p.write_text('b')
        with self.assertRaises(ValueError):m.verify(item)
        link=self.root/'alias';link.symlink_to(p)
        with self.assertRaises(ValueError):m.pin(link)

    def test_duplicate_or_short_inventory_fails(self):
        with self.assertRaises(ValueError):m.index([{'id':'a'},{'id':'a'}],'id',2)
        with self.assertRaises(ValueError):m.index([{'id':'a'}],'id',2)

    def test_scope_rejects_full_benchmark_or_second_model_selection(self):
        plan={'schema_version':1,'run_id':m.RUN_ID,'protocol':m.PROTOCOL,'evaluation_scope':'videomme_only',
              'all_benchmark_assets_ready':False,'mvbench_evaluation_ready':False,'benchmark_score_based_model_selection':False,
              'run_root':str(m.BASE/'evaluations'/m.RUN_ID),'media_root':str(m.MEDIA),'storage_roots':[str(m.BASE),str(m.P1)],
              'storage_limit_bytes':99_000_000_000,'working_reserve_bytes':1_000_000_000}
        m.check_scope(plan)
        for key in ('all_benchmark_assets_ready','mvbench_evaluation_ready','benchmark_score_based_model_selection'):
            with self.assertRaises(ValueError):m.check_scope(dict(plan,**{key:True}))

    def stage_fixture(self):
        scripts={'content_audit':'audit_benchmark_content.py','formal_gate':'prepare_formal_source_gate.py',
                 'manifest':'prepare_videomme_evaluation.py','cpu_preflight':'run_benchmark_eval.py'}
        for stage,script in scripts.items():
            (self.root/(stage+'.stdout')).write_text('{}');(self.root/(stage+'.stderr')).write_text('')
            self.write(self.root/(stage+'.exit.json'),{'returncode':0,'stdout_sha256':m.sha(self.root/(stage+'.stdout')),'stderr_sha256':m.sha(self.root/(stage+'.stderr'))})
            argv=[str(m.BASE/'env/bin/python'),str(m.P1/'benchmark-evaluation-code-20261013-v2/experiment/scripts/operations'/script)]
            if stage=='cpu_preflight':argv+=['--preflight-only']
            self.write(self.root/(stage+'.argv.json'),{'CUDA_VISIBLE_DEVICES':'','argv':argv})

    def test_scoped_gate_cannot_promote_underlying_failure(self):
        underlying={'formal_evaluation_ready':False,'evaluation_readiness_gates':{'content':{'status':'failed'}}}
        scoped=dict(underlying,evaluation_scope='videomme_only')
        m.preserved_underlying_gate(scoped,underlying)
        scoped['formal_evaluation_ready']=True
        with self.assertRaises(ValueError):m.preserved_underlying_gate(scoped,underlying)
        scoped=dict(underlying);del scoped['formal_evaluation_ready']
        with self.assertRaises(ValueError):m.preserved_underlying_gate(scoped,underlying)

    def test_formal_content_validation_rejects_a_different_audit(self):
        content={'binding':{'manifest_sha256':'a'*64},'passed':True}
        p=self.write(self.root/'CONTENT_AUDIT.json',content)
        valid={'provided':True,'passed':True,'errors':[],'path':str(p),'sha256':m.sha(p),'required_binding':content['binding']}
        m.bound_content_validation(valid,p,content)
        for key,value in [('path',str(self.root/'other.json')),('sha256','b'*64),('required_binding',{}),('passed',False),('errors',['bad'])]:
            with self.assertRaises(ValueError):m.bound_content_validation(dict(valid,**{key:value}),p,content)

    def test_stage_outputs_are_byte_bound(self):
        self.stage_fixture();self.assertEqual(len(m.stage_outputs(self.root)),4)
        (self.root/'content_audit.stdout').write_text('changed')
        with self.assertRaises(ValueError):m.stage_outputs(self.root)

    def test_stage_failure_or_gpu_mode_blocks(self):
        self.stage_fixture();p=self.root/'manifest.exit.json';d=m.load(p);d['returncode']=2;self.write(p,d)
        with self.assertRaises(ValueError):m.stage_outputs(self.root)
        self.stage_fixture();p=self.root/'cpu_preflight.argv.json';d=m.load(p);d['argv'].remove('--preflight-only');self.write(p,d)
        with self.assertRaises(ValueError):m.stage_outputs(self.root)

    def checkpoint_fixture(self):
        root=self.root/'final';root.mkdir()
        cfg={'num_loops':1,'num_frames':8,'pool_grid':4,'max_text_tokens':512}
        self.write(root/'deepprobe_config.json',cfg);self.write(root/'training_provenance.json',{'training_sources':['train:a'],'tuning_sources':[]})
        for i in range(18):(root/f'weight-{i}').write_bytes(bytes([i]))
        rows=[{'path':p.name,'bytes':p.stat().st_size,'sha256':m.sha(p)} for p in sorted(root.iterdir())]
        item={'path':str(root),'file_count':20,'files':rows,'run_id':'dp-p1-ouro-r1-f8-g4-s17-sft-a1'}
        pre={'checkpoint':str(root),'run_id':item['run_id'],'checkpoint_files_sha256':m.canonical(rows),
             'checkpoint_config_sha256':m.sha(root/'deepprobe_config.json'),'all_checkpoint_files_sha256_verified':True,'config':cfg,
             'token_budget':{'num_examples':2697,'include_answer':False,'truncation':False,'num_over_budget':0,'max_prompt_text_tokens':274,'max_text_tokens':512}}
        return root,item,pre

    def test_all_twenty_checkpoint_files_rehashed(self):
        root,item,pre=self.checkpoint_fixture();self.assertEqual(m.verify_checkpoint(item,1,pre,{'youtube:b'})['file_count'],20)
        (root/'weight-1').write_bytes(b'x')
        with self.assertRaises(ValueError):m.verify_checkpoint(item,1,pre,{'youtube:b'})

    def test_model_provenance_overlap_blocks(self):
        _,item,pre=self.checkpoint_fixture()
        with self.assertRaises(ValueError):m.verify_checkpoint(item,1,pre,{'train:a'})

    def test_wrong_model_order_or_token_truncation_blocks(self):
        _,item,pre=self.checkpoint_fixture()
        with self.assertRaises(ValueError):m.verify_checkpoint(item,2,pre,{'youtube:b'})
        pre['token_budget']['truncation']=True
        with self.assertRaises(ValueError):m.verify_checkpoint(item,1,pre,{'youtube:b'})

    def test_checkpoint_extra_file_blocks(self):
        root,item,pre=self.checkpoint_fixture();(root/'extra').write_bytes(b'')
        with self.assertRaises(ValueError):m.verify_checkpoint(item,1,pre,{'youtube:b'})

    def test_capacity_charges_external_aliases_without_internal_duplicate(self):
        a,b=self.root/'a',self.root/'b';a.mkdir();b.mkdir();(a/'payload').write_bytes(b'x'*9)
        ext=self.root/'external';ext.write_bytes(b'y'*7);(b/'e1').symlink_to(ext);(b/'e2').symlink_to(ext);(b/'internal').symlink_to(a/'payload')
        with patch.multiple(m,BASE=a,P1=b):
            result=m.capacity()
        self.assertEqual(sum(result['root_bytes'].values()),23)

    def test_external_directory_link_blocks(self):
        a,b=self.root/'a',self.root/'b';a.mkdir();b.mkdir();ext=self.root/'outside';ext.mkdir();(a/'unsafe').symlink_to(ext,target_is_directory=True)
        with patch.multiple(m,BASE=a,P1=b), self.assertRaises(ValueError):m.capacity()

    def test_audit_module_has_no_subprocess_submission_surface(self):
        import ast
        tree=ast.parse(MODULE.read_text())
        names={n.names[0].name for n in ast.walk(tree) if isinstance(n,ast.Import)}
        self.assertFalse({'subprocess','torch','oss2'} & names)


if __name__=='__main__':unittest.main(verbosity=2)
