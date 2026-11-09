"""Fresh a2 identity and all review evidence must precede the frozen helper."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reviewed_submit', ROOT / 'scripts/operations/submit_videomme_reviewed_eval_once.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


class ReviewedSubmitTests(unittest.TestCase):
    def fixture(self, root):
        original = root / 'original.json'; write(original, {'passed':False})
        review = root / 'review.json'; write(review, {'review':'complete'})
        independent = root / 'independent.json'; write(independent, {'reviewer':'root'})
        def pin(p): return {'path':str(p),'sha256':s.sha(p)}
        proof={'passed':True,'original_candidates_preserved':True,'original_audit_passed':False,
               'candidate_count':12,'matched_frame_count':30,'review':pin(review),
               'independent_review':dict(pin(independent),reviewer='root'),
               'artifacts':[pin(original),pin(review),pin(independent)]}
        validation = root / 'validation.json'
        write(validation, {'provided':True,'passed':True,'errors':[],'path':str(original),'sha256':s.sha(original),
            'admission_mode':'original_scan_plus_bound_candidate_review_v1','adjudication':proof})
        own=Path(s.__file__).resolve()
        plan={'run_id':s.RUN_ID,'run_root':str(root/s.RUN_ID),'evaluation_scope':'videomme_only',
              'all_benchmark_assets_ready':False,'mvbench_evaluation_ready':False,'benchmark_score_based_model_selection':False,
              'original_content_audit':pin(original),'content_review':pin(review),'independent_content_review':pin(independent),
              'reviewed_content_validation':pin(validation),'source_files':[pin(original),pin(review),pin(independent),pin(own)]}
        return plan, s.sha(original)

    def test_all_review_inputs_bound_before_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            plan,digest=self.fixture(Path(directory).resolve())
            with mock.patch.object(s,'ORIGINAL_AUDIT_SHA',digest): s.validate_review_plan(plan)
            for key in ('content_review','independent_content_review','original_content_audit'):
                bad=copy.deepcopy(plan); bad['source_files']=[r for r in bad['source_files'] if r!=bad[key]]
                with mock.patch.object(s,'ORIGINAL_AUDIT_SHA',digest),self.assertRaises(ValueError): s.validate_review_plan(bad)

    def test_old_attempt_is_forbidden(self):
        with tempfile.TemporaryDirectory() as directory:
            plan,digest=self.fixture(Path(directory).resolve()); plan['run_id']=plan['run_id'].replace('a2','a1')
            with self.assertRaisesRegex(ValueError,'new a2'): s.validate_review_plan(plan)

    def test_changed_review_or_unpreserved_raw_scan_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            plan,digest=self.fixture(Path(directory).resolve())
            Path(plan['content_review']['path']).write_text('{}')
            with mock.patch.object(s,'ORIGINAL_AUDIT_SHA',digest),self.assertRaises(ValueError): s.validate_review_plan(plan)

    def test_only_nonproject_cpu_dash_is_normalized(self):
        output='Job List:\nJob Name GPUs Nodes State Priority Created At Running Time\njinhaopeng-other - 1 Succeeded low - -\njinhaopeng-dpvlm-project - 1 Running low - -\n'
        actual=s.normalize_nonproject_cpu_rows(output)
        self.assertIn('jinhaopeng-other 0 1',actual)
        self.assertIn('jinhaopeng-dpvlm-project - 1',actual)

    def test_frozen_helper_gets_a2_identity_only(self):
        with tempfile.TemporaryDirectory() as directory:
            helper=Path(directory).resolve()/'helper.py'
            helper.write_text("JOB_SHORT='old'\nJOB_PREFIX='old-prefix'\ndef jobs_in_list(text): return text\n")
            with mock.patch.multiple(s,FROZEN_HELPER=helper,FROZEN_HELPER_SHA=s.sha(helper)):
                module=s.load_frozen_helper()
            self.assertEqual(module.JOB_SHORT,'dpvlm-videomme-r1r2-s17-a2')
            self.assertEqual(module.JOB_PREFIX,'jinhaopeng-dpvlm-videomme-r1r2-s17-a2-n1-')

    def test_ambiguous_helper_code_propagates_and_validation_precedes_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve(); plan,digest=self.fixture(root); path=root/'plan.json';write(path,plan)
            helper=types.SimpleNamespace(main=mock.Mock(return_value=2))
            with mock.patch.object(s,'ORIGINAL_AUDIT_SHA',digest),mock.patch.object(s,'load_frozen_helper',return_value=helper):
                result=s.main(['--plan',str(path),'--plan-sha256',s.sha(path),'--execute'])
            self.assertEqual(result,2);helper.main.assert_called_once()
            plan['run_id']='old';write(path,plan)
            with mock.patch.object(s,'load_frozen_helper') as load,self.assertRaises(ValueError):
                s.main(['--plan',str(path),'--plan-sha256',s.sha(path)])
            load.assert_not_called()


if __name__ == '__main__': unittest.main()
