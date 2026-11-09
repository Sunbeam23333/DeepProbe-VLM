"""Review must gate all CPU routing; CPU preparation never invokes submit."""
import pytest
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('reviewed_prep', ROOT / 'scripts/operations/prepare_videomme_reviewed_evaluation.py')
p = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(p)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


class ReviewedCPUPrepareTests(unittest.TestCase):
    @pytest.mark.private_evidence
    def test_fixed_scope_and_cpu_environment(self):
        scope = json.loads((ROOT / 'docs/operations/evidence/video-benchmark-execution-20261017/eval-preparation/VIDEOMME_A2_SCOPE.json').read_text())
        p.validate_scope(scope)
        for key, value in [('r4_success_dependency', True), ('question_count', 2700), ('greedy_max_new_tokens', 32),
                           ('audio', True), ('num_frames', 4), ('model_sequence', ['r2','r1'])]:
            with self.subTest(key=key):
                bad = dict(scope); bad[key] = value
                with self.assertRaises(ValueError): p.validate_scope(bad)
        self.assertEqual(p.cpu_environment()['CUDA_VISIBLE_DEVICES'], '')

    def test_raw_audit_and_submit_are_not_cpu_stage_routes(self):
        for name in ['content_audit', 'submit', 'train']:
            with self.assertRaisesRegex(ValueError, 'Unapproved CPU stage'):
                p.run_cpu_stage(Path('/not-created'), name, ['not-run'])

    def fixture(self, root):
        base, p1, experiment, media, output = [root / name for name in ['base','p1','experiment','media','out']]
        output.mkdir(); media.mkdir()
        (media / 'FRAME_CACHE_RECEIPT.json').write_text('{}')
        scope = json.loads((ROOT / 'docs/operations/evidence/video-benchmark-execution-20261017/eval-preparation/VIDEOMME_A2_SCOPE.json').read_text())
        human = root / 'human.json'; write(human, {'authorized': True}); scope['human_authorization'] = p.pin(human)
        auth = root / 'authorization.json'; write(auth, scope)
        review = root / 'review.json'; write(review, {})
        independent = root / 'independent.json'; write(independent, {})
        original_audit = root / 'original_failed_audit.json'; write(original_audit, {'passed': False})
        config_path = root / 'config.json'; write(config_path, {})
        paths = [Path(p.__file__).resolve()]
        for name in ['prepare_formal_source_gate_reviewed.py','validate_benchmark_content_review.py',
                     'submit_videomme_reviewed_eval_once.py','audit_videomme_reviewed_cpu_preparation.py']:
            path = root / name; path.write_text('# pinned control fixture\n'); paths.append(path)
        controls = [p.pin(path) for path in paths]
        config = {'kind':'videomme_reviewed_cpu_preparation_v1', 'run_id':p.RUN_ID, 'experiment_root':str(experiment),
                  'media_root':str(media), 'authorization':p.pin(auth), 'evaluation_source_files':[],
                  'training_fingerprint_files':[], 'control_source_files':controls,
                  'reviewed_formal_gate':controls[1], 'submission_helper':controls[3],
                  'content_review':p.pin(review), 'independent_review':p.pin(independent)}
        pins = [p.pin(original_audit),p.pin(review),p.pin(independent),p.pin(human),p.pin(auth),p.pin(config_path)]
        proof = {'passed':True,'original_candidates_preserved':True,'original_audit_passed':False,
                 'candidate_count':12,'matched_frame_count':30,'review':p.pin(review),
                 'independent_review':dict(p.pin(independent),reviewer='root'), 'artifacts':pins}
        validation = {'provided':True,'passed':True,'errors':[],'path':str(original_audit),
                      'sha256':p.ORIGINAL_AUDIT_SHA,'admission_mode':'original_scan_plus_bound_candidate_review_v1',
                      'adjudication':proof}
        return locals()

    def run_fixture(self, f, *, review_passes=True):
        calls = []; output = f['output']
        def stage(out, name, argv):
            calls.append((name, argv))
            if name == 'formal_gate':
                if not review_passes: raise ValueError('formal review failed')
                write(output / 'formal-gate/FORMAL_READY.json', {'formal_evaluation_ready':True,
                    'benchmark_media_gate':'passed','content_duplicate_gate':'passed'})
                write(output / 'formal-gate/content_audit_validation.json', f['validation'])
            if name == 'cpu_preflight':
                write(output / 'cpu_preflight.stdout', {'status':'CPU_PREFLIGHT_PASSED',
                    'plan_sha256':p.sha(output / 'EVALUATION_PLAN.json'),'num_gpus':8,'sequence':['r1','r2'],
                    'subtitles':False,'audio':False})
        original_pin = p.pin
        def pin(path):
            path=Path(path)
            return original_pin(path) if path.is_file() else {'path':str(path),'sha256':'a'*64}
        with mock.patch.multiple(p,BASE=f['base'],P1=f['p1'],EXPERIMENT=f['experiment'],MEDIA_ROOT=f['media'],
                    ORIGINAL_AUDIT=f['original_audit'], FROZEN_SOURCE_LIST_SHA=hashlib.sha256(b'[]\n').hexdigest()), \
             mock.patch.object(p,'verify',side_effect=lambda item:Path(item['path'])), \
             mock.patch.object(p,'verify_bucket_media',return_value={'evaluation_scope':'videomme_only'}), \
             mock.patch.object(p,'pin',side_effect=pin), mock.patch.object(p,'run_cpu_stage',side_effect=stage):
            try: p.prepare(f['config'],f['config_path'],output)
            except ValueError:
                if review_passes: raise
        return calls

    @pytest.mark.private_evidence
    def test_new_attempt_uses_three_cpu_stages_and_pins_all_review_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory).resolve()); calls=self.run_fixture(f)
            self.assertEqual([name for name, _ in calls],['formal_gate','manifest','cpu_preflight'])
            self.assertTrue(all('sslaunch' not in ' '.join(argv) for _, argv in calls))
            args=calls[0][1]
            self.assertIn('--content-review-independent',args)
            self.assertIn(str(f['original_audit']),args)
            self.assertNotIn('audit_benchmark_content.py',' '.join(args))
            plan=json.loads((f['output']/'EVALUATION_PLAN.json').read_text())
            self.assertEqual(plan['run_id'],'dp-benchmark-videomme-r1-r2-s17-a2')
            self.assertTrue(all(item in plan['source_files'] for item in f['pins']))
            ready=json.loads((f['output']/'READY_FOR_PARENT_SUBMISSION.json').read_text())
            self.assertFalse(ready['gpu_submitted'])
            self.assertEqual(ready['parent_submission_argv'][-1],'--execute')

    @pytest.mark.private_evidence
    def test_failed_review_stops_before_manifest_preflight_or_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory).resolve()); calls=self.run_fixture(f,review_passes=False)
            self.assertEqual([name for name, _ in calls],['formal_gate'])
            self.assertFalse((f['output']/'EVALUATION_PLAN.json').exists())
            self.assertFalse((f['output']/'READY_FOR_PARENT_SUBMISSION.json').exists())


if __name__ == '__main__': unittest.main()
