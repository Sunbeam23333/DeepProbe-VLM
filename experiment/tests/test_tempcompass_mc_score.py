"""No API calls: exact pinned official-rule parity and honest aggregate metrics."""
import pytest
import ast
import hashlib
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('tc_score', ROOT / 'scripts/operations/score_tempcompass_mc.py')
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


class ScoringTests(unittest.TestCase):
    @pytest.mark.private_evidence
    def test_actual_pinned_official_function_parity_with_llm_disabled(self):
        path = ROOT / 'docs/operations/evidence/video-benchmark-execution-20261017/tempcompass-protocol/eval_multi-choice.py'
        raw = path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), S.OFFICIAL_SCORER_SHA)
        module = ast.parse(raw)
        fn = next(x for x in module.body if isinstance(x, ast.FunctionDef) and x.name == 'main')
        # Remove only file output and print; retain every original matching branch.
        fn.body = fn.body[:1]
        env = {'tqdm': lambda values: values}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<pinned-official-scorer>', 'exec'), env)
        values = [None, '', 'A', 'B', ' A </s> ', 'A. exact', 'A. contradictory text', 'A) exact',
                  'A) anything', '(A)', 'a', 'The answer is A', 'A or B', 'A.B', 'D.', 'A)</s>',
                  'B) text', 'A. exact</s>', 'A\n', 'A, exact', 'A: exact']
        for value in values:
            predictions = {'123': {'action': [{'question': 'Question?', 'answer': 'A. exact', 'prediction': value}]}}
            evaluated = {}
            env['main'](predictions, evaluated, None, True)
            expected = evaluated['123']['action'][0]
            result = S.official_rule(value, 'A. exact')
            self.assertEqual(result, {'correct': bool(expected['rating']), 'match_success': expected['match_success']}, repr(value))

    def test_incomplete_duplicate_or_unbound_predictions_fail(self):
        row = self.record('a', 'action')
        for predictions in ([], [self.prediction(row)] * 2, [{'sample_id': 'a', 'prediction': 'A'}]):
            with self.assertRaises(ValueError):
                S.score([row], predictions)

    def test_micro_and_macro_are_distinct_and_nulls_remain(self):
        rows = [self.record('a', 'action'), self.record('b', 'action'), self.record('c', 'speed')]
        scored, summary = S.score(rows, [self.prediction(rows[0]), self.prediction(rows[1], None), self.prediction(rows[2])])
        self.assertEqual(summary['overall']['count'], 3)
        self.assertEqual(summary['overall']['correct'], 2)
        self.assertAlmostEqual(summary['overall']['accuracy'], 2/3)
        self.assertEqual(summary['dimension_macro_accuracy'], 0.75)
        self.assertFalse(scored[1]['match_success'])

    @staticmethod
    def record(identity, dim):
        return {'sample_id': identity, 'video_id': '123', 'source_video_id': 'component:123',
                'underlying_source_video_ids': ['shutterstock:123'], 'task_type': dim,
                'fine_grained_task_type': dim, 'original_answer': 'A. exact', 'answer': 'A'}

    @staticmethod
    def prediction(row, value='A'):
        return {'sample_id': row['sample_id'], 'manifest_row_sha256': hashlib.sha256(S.canonical(row)).hexdigest(),
                'prediction': value}


if __name__ == '__main__':
    unittest.main()
