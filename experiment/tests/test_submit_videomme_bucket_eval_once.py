"""Offline compatibility and fail-closed tests for the frozen-helper adapter."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile
import types
import unittest
from unittest import mock

OPERATIONS = Path(__file__).resolve().parents[1] / 'scripts/operations'
spec = importlib.util.spec_from_file_location('vm_submit_wrapper', OPERATIONS / 'submit_videomme_bucket_eval_once.py')
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def frozen_functions():
    """Use the accepted parser/admission source without importing torch."""
    source = (OPERATIONS / 'submit_benchmark_eval_once.py').read_bytes()
    assert hashlib.sha256(source).hexdigest() == wrapper.FROZEN_HELPER_SHA
    tree = ast.parse(source)
    names = {'jobs_in_list', 'live_snapshot', 'kube_states', 'verify_oss_empty'}
    selected = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names],
                          type_ignores=[])
    namespace = {'re': re, 'json': json, 'os': os, 'require': wrapper.require,
                 'PROJECT_PREFIX': wrapper.PROJECT_PREFIX,
                 'JOB_PREFIX': 'jinhaopeng-dpvlm-videomme-r1r2-s17-a1-n1-',
                 'TERMINAL': {'Succeeded', 'Failed', 'Cancelled', 'Canceled'},
                 'SSLAUNCH': '/frozen/sslaunch', 'KUBECTL': '/frozen/kubectl',
                 'KUBECONFIG': '/frozen/kubeconfig', 'OSSUTIL': '/frozen/ossutil',
                 'OSS_CONFIG': '/frozen/ossconfig', 'utc': lambda: 'offline-test'}
    exec(compile(selected, '<accepted-frozen-helper-functions>', 'exec'), namespace)
    return namespace


def scheduler(*rows, count=None):
    return ('Summary:\njinhaopeng ali-gpu - ' + str(len(rows) if count is None else count) +
            ' 1 0 1 0\n\nJob List:\nJob Name GPUs Nodes State Priority Created At Running Time\n' +
            '\n'.join(rows) + '\n')


CPU_ROW = 'jinhaopeng-s4n-l22-data-check-n1-261013184726 - 1 Succeeded low 2026-10-13T10:47:26Z 0m'
PROJECT = 'jinhaopeng-dpvlm-p1-ouro-r4-s17-align-a1-n1-example'


class ParserCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.frozen = frozen_functions()

    def parse(self, text):
        return self.frozen['jobs_in_list'](wrapper.normalize_nonproject_cpu_rows(text))

    def test_actual_nonproject_cpu_dash_is_accepted_without_project_rows(self):
        original = scheduler(CPU_ROW)
        with self.assertRaisesRegex(ValueError, 'Malformed scheduler row'):
            self.frozen['jobs_in_list'](original)
        self.assertEqual(self.parse(original), ({}, {}))
        self.assertEqual(wrapper.normalize_nonproject_cpu_rows(original), original.replace(' - 1 Succeeded ', ' 0 1 Succeeded '))

    def test_project_dash_is_never_normalized(self):
        row = CPU_ROW.replace('jinhaopeng-s4n-l22-data-check-n1-261013184726', PROJECT)
        self.assertEqual(wrapper.normalize_nonproject_cpu_rows(scheduler(row)), scheduler(row))
        with self.assertRaisesRegex(ValueError, 'Malformed scheduler row'):
            self.parse(scheduler(row))

    def test_project_states_and_optional_details_are_preserved(self):
        for state in ['Created', 'Pending', 'Running', 'Restarting', 'Unknown', 'Succeeded', 'Failed']:
            with self.subTest(state=state):
                row = f'{PROJECT} 8 1 {state} low 2026-10-13T10:47:26Z - NodeScheduling: FailedScheduling'
                jobs, details = self.parse(scheduler(CPU_ROW, row))
                self.assertEqual(jobs, {PROJECT: state})
                self.assertEqual(details, {PROJECT: 'NodeScheduling: FailedScheduling'})

    def test_nonproject_cpu_does_not_hide_active_project_from_admission(self):
        raw_parser = self.frozen['jobs_in_list']
        self.frozen['jobs_in_list'] = lambda text: raw_parser(wrapper.normalize_nonproject_cpu_rows(text))
        with tempfile.TemporaryDirectory() as directory:
            for state in ['Created', 'Pending', 'Running', 'Restarting', 'Unknown']:
                with self.subTest(state=state):
                    jobs = scheduler(CPU_ROW, f'{PROJECT} 8 1 {state} low 2026-10-13T10:47:26Z -')
                    def capture(root, label, argv):
                        if label.endswith('_jobs'): output = jobs
                        elif label.endswith('_oss'): output = 'Object Number is: 0\n'
                        else: output = '{"items": []}'
                        return {'returncode': 0, 'stdout': output}
                    self.frozen['capture'] = capture
                    with self.assertRaisesRegex(ValueError, 'Active project scheduler job exists'):
                        self.frozen['live_snapshot'](Path(directory), 'fresh', {'run_id': 'attempt',
                            'run_root': str(Path(directory) / 'absent')})

    def test_unknown_or_malformed_cpu_rows_still_fail(self):
        for row in [CPU_ROW.replace(' - 1 ', ' -1 1 '), CPU_ROW.replace(' - 1 ', ' NaN 1 '),
                    CPU_ROW.replace(' - 1 ', ' - 0 '), CPU_ROW.replace('Succeeded', 'Unrecognized'),
                    CPU_ROW.replace('jinhaopeng-s4n', 'other-s4n'), CPU_ROW.replace(' low ', ' urgent '),
                    CPU_ROW.replace('2026-10-13T10:47:26Z', 'yesterday')]:
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, 'Malformed scheduler row'):
                self.parse(scheduler(row))

    def test_count_duplicate_and_title_failures_are_preserved(self):
        for text in [scheduler(CPU_ROW, count=2), scheduler(CPU_ROW, CPU_ROW),
                     scheduler(CPU_ROW).replace('Job List:', 'missing'), scheduler(CPU_ROW) + 'Job List:\n']:
            with self.subTest(text=text), self.assertRaises(ValueError): self.parse(text)


class WrapperIdentityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.plan = self.root / 'plan.json'
        self.good = {'evaluation_scope': 'videomme_only', 'all_benchmark_assets_ready': False,
                     'mvbench_evaluation_ready': False,
                     'source_files': [{'path': str(Path(wrapper.__file__).resolve()),
                                       'sha256': wrapper.sha(wrapper.__file__)}]}

    def args(self, plan=None):
        self.plan.write_text(json.dumps(self.good if plan is None else plan) + '\n')
        return ['--plan', str(self.plan), '--plan-sha256', wrapper.sha(self.plan),
                '--output-dir', str(self.root / 'submission'), '--execute']

    def test_exact_plan_and_wrapper_pin_delegate_original_argv_once(self):
        args = self.args(); old = types.SimpleNamespace(main=mock.Mock(return_value=0))
        with mock.patch.object(wrapper, 'load_frozen_helper', return_value=old) as load:
            self.assertEqual(wrapper.main(args), 0)
        load.assert_called_once_with(); old.main.assert_called_once_with(args)

    def test_changed_plan_sha_blocks_before_helper_load(self):
        args = self.args(); self.plan.write_text(self.plan.read_text() + ' ')
        with mock.patch.object(wrapper, 'load_frozen_helper') as load:
            with self.assertRaisesRegex(ValueError, 'plan changed'): wrapper.main(args)
        load.assert_not_called()

    def test_wrong_scope_and_missing_changed_or_duplicate_self_pin_block(self):
        bad = []
        for field, value in [('evaluation_scope', 'all'), ('all_benchmark_assets_ready', True),
                             ('mvbench_evaluation_ready', True)]:
            plan = copy.deepcopy(self.good); plan[field] = value; bad.append(plan)
        plan = copy.deepcopy(self.good); plan['source_files'] = []; bad.append(plan)
        plan = copy.deepcopy(self.good); plan['source_files'][0]['sha256'] = '0' * 64; bad.append(plan)
        plan = copy.deepcopy(self.good); plan['source_files'] *= 2; bad.append(plan)
        for plan in bad:
            with self.subTest(plan=plan), mock.patch.object(wrapper, 'load_frozen_helper') as load:
                with self.assertRaises(ValueError): wrapper.main(self.args(plan))
                load.assert_not_called()

    def test_changed_original_helper_never_imports(self):
        helper = self.root / 'old.py'; helper.write_text('raise AssertionError("must not import")\n')
        with mock.patch.object(wrapper, 'FROZEN_HELPER', helper):
            with self.assertRaisesRegex(ValueError, 'Frozen submission helper changed'): wrapper.load_frozen_helper()

    def test_cli_propagates_original_helper_uncertain_return_code(self):
        tree = ast.parse(Path(wrapper.__file__).read_text())
        footer = ast.Module(body=[tree.body[-1]], type_ignores=[])
        with self.assertRaises(SystemExit) as raised:
            exec(compile(footer, '<wrapper-cli-footer>', 'exec'), {'__name__': '__main__', 'main': lambda: 2})
        self.assertEqual(raised.exception.code, 2)


if __name__ == '__main__': unittest.main(verbosity=2)
