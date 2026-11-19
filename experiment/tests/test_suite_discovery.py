"""Synthetic metadata discovery; no asset transfer or scheduler interaction."""
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stdout


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/operations/discover_experiment_suite_machine.py'
SPEC = importlib.util.spec_from_file_location('suite_discovery', SCRIPT)
discovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(discovery)


class SuiteDiscoveryTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.home = Path(temp.name).resolve()
        self.base = self.home / '.cache/Paper/DeepProbe-VLM'
        self.p1 = Path(str(self.base) + '-p1')
        self.source = self.base / 'suite-source/synthetic-v2/experiment'
        self.write(self.source / 'src/deepprobe_vlm/suite_workflow.py', '')
        (self.home / '.config/deepprobe-vlm').mkdir(parents=True)
        metadata = [
            'formal-training-data-20261009-v1/train.jsonl',
            'formal-training-data-20261009-v1/validation.jsonl',
            'formal-source-final-20261009-v1/FORMAL_READY.json',
            'releases/deepprobe-20261021-declared-v9/P1_SOURCE_MANIFEST.json',
            'videomme-reviewed-cpu-20261021-a1/manifest/videomme.jsonl',
            'benchmark-media-20261012/videomme/FRAME_CACHE_RECEIPT.json',
            'benchmark-media-tempcompass-20261021-a1/manifest.jsonl',
            'benchmark-media-tempcompass-20261021-a1/FRAME_CACHE_RECEIPT.json',
            'benchmark-videomme-oss-decode-20261013/release-v1/ACCEPTED_ARCHIVE_AUDIT.json',
        ]
        for name in metadata: self.write(self.p1 / name, '{}')
        proofs = {
            'smollm2': 'formal-preflight-v9-smollm2-s29-alignment-20261027-prepare-1/CPU_INPUTS_PREPARED.json',
            'ouro': 'formal-preflight-v9-r4-s29-alignment-20261021-prepare-1/CPU_INPUTS_VERIFIED_PENDING_CAPACITY.json',
        }
        for family, name in proofs.items():
            offline = {label: {'resolved_path': str(self.base / 'models' / family / label),
                'configured_id': family + '/' + label, 'revision': 'synthetic',
                'files': {'config.json': {'bytes': 2, 'sha256': 'a' * 64}}}
                for label in ('language', 'vision')}
            self.write(self.p1 / name, json.dumps({'offline': offline}))
        template = {'apiVersion': 'kubeflow.org/v1', 'kind': 'PyTorchJob',
            'spec': {'cleanPodPolicy': 'None', 'ttlSecondsAfterFinished': 86400,
                'pytorchReplicaSpecs': {'Master': {'template': {'spec': {
                    'containers': [{'name': 'pytorch', 'image': 'synthetic:test'}]}}}}}}
        self.write(self.p1 / 'smollm2-s29-align-a5-submit-20261118-v1/exact_job.stdout', json.dumps(template))

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(value)

    def discover(self, **kwargs):
        return discovery.discover(self.base, self.source, 'synthetic-suite', 1, 'normal', **kwargs)

    def test_default_keeps_finite_policy_and_fixed_roots_for_source_below_base(self):
        env = self.discover()['environment']
        self.assertEqual(env['storage'], {
            'schema_version': 2,
            'roots': [str(self.base), str(self.p1), str(self.home / 'Paper/DeepProbe-VLM'),
                      str(self.home / '.config/deepprobe-vlm')],
            'ledger_root': str(self.base / 'locks/suite-storage-ledger'),
            'home_limit_bytes': 150000000000, 'oss_limit_bytes': 150000000000,
            'oss_prefixes': ['oss://ss-devbox-stream/jinhaopeng/Paper/DeepProbe-VLM/'],
            'home_safety_bytes': 2000000000, 'oss_safety_bytes': 2000000000,
            'free_guard_bytes': 1000000000,
        })
        self.assertEqual(env['modelscope_token_file'], str(self.home / '.config/deepprobe-vlm/modelscope-token'))

    def test_explicit_unlimited_policy_preserves_all_other_candidate_fields(self):
        expected = deepcopy(self.discover())
        expected['environment']['storage']['oss_limit_bytes'] = None
        del expected['environment']['modelscope_token_file']
        self.assertEqual(self.discover(unlimited_oss=True), expected)
        with self.assertRaisesRegex(ValueError, 'explicit boolean'):
            self.discover(unlimited_oss='true')

    def test_cli_opt_in_keeps_exact_history_index_pins(self):
        indexes = {'history-import': {}, 'history-preparation': [], 'recovery-preparation': []}
        argv = [str(SCRIPT), '--base', str(self.base), '--source-root', str(self.source),
                '--suite-id', 'synthetic-suite', '--unlimited-oss']
        for arg, document in indexes.items():
            path = self.p1 / (arg + '.json'); self.write(path, json.dumps(document))
            argv += ['--' + arg + '-index', str(path)]
        output = io.StringIO()
        with mock.patch('sys.argv', argv), redirect_stdout(output): discovery.main()
        result = json.loads(output.getvalue())
        self.assertIsNone(result['environment']['storage']['oss_limit_bytes'])
        self.assertNotIn('modelscope_token_file', result['environment'])
        for arg in indexes:
            field = arg.replace('-', '_') + 's'
            self.assertEqual(result['input_indexes'][field], discovery.pin(self.p1 / (arg + '.json')))


if __name__ == '__main__': unittest.main()
