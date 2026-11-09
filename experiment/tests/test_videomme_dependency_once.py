"""Offline tests: exact dependencies, bounded waits and once-only handoff."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/operations/run_videomme_dependency_once.py'
spec = importlib.util.spec_from_file_location('vm_dependency', SOURCE)
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.base = self.root / 'base'; self.base.mkdir()
        self.p1 = self.root / 'p1'; self.p1.mkdir()
        self.prep = self.p1 / 'cpu'; self.audit = self.p1 / 'independent'
        self.media = self.p1 / 'media.json'
        self.terminal = self.p1 / 'TERMINAL.json'
        patch = mock.patch.multiple(d, BASE=self.base, P1=self.p1, PREPARATION=self.prep, AUDIT=self.audit, MEDIA=self.media, MEDIA_TERMINAL=self.terminal)
        patch.start(); self.addCleanup(patch.stop)

    def media_terminal(self, **changes):
        write(self.terminal, {'schema_version': 1, 'status': 'SUCCEEDED', 'worker_sha256': d.MEDIA_WORKER_SHA,
              'config_sha256': d.MEDIA_CONFIG_SHA, 'automatic_retry': False, 'gpu_submitted': False,
              'exit_code': 0, 'process_attempts': 1, **changes})

    def ready_fixture(self):
        plan = self.prep / 'EVALUATION_PLAN.json'
        write(plan, {'run_id': d.RUN, 'evaluation_scope': 'videomme_only', 'all_benchmark_assets_ready': False,
                     'mvbench_evaluation_ready': False})
        preflight = self.prep / 'cpu_preflight.stdout'; write(preflight, {'status': 'CPU_PREFLIGHT_PASSED'})
        scope = self.prep / 'VIDEOMME_ONLY_READY.json'; write(scope, {'evaluation_scope': 'videomme_only'})
        helper = self.p1 / 'wrapper.py'; helper.write_text('# accepted wrapper\n')
        ready = self.prep / 'READY_FOR_PARENT_SUBMISSION.json'
        argv = [d.PYTHON, str(helper), '--plan', str(plan), '--plan-sha256', d.sha(plan),
                '--output-dir', str(self.prep / 'submission'), '--execute']
        write(ready, {'status': 'VIDEOMME_ONLY_CPU_PREPARATION_COMPLETE', 'gpu_submitted': False,
                      'plan': d.pin(plan), 'cpu_preflight': d.pin(preflight), 'scope_ready': d.pin(scope),
                      'parent_submission_argv': argv})
        gate = self.audit / 'VIDEOMME_CPU_PREPARATION_INDEPENDENT_GATE.json'
        write(gate, {'kind': 'videomme_cpu_preparation_independent_gate_v1', 'status': 'passed',
                     'evaluation_scope': 'videomme_only', 'all_benchmark_assets_ready': False,
                     'mvbench_evaluation_ready': False, 'gpu_submitted': False, 'checks': {'full_hashes': True},
                     'plan': d.pin(plan), 'cpu_ready': d.pin(ready)})
        return helper, ready, gate

    def archive_fixture(self):
        folder = self.base / 'runs' / d.R4_RUN; folder.mkdir(parents=True)
        identity = {'run_id': d.R4_RUN, 'stage': 'alignment'}
        write(folder / 'EXIT_STATUS.json', {**identity, 'exit_code': 0, 'automatic_retry': False})
        write(folder / 'P1_STAGE_COMPLETE.json', {**identity, 'status': 'p1_formal_stage_complete',
            'formal_training_result': True, 'held_out_benchmark_result': False, 'world_size': 8, 'seed': 17,
            'rfg': {'num_frames': 8, 'num_loops': 4, 'pool_grid': 4}, 'training_result': {'optimizer_steps': 718},
            'requested_config_sha256': d.R4_CONFIG_SHA, 'source_identity': {'manifest_sha256': d.R4_SOURCE_SHA}})
        files = [{'path': name, 'bytes': (folder / name).stat().st_size, 'sha256': d.sha(folder / name)}
                 for name in ['EXIT_STATUS.json', 'P1_STAGE_COMPLETE.json']]
        files.append({'path': 'training/final/language_model/model.safetensors', 'bytes': 500, 'sha256': 'a' * 64})
        manifest = {**identity, 'uri': d.R4_URI, 'file_count': len(files), 'total_bytes': sum(v['bytes'] for v in files), 'files': files}
        write(folder / 'ARCHIVE_MANIFEST.json', manifest)
        write(folder / 'ARCHIVED.json', {**identity, 'uri': d.R4_URI, 'status': 'archive_verified',
            'archive_manifest_sha256': d.sha(folder / 'ARCHIVE_MANIFEST.json'),
            'remote_manifest_and_completion_sha256_readback': True, 'file_count': len(files), 'total_bytes': manifest['total_bytes']})
        return folder


class DependencyGateTest(Fixture):
    def test_absent_or_partial_media_never_prepares(self):
        self.assertFalse(d.media_probe()[0])
        self.media_terminal()
        write(self.media, {'kind': 'videomme_bucket_media_complete_v1', 'status': 'VIDEOMME_BUCKET_MEDIA_COMPLETE',
                          'complete': True, 'source_count': 547, 'frame_count': 4376, 'full_oss_get_sha256_verified': True})
        with self.assertRaisesRegex(ValueError, 'invalid'): d.media_probe()

    def test_complete899_media_allows_cpu_gate_only(self):
        self.media_terminal()
        write(self.media, {'kind': 'videomme_bucket_media_complete_v1', 'status': 'VIDEOMME_BUCKET_MEDIA_COMPLETE',
                          'complete': True, 'source_count': 899, 'frame_count': 7192, 'full_oss_get_sha256_verified': True})
        self.assertTrue(d.media_probe()[0])

    def test_media_failed_supervisor_stops_without_launch_or_wait(self):
        for changes in [{'status': 'FAILED', 'exit_code': 2}, {'status': 'FAILED', 'error_class': 'RuntimeError'}]:
            self.media_terminal(**changes)
            with self.assertRaisesRegex(ValueError, 'Media supervisor FAILED'): d.media_probe()

    def test_media_incomplete_terminal_publication_waits_without_preflight(self):
        self.terminal.write_text('{')
        with mock.patch.object(d.time, 'sleep') as pause:
            ready, state, _ = d.media_probe()
        self.assertFalse(ready)
        self.assertEqual(state, 'WAITING_FOR_MEDIA_TERMINAL_PUBLICATION')
        self.assertEqual(pause.call_count, 2)

    def test_complete_media_alone_cannot_bypass_supervisor_barrier(self):
        write(self.media, {'complete': True, 'source_count': 899, 'frame_count': 7192})
        self.assertFalse(d.media_probe()[0])

    def test_real_ready_and_matching_independent_gate_required(self):
        self.assertIsNone(d.validate_ready_and_gate(self.p1 / 'missing'))
        helper, ready, gate = self.ready_fixture()
        self.assertEqual(d.validate_ready_and_gate(helper)['ready'], d.pin(ready))
        good = json.loads(gate.read_text())
        for key, value in [('checks', {}), ('checks', {'full_hashes': 1}), ('gpu_submitted', True),
                           ('mvbench_evaluation_ready', True), ('status', 'pending'), ('plan', {'path': '/other', 'sha256': 'a' * 64})]:
            with self.subTest(key=key):
                bad = copy.deepcopy(good); bad[key] = value; write(gate, bad)
                with self.assertRaises(ValueError): d.validate_ready_and_gate(helper)

    def test_changed_ready_or_substitution_command_rejected(self):
        helper, ready, gate = self.ready_fixture()
        document = json.loads(ready.read_text()); document['parent_submission_argv'][1] = '/other/submit.py'
        write(ready, document)
        audit = json.loads(gate.read_text()); audit['cpu_ready'] = d.pin(ready); write(gate, audit)
        with self.assertRaisesRegex(ValueError, 'Parent command differs'): d.validate_ready_and_gate(helper)

    def test_cpu_failure_blocks_without_waiting_or_retry(self):
        write(self.prep / 'CPU_PREPARATION_FAILED.json', {'error': 'failed'})
        with self.assertRaisesRegex(ValueError, 'CPU preparation failed'): d.validate_ready_and_gate(self.p1 / 'wrapper')

    def test_exact_r4_identity_and_failed_cancelled_unknown_stop(self):
        frozen = types.SimpleNamespace(SSLAUNCH='/frozen/sslaunch')
        for state in ['Failed', 'Cancelled', 'Canceled', 'Unknown', 'Restarting']:
            with self.subTest(state=state), mock.patch.object(d.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0, stdout=f'Job: {d.R4_JOB}\nState: {state}\n')) as call:
                with self.assertRaisesRegex(ValueError, 'stop without retry'): d.exact_r4_state(frozen)
                self.assertEqual(call.call_count, 1)
                self.assertEqual(call.call_args.args[0][-1], d.R4_JOB)
        with mock.patch.object(d.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0, stdout='Job: another\nState: Succeeded\n')):
            with self.assertRaisesRegex(ValueError, 'identity'): d.exact_r4_state(frozen)

    def test_wait_is_bounded_and_only_records_state_changes(self):
        clock = [0.0]; probes = [0]
        def probe(remaining):
            probes[0] += 1
            return probes[0] == 3, 'READY' if probes[0] == 3 else 'WAIT', {}
        with mock.patch.object(d.time, 'monotonic', side_effect=lambda: clock[0]), \
             mock.patch.object(d.time, 'sleep', side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)):
            d.wait_for(self.p1, probe, 5, 1)
        events = [json.loads(line) for line in (self.p1 / 'events.jsonl').read_text().splitlines()]
        self.assertEqual([row['state'] for row in events], ['WAIT', 'READY'])
        clock[0] = 0
        with mock.patch.object(d.time, 'monotonic', side_effect=lambda: clock[0]), \
             mock.patch.object(d.time, 'sleep', side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)):
            with self.assertRaises(TimeoutError): d.wait_for(self.p1, lambda remaining: (False, 'WAIT', {}), 2, 1)
        self.assertEqual(clock[0], 2)


class ArchiveDependencyTest(Fixture):
    def test_r4_success_control_and_manifest_are_bound(self):
        folder = self.archive_fixture()
        self.assertEqual(len(d.validate_r4_controls(folder)), 3)
        target = folder / 'P1_STAGE_COMPLETE.json'; good = json.loads(target.read_text())
        for key, value in [('world_size', 4), ('requested_config_sha256', 'b' * 64),
                           ('training_result', {'optimizer_steps': 717}), ('stage', 'sft'),
                           ('rfg', {'num_frames': 8, 'num_loops': 2, 'pool_grid': 4})]:
            with self.subTest(key=key):
                bad = copy.deepcopy(good); bad[key] = value; write(target, bad)
                with self.assertRaises(ValueError): d.validate_r4_controls(folder)

    def test_failed_exit_and_mismatched_manifest_sha_reject(self):
        folder = self.archive_fixture(); target = folder / 'EXIT_STATUS.json'
        good = target.read_bytes(); doc = json.loads(good); doc['exit_code'] = 1; write(target, doc)
        with self.assertRaisesRegex(ValueError, 'exit0'): d.validate_r4_controls(folder)
        target.write_bytes(good); manifest = folder / 'ARCHIVE_MANIFEST.json'; manifest.write_text(manifest.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'manifest/control SHA'): d.validate_r4_controls(folder)

    def test_exact_oss_controls_and_complete_payload_lengths(self):
        folder = self.archive_fixture(); calls = []
        def capture(root, label, argv, timeout=180):
            calls.append(argv)
            if argv[1] == 'cp':
                shutil.copyfile(folder / Path(argv[2]).name, Path(argv[3])); output = ''
            else:
                files = d.validate_r4_controls(folder)
                lengths = {key: value['bytes'] for key, value in files.items()}
                lengths.update({name: (folder / name).stat().st_size for name in ['ARCHIVE_MANIFEST.json', 'ARCHIVED.json']})
                output = '\n'.join(f'2026-10-13 14:00:00 +0000 UTC {size} Standard {d.R4_URI}/{key}' for key, size in lengths.items())
                output += f'\nObject Number is: {len(lengths)}\n'
            return {'returncode': 0, 'stdout': output}
        frozen = types.SimpleNamespace(capture=capture, OSSUTIL='ossutil', OSS_CONFIG='/config')
        result = d.verify_r4_archive(self.p1, frozen)
        self.assertEqual(len(calls), 5)
        self.assertEqual(result['object_count'], 5)
        self.assertIs(result['payload_full_SHA_readback_performed'], False)
        self.assertTrue(all(argv[2].startswith(d.R4_URI + '/') for argv in calls))

    def test_changed_oss_control_bytes_stop_without_later_requests(self):
        self.archive_fixture(); calls = []
        def capture(root, label, argv, timeout=180):
            calls.append(argv); Path(argv[3]).write_text('{}'); return {'returncode': 0, 'stdout': ''}
        frozen = types.SimpleNamespace(capture=capture, OSSUTIL='ossutil', OSS_CONFIG='/config')
        with self.assertRaisesRegex(ValueError, 'readback mismatch'): d.verify_r4_archive(self.p1, frozen)
        self.assertEqual(len(calls), 1)


class OnceOnlyTest(Fixture):
    def test_cpu_cuda_mask_is_not_inherited_by_scheduler(self):
        with mock.patch.dict(d.os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            self.assertNotIn('CUDA_VISIBLE_DEVICES', d.submission_environment())

    def test_failed_child_is_not_called_twice(self):
        process = types.SimpleNamespace(wait=mock.Mock(return_value=2), pid=99999)
        with mock.patch.object(d.subprocess, 'Popen', return_value=process) as popen:
            with self.assertRaisesRegex(ValueError, 'never invoked again'):
                d.invoke_once(self.p1, 'submission_wrapper', ['frozen-wrapper'], {}, timeout=1)
            with self.assertRaises(FileExistsError):
                d.invoke_once(self.p1, 'submission_wrapper', ['frozen-wrapper'], {}, timeout=1)
        self.assertEqual(popen.call_count, 1)

    def test_full_mode_routes_prepare_then_submit_once_and_intent_blocks_restart(self):
        config = self.p1 / 'config.json'; write(config, {'wait_seconds': 86400})
        out = self.p1 / 'handoff'; order = []
        args = ['--config', str(config), '--config-sha256', d.sha(config), '--output-dir', str(out), '--execute']
        with mock.patch.object(d.sys, 'platform', 'linux'), mock.patch.object(d.getpass, 'getuser', return_value='jinhaopeng'), \
             mock.patch.object(d, 'validate_config', return_value=()), \
             mock.patch.object(d, 'prepare_once', side_effect=lambda *args: order.append('prepare')), \
             mock.patch.object(d, 'submit_once', side_effect=lambda *args: order.append('submit')):
            d.main(args)
            args[-2] = str(self.p1 / 'another-evidence')
            with self.assertRaises(FileExistsError): d.main(args)
        self.assertEqual(order, ['prepare', 'submit'])

    def test_prepare_failure_prevents_submit(self):
        config = self.p1 / 'config.json'; write(config, {'wait_seconds': 86400})
        out = self.p1 / 'handoff'
        args = ['--config', str(config), '--config-sha256', d.sha(config), '--output-dir', str(out), '--execute']
        with mock.patch.object(d.sys, 'platform', 'linux'), mock.patch.object(d.getpass, 'getuser', return_value='jinhaopeng'), \
             mock.patch.object(d, 'validate_config', return_value=()), \
             mock.patch.object(d, 'prepare_once', side_effect=ValueError('CPU failed')), \
             mock.patch.object(d, 'submit_once') as submit:
            with self.assertRaisesRegex(ValueError, 'CPU failed'): d.main(args)
        submit.assert_not_called()
        result = json.loads((out / 'DEPENDENCY_STOPPED.json').read_text())
        self.assertIs(result['submission_call_attempted'], False)


if __name__ == '__main__': unittest.main(verbosity=2)
