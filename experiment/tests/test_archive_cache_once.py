"""Offline orchestration/failure tests; no OSS, real credentials, or uploads."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = ROOT / 'scripts/operations/run_archive_cache_once.py'
SPEC = importlib.util.spec_from_file_location('archive_once', WRAPPER_PATH)
W = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(W)
CATALOG = 'a' * 64


def digest(data):
    return hashlib.sha256(data).hexdigest()


def marker(**changes):
    return {'event': W.COMPLETION, 'archives': 31, 'bytes': W.ARCHIVE_BYTES,
            'catalog_sha256': CATALOG, 'all_benchmark_assets_ready': False, **changes}


def audit(**changes):
    return {'status': W.AUDIT_PASS, 'plan_sha256': W.PLAN_SHA, 'exact_object_count': 64,
            'archive_count': 31, 'archive_bytes': W.ARCHIVE_BYTES, 'small_object_count': 33,
            'catalog_sha256': CATALOG, 'exact_prefix_total_bytes': W.ARCHIVE_BYTES + 1000,
            'all_pinned_zip_archives_cached': True, 'all_benchmark_assets_ready': False,
            'formal_evaluation_ready': False, 'ntu_media_included': False, 'zip_body_get_count': 0, **changes}


class FakeCalls:
    def __init__(self, lines=None, client_rc=0, remote_rc=0, offline_rc=0,
                 remote_audit=None, offline_audit=None, after_client=None, interrupted=False):
        self.lines = [marker()] if lines is None else lines
        self.client_rc, self.remote_rc, self.offline_rc = client_rc, remote_rc, offline_rc
        self.remote_audit = remote_audit if remote_audit is not None else audit()
        self.offline_audit = offline_audit if offline_audit is not None else self.remote_audit
        self.after_client, self.interrupted = after_client, interrupted
        self.client_calls, self.audit_calls = [], []

    def popen(self, argv, stdin, stdout, stderr):
        self.client_calls.append({'argv': argv, 'stdin': stdin, 'stderr': stderr})
        assert stdin is None  # fd0 is inherited, not read or copied by the wrapper.
        for row in self.lines:
            stdout.write((row if isinstance(row, str) else json.dumps(row)).encode() + b'\n')
        calls = self
        class Process:
            pid = 12345
            def wait(self):
                if calls.after_client:
                    calls.after_client()
                if calls.interrupted:
                    raise W.WorkflowError('wrapper_interrupted')
                return calls.client_rc
            def poll(self):
                return None if calls.interrupted else calls.client_rc
            def kill(self):
                raise AssertionError('wrapper must not kill uploads')
            terminate = kill
        return Process()

    def run(self, argv, stdin, stdout, stderr, timeout):
        assert stdin == subprocess.DEVNULL
        self.audit_calls.append(argv)
        if argv[0] == 'ssh':
            value = {'snapshot': {}, 'snapshot_sha256': 'b' * 64, 'audit': self.remote_audit}
            rc = self.remote_rc
        else:
            value, rc = self.offline_audit, self.offline_rc
        stdout.write(json.dumps(value).encode())
        return SimpleNamespace(returncode=rc)


class ArchiveOnceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        client = self.root / 'coordinate_hf_stream_oss_cache_v4.py'
        helper = self.root / 'hf_to_stream_oss_cache_v3.py'
        auditor = self.root / 'audit_hf_stream_oss_archives.py'
        plan = self.root / 'plan.json'
        for path, content in [(client, b'# synthetic client\n'), (helper, b'# synthetic helper\n'),
                              (auditor, b'# synthetic auditor\n'), (plan, b'{"synthetic":true}\n')]:
            path.write_bytes(content)
        self.addCleanup(patch.stopall)
        patch.object(W, 'AUDITOR_SHA', digest(auditor.read_bytes())).start()
        patch.object(W, 'PLAN_SHA', digest(plan.read_bytes())).start()
        self.client_config = self.root / 'client-config.json'
        cfg = {'python': W.REMOTE_PYTHON, 'worker': W.REMOTE_ROOT + 'release/hf_to_stream_oss_cache_v3.py',
               'worker_sha256': digest(helper.read_bytes()), 'local_plan': str(plan),
               'remote_plan': W.REMOTE_ROOT + 'control/plan.json', 'plan_sha256': W.PLAN_SHA}
        self.client_config.write_text(json.dumps(cfg))
        self.config = {'schema_version': 1, 'ssh_alias': W.HOST, 'local_python': sys.executable,
            'remote_python': W.REMOTE_PYTHON,
            'client': {'path': str(client), 'sha256': digest(client.read_bytes())},
            'client_helper': {'path': str(helper), 'sha256': digest(helper.read_bytes())},
            'client_config': {'path': str(self.client_config), 'sha256': digest(self.client_config.read_bytes())},
            'auditor': {'local_path': str(auditor), 'remote_path': W.REMOTE_ROOT + 'audit/audit_hf_stream_oss_archives.py',
                        'sha256': W.AUDITOR_SHA},
            'plan': {'local_path': str(plan), 'remote_path': cfg['remote_plan'], 'sha256': W.PLAN_SHA},
            'evidence_dir': str(self.root / 'run')}
        self.config_path = self.root / 'wrapper-config.json'
        self.write_config()

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config))
        self.config_sha = digest(self.config_path.read_bytes())

    def execute(self, calls):
        W.load_config(self.config_path, self.config_sha)
        return W.run_once(self.config, self.config_path, self.config_sha, calls.popen, calls.run)

    def test_single_client_then_remote_and_offline_acceptance(self):
        calls = FakeCalls()
        code, terminal, path = self.execute(calls)
        self.assertEqual(code, 0)
        self.assertEqual(len(calls.client_calls), 1)
        self.assertEqual(len(calls.audit_calls), 2)
        self.assertIn(W.HOST, calls.audit_calls[0])
        self.assertIn('--snapshot', calls.audit_calls[1])
        self.assertEqual(terminal['status'], 'PINNED_ARCHIVE_CACHE_INDEPENDENTLY_ACCEPTED')
        self.assertFalse(terminal['automatic_evaluation_enabled'])
        self.assertFalse(terminal['all_benchmark_assets_ready'])
        self.assertEqual(terminal['upload_restarts'], 0)
        self.assertEqual(json.loads(path.read_text()), terminal)

    def test_nonzero_client_never_audits_even_with_success_marker(self):
        calls = FakeCalls(client_rc=2)
        code, terminal, _ = self.execute(calls)
        self.assertEqual(code, 2)
        self.assertEqual(terminal['error_code'], 'client_nonzero_exit')
        self.assertEqual(len(calls.client_calls), 1)
        self.assertFalse(calls.audit_calls)

    def test_missing_completion_never_audits(self):
        calls = FakeCalls(lines=[{'status': 'CACHED_VERIFIED'}])
        self.assertEqual(self.execute(calls)[0], 2)
        self.assertFalse(calls.audit_calls)

    def test_duplicate_completion_never_audits(self):
        calls = FakeCalls(lines=[marker(), marker()])
        self.assertEqual(self.execute(calls)[0], 2)
        self.assertFalse(calls.audit_calls)

    def test_wrong_completion_counts_never_audit(self):
        calls = FakeCalls(lines=[marker(archives=30)])
        self.assertEqual(self.execute(calls)[0], 2)
        self.assertFalse(calls.audit_calls)

    def test_prior_failure_event_rejects_success_marker(self):
        calls = FakeCalls(lines=[{'event': 'HF_OSS_CACHE_COORDINATOR_FAILED'}, marker()])
        self.assertEqual(self.execute(calls)[0], 2)
        self.assertFalse(calls.audit_calls)

    def test_recovered_worker_failure_allows_terminal_completion_and_audits(self):
        calls = FakeCalls(lines=[
            {'status': 'FAILED', 'error_code': 'operation_TimeoutError'},
            {'status': 'NEEDS_SIGNED_URL', 'asset_id': 'synthetic_archive'},
            {'event': 'HF_SIGNED_URL_REFRESH_RESPONSE_SENT', 'asset_id': 'synthetic_archive',
             'refresh_count': 1},
            {'status': 'CACHED_VERIFIED', 'asset_id': 'synthetic_archive', 'bytes': 42,
             'sha256': 'd' * 64, 'receipt_sha256': 'e' * 64},
            marker()])
        code, terminal, _ = self.execute(calls)
        self.assertEqual(code, 0)
        self.assertEqual(terminal['status'], 'PINNED_ARCHIVE_CACHE_INDEPENDENTLY_ACCEPTED')
        self.assertEqual(len(calls.client_calls), 1)
        self.assertEqual(len(calls.audit_calls), 2)
        self.assertEqual(terminal['upload_restarts'], 0)

    def test_completion_must_be_terminal_event(self):
        calls = FakeCalls(lines=[marker(), {'status': 'unexpected_after_completion'}])
        self.assertEqual(self.execute(calls)[0], 2)
        self.assertFalse(calls.audit_calls)

    def test_remote_audit_failure_has_no_offline_run_or_upload_restart(self):
        calls = FakeCalls(remote_rc=2)
        code, terminal, _ = self.execute(calls)
        self.assertEqual(code, 2)
        self.assertEqual(terminal['error_code'], 'remote_readonly_audit_failed')
        self.assertEqual(len(calls.client_calls), 1)
        self.assertEqual(len(calls.audit_calls), 1)

    def test_offline_audit_failure_is_terminal(self):
        calls = FakeCalls(offline_rc=2)
        code, terminal, _ = self.execute(calls)
        self.assertEqual(code, 2)
        self.assertEqual(terminal['error_code'], 'local_offline_audit_failed')
        self.assertEqual(len(calls.audit_calls), 2)

    def test_remote_and_offline_audits_must_agree(self):
        calls = FakeCalls(offline_audit=audit(catalog_sha256='c' * 64))
        code, terminal, _ = self.execute(calls)
        self.assertEqual(code, 2)
        self.assertEqual(terminal['error_code'], 'offline_audit_differs_from_remote')

    def test_audit_catalog_must_match_client_event(self):
        calls = FakeCalls(remote_audit=audit(catalog_sha256='c' * 64))
        self.assertEqual(self.execute(calls)[1]['error_code'], 'client_catalog_audit_mismatch')

    def test_audit_cannot_open_evaluation_gate(self):
        calls = FakeCalls(remote_audit=audit(formal_evaluation_ready=True))
        terminal = self.execute(calls)[1]
        self.assertEqual(terminal['status'], 'FAILED')
        self.assertFalse(terminal['formal_evaluation_ready'])

    def test_pins_are_rechecked_after_long_running_client(self):
        calls = FakeCalls(after_client=lambda: Path(self.config['auditor']['local_path']).write_text('changed'))
        terminal = self.execute(calls)[1]
        self.assertEqual(terminal['error_code'], 'pinned_file_sha_changed')
        self.assertFalse(calls.audit_calls)

    def test_changed_alias_is_rejected_before_any_process(self):
        self.config['ssh_alias'] = 'unapproved'
        self.write_config()
        with self.assertRaisesRegex(W.WorkflowError, 'execution_host_or_runtime_changed'):
            W.load_config(self.config_path, self.config_sha)

    def test_changed_wrapper_config_sha_is_rejected(self):
        with self.assertRaisesRegex(W.WorkflowError, 'pinned_file_sha_changed'):
            W.load_config(self.config_path, 'f' * 64)

    def test_preexisting_evidence_directory_is_not_reused(self):
        path = Path(self.config['evidence_dir'])
        path.mkdir()
        (path / 'keep').write_text('preserve')
        calls = FakeCalls()
        with self.assertRaises(FileExistsError):
            self.execute(calls)
        self.assertEqual((path / 'keep').read_text(), 'preserve')
        self.assertFalse(calls.client_calls)

    def test_duplicate_json_event_key_is_rejected(self):
        calls = FakeCalls(lines=['{"event":"ignored","event":"PINNED_HF_ZIP_CACHE_COMPLETE"}'])
        self.assertEqual(self.execute(calls)[1]['error_code'], 'duplicate_json_key')
        self.assertFalse(calls.audit_calls)

    def test_oversized_log_is_rejected_without_audit(self):
        with patch.object(W, 'MAX_LOG', 1):
            calls = FakeCalls()
            self.assertEqual(self.execute(calls)[1]['error_code'], 'client_log_oversized')
            self.assertFalse(calls.audit_calls)

    def test_interruption_records_unknown_client_state_without_kill_or_restart(self):
        calls = FakeCalls(interrupted=True)
        code, terminal, _ = self.execute(calls)
        self.assertEqual(code, 2)
        self.assertTrue(terminal['client_still_running_at_wrapper_exit'])
        self.assertFalse(terminal['client_was_killed_or_restarted_by_wrapper'])
        self.assertFalse(terminal['artifacts']['client.log']['stable'])
        self.assertIsNone(terminal['artifacts']['client.log']['sha256'])
        self.assertFalse(calls.audit_calls)

    def test_default_mode_does_not_read_stdin_or_start_process(self):
        class ForbiddenStdin:
            def read(self, *args):
                raise AssertionError('wrapper read token stdin')
            readline = read
        output = io.StringIO()
        with patch.object(W.sys, 'stdin', ForbiddenStdin()), patch.object(W.sys, 'argv',
             ['wrapper', '--config', str(self.config_path), '--config-sha256', self.config_sha]), \
             patch.object(W, 'run_once') as run, contextlib.redirect_stdout(output):
            self.assertEqual(W.main(), 0)
        run.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())['status'], 'PREPARED_NO_PROCESS_START')

    def test_real_fd0_passthrough_without_token_in_wrapper_artifacts(self):
        token = 'hf_unit_test_memory_only'
        client = Path(self.config['client']['path'])
        client.write_text("import sys,json\nvalue=sys.stdin.readline().strip()\n"
                          f"assert len(value)=={len(token)} and value.startswith('hf_')\n"
                          "print(" + repr(json.dumps(marker())) + ")\n")
        self.config['client']['sha256'] = digest(client.read_bytes())
        self.write_config()
        program = """
import importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
spec=importlib.util.spec_from_file_location('wrap',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
config=json.loads(Path(sys.argv[2]).read_text());m.PLAN_SHA=config['plan']['sha256'];m.AUDITOR_SHA=config['auditor']['sha256']
audit=json.loads(sys.argv[4])
def offline_runner(argv,stdin,stdout,stderr,timeout):
 assert stdin==m.subprocess.DEVNULL
 result={'snapshot':{},'snapshot_sha256':'b'*64,'audit':audit} if argv[0]=='ssh' else audit
 stdout.write(json.dumps(result).encode());return SimpleNamespace(returncode=0)
m.load_config(Path(sys.argv[2]),sys.argv[3])
code,terminal,path=m.run_once(config,Path(sys.argv[2]),sys.argv[3],runner=offline_runner)
print(json.dumps({'code':code,'status':terminal['status']}));sys.exit(code)
"""
        result = subprocess.run([sys.executable, '-c', program, str(WRAPPER_PATH), str(self.config_path),
                                 self.config_sha, json.dumps(audit())], input=token + '\n',
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(token, result.stdout + result.stderr)
        for path in Path(self.config['evidence_dir']).iterdir():
            self.assertNotIn(token.encode(), path.read_bytes())


if __name__ == '__main__':
    unittest.main()
