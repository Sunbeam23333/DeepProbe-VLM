#!/usr/bin/env python3
"""One client run, then independent read-only remote/offline archive acceptance.

The client inherits fd0 directly. This wrapper never reads stdin, a token file,
or credentials; it never restarts an upload, schedules itself, or evaluates.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import signal
import subprocess
import sys

HOST = 'haopengjin2ali'
PLAN_SHA = '89cf4eddb4ffc62a89b46a4b7966a68c3fce3af46dcb8c6fb32fa35ea96b1512'
AUDITOR_SHA = '0c28cc788c5d32839ae60e2453e23bef3fadf360974c1b9eb05022333059a620'
REMOTE_ROOT = '/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/'
REMOTE_PYTHON = REMOTE_ROOT + 'benchmark-oss-stream-runtime-20261013/venv/bin/python'
ARCHIVE_BYTES = 118322902217
COMPLETION = 'PINNED_HF_ZIP_CACHE_COMPLETE'
AUDIT_PASS = 'ARCHIVE_METADATA_INDEPENDENT_AUDIT_PASSED'
MAX_LOG = 100_000_000
MAX_CAPTURE = 100_000_000
BOOTSTRAP = (
    'import hashlib,runpy,sys; from pathlib import Path; '
    'p,expected,q,qsha=sys.argv[1:]; '
    'assert not any(x.is_symlink() for x in [Path(p),*Path(p).parents,Path(q),*Path(q).parents]); '
    'assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==expected; '
    'assert hashlib.sha256(Path(q).read_bytes()).hexdigest()==qsha; '
    'sys.argv=[p,"--plan",q,"--remote-read-only"]; runpy.run_path(p,run_name="__main__")'
)


class WorkflowError(Exception):
    pass


def require(value, code):
    if not value:
        raise WorkflowError(code)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def safe_path(value):
    path = Path(value)
    require(path.is_absolute() and '..' not in path.parts
            and all(not p.is_symlink() for p in [path, *path.parents]), 'unsafe_local_path')
    return path


def hash_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_file(path, expected, maximum=MAX_CAPTURE):
    path = safe_path(path)
    require(path.is_file() and path.stat().st_size <= maximum, 'missing_or_oversized_pinned_file')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'pinned_file_sha_changed')
    return raw


def strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate_json_key')
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=unique,
                          parse_constant=lambda _: (_ for _ in ()).throw(WorkflowError('nonfinite_json')))
    except (ValueError, UnicodeError):
        raise WorkflowError('invalid_json') from None


def remote_path(value, filename=None):
    path = PurePosixPath(value)
    require(path.is_absolute() and '..' not in path.parts and str(path) == value
            and value.startswith(REMOTE_ROOT), 'unapproved_remote_path')
    if filename:
        require(path.name == filename, 'unexpected_remote_filename')


def validate_bindings(config):
    require(set(config) == {'schema_version', 'ssh_alias', 'local_python', 'remote_python',
        'client', 'client_helper', 'client_config', 'auditor', 'plan', 'evidence_dir'}, 'config_fields_changed')
    require(config['schema_version'] == 1 and config['ssh_alias'] == HOST
            and config['remote_python'] == REMOTE_PYTHON, 'execution_host_or_runtime_changed')
    require(Path(config['local_python']).is_absolute() and os.access(config['local_python'], os.X_OK),
            'local_python_unavailable')
    for key in ('client', 'client_helper', 'client_config'):
        pin = config[key]
        require(set(pin) == {'path', 'sha256'} and re.fullmatch('[0-9a-f]{64}', pin['sha256']),
                'invalid_file_pin')
        checked_file(pin['path'], pin['sha256'])
    require(Path(config['client']['path']).name == 'coordinate_hf_stream_oss_cache_v4.py',
            'client_v4_required')
    require(Path(config['client_helper']['path']) ==
            Path(config['client']['path']).with_name('hf_to_stream_oss_cache_v3.py'),
            'client_helper_path_changed')
    for key, expected, filename in [('auditor', AUDITOR_SHA, 'audit_hf_stream_oss_archives.py'),
                                     ('plan', PLAN_SHA, None)]:
        pin = config[key]
        require(set(pin) == {'local_path', 'remote_path', 'sha256'} and pin['sha256'] == expected,
                'auditor_or_plan_pin_changed')
        checked_file(pin['local_path'], expected)
        remote_path(pin['remote_path'], filename)
    client_config = strict_json(checked_file(config['client_config']['path'], config['client_config']['sha256']))
    require(set(client_config) == {'python', 'worker', 'worker_sha256', 'local_plan', 'remote_plan', 'plan_sha256'},
            'client_config_fields_changed')
    require(client_config['python'] == REMOTE_PYTHON
            and client_config['worker_sha256'] == config['client_helper']['sha256']
            and client_config['local_plan'] == config['plan']['local_path']
            and client_config['remote_plan'] == config['plan']['remote_path']
            and client_config['plan_sha256'] == PLAN_SHA, 'client_config_binding_changed')
    remote_path(client_config['worker'], 'hf_to_stream_oss_cache_v3.py')
    safe_path(config['evidence_dir'])
    return config


def load_config(path, expected):
    require(bool(re.fullmatch('[0-9a-f]{64}', expected)), 'invalid_config_sha')
    return validate_bindings(strict_json(checked_file(path, expected, maximum=1_000_000)))


def client_command(config):
    return [config['local_python'], '-B', config['client']['path'], '--config',
            config['client_config']['path'], '--config-sha256', config['client_config']['sha256'], '--execute']


def remote_audit_command(config):
    remote = ['env', 'PYTHONNOUSERSITE=1', 'PYTHONDONTWRITEBYTECODE=1', config['remote_python'],
              '-c', BOOTSTRAP, config['auditor']['remote_path'], AUDITOR_SHA,
              config['plan']['remote_path'], PLAN_SHA]
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=30',
            '-o', 'ServerAliveCountMax=3', HOST, shlex.join(remote)]


def completion_from_log(path):
    require(path.stat().st_size <= MAX_LOG, 'client_log_oversized')
    completions, failed, last = [], False, None
    with path.open('rb') as stream:
        while True:
            line = stream.readline(1_000_001)
            if not line:
                break
            require(len(line) <= 1_000_000, 'client_log_line_oversized')
            if not line.strip():
                continue
            event = strict_json(line)
            require(isinstance(event, dict), 'client_log_not_object')
            if event.get('event') == COMPLETION:
                completions.append(event)
            # Worker operation failures may recover inside the single pinned
            # client after exact-state inspection. Only coordinator failure is
            # terminal here; independent acceptance still checks every archive.
            if event.get('event') == 'HF_OSS_CACHE_COORDINATOR_FAILED':
                failed = True
            last = event
    require(len(completions) == 1 and not failed and last == completions[0],
            'client_completion_event_not_unique_terminal_success')
    marker = completions[0]
    require(set(marker) == {'event', 'archives', 'bytes', 'catalog_sha256', 'all_benchmark_assets_ready'}
            and marker['archives'] == 31 and marker['bytes'] == ARCHIVE_BYTES
            and isinstance(marker['catalog_sha256'], str)
            and re.fullmatch('[0-9a-f]{64}', marker['catalog_sha256'])
            and marker['all_benchmark_assets_ready'] is False, 'client_completion_binding_changed')
    return marker


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def capture(argv, stdout_path, stderr_path, runner=subprocess.run):
    with stdout_path.open('xb') as out, stderr_path.open('xb') as err:
        # Auditors must not inherit any unread bytes from the client's token fd.
        result = runner(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=900)
    return result.returncode


def verified_audit(capture_path, offline_path, marker):
    require(capture_path.stat().st_size <= MAX_CAPTURE and offline_path.stat().st_size <= MAX_CAPTURE,
            'audit_capture_oversized')
    remote = strict_json(capture_path.read_bytes())
    offline = strict_json(offline_path.read_bytes())
    require(isinstance(remote, dict) and set(remote) == {'snapshot', 'snapshot_sha256', 'audit'},
            'remote_audit_output_changed')
    require(offline == remote['audit'], 'offline_audit_differs_from_remote')
    require(offline.get('status') == AUDIT_PASS and offline.get('plan_sha256') == PLAN_SHA
            and offline.get('exact_object_count') == 64 and offline.get('archive_count') == 31
            and offline.get('archive_bytes') == ARCHIVE_BYTES and offline.get('small_object_count') == 33,
            'archive_audit_totals_or_plan_changed')
    require(offline.get('catalog_sha256') == marker['catalog_sha256'], 'client_catalog_audit_mismatch')
    require(offline.get('all_pinned_zip_archives_cached') is True
            and offline.get('all_benchmark_assets_ready') is False
            and offline.get('formal_evaluation_ready') is False and offline.get('ntu_media_included') is False
            and offline.get('zip_body_get_count') == 0, 'archive_audit_readiness_changed')
    return offline


def run_once(config, config_path, config_sha, popen=subprocess.Popen, runner=subprocess.run):
    directory = safe_path(config['evidence_dir'])
    require(directory.parent == safe_path(config_path).parent, 'evidence_directory_must_be_new_config_sibling')
    directory.mkdir()  # Exclusive one-shot run: an existing directory is never reused.
    state = {'schema_version': 1, 'wrapper_pid': os.getpid(), 'started_utc': now(),
             'config_path': str(config_path), 'config_sha256': config_sha, 'phase': 'client_not_started',
             'client_attempts': 0, 'upload_restarts': 0, 'automatic_evaluation_enabled': False,
             'automation_created': False, 'formal_evaluation_ready': False, 'all_benchmark_assets_ready': False}
    write_json(directory / 'WRAPPER_STARTED.json', state)
    client, terminal, code = None, None, 2
    try:
        # fd0 goes straight to the pinned client; no read/copy/log/argv token handling here.
        with (directory / 'client.log').open('xb') as log:
            client = popen(client_command(config), stdin=None, stdout=log, stderr=subprocess.STDOUT)
            state.update(phase='client_running', client_pid=client.pid, client_attempts=1)
            write_json(directory / 'CLIENT_STARTED.json', {**state, 'client_argv': client_command(config)})
            rc = client.wait()  # No upload timeout or automatic retry.
        state.update(phase='client_terminal', client_returncode=rc)
        require(rc == 0, 'client_nonzero_exit')
        marker = completion_from_log(directory / 'client.log')
        # Recheck all frozen local inputs after a potentially hours-long transfer.
        load_config(config_path, config_sha)
        state.update(phase='remote_readonly_audit', completion_event=marker)
        remote_capture, remote_stderr = directory / 'remote_audit_capture.json', directory / 'remote_audit.stderr'
        audit_rc = capture(remote_audit_command(config), remote_capture, remote_stderr, runner)
        state['remote_audit_returncode'] = audit_rc
        require(audit_rc == 0, 'remote_readonly_audit_failed')
        state['phase'] = 'local_offline_audit'
        checked_file(config['auditor']['local_path'], AUDITOR_SHA)
        checked_file(config['plan']['local_path'], PLAN_SHA)
        local_capture, local_stderr = directory / 'local_audit.json', directory / 'local_audit.stderr'
        offline_argv = [config['local_python'], '-B', config['auditor']['local_path'], '--plan',
                        config['plan']['local_path'], '--snapshot', str(remote_capture)]
        offline_rc = capture(offline_argv, local_capture, local_stderr, runner)
        state['local_audit_returncode'] = offline_rc
        require(offline_rc == 0, 'local_offline_audit_failed')
        audit = verified_audit(remote_capture, local_capture, marker)
        state.update(phase='accepted', archive_count=31, archive_bytes=ARCHIVE_BYTES,
                     exact_object_count=64, catalog_sha256=audit['catalog_sha256'],
                     exact_prefix_total_bytes=audit['exact_prefix_total_bytes'],
                     ntu_media_included=False, all_pinned_zip_archives_cached=True)
        terminal, code = {**state, 'status': 'PINNED_ARCHIVE_CACHE_INDEPENDENTLY_ACCEPTED'}, 0
    except Exception as error:
        terminal = {**state, 'status': 'FAILED',
                    'error_code': str(error) if isinstance(error, WorkflowError) else 'wrapper_operation_failed'}
    finally:
        if terminal is not None:
            terminal['finished_utc'] = now()
            active = client is not None and client.poll() is None
            terminal['client_still_running_at_wrapper_exit'] = active
            terminal['client_was_killed_or_restarted_by_wrapper'] = False
            artifacts = {}
            for path in sorted(directory.iterdir()):
                if not path.is_file() or path.is_symlink():
                    continue
                if path.name == 'client.log' and active:
                    artifacts[path.name] = {'stable': False, 'sha256': None}
                else:
                    artifacts[path.name] = {'stable': True, 'size_bytes': path.stat().st_size,
                                            'sha256': hash_file(path)}
            terminal['artifacts'] = artifacts
            write_json(directory / 'TERMINAL.json', terminal)
    return code, terminal, directory / 'TERMINAL.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config, args.config_sha256)
    if not args.execute:
        print(json.dumps({'status': 'PREPARED_NO_PROCESS_START', 'client_attempts': 0,
                          'automatic_evaluation_enabled': False}), flush=True)
        return 0
    def interrupted(signum, frame):
        raise WorkflowError('wrapper_interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    code, terminal, path = run_once(config, args.config, args.config_sha256)
    print(json.dumps({'status': terminal['status'], 'terminal_receipt': str(path),
                      'terminal_sha256': hash_file(path), 'automatic_evaluation_enabled': False}), flush=True)
    return code


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        code = str(error) if isinstance(error, WorkflowError) else 'wrapper_operation_failed'
        print(json.dumps({'status': 'FAILED', 'error_code': code, 'automatic_evaluation_enabled': False}), flush=True)
        sys.exit(2)
