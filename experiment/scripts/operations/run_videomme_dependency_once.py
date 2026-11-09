#!/usr/bin/env python3
"""Bounded Video-MME CPU preparation or exact-R4 dependency handoff, once.

--prepare-only waits for complete899 media, then calls CPU preparation and an
independently authored CPU audit once. --execute performs that same preparation,
waits for exact R4 success/archive, then calls the submission wrapper once.
No recurring automation, training retry, archive retry or input cleanup exists.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
BASE = Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM')
P1 = Path(str(BASE) + '-p1')
PYTHON = str(BASE / 'env/bin/python')
RUN = 'dp-benchmark-videomme-r1-r2-s17-a1'
R4_RUN = 'dp-p1-ouro-r4-f8-g4-s17-alignment-a1'
R4_JOB = 'jinhaopeng-dpvlm-p1-ouro-r4-s17-align-a1-n1-2748f'
R4_CONFIG_SHA = 'b1c3fc9ce9bcc65335ebc68435730ef8f32a0e2d9cb953babf13a871d5b0c865'
R4_SOURCE_SHA = 'ccc99b4780849055ea1df69ef69dfc645f3bdc66740bd0366f36958a07ac0cd2'
R4_URI = 'oss://ss-devbox-stream/jinhaopeng/Paper/DeepProbe-VLM/runs/' + R4_RUN
PREPARATION = P1 / 'videomme-bucket-evaluation-cpu-20261013-a1'
AUDIT = P1 / 'videomme-bucket-evaluation-independent-20261013-a1'
MEDIA = P1 / 'benchmark-media-20261012/videomme/VIDEOMME_BUCKET_MEDIA_COMPLETE.json'
MEDIA_TERMINAL = P1 / 'benchmark-videomme-oss-decode-20261013/run-a1/TERMINAL.json'
MEDIA_WORKER_SHA = '444187e3155686f8f82344ccc1e798d595b2258472f6afee6b9f3c8bea087c34'
MEDIA_CONFIG_SHA = '26181ab9747eeb696ff018ba25cd2d42bdcf5913a579af566ffee555a5cc64e8'
CONTROL_NAMES = ('EXIT_STATUS.json', 'P1_STAGE_COMPLETE.json', 'ARCHIVE_MANIFEST.json', 'ARCHIVED.json')


def require(value, message):
    if not value: raise ValueError(message)


def utc(): return datetime.now(timezone.utc).isoformat()


def path(value):
    result = Path(value)
    require(result.is_absolute() and '..' not in result.parts and
            not any(p.is_symlink() for p in [result, *result.parents]), 'Unsafe artifact path')
    return result


def sha(value):
    digest = hashlib.sha256()
    with path(value).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''): digest.update(chunk)
    return digest.hexdigest()


def pin(value): return {'path': str(path(value)), 'sha256': sha(value)}


def verify(item):
    result = path(item['path'])
    require(result.is_file() and sha(result) == item['sha256'], 'Pinned input changed: ' + str(result))
    return result


def load(value):
    value = json.loads(path(value).read_text())
    require(isinstance(value, dict), 'JSON object required')
    return value


def save(value, document):
    target = path(value)
    with target.open('x') as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def event(root, state, **extra):
    document = {'observed_utc': utc(), 'state': state, **extra}
    with (root / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(document, sort_keys=True, allow_nan=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(document, sort_keys=True), flush=True)


def load_module(source, name):
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def validate_config(config):
    require(config.get('kind') == 'videomme_bounded_dependency_once_v1' and config.get('run_id') == RUN,
            'Unexpected dependency identity')
    require(config.get('predecessor') == {'exact_job': R4_JOB, 'run_id': R4_RUN,
            'run_root': str(BASE / 'runs' / R4_RUN), 'oss_prefix': R4_URI + '/',
            'requested_config_sha256': R4_CONFIG_SHA, 'source_manifest_sha256': R4_SOURCE_SHA,
            'optimizer_steps': 718}, 'Exact R4 predecessor binding differs')
    require(type(config.get('wait_seconds')) is int and 1 <= config['wait_seconds'] <= 86400 and
            type(config.get('poll_seconds')) is int and 1 <= config['poll_seconds'] <= 60,
            'Wait must be bounded to24h, polling to60s')
    require(verify(config['pipeline_script']).resolve() == Path(__file__).resolve(), 'Pin this exact dependency script')
    preparation_config_path = verify(config['cpu_preparation_config'])
    preparation = load(preparation_config_path)
    require(preparation.get('candidate_output_dir') == str(PREPARATION), 'CPU output root differs')
    prep_script = verify(preparation['pipeline_script'])
    scope = verify(preparation['authorization'])
    helper = verify(preparation['submission_helper'])
    auditor = verify(config['independent_audit_script'])
    for item in preparation['evaluation_source_files']: verify(item)
    prepare_module = load_module(prep_script, 'accepted_vm_cpu_preparation')
    prepare_module.validate_scope(load(scope))
    package = (json.dumps(preparation['evaluation_source_files'], sort_keys=True, separators=(',', ':')) + '\n').encode()
    require(hashlib.sha256(package).hexdigest() == prepare_module.FROZEN_SOURCE_LIST_SHA, 'Frozen package changed')
    return preparation, preparation_config_path, prep_script, helper, auditor


def wait_for(root, probe, deadline, poll_seconds):
    last = None
    while time.monotonic() < deadline:
        ready, state, evidence = probe(max(0.001, deadline - time.monotonic()))
        if state != last:
            event(root, state, **evidence); last = state
        require(time.monotonic() < deadline, 'Dependency wait exceeded deadline; no call performed')
        if ready: return evidence
        time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))
    raise TimeoutError('Dependency wait reached24h/configured bound; no call performed')


def budget(deadline):
    remaining = deadline - time.monotonic()
    require(remaining > 0, 'Dependency deadline reached; no call performed')
    return min(14400, remaining)


def invoke_once(root, label, argv, env, timeout=14400):
    """New evidence/phase intent plus no retry, even on timeout or client error."""
    save(root / (label + '.argv.json'), {'argv': argv, 'invoked_utc': utc(), 'automatic_retry': False})
    error = None
    with (root / (label + '.stdout')).open('xb') as out, (root / (label + '.stderr')).open('xb') as err:
        process = subprocess.Popen(argv, stdout=out, stderr=err, env=env, cwd='/home/jinhaopeng', start_new_session=True)
        try: returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Only this wrapper's process group, never a scheduler/GPU job.
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
            returncode, error = process.returncode, 'TimeoutExpired inspect exact job/intent before any manual action'
    result = {'returncode': returncode, 'error': error, 'stdout': pin(root / (label + '.stdout')),
              'stderr': pin(root / (label + '.stderr')), 'automatic_retry': False}
    save(root / (label + '.exit.json'), result)
    require(returncode == 0 and error is None, label + ' failed/ambiguous; never invoked again')
    return result


def media_probe(remaining=None):
    # TERMINAL is the real supervisor's last fsynced record after worker exit.
    # Its exclusive publication briefly exposes a partial JSON; never launch a
    # CPU gate from a partially published receipt or before this stable barrier.
    if not MEDIA_TERMINAL.is_file(): return False, 'WAITING_FOR_COMPLETE899_MEDIA', {}
    terminal = None
    for attempt in range(3):
        try: terminal = load(MEDIA_TERMINAL); break
        except json.JSONDecodeError:
            if attempt < 2: time.sleep(0.1)
    if terminal is None: return False, 'WAITING_FOR_MEDIA_TERMINAL_PUBLICATION', {}
    require(terminal.get('schema_version') == 1 and terminal.get('worker_sha256') == MEDIA_WORKER_SHA and
            terminal.get('config_sha256') == MEDIA_CONFIG_SHA and terminal.get('automatic_retry') is False and
            terminal.get('gpu_submitted') is False and terminal.get('status') in {'SUCCEEDED', 'FAILED'},
            'Media supervisor terminal identity differs')
    require(terminal['status'] != 'FAILED', 'Media supervisor FAILED; stop without retry: ' + str(MEDIA_TERMINAL))
    require(type(terminal.get('exit_code')) is int and terminal['exit_code'] == 0 and
            type(terminal.get('process_attempts')) is int and terminal['process_attempts'] == 1,
            'Media supervisor did not complete exactly one successful worker')
    require(MEDIA.is_file(), 'Media supervisor succeeded without complete899 receipt')
    media = load(MEDIA)
    require(media.get('kind') == 'videomme_bucket_media_complete_v1' and
            media.get('status') == 'VIDEOMME_BUCKET_MEDIA_COMPLETE' and media.get('complete') is True and
            media.get('source_count') == 899 and media.get('frame_count') == 7192 and
            media.get('full_oss_get_sha256_verified') is True, 'Existing media completion is invalid')
    return True, 'COMPLETE899_MEDIA_OBSERVED', {'media': pin(MEDIA), 'media_terminal': pin(MEDIA_TERMINAL)}


def validate_ready_and_gate(helper):
    ready_path = PREPARATION / 'READY_FOR_PARENT_SUBMISSION.json'
    gate_path = AUDIT / 'VIDEOMME_CPU_PREPARATION_INDEPENDENT_GATE.json'
    if (PREPARATION / 'CPU_PREPARATION_FAILED.json').is_file(): raise ValueError('CPU preparation failed; no retry')
    if not ready_path.is_file() or not gate_path.is_file(): return None
    ready, gate = load(ready_path), load(gate_path)
    require(ready.get('status') == 'VIDEOMME_ONLY_CPU_PREPARATION_COMPLETE' and ready.get('gpu_submitted') is False,
            'CPU preparation status differs')
    require(gate.get('kind') == 'videomme_cpu_preparation_independent_gate_v1' and gate.get('status') == 'passed' and
            gate.get('evaluation_scope') == 'videomme_only' and gate.get('all_benchmark_assets_ready') is False and
            gate.get('mvbench_evaluation_ready') is False and gate.get('gpu_submitted') is False,
            'Independent CPU gate not passed for VM-only scope')
    require(isinstance(gate.get('checks'), dict) and gate['checks'] and all(v is True for v in gate['checks'].values()),
            'Independent CPU checks incomplete/failed')
    require(gate.get('cpu_ready') == pin(ready_path) and gate.get('plan') == ready.get('plan'),
            'Independent audit binds different CPU ready/plan')
    plan_path = verify(ready['plan']); plan = load(plan_path)
    require(plan_path == PREPARATION / 'EVALUATION_PLAN.json' and plan.get('run_id') == RUN and
            plan.get('evaluation_scope') == 'videomme_only' and plan.get('all_benchmark_assets_ready') is False and
            plan.get('mvbench_evaluation_ready') is False, 'Independent plan identity differs')
    expected = [PYTHON, str(helper), '--plan', str(plan_path), '--plan-sha256', sha(plan_path),
                '--output-dir', str(PREPARATION / 'submission'), '--execute']
    require(ready.get('parent_submission_argv') == expected, 'Parent command differs from exact accepted wrapper')
    verify(ready['cpu_preflight']); verify(ready['scope_ready'])
    return {'ready': pin(ready_path), 'gate': pin(gate_path), 'plan': pin(plan_path), 'argv': expected}


def exact_r4_state(frozen, timeout=60):
    result = subprocess.run([frozen.SSLAUNCH, 'job', 'info', '-c', 'ali-gpu', R4_JOB],
                            capture_output=True, text=True, timeout=min(60, timeout), check=False)
    require(result.returncode == 0, 'Exact R4 scheduler query failed')
    require(re.findall(r'^Job:\s*(\S+)\s*$', result.stdout, re.M) == [R4_JOB], 'Exact R4 job identity missing/repeated')
    states = re.findall(r'^State:\s*(\w+)\s*$', result.stdout, re.M)
    require(len(states) == 1, 'Exact R4 state missing/repeated')
    require(states[0] in {'Created', 'Pending', 'Running', 'Succeeded'}, 'R4 failed/cancelled/unknown/restarting; stop without retry')
    return states[0], result.stdout


def validate_r4_controls(directory):
    documents = {name: load(directory / name) for name in CONTROL_NAMES}
    for value in documents.values():
        require(value.get('run_id') == R4_RUN and value.get('stage') == 'alignment', 'R4 control run/stage mismatch')
    exit_status, complete, manifest, archived = (documents[name] for name in CONTROL_NAMES)
    require(type(exit_status.get('exit_code')) is int and exit_status['exit_code'] == 0 and
            exit_status.get('automatic_retry') is False, 'R4 did not exit0')
    require(complete.get('status') == 'p1_formal_stage_complete' and complete.get('formal_training_result') is True and
            complete.get('held_out_benchmark_result') is False and complete.get('world_size') == 8 and
            complete.get('seed') == 17 and complete.get('rfg') == {'num_frames': 8, 'num_loops': 4, 'pool_grid': 4} and
            complete.get('training_result', {}).get('optimizer_steps') == 718 and
            complete.get('requested_config_sha256') == R4_CONFIG_SHA and
            complete.get('source_identity', {}).get('manifest_sha256') == R4_SOURCE_SHA, 'R4 completion/config/source mismatch')
    require(archived.get('status') == 'archive_verified' and archived.get('uri') == R4_URI and
            archived.get('archive_manifest_sha256') == sha(directory / 'ARCHIVE_MANIFEST.json') and
            archived.get('remote_manifest_and_completion_sha256_readback') is True and manifest.get('uri') == R4_URI,
            'R4 archive manifest/control SHA mismatch')
    files, total = {}, 0
    require(isinstance(manifest.get('files'), list) and manifest['files'], 'R4 archive manifest empty')
    for row in manifest['files']:
        relative = PurePosixPath(row['path'])
        require(not relative.is_absolute() and '..' not in relative.parts and str(relative) == row['path'] and
                row['path'] not in files and row['path'] not in {'ARCHIVE_MANIFEST.json', 'ARCHIVED.json'} and
                type(row['bytes']) is int and row['bytes'] >= 0 and re.fullmatch(r'[a-f0-9]{64}', row['sha256']),
                'Invalid R4 archive payload row')
        files[row['path']] = row; total += row['bytes']
    require(manifest.get('file_count') == archived.get('file_count') == len(files) and
            manifest.get('total_bytes') == archived.get('total_bytes') == total, 'R4 archive count/size mismatch')
    for name in ['EXIT_STATUS.json', 'P1_STAGE_COMPLETE.json']:
        require(name in files and files[name]['sha256'] == sha(directory / name) and
                files[name]['bytes'] == (directory / name).stat().st_size, 'R4 manifest does not bind success control')
    return files


def verify_r4_archive(root, frozen):
    local = BASE / 'runs' / R4_RUN
    files = validate_r4_controls(local)
    readback = root / 'r4-oss-readback'; readback.mkdir()
    local_pins = {name: pin(local / name) for name in CONTROL_NAMES}
    for name in CONTROL_NAMES:
        result = frozen.capture(root, 'r4_readback_' + name.replace('.', '_'),
            [frozen.OSSUTIL, 'cp', R4_URI + '/' + name, str(readback / name), '-c', frozen.OSS_CONFIG, '--no-progress'], timeout=180)
        require(result['returncode'] == 0 and sha(readback / name) == local_pins[name]['sha256'],
                'R4 OSS control readback mismatch: ' + name)
    validate_r4_controls(readback)
    listing = frozen.capture(root, 'r4_exact_oss', [frozen.OSSUTIL, 'ls', R4_URI + '/', '-c', frozen.OSS_CONFIG])
    require(listing['returncode'] == 0, 'R4 exact OSS listing failed')
    expected = {R4_URI + '/' + name: row['bytes'] for name, row in files.items()}
    expected.update({R4_URI + '/' + name: (readback / name).stat().st_size for name in ['ARCHIVE_MANIFEST.json', 'ARCHIVED.json']})
    actual = {}
    for line in listing['stdout'].splitlines():
        fields = line.split()
        if not fields or not fields[-1].startswith('oss://'): continue
        require(len(fields) >= 6 and fields[-1] not in actual and fields[4].isdigit(), 'Malformed exact R4 OSS row')
        actual[fields[-1]] = int(fields[4])
    require(actual == expected and re.findall(r'^Object Number is:\s*(\d+)\s*$', listing['stdout'], re.M) == [str(len(expected))],
            'R4 exact OSS payload/control key set or lengths differ')
    for item in local_pins.values(): verify(item)
    result = {'status': 'EXACT_R4_SUCCESS_ARCHIVE_DEPENDENCY_VERIFIED', 'exact_job': R4_JOB,
              'controls': local_pins, 'oss_prefix': R4_URI + '/', 'object_count': len(expected),
              'bytes': sum(expected.values()), 'payload_full_SHA_readback_performed': False,
              'engineering_dependency_only': True, 'r4_held_out_quality_required': False}
    save(root / 'R4_DEPENDENCY_VERIFIED.json', result)
    return result


def prepare_once(config, root, prepared, deadline, check_predecessor=False):
    prep, config_path, script, helper, auditor = prepared
    require(not PREPARATION.exists() and not AUDIT.exists(), 'CPU/audit output already exists; inspect without retry')
    probe = media_probe
    if check_predecessor:
        frozen = load_module(helper, 'accepted_vm_early_dependency_wrapper').load_frozen_helper()
        def probe(remaining):
            state, _ = exact_r4_state(frozen, remaining)
            ready, label, evidence = media_probe(remaining)
            return ready, label + '_R4_' + state, {**evidence, 'exact_job': R4_JOB, 'r4_state': state}
    wait_for(root, probe, deadline, config['poll_seconds'])
    validate_config(config)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1')
    invoke_once(root, 'cpu_preparation', [PYTHON, str(script), '--config', str(config_path),
                '--config-sha256', sha(config_path), '--output-dir', str(PREPARATION)], env, timeout=budget(deadline))
    ready = PREPARATION / 'READY_FOR_PARENT_SUBMISSION.json'
    invoke_once(root, 'independent_audit', [PYTHON, str(auditor), '--preparation-root', str(PREPARATION),
                '--ready-sha256', sha(ready), '--output-dir', str(AUDIT)], env, timeout=budget(deadline))
    gate = validate_ready_and_gate(helper)
    require(gate is not None, 'CPU/audit command returned without accepted completion')
    save(root / 'PREPARATION_AND_INDEPENDENT_GATE_COMPLETE.json', {**gate, 'gpu_submitted': False})


def submission_environment():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    env.pop('CUDA_VISIBLE_DEVICES', None)
    return env


def submit_once(config, root, prepared, deadline):
    _, _, _, helper, _ = prepared
    wrapper = load_module(helper, 'accepted_vm_submission_wrapper')
    frozen = wrapper.load_frozen_helper()
    def inspect(remaining):
        state, stdout = exact_r4_state(frozen, remaining)
        gate = validate_ready_and_gate(helper)
        controls_exist = all((BASE / 'runs' / R4_RUN / name).is_file() for name in CONTROL_NAMES)
        ready = state == 'Succeeded' and gate is not None and controls_exist
        return ready, ('DEPENDENCIES_READY' if ready else f'WAITING_R4_{state}_CPU_{gate is not None}_ARCHIVE_{controls_exist}'), {'exact_job': R4_JOB, 'r4_state': state}
    wait_for(root, inspect, deadline, config['poll_seconds'])
    validate_config(config)
    verify_r4_archive(root, frozen)
    state, stdout = exact_r4_state(frozen); require(state == 'Succeeded', 'R4 state changed before handoff')
    (root / 'r4_exact_success.stdout').write_text(stdout)
    gate = validate_ready_and_gate(helper); require(gate is not None, 'Independent CPU gate disappeared')
    plan = load(verify(gate['plan']))
    snapshot = frozen.live_snapshot(root, 'bridge_fresh_gate', plan)
    require(snapshot['jobs'].get(R4_JOB) == 'Succeeded' and snapshot['kubernetes'].get(R4_JOB) == 'Succeeded',
            'R4 scheduler/Kubernetes success disagree')
    save(root / 'READY_FOR_SINGLE_SUBMISSION_CALL.json', {**gate, 'r4_dependency': pin(root / 'R4_DEPENDENCY_VERIFIED.json'),
         'fresh_snapshot': snapshot, 'automatic_retry': False})
    # The accepted helper repeats CPU/capacity/exact OSS/jobs/K8s/dry-run checks
    # and writes its own durable global intent before its one real submit.
    invoke_once(root, 'submission_wrapper', gate['argv'], submission_environment(), timeout=budget(deadline))
    outcome_path = PREPARATION / 'submission/SUBMISSION_OUTCOME.json'
    outcome = load(outcome_path)
    require(outcome.get('status') == 'SUBMITTED_EXACT_JOB_OBSERVED' and outcome.get('never_retry_this_run_id') is True,
            'Submission outcome ambiguous; inspect exact job, never retry')
    save(root / 'SINGLE_HANDOFF_COMPLETE.json', {'status': 'SUBMITTED_EXACT_JOB_OBSERVED', 'outcome': pin(outcome_path),
         'automatic_retry': False, 'recurring_automation_created': False, 'evaluation_scope': 'videomme_only'})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare-only', action='store_true'); modes.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    require(sys.platform == 'linux' and getpass.getuser() == 'jinhaopeng' and BASE.is_dir(), 'Authorized Ali devbox only')
    config = load(verify({'path': str(args.config), 'sha256': args.config_sha256}))
    prepared = validate_config(config)
    root = path(args.output_dir); require(root.is_relative_to(P1) and not root.exists(), 'New dependency evidence root required')
    phase = 'prepare' if args.prepare_only else 'submit'
    locks = path(BASE / 'locks'); locks.mkdir(exist_ok=True)
    with (locks / ('videomme-dependency-' + phase + '.lock')).open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        intent = locks / ('videomme-dependency-' + phase + '-' + RUN + '.json')
        save(intent, {'config': pin(args.config), 'phase': phase, 'evidence_root': str(root), 'created_utc': utc(),
                      'one_process_one_call': True, 'automatic_retry': False})
        root.mkdir(parents=True, exist_ok=False)
        try:
            event(root, 'STARTED_' + phase.upper(), config=pin(args.config), bounded_wait_seconds=config['wait_seconds'])
            deadline = time.monotonic() + config['wait_seconds']
            prepare_once(config, root, prepared, deadline, args.execute)
            if args.execute: submit_once(config, root, prepared, deadline)
        except BaseException as error:
            save(root / 'DEPENDENCY_STOPPED.json', {'error': str(error), 'error_type': type(error).__name__,
                 'phase': phase, 'automatic_retry': False, 'submission_call_attempted': (root / 'submission_wrapper.argv.json').exists(),
                 'observed_utc': utc(), 'next_action': 'Inspect exact job/intents and preserved evidence; never blindly restart this phase'})
            raise


if __name__ == '__main__': raise SystemExit(main())
