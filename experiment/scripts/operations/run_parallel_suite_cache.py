"""Explicit, supervised F4/F16 CPU producer; no training/history entry point."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import uuid

SOURCE = Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM/suite-source/20261121-v5/experiment')
CONTROL = Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/suite-suite-remaining-20261119/control-v9')
PYTHON = '/home/jinhaopeng/.cache/Paper/DeepProbe-VLM/suite-runtime/20261119-v1/bin/python'
PREDECESSOR = CONTROL.with_name('control-v6') / 'cpu-cache-prepare/747f1a06e69a47c8aeffda3edf4c4758'


def exact_process_quiet(pid, start):
    path = Path('/proc', str(pid), 'stat')
    return not path.exists() or path.read_text().split(') ', 1)[1].split()[19] != start


def process(pid):
    return {'pid': pid, 'linux_process_start': Path('/proc', str(pid), 'stat').read_text().split(') ', 1)[1].split()[19]}


def verify_supervisor_start(runner, start_pin, start, proof):
    """Only the live direct child of this control's active supervisor may write."""
    runner.verify_pin(start_pin)
    path = runner.safe(start_pin['path'])
    attempt = path.parent
    runner.require(path.name == 'START.json' and attempt.parent == CONTROL / 'cpu-cache-prepare'
                   and len(attempt.name) == 32 and all(c in '0123456789abcdef' for c in attempt.name),
                   'Supervisor START is outside this exact parallel control attempt')
    runner.require(start.get('status') == 'CPU_CACHE_PREPARATION_STARTED'
                   and start.get('output_root') == str(attempt), 'Supervisor START status/output identity differs')
    active_pin = runner.artifact(CONTROL / 'CPU_CACHE_PREPARE_ACTIVE.json')
    runner.require(all(active_pin[key] == start_pin[key] for key in ('bytes', 'sha256'))
                   and runner.read(active_pin['path']) == start, 'Supervisor START is not the current ACTIVE owner')
    pid, identity = start.get('pid'), start.get('linux_process_start')
    runner.require(type(pid) is int and pid > 0 and isinstance(identity, str) and identity.isdigit()
                   and os.getppid() == pid and not exact_process_quiet(pid, identity),
                   'Cache preparation is not the direct child of the exact live supervisor')
    runner.require(start['candidate'] == proof['candidate'] and start['driver'] == proof['driver']
                   and start['source_manifest'] == proof['source_manifest'] and start['execution'] == proof['execution'],
                   'Child runtime is not bound to exact supervisor inputs')


def inputs(args):
    assert os.getuid() == 83980 and 'haopengjin3ali' in os.uname().nodename
    assert str(Path(sys.executable).absolute()) == PYTHON
    sys.path.insert(0, str(SOURCE / 'src'))
    from deepprobe_vlm import suite_runner as runner, suite_media as media
    from deepprobe_vlm.formal_integrity import verify_frozen_source_manifest
    source_pin = runner.artifact(SOURCE.parent / 'P1_SOURCE_MANIFEST.json')
    runner.require(source_pin['sha256'] == args.source_sha256, 'Parallel CPU source manifest changed')
    source_proof = verify_frozen_source_manifest(SOURCE, Path(source_pin['path']), source_pin['sha256'])
    driver_pin = runner.artifact(Path(__file__))
    runner.require(driver_pin['sha256'] == args.driver_sha256, 'Parallel CPU driver changed')
    candidate_pin = runner.artifact(CONTROL / 'CACHE_ONLY_CANDIDATE.json')
    runner.require(candidate_pin['sha256'] == args.candidate_sha256, 'Cache-only candidate changed')
    candidate = runner.read(candidate_pin['path'])
    runner.require(candidate['environment']['experiment_root'] == str(SOURCE), 'Cache-only source differs')
    runner.require(not candidate['environment'].get('history') and not any(candidate.get(name) for name in
        ('history_imports', 'history_preparations', 'recovery_preparations')), 'History work forbidden in CPU cache candidate')
    runner.require(not any(name in candidate for name in ('job_template', 'job_template_source', 'job_template_document')),
                   'GPU template entry points forbidden')
    requests = candidate['benchmark_preparations']
    runner.require(len(requests) == 4 and {(r['benchmark'], r['frames']) for r in requests}
                   == {(b, f) for b in ('videomme899', 'tempcompass_mc') for f in (4, 16)},
                   'Parallel producer owns exactly four F4/F16 shapes; F8 belongs to legacy reuse')
    runner.require(1 <= args.workers <= 8 and 0 < args.max_inflight_bytes <= 8 * 1024**3,
                   'Explicit workers1..8 and bounded payload budget up to8GiB required')
    previous_start = runner.artifact(PREDECESSOR / 'START.json')
    runner.require(previous_start['sha256'] == 'a8393390a2762d7c55b506ea7b28e1d4d9cbc4c4b035622a85d9d12e087742fd',
                   'Predecessor producer identity changed')
    previous_exit = runner.artifact(PREDECESSOR / 'EXIT.json')
    runner.require(previous_exit['sha256'] == args.predecessor_exit_sha256, 'Exact predecessor exit changed')
    prior = runner.read(previous_exit['path'])
    runner.require(prior.get('status') == 'CPU_CACHE_PREPARATION_EXITED' and prior.get('group_cleanup_verified') is True
                   and type(prior.get('returncode')) is int, 'Predecessor is not quiescent')
    runner.require(not (CONTROL.with_name('control-v6') / 'CPU_CACHE_PREPARE_ACTIVE.json').exists()
                   and exact_process_quiet(58197, '1273577477') and exact_process_quiet(58198, '1273577511'),
                   'Old producer still owns the shared future F4/F16 prefixes')
    bounds = {}
    for request in requests:
        runner.verify_pin(request['accepted_decode_receipt'])
        original = runner.read(request['accepted_decode_receipt']['path'])
        sizes = [row['original_size_bytes'] for row in original['entries']]
        runner.require(sizes and all(type(size) is int and 0 < size <= media.MAX_MEMBER_BYTES for size in sizes),
                       'Accepted original video sizes are needed for memory admission')
        largest = media.frame_build_memory_bound({'member': {'size_bytes': max(sizes)}}, request['frames'])
        runner.require(largest <= args.max_inflight_bytes, 'Largest accepted original cannot fit payload budget')
        bounds[request['benchmark'] + '-f' + str(request['frames'])] = {
            'largest_accepted_original_bytes': max(sizes), 'largest_video_payload_bound_bytes': largest,
            'guaranteed_slots_at_largest_size': min(args.workers, args.max_inflight_bytes // largest),
            'scope': 'conservative bound over all accepted originals, including any outside selected subset'}
    return runner, candidate, {
        'source_manifest': source_pin, 'source_verification': source_proof, 'driver': driver_pin,
        'candidate': candidate_pin, 'predecessor_start': previous_start, 'predecessor_exit': previous_exit,
        'execution': {'workers': args.workers, 'max_inflight_bytes': args.max_inflight_bytes,
                      'maximum_pending_videos': args.workers, 'all_sdk_and_writes_on_coordinator': True,
                      'per_shape_memory_bounds': bounds, 'native_allocator_rss_is_not_a_hard_limit': True}}


def prepare(args):
    runner, candidate, proof = inputs(args)
    from deepprobe_vlm.suite_workflow import prepare_benchmark
    runner.require(args.supervisor_start and args.supervisor_start_sha256, 'Explicit supervisor START required')
    start_pin = runner.artifact(args.supervisor_start)
    runner.require(start_pin['sha256'] == args.supervisor_start_sha256, 'Supervisor START changed')
    start = runner.read(start_pin['path'])
    verify_supervisor_start(runner, start_pin, start, proof)
    for request in candidate['benchmark_preparations']:
        key = request['benchmark'] + '-f' + str(request['frames'])
        output = CONTROL / 'cache-assets' / key / 'PREPARED_ASSET.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            prior = runner.read(output)
            runner.require(prior['request'] == request and prior['candidate'] == proof['candidate']
                           and prior['producer_source']['manifest'] == proof['source_manifest'],
                           'Previously prepared cache belongs to a different immutable attempt')
            print(json.dumps({'status': 'CACHE_PREPARATION_ALREADY_RECORDED', 'asset': runner.artifact(output)}), flush=True)
            continue
        print(json.dumps({'status': 'CACHE_PREPARATION_STARTED', 'utc': runner.utc(), 'key': key,
                          'prefix': request['oss_prefix'], 'execution': proof['execution']}), flush=True)
        asset, complete = prepare_benchmark(candidate['environment'], request, resume=args.resume,
            workers=args.workers, max_inflight_bytes=args.max_inflight_bytes)
        runner.write_once(output, {'status': 'CPU_CACHE_ASSET_PREPARED_NOT_FLEET_ADMITTED', 'utc': runner.utc(),
            'request': request, 'request_sha256': runner.digest(request), 'candidate': proof['candidate'],
            'producer_source': {'root': str(SOURCE), 'manifest': proof['source_manifest'], 'verification': proof['source_verification']},
            'asset': asset, 'cache_complete': complete, 'execution': proof['execution'], 'driver': proof['driver'],
            'supervisor_start': start_pin, 'checkpoint_payload_gets': 0, 'gpu_submitted': False, 'fleet_ready_published': False})
        print(json.dumps({'status': 'CPU_CACHE_ASSET_PREPARED_NOT_FLEET_ADMITTED', 'key': key,
                          'asset': runner.artifact(output), 'execution': proof['execution']}), flush=True)


def supervise(args):
    runner, candidate, proof = inputs(args)
    from deepprobe_vlm.suite_process import run_group
    from deepprobe_vlm.suite_storage import budget_for
    lock = (CONTROL / 'CPU_CACHE_PREPARE.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active = CONTROL / 'CPU_CACHE_PREPARE_ACTIVE.json'
    runner.require(not active.exists(), 'Prior parallel producer needs explicit process reconciliation')
    output = CONTROL / 'cpu-cache-prepare' / uuid.uuid4().hex
    output.mkdir(parents=True)
    budget = budget_for(candidate['environment'])
    owner = {'suite_id': candidate['environment']['suite_id'], 'operation': 'parallel-cache-supervision', 'attempt': str(output)}
    token = budget.reserve(owner, 1024 * 1024, 0)
    start = {'status': 'CPU_CACHE_PREPARATION_STARTED', 'utc': runner.utc(), **process(os.getpid()), **proof,
             'output_root': str(output), 'storage_token': token, 'gpu_submitted': False, 'automatic_retry': False}
    runner.write_once(output / 'START.json', start)
    runner.write_once(active, start)
    start_pin = runner.artifact(output / 'START.json')
    result = {'status': 'CPU_CACHE_PREPARATION_EXITED', 'returncode': None, 'group_cleanup_verified': False,
              'source_manifest': proof['source_manifest'], 'driver': proof['driver'], 'candidate': proof['candidate'],
              'execution': proof['execution'], 'start': start_pin, 'gpu_submitted': False, 'automatic_retry': False}
    try:
        with (output / 'stdout').open('xb') as stdout, (output / 'stderr').open('xb') as stderr:
            run = run_group([sys.executable, '-B', str(Path(__file__)), 'prepare', *sys.argv[2:],
                '--supervisor-start', start_pin['path'], '--supervisor-start-sha256', start_pin['sha256']],
                cwd=SOURCE, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(SOURCE / 'src')),
                stdout=stdout, stderr=stderr, timeout=7 * 24 * 3600,
                on_started=lambda child: runner.write_once(output / 'CHILD.json', process(child.pid)))
        result.update(returncode=run.returncode, group_cleanup_verified=run.group_cleanup_verified)
    except BaseException as error:
        result.update(error_type=type(error).__name__, returncode=getattr(error, 'group_returncode', None),
                      group_cleanup_verified=getattr(error, 'group_cleanup_verified', False))
    finally:
        result['utc'] = runner.utc()
        runner.write_once(output / 'EXIT.json', result)
        if result['group_cleanup_verified']:
            runner.require(runner.read(active) == start, 'Parallel producer active owner changed')
            budget.release(token, quiescent=lambda observed: {'owner': observed, 'active_writers': [], 'active_uploads': []})
            active.unlink()
    print(json.dumps(result), flush=True)
    return 0 if result['returncode'] == 0 and result['group_cleanup_verified'] else 2


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('mode', choices=('supervise', 'prepare'))
    for name in ('source-sha256', 'candidate-sha256', 'driver-sha256', 'predecessor-exit-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=8)
    parser.add_argument('--max-inflight-bytes', type=int, default=8 * 1024**3)
    parser.add_argument('--resume', action='store_true', help='explicitly reverify exact owned same-builder partial caches')
    parser.add_argument('--supervisor-start', type=Path)
    parser.add_argument('--supervisor-start-sha256')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args)
        return 0
    return supervise(args)


if __name__ == '__main__':
    raise SystemExit(main())
