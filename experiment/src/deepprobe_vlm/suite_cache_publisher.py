"""CPU-only publication of independently verified progressive cache receipts."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import suite_progressive as progressive, suite_runner as runner
from .suite_storage import budget_for


PUBLICATION_RESERVE_BYTES = 64 * 1024 * 1024


def record_failure(environment, declaration, error, evidence=()):
    """Durably stop waiting consumers; explicit recovery must preserve this marker."""
    path = runner.safe(declaration['ready_path'])
    failure = path.with_name(path.stem + '.FAILED.json')
    if failure.exists():
        return runner.artifact(failure)
    budget = budget_for(environment)
    owner = {'suite_id': environment['suite_id'], 'operation': 'cache-failure-publication',
             'contract_sha256': declaration['contract_sha256']}
    with budget.asset_lock(owner):
        if failure.exists():
            return runner.artifact(failure)
        token = budget.reserve(owner, PUBLICATION_RESERVE_BYTES, 0)
        try:
            failure.parent.mkdir(parents=True, exist_ok=True)
            runner.write_once(failure, {'status': 'SUITE_BENCHMARK_PREPARATION_FAILED',
                'utc': runner.utc(), 'contract': declaration['contract'],
                'contract_sha256': declaration['contract_sha256'], 'error_type': type(error).__name__,
                'error': str(error) if type(error) in {ValueError, KeyError, TypeError} else type(error).__name__,
                'evidence': list(evidence), 'automatic_retry': False})
            return runner.artifact(failure)
        finally:
            budget.release(token, quiescent=lambda observed: {
                'owner': observed, 'active_writers': [], 'active_uploads': []})


def producer_outcome(start_pin, declarations):
    if start_pin is None:
        return None
    runner.verify_pin(start_pin)
    start = runner.read(start_pin['path'])
    root = runner.safe(start_pin['path']).parent
    runner.require(root / 'START.json' == Path(start_pin['path'])
                   and start.get('output_root') == str(root)
                   and start.get('status') == 'CPU_CACHE_PREPARATION_STARTED', 'Exact CPU producer start required')
    runner.verify_pin(start['source_manifest']); runner.verify_pin(start['candidate']); runner.verify_pin(start['driver'])
    candidate = runner.read(start['candidate']['path'])
    for declaration in declarations.values():
        contract = declaration['contract']
        if 'legacy_reuse' in contract['request']:
            continue  # The read-only reuse publisher owns this shape, not the decoder.
        runner.require(contract['producer']['source_manifest'] == start['source_manifest']
                       and contract['producer']['source_root'] == candidate['environment']['experiment_root']
                       and contract['request'] in candidate['benchmark_preparations'],
                       'Observed CPU producer does not own every declared cache')
    exit_path = root / 'EXIT.json'
    if not exit_path.exists():
        return None
    pin = runner.artifact(exit_path)
    value = runner.read(exit_path)
    runner.require(value.get('status') == 'CPU_CACHE_PREPARATION_EXITED'
                   and type(value.get('group_cleanup_verified')) is bool,
                   'Invalid CPU producer exit evidence')
    runner.verify_pin(pin)
    return {'successful': value.get('returncode') == 0 and value['group_cleanup_verified']
            and 'error' not in value, 'evidence': [start_pin, pin]}


def publish_prepared(environment, declaration):
    """Consume one immutable producer descriptor; pending does not reserve space."""
    request = declaration['contract']['request']
    prepared_path = runner.safe(declaration['ready_path']).with_name('PREPARED_ASSET.json')
    existing = progressive.resolve_ready(environment, request['benchmark'], request['frames'])
    if existing is not None:
        return existing
    if 'legacy_reuse' in request:
        from .suite_legacy_cache import prepare_asset
        budget = budget_for(environment)
        owner = {'suite_id': environment['suite_id'], 'operation': 'legacy-f8-cache-publication',
                 'contract_sha256': declaration['contract_sha256']}
        with budget.asset_lock(owner):
            existing = progressive.resolve_ready(environment, request['benchmark'], request['frames'])
            if existing is not None:
                return existing
            token = budget.reserve(owner, 2 * request['metadata_reserve_bytes'] + PUBLICATION_RESERVE_BYTES, 0)
            try:
                asset = prepare_asset(environment, request)
                return progressive.publish_ready(environment, declaration, asset)
            finally:
                budget.release(token, quiescent=lambda observed: {
                    'owner': observed, 'active_writers': [], 'active_uploads': []})
    try:
        prepared_path.stat()
    except FileNotFoundError:
        return None
    pin = runner.artifact(prepared_path)
    prepared = runner.read(prepared_path)
    producer = declaration['contract']['producer']
    runner.require(prepared.get('status') == 'CPU_CACHE_ASSET_PREPARED_NOT_FLEET_ADMITTED'
                   and prepared.get('request') == request
                   and prepared.get('request_sha256') == runner.digest(request)
                   and prepared.get('producer_source', {}).get('root') == producer['source_root']
                   and prepared['producer_source'].get('manifest') == producer['source_manifest'],
                   'Prepared cache descriptor differs from its frozen declaration')
    runner.verify_pin(prepared['candidate'])
    candidate = runner.read(prepared['candidate']['path'])
    runner.require(request in candidate.get('benchmark_preparations', [])
                   and candidate.get('environment', {}).get('experiment_root') == producer['source_root'],
                   'Prepared cache candidate does not bind this producer and request')
    runner.verify_pin(pin)
    budget = budget_for(environment)
    owner = {'suite_id': environment['suite_id'], 'operation': 'cache-readiness-publication',
             'contract_sha256': declaration['contract_sha256']}
    # Synchronous remote reads have no child writers or upload workers. A crash
    # retains this reservation for explicit process/ledger reconciliation.
    with budget.asset_lock(owner):
        existing = progressive.resolve_ready(environment, request['benchmark'], request['frames'])
        if existing is not None:
            runner.require(existing['asset'] == prepared['asset'], 'Existing READY asset differs from producer')
            return existing
        token = budget.reserve(owner, PUBLICATION_RESERVE_BYTES, 0)
        try:
            result = progressive.publish_ready(environment, declaration, prepared['asset'])
            runner.verify_pin(pin)
            return result
        finally:
            budget.release(token, quiescent=lambda observed: {
                'owner': observed, 'active_writers': [], 'active_uploads': []})


def publish_available(environment, plan, *, producer_start=None):
    declarations = progressive.validate_declarations(environment, plan)
    # Reuse is independent of the decoder's lifecycle. In particular, an F4
    # decoder failure must not mark accepted F8 inputs failed before they can
    # publish their own complete remote verification.
    for declaration in declarations.values():
        if 'legacy_reuse' in declaration.get('contract', {}).get('request', {}):
            try:
                publish_prepared(environment, declaration)
            except Exception as exc:
                record_failure(environment, declaration, exc)
                raise
    try:
        outcome = producer_outcome(producer_start, declarations)
    except Exception as exc:
        fail_pending(environment, declarations, exc)
        raise
    if outcome is not None:
        # A terminal producer can no longer publish missing descriptors. Mark
        # every missing shape before raising: DAG-ready F8 consumers must not
        # wait forever merely because the first declared failure was F4.
        missing = [declaration for declaration in declarations.values()
                   if 'legacy_reuse' not in declaration['contract']['request']
                   and not runner.safe(declaration['ready_path']).exists()
                   and not runner.safe(declaration['ready_path']).with_name('PREPARED_ASSET.json').exists()]
        if missing:
            error = ValueError('CPU cache producer exited ' + ('without every descriptor' if outcome['successful']
                               else 'unsuccessfully') + '; exact attempt evidence requires reconciliation')
            fail_pending(environment, declarations, error, outcome['evidence'])
            raise error
    ready, pending = {}, []
    for identity, declaration in declarations.items():
        try:
            value = publish_prepared(environment, declaration)
            if value is None and outcome is not None:
                raise ValueError('CPU cache producer exited ' + ('without every descriptor' if outcome['successful']
                                 else 'unsuccessfully') + '; exact attempt evidence requires reconciliation')
        except Exception as exc:
            fail_pending(environment, declarations, exc, outcome['evidence'] if outcome else [])
            raise
        if value is None:
            pending.append(identity)
        else:
            ready[identity] = {key: item for key, item in value.items() if key != 'asset'}
    return {'status': 'SUITE_CACHE_READINESS_COMPLETE' if not pending else 'SUITE_CACHE_READINESS_PENDING',
            'utc': runner.utc(), 'suite_id': environment['suite_id'], 'ready': ready, 'pending': pending,
            'checkpoint_payload_gets': 0, 'gpu_submitted': False}


def fail_pending(environment, declarations, error, evidence=()):
    # This publisher is stopping, so even prepared-but-unpublished shapes need
    # durable terminal evidence. Existing accepted READY receipts are retained.
    return [record_failure(environment, declaration, error, evidence)
            for declaration in declarations.values()
            if not runner.safe(declaration['ready_path']).exists()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--environment-sha256', required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--producer-start', type=Path)
    parser.add_argument('--producer-start-sha256')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=30)
    parser.add_argument('--timeout-seconds', type=int, default=7 * 24 * 3600)
    args = parser.parse_args(argv)
    runner.require(1 <= args.poll_seconds <= 60 and 0 < args.timeout_seconds <= 7 * 24 * 3600,
                   'Bounded polling and timeout required')
    pins = {key: runner.artifact(getattr(args, key)) for key in ('environment', 'plan')}
    for name, pin in pins.items():
        runner.require(pin['sha256'] == getattr(args, name + '_sha256'), 'Frozen ' + name + ' SHA differs')
    environment, plan = runner.read(args.environment), runner.read(args.plan)
    runner.require(bool(args.producer_start) == bool(args.producer_start_sha256), 'CPU producer START path and SHA must be paired')
    producer_start = runner.artifact(args.producer_start) if args.producer_start else None
    if producer_start:
        runner.require(producer_start['sha256'] == args.producer_start_sha256, 'CPU producer START SHA differs')
    from .suite_bindings import validate_environment
    from .experiment_suite import validate_plan
    validate_plan(plan, config_root=environment['experiment_root'])
    validate_environment(environment, plan)
    end = time.monotonic() + args.timeout_seconds
    previous = None
    try:
        while True:
            for pin in pins.values():
                runner.verify_pin(pin)
            result = publish_available(environment, plan, producer_start=producer_start)
            identity = runner.digest({key: value for key, value in result.items() if key != 'utc'})
            if identity != previous:
                print(json.dumps(result, sort_keys=True), flush=True)
                previous = identity
            if not result['pending'] or not args.watch:
                return 0
            runner.require(time.monotonic() + args.poll_seconds < end, 'Cache readiness publisher timed out')
            time.sleep(args.poll_seconds)
    except Exception as exc:
        fail_pending(environment, environment['benchmark_preparations'], exc)
        if type(exc) not in {ValueError, KeyError, TypeError}:
            raise ValueError('Cache publisher failed: ' + type(exc).__name__) from None
        raise


if __name__ == '__main__':
    raise SystemExit(main())
