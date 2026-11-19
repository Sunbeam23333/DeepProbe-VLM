"""Reduce superseded P0 storage without allocating a GPU or growing OSS usage.

Plans bind the existing archive manifests and a real inventory. Execution is
separate, requires the shared maintenance lock plus a fresh no-job/no-reference
guard, verifies the preserved accepted P0 final, and removes only individually
listed weight/optimizer objects. Small experiment evidence is never a target.
"""
from __future__ import annotations

import json
import argparse
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

from . import checkpoint_retention as r


ROOT = 'oss://ss-devbox-stream/jinhaopeng/Paper/DeepProbe-VLM/runs/'
RUN = re.compile(r'p0-(?:05-20261009-v[1-8]|ddp-20261008-v1|single-20261008-v2)\Z')
KEPT_RUN = 'p0-05-20261009-v8'
KEPT_FINAL = 'strict/control/final/'


def weight(path):
    value = r.relative(path)
    return (any(part == 'final' or re.fullmatch(r'checkpoint-\d+', part) for part in value.parts[:-1])
            and (value.name.endswith('.safetensors') or value.name == 'optimizer.pt'
                 or value.name.startswith('pytorch_model') and value.name.endswith('.bin')))


def _read(pin):
    return json.loads(r.check_pin(pin).read_text())


def prepare(snapshot_pin, manifest_pins, accepted_pin, scientific_plan_pin):
    """Prepare exact SHA-pinned candidates; this does not attest fresh payload bytes."""
    snapshot, accepted, scientific = _read(snapshot_pin), _read(accepted_pin), _read(scientific_plan_pin)
    r.require(accepted.get('run_id') == KEPT_RUN and accepted.get('status') == 'p0_05_engineering_gate_and_archive_verified'
              and accepted.get('job', {}).get('scheduler_state') == 'Succeeded'
              and accepted.get('job', {}).get('pod_container_exit_codes') == [0], 'Accepted P0 v8 terminal evidence required')
    r.require(isinstance(scientific.get('tasks'), list) and scientific['tasks'], 'Frozen remaining scientific task plan required')
    # The current scientific DAG must never initialize from one of these P0
    # engineering models. References to P0 *gate reports* do not use its weights.
    for task in scientific['tasks']:
        for key in ('initializer', 'historical_hint', 'checkpoint', 'checkpoint_task_id'):
            r.require(not re.search(r'p0-(?:05-|single-|ddp-)', json.dumps(task.get(key))),
                      'A remaining scientific task still references a P0 checkpoint')
    rows = snapshot.get('oss_inventory', {}).get('objects')
    r.require(isinstance(rows, list) and snapshot.get('oss_inventory', {}).get('complete') is True,
              'Complete real OSS snapshot required')
    current = {row['uri']: row for row in rows}
    r.require(len(current) == len(rows), 'P0 plan needs a current-object inventory without duplicate keys')
    manifests, targets, preserved_final = [], [], []
    for pin in manifest_pins:
        manifest = _read(pin)
        run_id = manifest.get('run_id')
        r.require(isinstance(run_id, str) and RUN.fullmatch(run_id), 'Only explicitly scoped old P0 runs may retire')
        prefix = ROOT + run_id + '/'
        r.require(manifest.get('uri', '').rstrip('/') + '/' == prefix
                  and manifest.get('file_count') == len(manifest.get('files', []))
                  and manifest.get('total_bytes') == sum(row['bytes'] for row in manifest['files']),
                  'Historical archive inventory binding differs')
        if run_id == KEPT_RUN:
            r.require(accepted.get('archive_manifest_sha256') == pin['sha256'], 'Accepted P0 v8 archive manifest differs')
        for row in manifest['files']:
            r.relative(row['path'])
            r.require(set(row) == {'path', 'bytes', 'sha256'} and type(row['bytes']) is int
                      and row['bytes'] >= 0 and r.valid_sha(row['sha256']), 'Historical file pin invalid')
            uri = prefix + row['path']
            r.require(uri in current and current[uri]['bytes'] == row['bytes'],
                      'Historical archive has missing/changed-size objects in real inventory')
            if run_id == KEPT_RUN and row['path'].startswith(KEPT_FINAL):
                preserved_final.append({**row, 'uri': uri, 'run_id': run_id})
            elif weight(row['path']):
                targets.append({**row, 'uri': uri, 'run_id': run_id})
        manifests.append({'run_id': run_id, 'prefix': prefix, 'manifest': pin,
                          'remote_manifest_uri': prefix + 'ARCHIVE_MANIFEST.json'})
    r.require(preserved_final and any(row['path'].endswith('.safetensors') for row in preserved_final),
              'Accepted P0 v8 complete control/final must be retained')
    r.require(len({row['uri'] for row in targets}) == len(targets) and targets, 'Duplicate or empty P0 retirement selection')
    return {'schema_version': 1, 'status': 'P0_RETIREMENT_PREPARED_NOT_EXECUTED',
            'snapshot': snapshot_pin, 'accepted_engineering_gate': accepted_pin,
            'scientific_plan': scientific_plan_pin, 'manifests': manifests,
            'targets': sorted(targets, key=lambda row: row['uri']),
            'preserved_accepted_final': sorted(preserved_final, key=lambda row: row['uri']),
            'reclaim_bytes': sum(row['bytes'] for row in targets),
            'preserved_final_bytes': sum(row['bytes'] for row in preserved_final),
            'remaining_oss_logical_bytes_if_executed': snapshot['oss']['bytes'] - sum(row['bytes'] for row in targets),
            'new_oss_bytes': 0, 'modelscope_backup_required_for_superseded_p0': False,
            'requires_fresh': ['no_active_project_jobs', 'no_checkpoint_references',
                               'preserved_final_full_get_sha256', 'each_target_full_get_sha256'],
            'non_weight_results_logs_manifests_preserved': True}


def execute(plan, *, budget, stores, guard, transaction_root, apply=False):
    """An intentionally separate decrement-only maintenance path above the cap."""
    expected = prepare(plan['snapshot'], [entry['manifest'] for entry in plan['manifests']],
                       plan['accepted_engineering_gate'], plan['scientific_plan'])
    r.require(plan == expected, 'Historical retirement plan changed')
    if not apply:
        return {'status': 'P0_RETIREMENT_PLAN_ONLY', 'plan_sha256': r.digest(plan),
                'reclaim_bytes': plan['reclaim_bytes'], 'objects_deleted': False}
    r.require(callable(guard), 'Fresh project-wide no-job/reference proof required')
    transaction = r.safe(transaction_root) / r.digest(plan)
    transaction.mkdir(parents=True, exist_ok=True)
    def checked_guard():
        proof = guard(plan)
        r.require(isinstance(proof, dict) and proof.get('plan_sha256') == r.digest(plan)
                  and proof.get('active_project_jobs') == [] and proof.get('active_writers') == []
                  and proof.get('checkpoint_references') == [], 'Historical P0 still has an active writer/job/reference')
        return proof
    with budget.lock():
        ledger = budget._read()
        r.require(not any(row['state'] == 'ACTIVE' for row in ledger['reservations'].values()),
                  'Maintenance waits for all active or unreconciled worker reservations')
        targeted = {row['uri'] for row in plan['targets']}
        for artifact in ledger['artifacts'].values():
            if any(row['path'] in targeted for row in artifact['locations']):
                r.require(not artifact['permanent'] and set(artifact['references']) == set(artifact['completed_references']),
                          'Shared registry retains a consumer/recovery reference to historical P0 weights')
        proof = checked_guard()
        r.write_once(transaction / 'PLAN.json', plan)
        r.write_once(transaction / ('GUARD-' + uuid.uuid4().hex + '.json'), proof)
        # Verify retained recovery bytes even when resuming a partially applied
        # retirement. Neither an old acceptance nor an ETag substitutes for GET.
        for row in plan['preserved_accepted_final']:
            r.require(r._object_digest(stores[row['run_id']], row['uri']) == {
                'bytes': row['bytes'], 'sha256': row['sha256']}, 'Preserved P0 final is missing or corrupt')
        for entry in plan['manifests']:
            r.require(r._object_digest(stores[entry['run_id']], entry['remote_manifest_uri']) == {
                'bytes': entry['manifest']['bytes'], 'sha256': entry['manifest']['sha256']},
                'Historical remote archive manifest differs from the reviewed plan')
        removed, already_absent = [], []
        for row in plan['targets']:
            store = stores[row['run_id']]
            prefix = ROOT + row['run_id'] + '/'
            listed = r._listing(store, prefix)
            intent = transaction / (r.digest(row) + '.intent.json')
            if row['uri'] not in listed:
                r.require(intent.is_file() and json.loads(intent.read_text()) == row,
                          'P0 object missing without a durable exact retirement intent')
                already_absent.append(row['uri'])
                continue
            r.require(listed[row['uri']]['bytes'] == row['bytes'], 'P0 object length changed before retirement')
            checked_guard()
            r.write_once(intent, row)
            # Adapter rechecks metadata and streams the exact SHA immediately
            # before one-object deletion. It cannot grow the bucket.
            store.delete_exact_verified(row['uri'], {**listed[row['uri']], 'sha256': row['sha256']})
            r.write_once(transaction / (r.digest(row) + '.done.json'), row)
            removed.append(row['uri'])
        result = {'status': 'SUPERSEDED_P0_WEIGHTS_RETIRED', 'plan_sha256': r.digest(plan),
                  'removed_this_invocation': removed, 'absent_after_durable_intent': already_absent,
                  'reclaimed_target_bytes': plan['reclaim_bytes'], 'preserved_accepted_final': plan['preserved_accepted_final'],
                  'non_weight_results_logs_manifests_preserved': True, 'new_oss_bytes': 0}
        final = transaction / 'DELETION.json'
        if final.exists():
            return {**json.loads(final.read_text()), 'receipt': r.file_pin(final)}
        r.write_once(final, result)
        return {**result, 'receipt': r.file_pin(final)}


def live_guard(plan, *, kubectl, kubeconfig, reference_environment=None):
    """Read a complete cluster snapshot; no scheduling or remote writes occur."""
    # Every project launch is bound to namespace default. Avoid requiring
    # unrelated cluster-wide listing privileges while checking that full scope.
    result = subprocess.run([kubectl, '--request-timeout=90s', 'get', 'pytorchjobs,pods',
        '-n', 'default', '-o', 'json'], env={**os.environ, 'KUBECONFIG': kubeconfig},
        capture_output=True, timeout=120)
    r.require(result.returncode == 0, 'Cannot verify current project workload quiescence')
    document = json.loads(result.stdout)
    r.require(isinstance(document.get('items'), list) and not document.get('metadata', {}).get('continue'),
              'Incomplete cluster listing cannot authorize maintenance')
    active = []
    for item in document['items']:
        meta, status = item.get('metadata', {}), item.get('status', {})
        if not meta.get('name', '').startswith(('jinhaopeng-dpvlm-', 'deepprobe-')):
            continue
        if item.get('kind') == 'Pod':
            inactive = status.get('phase') in {'Failed', 'Succeeded'} and not meta.get('deletionTimestamp')
        elif item.get('kind') == 'PyTorchJob':
            true = {row.get('type') for row in status.get('conditions', []) if row.get('status') == 'True'}
            inactive = (len(true & {'Succeeded', 'Failed'}) == 1 and true <= {'Created', 'Succeeded', 'Failed'}
                        and not meta.get('deletionTimestamp')
                        and all(row.get('active', 0) == 0 for row in status.get('replicaStatuses', {}).values()))
        else:
            inactive = False
        if not inactive:
            active.append({'kind': item.get('kind'), 'name': meta.get('name'), 'uid': meta.get('uid')})
    run_ids = [entry['run_id'] for entry in plan['manifests']]
    writers = []
    proc = Path('/proc')
    r.require(proc.is_dir(), 'Maintenance requires the live Linux process view')
    for process in proc.iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            command = (process / 'cmdline').read_bytes()
            if any(run_id.encode() in command for run_id in run_ids) and any(
                    name in command for name in (b'ossutil', b'torchrun', b'deepprobe_vlm.train', b'archive_payload')):
                writers.append(int(process.name))
        except (FileNotFoundError, ProcessLookupError):
            pass
        except PermissionError:
            r.require(process.stat().st_uid != os.getuid(), 'Cannot inspect a same-user potential P0 writer')
    references = []
    if reference_environment is not None:
        environment = _read(reference_environment)
        r.require(environment.get('schema_version') == 1 and isinstance(environment.get('suite_id'), str)
                  and isinstance(environment.get('assets', {}).get('backbones'), dict)
                  and environment['assets']['backbones'] and isinstance(environment.get('history'), dict),
                  'Current reference environment must contain actual model assets')
        for asset in environment['assets']['backbones'].values():
            r.require(isinstance(asset.get('language_root'), str) and isinstance(asset.get('vision_root'), str)
                      and isinstance(asset.get('files'), list) and asset['files']
                      and all(isinstance(row.get('path'), str) and r.valid_sha(row.get('sha256'))
                              and type(row.get('bytes')) is int for row in asset['files']),
                      'Machine reference assets need real byte/SHA file inventories')
        live_inputs = {'backbones': environment.get('assets', {}).get('backbones'),
                       'historical_checkpoints': [value.get('checkpoint') for value in environment.get('history', {}).values()]}
        encoded = json.dumps(live_inputs)
        references = [run_id for run_id in run_ids if run_id in encoded]
    return {'plan_sha256': r.digest(plan), 'active_project_jobs': active,
            'active_writers': writers, 'checkpoint_references': references,
            'cluster_listing_sha256': r.digest(document), 'observed_utc': r.utc(),
            'project_workload_namespace': 'default',
            'reference_environment': reference_environment,
            'scientific_plan': plan['scientific_plan']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'dry-run', 'probe', 'apply'])
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--archive-manifest', type=Path, action='append')
    parser.add_argument('--accepted', type=Path)
    parser.add_argument('--scientific-plan', type=Path)
    parser.add_argument('--storage', type=Path)
    parser.add_argument('--reference-environment', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('--ossutil', default='/home/jinhaopeng/.local/bin/ossutil')
    parser.add_argument('--oss-config', default='/home/jinhaopeng/.ossutilconfig')
    parser.add_argument('--kubectl', default='/home/jinhaopeng/.local/bin/kubectl')
    parser.add_argument('--kubeconfig', default='/mnt/vepfs/base/infra/.kube/aliconfig')
    args = parser.parse_args(argv)
    if args.mode == 'prepare':
        r.require(all((args.snapshot, args.archive_manifest, args.accepted, args.scientific_plan)), 'Preparation input pins required')
        plan = prepare(r.file_pin(args.snapshot), [r.file_pin(p) for p in args.archive_manifest],
                       r.file_pin(args.accepted), r.file_pin(args.scientific_plan))
        r.write_once(args.plan, plan)
        print(json.dumps({'status': plan['status'], 'plan': r.file_pin(args.plan), 'reclaim_bytes': plan['reclaim_bytes']}))
        return 0
    plan_pin = r.file_pin(args.plan)
    r.require(plan_pin['sha256'] == args.plan_sha256, 'Reviewed exact retirement plan SHA required')
    plan = _read(plan_pin)
    r.require(args.state_dir and args.output, 'Separate maintenance state/output paths required')
    from .checkpoint_store import OssutilStore
    stores = {entry['run_id']: OssutilStore(args.ossutil, args.oss_config, allowed_prefix=entry['prefix'],
              state_dir=args.state_dir / entry['run_id'] / 'oss') for entry in plan['manifests']}
    reference = r.file_pin(args.reference_environment) if args.reference_environment else None
    guard = lambda candidate: live_guard(candidate, kubectl=args.kubectl, kubeconfig=args.kubeconfig,
                                         reference_environment=reference)
    if args.mode == 'probe':
        # One retained 1.7 GB vision file is a useful throughput sample without
        # creating a second copy or re-reading all old checkpoint payloads.
        row = next(row for row in plan['preserved_accepted_final'] if row['path'].endswith('vision_model/model.safetensors'))
        started = time.monotonic()
        observed = r._object_digest(stores[row['run_id']], row['uri'])
        elapsed = time.monotonic() - started
        r.require(observed == {'bytes': row['bytes'], 'sha256': row['sha256']}, 'Retained P0 final probe SHA differs')
        value = {'status': 'PRESERVED_FINAL_ONE_OBJECT_FULL_GET_SHA256_VERIFIED', 'object': row,
                 'seconds': elapsed, 'bytes_per_second': row['bytes'] / elapsed,
                 'estimated_target_get_seconds_at_observed_rate': plan['reclaim_bytes'] * elapsed / row['bytes'],
                 'estimate_limitation': 'One-object network sample only; not an execution ETA.',
                 'remote_objects_deleted': False}
    else:
        r.require(args.storage, 'Exact shared storage configuration required')
        from .suite_storage import budget_for
        environment = {'storage': json.loads(args.storage.read_text()),
                       'tools': {'ossutil': args.ossutil, 'oss_config': args.oss_config}}
        budget = budget_for(environment)
        if args.mode == 'dry-run':
            observed = guard(plan)
            r.require(not observed['active_project_jobs'] and not observed['checkpoint_references'] and not observed['active_writers'],
                      'Fresh maintenance guard is not clear')
            value = {**execute(plan, budget=budget, stores=stores, guard=guard,
                              transaction_root=args.state_dir, apply=False), 'guard': observed,
                     'storage_config': r.file_pin(args.storage),
                     'reference_environment_still_required_before_apply': reference is None}
        else:
            r.require(reference is not None, 'Apply requires the exact current machine environment to inspect input references')
            value = execute(plan, budget=budget, stores=stores, guard=guard,
                            transaction_root=args.state_dir, apply=True)
    r.write_once(args.output, value)
    print(json.dumps({'status': value['status'], 'receipt': r.file_pin(args.output)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
