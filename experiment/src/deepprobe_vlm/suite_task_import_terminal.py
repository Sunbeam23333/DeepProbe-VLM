"""Verify pinned old-fleet quiescence before importing scientific task facts.

The importer must independently authenticate the request/source binding and
review before calling this helper. A pinned operator attestation is evidence
of quiescence, not an automatic authorisation to stop, requeue, or delete jobs.
"""
from . import suite_runner as runner
from .suite_actions import terminal_job
from .suite_cluster import project_workloads
from .suite_fleet import validate_execution_profile, verify_job_identity


def _terminal_pin(pin):
    runner.require(isinstance(pin, dict) and set(pin) == {'path', 'bytes', 'sha256'}
                   and type(pin['bytes']) is int and pin['bytes'] > 0,
                   'Terminal proof requires complete nonempty artifact pins')
    observed = runner.verify_pin(pin)
    runner.require(observed == pin, 'Terminal proof pin is not canonical')
    return pin


def read_pin(pin):
    """Read authenticated JSON without rewriting the original artifact."""
    value = runner.read(_terminal_pin(pin)['path'])
    runner.require(isinstance(value, dict), 'Pinned terminal metadata must be an object')
    return value


def _terminal_equal(value, expected, message):
    runner.require(isinstance(value, dict) and all(
        type(value.get(key)) is type(wanted) and value[key] == wanted
        for key, wanted in expected.items()), message)


def _terminal_scope(items, exact_jobs, original_uids, original_fleet):
    selected = {id(item) for item in project_workloads(items, ('jinhaopeng-dpvlm-', 'deepprobe-'))}
    for item in items:
        meta = item['metadata']
        if (meta['uid'] in original_uids
                or (meta['namespace'], meta['name']) in exact_jobs
                or meta.get('labels', {}).get('deepprobe/fleet') == original_fleet):
            selected.add(id(item))
    # The original UID and exact name are independent roots of the census.
    # Close ownership again so arbitrary renamed/unlabelled descendants cannot
    # disappear merely because none of their names use a project prefix.
    while True:
        uids = {item['metadata']['uid'] for item in items if id(item) in selected}
        owners = {ref.get('uid') for item in items if id(item) in selected
                  for ref in item['metadata'].get('ownerReferences', [])}
        expanded = selected | {id(item) for item in items
            if item['metadata']['uid'] in owners
            or any(ref.get('uid') in uids | owners for ref in item['metadata'].get('ownerReferences', []))}
        if expanded == selected:
            return [item for item in items if id(item) in selected]
        selected = expanded


def verify_terminal(request, source_binding, source_identity):
    """Authenticate every old worker's Job, original Pod, and writer closure.

    Deliberately fail closed if a declared worker never registered: that case
    needs a distinct reviewed non-submission proof, not an omitted registration.
    Terminal Jobs must remain in the census. Original Pods may have disappeared,
    but no replacement Pod may belong to that Job or the original fleet.
    """
    state = runner.safe(request['source_state_root'])
    runner.require(str(state) == request['source_state_root'], 'Noncanonical old state root')
    profile = validate_execution_profile(source_binding['execution_profile'], source_binding['suite_id'])
    _terminal_equal(source_identity, {
        'schema_version': 2, 'suite_id': source_binding['suite_id'],
        'execution_profile_sha256': runner.digest(profile),
        'source_identity_sha256': runner.source_identity(source_binding),
    }, 'Terminal source identity differs from its original binding')
    binding_sha = source_identity.get('binding_sha256')
    runner.require(isinstance(binding_sha, str) and runner.SHA.fullmatch(binding_sha),
                   'Terminal source binding SHA is missing')
    proof = read_pin(request['terminal_proof'])
    fields = {'schema_version', 'status', 'source_identity', 'complete_project_workload_snapshot',
              'automatic_requeue_disabled', 'workloads', 'worker_registrations',
              'process_closures', 'active_writers', 'active_uploads'}
    runner.require(set(proof) == fields, 'Incomplete or unexpected terminal proof fields')
    _terminal_equal(proof, {
        'schema_version': 1, 'status': 'SUITE_FLEET_TERMINAL_QUIESCENT',
        'source_identity': source_identity, 'complete_project_workload_snapshot': True,
        'automatic_requeue_disabled': True, 'active_writers': [], 'active_uploads': [],
    }, 'Old fleet is not explicitly terminal, quiescent, and requeue-disabled')

    workers = {row['worker_id']: row for row in profile['workers']}
    registrations = proof['worker_registrations']
    closures = proof['process_closures']
    runner.require(isinstance(registrations, list) and len(registrations) == len(workers)
                   and isinstance(closures, list) and len(closures) == len(workers),
                   'Every declared worker needs exactly one registration and closure')
    registered, registration_paths, job_uids, pod_uids = {}, set(), set(), set()
    for pin in registrations:
        row = read_pin(pin)
        worker_id = row.get('worker_id')
        runner.require(worker_id in workers and worker_id not in registered,
                       'Duplicate or undeclared terminal worker registration')
        expected_path = state / 'workers' / worker_id / 'JOB_OBSERVED.json'
        runner.require(pin['path'] == str(expected_path), 'Registration is not the exact old-state worker file')
        runner.require(set(row) == {'worker_id', 'job_uid', 'pod_uid', 'node', 'utc'}
                       and all(isinstance(row[key], str) and row[key] for key in row),
                       'Worker registration lacks original identities')
        runner.require(row['job_uid'] not in job_uids and row['pod_uid'] not in pod_uids
                       and row['job_uid'] != row['pod_uid'], 'Duplicate worker Job or Pod UID')
        registered[worker_id] = row
        registration_paths.add(str(expected_path))
        job_uids.add(row['job_uid'])
        pod_uids.add(row['pod_uid'])
    runner.require(not job_uids.intersection(pod_uids), 'Job and Pod UID identities overlap')
    runner.require(set(registered) == set(workers), 'Terminal registration coverage is incomplete')
    actual_registration_paths = {str(path) for path in (state / 'workers').glob('*/JOB_OBSERVED.json')}
    runner.require(actual_registration_paths == registration_paths,
                   'Old-state worker registrations are absent from the terminal proof')

    closed = set()
    for pin in closures:
        row = read_pin(pin)
        worker_id = row.get('worker_id')
        runner.require(worker_id in registered and worker_id not in closed,
                       'Duplicate or undeclared worker process closure')
        reg = registered[worker_id]
        runner.require(set(row) == {'schema_version', 'status', 'worker_id', 'job_uid', 'pod_uid',
                                   'controller_exited', 'active_writers', 'active_uploads', 'evidence'},
                       'Incomplete process closure fields')
        _terminal_equal(row, {
            'schema_version': 1, 'status': 'SUITE_WORKER_PROCESSES_QUIESCENT',
            'worker_id': worker_id, 'job_uid': reg['job_uid'], 'pod_uid': reg['pod_uid'],
            'controller_exited': True, 'active_writers': [], 'active_uploads': [],
        }, 'Worker processes are not closed for their exact original Job and Pod')
        runner.require(isinstance(row['evidence'], list) and row['evidence'],
                       'Process closure needs real nonempty process evidence')
        local_paths = set()
        for evidence in row['evidence']:
            _terminal_pin(evidence)  # Evidence may be a raw process listing, not JSON.
            runner.require(evidence['path'] not in local_paths, 'Duplicate process evidence pin')
            local_paths.add(evidence['path'])
        closed.add(worker_id)
    runner.require(closed == set(workers), 'Worker process closure coverage is incomplete')

    census = read_pin(proof['workloads'])
    header = census.get('metadata', {})
    runner.require(isinstance(census.get('items'), list) and isinstance(header, dict)
                   and header.get('continue', '') in {'', None}
                   and (header.get('remainingItemCount') is None
                        or type(header['remainingItemCount']) is int and header['remainingItemCount'] == 0),
                   'A complete non-paginated workload census is required')
    by_uid, identities = {}, set()
    for item in census['items']:
        runner.require(isinstance(item, dict) and isinstance(item.get('metadata'), dict),
                       'Malformed workload census item')
        meta = item['metadata']
        identity = (item.get('kind'), meta.get('namespace'), meta.get('name'))
        uid = meta.get('uid')
        runner.require(all(isinstance(value, str) and value for value in identity)
                       and isinstance(uid, str) and uid and uid not in by_uid and identity not in identities,
                       'Missing or duplicate workload UID or exact identity')
        runner.require(isinstance(meta.get('labels', {}), dict)
                       and isinstance(meta.get('annotations', {}), dict)
                       and isinstance(meta.get('ownerReferences', []), list)
                       and all(isinstance(ref, dict) for ref in meta.get('ownerReferences', [])),
                       'Malformed workload identity metadata')
        by_uid[uid] = item
        identities.add(identity)
    original_fleet = runner.digest(profile)[:32]
    expected_jobs = {(row['namespace'], row['job_name']): worker_id for worker_id, row in workers.items()}
    scoped = _terminal_scope(census['items'], expected_jobs, job_uids | pod_uids, original_fleet)
    runner.require(all(item['kind'] in {'PyTorchJob', 'Pod'} for item in scoped),
                   'Unexpected project workload kind in terminal proof')
    seen_jobs, seen_pods = set(), set()
    for item in scoped:
        meta = item['metadata']
        if item['kind'] != 'PyTorchJob':
            continue
        worker_id = expected_jobs.get((meta['namespace'], meta['name']))
        if worker_id is None:
            runner.require(meta.get('uid') not in job_uids, 'Original Job UID has another exact identity')
            runner.require(meta.get('labels', {}).get('deepprobe/fleet') != original_fleet,
                           'An undeclared Job belongs to the original fleet')
            continue
        verify_job_identity(item, profile, worker_id, binding_sha)
        runner.require(meta['uid'] == registered[worker_id]['job_uid'], 'Original exact Job was replaced or recreated')
        runner.require(terminal_job(item), 'Original exact Job is not terminal')
        seen_jobs.add(worker_id)
    runner.require(seen_jobs == set(workers), 'An original exact Job is absent from the complete census')

    jobs_by_uid = {reg['job_uid']: worker_id for worker_id, reg in registered.items()}
    pods_by_uid = {reg['pod_uid']: worker_id for worker_id, reg in registered.items()}
    for item in scoped:
        if item['kind'] != 'Pod':
            continue
        meta = item['metadata']
        references = meta.get('ownerReferences', [])
        runner.require(isinstance(references, list) and all(isinstance(ref, dict) for ref in references),
                       'Malformed Pod owner references')
        owners = {jobs_by_uid[ref['uid']] for ref in references if ref.get('uid') in jobs_by_uid}
        named_owners = {expected_jobs[(meta['namespace'], ref['name'])] for ref in references
                        if ref.get('kind') == 'PyTorchJob'
                        and (meta['namespace'], ref.get('name')) in expected_jobs}
        registered_worker = pods_by_uid.get(meta['uid'])
        original_label = meta.get('labels', {}).get('deepprobe/fleet') == original_fleet
        if not owners and not named_owners and registered_worker is None and not original_label:
            continue
        runner.require(len(owners) == 1 and named_owners == owners,
                       'Original fleet Pod has a missing, changed, or ambiguous Job owner')
        worker_id = next(iter(owners))
        reg, declared = registered[worker_id], workers[worker_id]
        runner.require(registered_worker == worker_id and meta['uid'] == reg['pod_uid'],
                       'A replacement Pod exists for the original fleet worker')
        matching = [ref for ref in references if ref.get('uid') == reg['job_uid']]
        runner.require(len(matching) == 1 and matching[0].get('kind') == 'PyTorchJob'
                       and matching[0].get('name') == declared['job_name']
                       and meta['namespace'] == declared['namespace'], 'Original Pod owner identity differs')
        runner.require(item.get('status', {}).get('phase') in {'Succeeded', 'Failed'}
                       and not meta.get('deletionTimestamp')
                       and item.get('spec', {}).get('restartPolicy') == 'Never'
                       and item['spec'].get('nodeName') == reg['node'],
                       'Original Pod is not terminal on its registered node with restart Never')
        seen_pods.add(worker_id)
    # Absent originals are accepted only by the complete, pinned census above;
    # UID membership also catches any original Pod misclassified outside scope.
    for worker_id, reg in registered.items():
        runner.require(reg['pod_uid'] not in by_uid or worker_id in seen_pods,
                       'Original Pod UID exists outside the authenticated fleet Pod set')
    return {'status': 'SUITE_FLEET_TERMINAL_QUIESCENT', 'source_identity': source_identity,
            'terminal_proof': request['terminal_proof'], 'worker_ids': sorted(workers),
            'job_uids': sorted(job_uids), 'pod_uids': sorted(pod_uids),
            'absent_original_pod_workers': sorted(set(workers) - seen_pods)}
