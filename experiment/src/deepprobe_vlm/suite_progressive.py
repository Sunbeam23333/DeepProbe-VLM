"""Immutable CPU cache readiness for a frozen, progressively admitted fleet.

Only publication performs remote cache verification. Resolution checks local
evidence and never fetches checkpoint weights or constructs benchmark media.
An absent receipt means pending; invalid evidence is always an error.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path

from . import suite_runner as runner


BENCHMARKS = {'videomme899': 'videomme', 'tempcompass_mc': 'tempcompass'}
FRAMES = (4, 8, 16)
STATUS = 'SUITE_BENCHMARK_CACHE_READY'


def key(benchmark, frames):
    runner.require(benchmark in BENCHMARKS and type(frames) is int and frames in FRAMES,
                   'Unsupported progressive benchmark shape')
    return benchmark + ':f' + str(frames)


def _context(environment):
    from .suite_fleet import validate_execution_profile
    from .suite_storage import validate_config
    runner.require(isinstance(environment, dict) and environment.get('execution_profile', {}).get('schema_version') == 2
                   and environment.get('storage', {}).get('schema_version') == 2,
                   'Progressive caches require a schema-v2 fleet and storage')
    validate_execution_profile(environment['execution_profile'], suite_id=environment['suite_id'])
    validate_config(environment['storage'])


def _accounted(environment, value):
    runner.require(isinstance(value, str), 'Accounted cache paths must be strings')
    path = runner.safe(value)
    runner.require(str(path) == value and any(path != runner.safe(root) and path.is_relative_to(runner.safe(root))
                   for root in environment['storage']['roots']), 'Progressive cache path escapes accounted roots')
    return path


def _metadata_pin(environment, pin):
    from .suite_media import MAX_METADATA_BYTES
    runner.require(isinstance(pin, dict) and set(pin) == {'path', 'bytes', 'sha256'}
                   and type(pin['bytes']) is int and 0 < pin['bytes'] <= MAX_METADATA_BYTES,
                   'Complete bounded metadata pin required')
    _accounted(environment, pin['path'])
    runner.verify_pin(pin)


def _request(environment, request):
    from .suite_media import _key, PROJECT_PREFIX
    fields = {'benchmark', 'frames', 'source_manifest', 'accepted_decode_receipt', 'output_dir', 'oss_prefix',
              'media_root', 'max_png_bytes', 'metadata_reserve_bytes'}
    runner.require(isinstance(request, dict) and fields <= set(request) <= fields | {'archive_audit', 'legacy_reuse'},
                   'Invalid progressive benchmark request')
    identity = key(request['benchmark'], request['frames'])
    for name in ('source_manifest', 'accepted_decode_receipt', 'archive_audit'):
        if name in request:
            _metadata_pin(environment, request[name])
    if 'legacy_reuse' in request:
        from .suite_legacy_cache import input_inventory
        runner.require(isinstance(request['legacy_reuse'], dict)
                       and set(request['legacy_reuse']) == {'inventory', 'acceptance'}, 'Invalid legacy reuse evidence')
        for pin in request['legacy_reuse'].values():
            _metadata_pin(environment, pin)
        input_inventory(request)
    runner.require(type(request['max_png_bytes']) is int and request['max_png_bytes'] > 0
                   and type(request['metadata_reserve_bytes']) is int
                   and request['metadata_reserve_bytes'] >= 64 * 1024 * 1024, 'Invalid progressive cache byte reserves')
    output = _accounted(environment, request['output_dir'])
    _accounted(environment, str(output.with_name(output.name + '-inputs')))
    _accounted(environment, request['media_root'])
    prefix = _key(request['oss_prefix'])
    runner.require(prefix.endswith('/') and prefix != PROJECT_PREFIX
                   and all(part and part not in {'.', '..'} for part in prefix[:-1].split('/'))
                   and any(('oss://ss-devbox-stream/' + prefix).startswith(root)
                           for root in environment['storage']['oss_prefixes']),
                   'Cache prefix must be an exact accounted stream prefix')
    return identity


def _producer(environment, producer, *, live=False):
    runner.require(isinstance(producer, dict) and set(producer) == {'source_root', 'source_manifest'},
                   'Exact cache producer source required')
    root = _accounted(environment, producer['source_root'])
    pin = producer['source_manifest']
    _metadata_pin(environment, pin)
    runner.require(not Path(pin['path']).is_relative_to(root), 'Producer source manifest must be outside its source tree')
    document = runner.read(pin['path'])
    runner.require(isinstance(document, dict) and document.get('schema_version') == 1
                   and document.get('purpose') == 'DEEPPROBE_FROZEN_P1_SOURCE'
                   and isinstance(document.get('files'), dict) and document['files'],
                   'Invalid frozen cache producer source manifest')
    if live:
        from .formal_integrity import verify_frozen_source_manifest
        return verify_frozen_source_manifest(root, Path(pin['path']), pin['sha256'])
    return None


def _declaration(environment, declaration):
    _context(environment)
    runner.require(isinstance(declaration, dict) and set(declaration) == {'contract', 'contract_sha256', 'ready_path'},
                   'Invalid progressive cache declaration')
    contract = declaration['contract']
    runner.require(isinstance(contract, dict) and set(contract) == {'schema_version', 'suite_id', 'request', 'producer'}
                   and type(contract['schema_version']) is int and contract['schema_version'] == 1
                   and contract['suite_id'] == environment['suite_id']
                   and declaration['contract_sha256'] == runner.digest(contract), 'Progressive cache contract differs')
    identity = _request(environment, contract['request'])
    _producer(environment, contract['producer'])
    if 'legacy_reuse' in contract['request']:
        runner.require(contract['producer']['source_root'] == environment['experiment_root'],
                       'Legacy reuse must bind the current publisher source, not the old decoder')
    ready = _accounted(environment, declaration['ready_path'])
    request = contract['request']
    output = Path(request['output_dir'])
    owned_dirs = (output, output.with_name(output.name + '-inputs'), Path(request['media_root']),
                  Path(contract['producer']['source_root']))
    runner.require(not any(ready == root or ready.is_relative_to(root) or root.is_relative_to(ready)
                           for root in owned_dirs), 'READY path overlaps cache or producer directories')
    runner.require(ready.suffix == '.json', 'READY must use an exact JSON artifact path')
    runner.require(all(ready != Path(request[name]['path']) for name in ('source_manifest', 'accepted_decode_receipt', 'archive_audit')
                       if name in request) and ready != Path(contract['producer']['source_manifest']['path']),
                   'READY path overlaps pinned input evidence')
    return identity


def make_declaration(environment, request, producer_source_manifest, ready_path, producer_source_root=None):
    declaration = {'contract': {'schema_version': 1, 'suite_id': environment['suite_id'], 'request': deepcopy(request),
        'producer': {'source_root': str(producer_source_root or environment['experiment_root']),
                     'source_manifest': deepcopy(producer_source_manifest)}}, 'ready_path': str(ready_path)}
    declaration['contract_sha256'] = runner.digest(declaration['contract'])
    _declaration(environment, declaration)
    _producer(environment, declaration['contract']['producer'], live=True)
    return declaration


def validate_declarations(environment, plan):
    """Validate all six frozen shapes without waiting for any READY receipt."""
    _context(environment)
    declarations = environment.get('benchmark_preparations')
    expected = {key(benchmark, frames) for benchmark in BENCHMARKS for frames in FRAMES}
    runner.require(isinstance(declarations, dict) and set(declarations) == expected,
                   'Progressive admission requires all six benchmark declarations')
    shapes = {key(task['benchmark'], task['frames']) for task in plan['tasks'] if task['kind'] == 'evaluate'}
    runner.require(shapes == expected, 'Scientific plan must retain all six benchmark shapes')
    paths, ready_paths, prefixes = [], [], []
    for identity, declaration in declarations.items():
        runner.require(_declaration(environment, declaration) == identity, 'Benchmark declaration key differs')
        request = declaration['contract']['request']
        output = Path(request['output_dir'])
        paths.extend((output, output.with_name(output.name + '-inputs'), Path(request['media_root'])))
        ready_paths.append(Path(declaration['ready_path']))
        prefixes.append(request['oss_prefix'])
    runner.require(all(a != b and not a.is_relative_to(b) and not b.is_relative_to(a)
                       for i, a in enumerate(paths) for b in paths[i + 1:]), 'Progressive cache directories overlap')
    runner.require(len(set(ready_paths)) == len(ready_paths) and all(
        not (ready == root or ready.is_relative_to(root) or root.is_relative_to(ready))
        for ready in ready_paths for root in paths), 'Progressive READY paths overlap')
    runner.require(all(not a.startswith(b) and not b.startswith(a)
                       for i, a in enumerate(prefixes) for b in prefixes[i + 1:]), 'Progressive cache prefixes overlap')
    return declarations


def validate_admission(preflight, environment, plan):
    """Accept only a complete, explicit progressive CPU admission snapshot.

    A deferred cache may have become ready since this snapshot was produced.
    Caches claimed ready by the snapshot must still have the exact same pin.
    Launcher separately verifies plan/environment/binding hashes and freshness.
    """
    declarations = validate_declarations(environment, plan)
    runner.require(isinstance(preflight, dict) and preflight.get('status') == 'SUITE_CPU_PROGRESSIVE_ADMITTED'
                   and preflight.get('benchmark_preparation_mode') == 'external_progressive'
                   and preflight.get('failures') == [], 'Explicit progressive CPU admission required')
    checks = preflight.get('checks')
    runner.require(isinstance(checks, list) and checks, 'Progressive CPU checks are missing')
    expected = {'benchmark/' + benchmark + '/' + str(frames): key(benchmark, frames)
                for benchmark in BENCHMARKS for frames in FRAMES}
    seen, deferred = set(), []
    for row in checks:
        runner.require(isinstance(row, dict) and isinstance(row.get('check'), str)
                       and row.get('status') in {'passed', 'deferred'}, 'Invalid progressive CPU check')
        name = row['check']
        if name not in expected:
            runner.require(not name.startswith('benchmark/') and row['status'] == 'passed',
                           'Only declared benchmark checks may be deferred')
            continue
        runner.require(name not in seen, 'Duplicate progressive benchmark check')
        seen.add(name)
        declaration = declarations[expected[name]]
        request = declaration['contract']['request']
        result = row.get('result')
        if row['status'] == 'deferred':
            wanted = {'status': 'DEFERRED_BENCHMARK', 'benchmark': request['benchmark'], 'frames': request['frames'],
                      'contract_sha256': declaration['contract_sha256'], 'ready_path': declaration['ready_path']}
            runner.require(result == wanted, 'Deferred benchmark differs from frozen declaration')
            deferred.append(wanted)
        else:
            runner.require(isinstance(result, dict) and set(result) == {'ready_pin', 'contract_sha256', 'cache_binding_sha256'},
                           'Ready benchmark admission must pin its exact receipt')
            resolved = resolve_ready(environment, request['benchmark'], request['frames'], expected_pin=result['ready_pin'])
            runner.require(resolved is not None and result == {name: resolved[name] for name in result},
                           'Admitted READY receipt differs from current evidence')
    runner.require(seen == set(expected), 'Progressive CPU admission omits benchmark shapes')
    supplied = preflight.get('deferred_benchmarks')
    runner.require(isinstance(supplied, list) and all(isinstance(item, dict) for item in supplied)
                   and sorted(map(runner.canonical, supplied)) == sorted(map(runner.canonical, deferred)),
                   'Deferred benchmark inventory differs from CPU checks')
    return True


def _asset(environment, declaration, asset):
    from . import suite_media as media
    request = declaration['contract']['request']
    runner.require(isinstance(asset, dict) and set(asset) == {'cache', 'media_root', 'max_bytes'}
                   and asset['media_root'] == request['media_root'] and type(asset['max_bytes']) is int
                   and asset['max_bytes'] == request['max_png_bytes'], 'READY asset differs from frozen request')
    if 'legacy_reuse' in request:
        from .suite_legacy_cache import validate_asset
        for name in ('source_manifest', 'source_binding', 'manifest', 'receipt'):
            _metadata_pin(environment, asset['cache'][name])
        return validate_asset(request, asset)
    cache = asset['cache']
    runner.require(isinstance(cache, dict) and set(cache) == {'kind', 'source_manifest', 'source_binding', 'manifest',
                   'receipt', 'complete', 'plan', 'prefix'} and cache['kind'] == 'suite'
                   and cache['prefix'] == request['oss_prefix'] and cache['source_manifest'] == request['source_manifest'],
                   'READY cache prefix/source differs from frozen request')
    output = Path(request['output_dir'])
    inputs = output.with_name(output.name + '-inputs')
    names = {'source_binding': inputs / 'SOURCE_BINDING.json', 'manifest': output / 'manifest.jsonl',
             'receipt': output / 'FRAME_CACHE_RECEIPT.json', 'complete': output / 'CACHE_COMPLETE.json',
             'plan': output / 'BUILD_PLAN.json'}
    for name, path in names.items():
        _metadata_pin(environment, cache[name])
        runner.require(cache[name]['path'] == str(path), 'READY cache metadata path differs from frozen request')
    rows, binding, inventory, frames, _, _ = media._bound_cache_inventory(cache)
    runner.require(frames == request['frames'] and binding['dataset'] == BENCHMARKS[request['benchmark']],
                   'READY benchmark or frame shape differs')
    runner.require(binding == runner.read(cache['source_binding']['path']), 'Cache source binding evidence differs')
    evidence = binding.get('normalization_evidence')
    runner.require(isinstance(evidence, dict) and set(evidence) == {'original_decode_receipt', 'archive_audit', 'archive_bound_receipt'}
                   and evidence['original_decode_receipt'] == request['accepted_decode_receipt']
                   and evidence['archive_audit'] == request.get('archive_audit'), 'Cache decode/archive evidence differs from request')
    normalized = evidence['archive_bound_receipt']
    _metadata_pin(environment, normalized)
    normalized_value = runner.read(normalized['path'])
    runner.require(normalized['path'] == str(inputs / 'ARCHIVE_BOUND_SOURCE_RECEIPT.json')
                   and binding.get('accepted_original_decode_receipt_sha256') == media.digest(media.encode(normalized_value)),
                   'Cache normalized decode receipt differs')
    catalog_path = inputs / 'ARCHIVE_CATALOG.json'
    _metadata_pin(environment, runner.artifact(catalog_path))
    catalog = runner.read(catalog_path)
    runner.require(isinstance(catalog, list)
                   and binding.get('archive_catalog_sha256') == media.digest(media.encode(catalog)),
                   'Cache original archive catalog differs')
    from .suite_workflow import bind_legacy_archive_identities
    runner.require(normalized_value == bind_legacy_archive_identities(
        runner.read(request['accepted_decode_receipt']['path']), catalog),
        'Cache normalized decode evidence differs from frozen input')
    runner.require(inventory and 0 < sum(item['bytes'] for item in inventory) <= asset['max_bytes'],
                   'READY cache exceeds frozen hydration reserve')
    return rows, inventory


def _remote_result(asset, declaration, rows, inventory):
    result = {'qa': len(rows), 'frames': declaration['contract']['request']['frames'],
        'acceptance': asset['cache'].get('complete', asset['cache']['receipt']),
        'cache_binding_sha256': runner.digest(asset['cache']), 'frame_objects': len(inventory),
        'frame_bytes': sum(item['bytes'] for item in inventory), 'all_png_full_get_and_rgb_verified': True,
        'local_media_created': False, 'local_hydration_still_required_before_evaluation': True,
        'formal_evaluation_completed': False, 'network_or_hydration_requested': True}
    if 'legacy_reuse' in declaration['contract']['request']:
        result.update(cache_preparation='accepted_legacy_f8_reuse', original_videos_decoded=False,
                      new_pngs_uploaded=False, exact_legacy_prefix_verified=True)
    return result


def _registered(environment, declaration):
    identity = _declaration(environment, declaration)
    runner.require(environment.get('benchmark_preparations', {}).get(identity) == declaration,
                   'READY declaration is not frozen in this environment')
    return identity


def resolve_ready(environment, benchmark, frames, expected_pin=None):
    """Read the exact declared READY; missing claimed evidence is never pending."""
    identity = key(benchmark, frames)
    declaration = environment.get('benchmark_preparations', {}).get(identity)
    runner.require(declaration is not None, 'Benchmark has no frozen progressive declaration')
    _registered(environment, declaration)
    path = runner.safe(declaration['ready_path'])
    failure = path.with_name(path.stem + '.FAILED.json')
    if failure.exists():
        failed_pin = runner.artifact(failure)
        failed = runner.read(failure)
        runner.require(failed.get('status') == 'SUITE_BENCHMARK_PREPARATION_FAILED'
                       and failed.get('contract_sha256') == declaration['contract_sha256']
                       and failed.get('contract') == declaration['contract'], 'Malformed cache producer failure marker')
        for item in failed.get('evidence', []):
            runner.verify_pin(item)
        runner.verify_pin(failed_pin)
        raise ValueError('CPU cache preparation failed; preserve evidence and reconcile before recovery: ' + str(failure))
    if expected_pin is not None:
        runner.require(isinstance(expected_pin, dict) and expected_pin.get('path') == str(path), 'Claimed READY path differs')
        runner.verify_pin(expected_pin)
    try:
        path.stat()
    except FileNotFoundError:
        return None
    pin = runner.artifact(path)
    _metadata_pin(environment, pin)
    value = runner.read(path)
    fields = {'schema_version', 'status', 'generated_utc', 'contract', 'contract_sha256', 'asset',
              'cache_binding_sha256', 'inventory_sha256', 'remote_verification', 'producer_source_verification'}
    runner.require(isinstance(value, dict) and set(value) == fields and type(value['schema_version']) is int
                   and value['schema_version'] == 1 and value['status'] == STATUS
                   and value['contract'] == declaration['contract']
                   and value['contract_sha256'] == declaration['contract_sha256'], 'Published READY contract differs')
    generated = datetime.fromisoformat(value['generated_utc'].replace('Z', '+00:00'))
    runner.require(generated.tzinfo is not None, 'READY timestamp must include a timezone')
    producer = _producer(environment, declaration['contract']['producer'], live=True)
    runner.require(value['producer_source_verification'] == producer, 'READY producer source verification differs')
    rows, inventory = _asset(environment, declaration, value['asset'])
    runner.require(value['cache_binding_sha256'] == runner.digest(value['asset']['cache'])
                   and value['inventory_sha256'] == runner.digest(inventory)
                   and value['remote_verification'] == _remote_result(value['asset'], declaration, rows, inventory),
                   'READY full GET/RGB verification or inventory differs')
    runner.verify_pin(pin)
    if expected_pin is not None:
        runner.verify_pin(expected_pin)
    return {'asset': value['asset'], 'ready_pin': pin, 'contract_sha256': declaration['contract_sha256'],
            'cache_binding_sha256': value['cache_binding_sha256']}


def publish_ready(environment, declaration, asset):
    """Publish only after actual complete remote verification; reuse is read-only."""
    from .suite_preflight import _remote_benchmark_gate
    _registered(environment, declaration)
    request = declaration['contract']['request']
    existing = resolve_ready(environment, request['benchmark'], request['frames'])
    if existing is not None:
        runner.require(existing['asset'] == asset, 'An immutable READY already binds a different asset')
        return existing
    producer = _producer(environment, declaration['contract']['producer'], live=True)
    rows, inventory = _asset(environment, declaration, asset)
    checked = _remote_benchmark_gate(asset, request['benchmark'], request['frames'], environment)
    runner.require(checked == _remote_result(asset, declaration, rows, inventory), 'Remote cache acceptance is incomplete')
    final_rows, final_inventory = _asset(environment, declaration, asset)
    runner.require(rows == final_rows and inventory == final_inventory
                   and producer == _producer(environment, declaration['contract']['producer'], live=True),
                   'Cache evidence or producer changed during acceptance')
    value = {'schema_version': 1, 'status': STATUS, 'generated_utc': runner.utc(),
             'contract': deepcopy(declaration['contract']), 'contract_sha256': declaration['contract_sha256'],
             'asset': deepcopy(asset), 'cache_binding_sha256': runner.digest(asset['cache']),
             'inventory_sha256': runner.digest(inventory), 'remote_verification': checked,
             'producer_source_verification': producer}
    path = runner.safe(declaration['ready_path'])
    path.parent.mkdir(parents=True, exist_ok=True)
    runner.write_once(path, value)
    return resolve_ready(environment, request['benchmark'], request['frames'])


def task_readiness(environment, plan, task):
    """Return claim evidence, or None only when a new evaluation awaits cache.

    Accepted historical evaluation results need no new frames. Authenticate
    their full existing receipt before bypassing the cache gate; a hint or a
    malformed import never becomes permission to run or silently wait.
    """
    if task['kind'] != 'evaluate' or 'benchmark_preparations' not in environment:
        return {}
    history = environment.get('history', {})
    if task['id'] in history:
        from .suite_bindings import verify_history_binding
        proof = verify_history_binding(plan, task, history[task['id']])
        runner.require(proof.get('eligible') is True and proof.get('mode') == 'accepted',
                       'Only verified accepted evaluation history bypasses the cache gate')
        return {}
    ready = resolve_ready(environment, task['benchmark'], task['frames'])
    return None if ready is None else {name: value for name, value in ready.items() if name != 'asset'}
