"""Explicit reuse of the two accepted original-video F8 bucket caches.

This adapter imports old evidence, never decodes video, uploads PNGs or invents
new-cache completion records. Publication and hydration still GET/hash every
PNG and verify RGB pixels against the original receipt.
"""
from __future__ import annotations

from pathlib import Path

from . import suite_media as media, suite_runner as runner


def input_inventory(request):
    """Bind the exact old prefix to its independent acceptance and source pins."""
    runner.require(request['frames'] == 8 and request['benchmark'] in {'videomme899', 'tempcompass_mc'},
                   'Legacy reuse supports only the two accepted F8 protocols')
    spec = request.get('legacy_reuse')
    runner.require(isinstance(spec, dict) and set(spec) == {'inventory', 'acceptance'},
                   'Legacy reuse requires exact inventory and independent acceptance pins')
    for pin in [*spec.values(), request['source_manifest'], request['accepted_decode_receipt']]:
        runner.verify_pin(pin)
    old = runner.read(spec['inventory']['path'])
    accepted = runner.read(spec['acceptance']['path'])
    receipt = runner.read(request['accepted_decode_receipt']['path'])
    runner.require(receipt.get('complete') is True and receipt.get('failures', []) == []
                   and receipt.get('num_frames') == 8 and receipt.get('sampling') == media.SAMPLING,
                   'Legacy original decode receipt is incomplete or uses another protocol')
    prefix = request['oss_prefix']
    if request['benchmark'] == 'videomme899':
        runner.require(old.get('status') == 'VIDEOMME_BUCKET_MEDIA_COMPLETE'
                       and old.get('complete') is True and old.get('full_oss_get_sha256_verified') is True
                       and old.get('exact_frame_key_set_verified') is True and old.get('frame_prefix') == prefix
                       and old.get('source_count') == 899 and old.get('frame_count') == 7192
                       and old.get('frame_cache_receipt', {}).get('sha256') == request['accepted_decode_receipt']['sha256'],
                       'VideoMME legacy inventory does not bind the accepted F8 receipt/prefix')
        runner.require(accepted.get('status') == 'accepted_media_only_not_formal_evaluation'
                       and accepted.get('all_local_png_sha256_rgb_and_dimensions_rehashed') is True
                       and accepted.get('two_exact_oss_key_size_etag_lists_stable') is True
                       and accepted.get('control_sha256', {}).get('VIDEOMME_BUCKET_MEDIA_COMPLETE.json') == spec['inventory']['sha256']
                       and accepted.get('control_sha256', {}).get('FRAME_CACHE_RECEIPT.json') == request['accepted_decode_receipt']['sha256'],
                       'VideoMME independent media acceptance differs')
        mappings = [{'relative_path': x['relative_path'], 'key': x['bucket_key'],
                     'size_bytes': x['size_bytes'], 'sha256': x['sha256']} for x in old['frames']]
        inventory = {x['key']: {'bytes': x['size_bytes'], 'sha256': x['sha256']} for x in mappings}
        checked = {x['key']: {'bytes': x['bytes'], 'sha256': x['sha256']}
                   for x in accepted['head_evidence'] if x['key'].startswith(prefix)}
        runner.require(len(inventory) == len(mappings) == 7192 and checked == inventory
                       and sum(x['bytes'] for x in inventory.values()) == old['cache_bytes'] == accepted['png_bytes'],
                       'VideoMME independent exact frame inventory differs')
    else:
        runner.require(old.get('tempcompass_frame_cache_ready') is True
                       and old.get('archive_full_oss_get_sha256_revalidated') is True
                       and old.get('qa_count') == 1580 and old.get('clip_count') == 410 and old.get('png_count') == 3280,
                       'TempCompass old cache is incomplete')
        runner.require(accepted.get('status') == 'TEMP_COMPASS_REMOTE_AUDIT_PASSED'
                       and accepted.get('all_original_candidates_resolved_by_two_reviews') is True
                       and accepted.get('review', {}).get('passed') is True
                       and accepted.get('exact_oss_prefix') == prefix and accepted.get('media_inventory') == receipt,
                       'TempCompass independent media acceptance differs')
        inventory = dict(old['objects_before_terminal'])
        runner.require(prefix + 'TERMINAL.json' not in inventory, 'Legacy terminal inventory is recursive')
        inventory[prefix + 'TERMINAL.json'] = {k: spec['inventory'][k] for k in ('bytes', 'sha256')}
        for key, pin in [('manifest.jsonl', request['source_manifest']),
                         ('FRAME_CACHE_RECEIPT.json', request['accepted_decode_receipt'])]:
            runner.require(inventory.get(prefix + key) == {k: pin[k] for k in ('bytes', 'sha256')},
                           'TempCompass source manifest or original receipt changed')
        mappings = [{'relative_path': frame['relative_path'], 'key': prefix + frame['relative_path'],
                     'size_bytes': frame['size_bytes'], 'sha256': frame['sha256']}
                    for entry in receipt['entries'] for frame in entry['frames']]
        runner.require(len(mappings) == 3280 and len(inventory) == accepted['local_and_oss_objects']
                       and sum(x['bytes'] for x in inventory.values()) == accepted['local_and_oss_bytes'],
                       'TempCompass accepted exact inventory differs')
    runner.require(all(media._key(key).startswith(prefix) and type(item['bytes']) is int and item['bytes'] > 0
                       and media._sha(item['sha256']) for key, item in inventory.items()),
                   'Invalid or escaping legacy object inventory')
    runner.require(len({x['relative_path'] for x in mappings}) == len(mappings)
                   and len({x['key'] for x in mappings}) == len(mappings), 'Duplicate legacy frame object')
    for item in mappings:
        runner.require(inventory.get(item['key']) == {'bytes': item['size_bytes'], 'sha256': item['sha256']},
                       'Legacy frame mapping differs from exact inventory')
    return mappings, inventory


def prepare_asset(environment, request, *, store=None):
    """Write only small, derived evidence; all original cache bytes stay put."""
    from .suite_workflow import source_catalog, bind_legacy_archive_identities, save
    mappings, _ = input_inventory(request)
    output = runner.safe(request['output_dir'])
    inputs = output.with_name(output.name + '-inputs')
    original = runner.read(request['accepted_decode_receipt']['path'])
    store = store or media.OSSStore(environment['tools']['oss_config'], allowed_prefix=request['oss_prefix'])
    audit = runner.read(request['archive_audit']['path']) if request.get('archive_audit') else None
    if request.get('archive_audit'):
        runner.verify_pin(request['archive_audit'])
    catalog = source_catalog(original, store, audit)
    normalized = bind_legacy_archive_identities(original, catalog)
    normalized_pin = save(inputs / 'ARCHIVE_BOUND_SOURCE_RECEIPT.json', normalized)
    source = media.create_source_binding(Path(request['source_manifest']['path']).read_bytes(),
        Path(normalized_pin['path']).read_bytes(), catalog, store=store)
    runner.require(normalized_pin['sha256'] == source['accepted_original_decode_receipt_sha256'],
                   'Normalized legacy receipt serialization differs')
    source['normalization_evidence'] = {'original_decode_receipt': request['accepted_decode_receipt'],
        'archive_audit': request.get('archive_audit'), 'archive_bound_receipt': normalized_pin}
    source_pin = save(inputs / 'SOURCE_BINDING.json', source)
    save(inputs / 'ARCHIVE_CATALOG.json', catalog)
    cache = {'kind': 'legacy', 'source_manifest': request['source_manifest'],
        'source_binding': source_pin, 'manifest': request['source_manifest'], 'receipt': normalized_pin,
        'prefix': request['oss_prefix'], 'frame_objects': mappings, 'legacy_request': request}
    asset = {'cache': cache, 'media_root': request['media_root'], 'max_bytes': request['max_png_bytes']}
    validate_asset(request, asset)
    save(output / 'LEGACY_REUSE_ASSET.json', asset)
    allowed = {output: {'LEGACY_REUSE_ASSET.json'}, inputs: {
        'ARCHIVE_BOUND_SOURCE_RECEIPT.json', 'SOURCE_BINDING.json', 'ARCHIVE_CATALOG.json'}}
    written = []
    for directory, names in allowed.items():
        runner.require({p.name for p in directory.iterdir()} == names, 'Unexpected legacy reuse metadata')
        for name in names:
            path = runner.safe(directory / name)
            runner.require(path.is_file() and path.stat().st_nlink == 1, 'Invalid legacy reuse metadata file')
            written.append(path.stat().st_size)
    runner.require(sum(written) <= 2 * request['metadata_reserve_bytes'], 'Legacy metadata exceeds reserved home bytes')
    return asset


def validate_asset(request, asset):
    """Prove normalized evidence only adds archive identity, never frame facts."""
    mappings, _ = input_inventory(request)
    cache = asset['cache']
    runner.require(set(cache) == {'kind', 'source_manifest', 'source_binding', 'manifest', 'receipt',
                   'prefix', 'frame_objects', 'legacy_request'} and cache['kind'] == 'legacy'
                   and cache['legacy_request'] == request and cache['frame_objects'] == mappings
                   and cache['source_manifest'] == cache['manifest'] == request['source_manifest']
                   and cache['prefix'] == request['oss_prefix'], 'Legacy asset differs from frozen reuse request')
    rows, binding, inventory, frames, _, _ = media._bound_cache_inventory(cache)
    runner.require(frames == 8 and binding['dataset'] == ('videomme' if request['benchmark'] == 'videomme899' else 'tempcompass'),
                   'Legacy cache protocol differs')
    inputs = runner.safe(request['output_dir']).with_name(Path(request['output_dir']).name + '-inputs')
    evidence = binding.get('normalization_evidence', {})
    runner.require(evidence == {'original_decode_receipt': request['accepted_decode_receipt'],
        'archive_audit': request.get('archive_audit'), 'archive_bound_receipt': cache['receipt']}
        and cache['receipt']['path'] == str(inputs / 'ARCHIVE_BOUND_SOURCE_RECEIPT.json')
        and cache['source_binding']['path'] == str(inputs / 'SOURCE_BINDING.json'),
        'Legacy normalization provenance differs')
    catalog = runner.read(inputs / 'ARCHIVE_CATALOG.json')
    from .suite_workflow import bind_legacy_archive_identities
    runner.require(binding['archive_catalog_sha256'] == media.digest(media.encode(catalog))
                   and runner.read(cache['receipt']['path']) == bind_legacy_archive_identities(
                       runner.read(request['accepted_decode_receipt']['path']), catalog),
                   'Legacy normalization changed the accepted original decode evidence')
    runner.require(0 < sum(x['bytes'] for x in inventory) <= asset['max_bytes'], 'Legacy cache exceeds hydration reserve')
    return rows, inventory


def verify_remote_inventory(cache, store, *, metadata=True):
    """Recheck the exact legacy prefix; PNG bodies are checked by the caller."""
    mappings, inventory = input_inventory(cache['legacy_request'])
    runner.require(cache['frame_objects'] == mappings, 'Legacy mapping changed')
    prefix = cache['prefix']
    snapshot = store.list_metadata(prefix)
    runner.require(isinstance(snapshot, dict) and set(snapshot) == set(inventory),
                   'Exact legacy remote cache inventory changed')
    runner.require(all(snapshot[key]['size_bytes'] == item['bytes'] for key, item in inventory.items()),
                   'Legacy remote object size changed')
    if metadata:
        frames = {row['key'] for row in mappings}
        for key, item in inventory.items():
            if key not in frames:
                runner.require(item['bytes'] <= media.MAX_METADATA_BYTES, 'Legacy metadata exceeds bounded GET size')
                raw = store.get(key, item['bytes'])
                runner.require(raw is not None and len(raw) == item['bytes'] and media.digest(raw) == item['sha256'],
                               'Legacy remote metadata full GET SHA differs')
    return snapshot
