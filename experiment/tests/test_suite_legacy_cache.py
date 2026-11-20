"""Legacy F8 reuse preserves accepted frames without any original-video decode."""
from copy import deepcopy
from contextlib import nullcontext
import io
import json
from pathlib import Path
from unittest.mock import Mock
import zipfile

import pytest
from PIL import Image

from deepprobe_vlm import suite_legacy_cache as legacy, suite_media as media
from deepprobe_vlm import suite_progressive as progressive, suite_runner as runner
from deepprobe_vlm import suite_cache_publisher as publisher
from test_suite_media import FakeStore
from test_suite_progressive import declared, write


@pytest.fixture(params=['tempcompass_mc', 'videomme899'])
def old_cache(declared, monkeypatch, request):
    benchmark = request.param
    is_video = benchmark == 'videomme899'
    clips, qa = (899, 2697) if is_video else (410, 1580)
    env, plan, producer = declared
    env['experiment_root'] = str(producer)
    previous = env['benchmark_preparations'][benchmark + ':f8']
    request = deepcopy(previous['contract']['request'])
    request.pop('archive_audit')
    root = Path(env['storage']['roots'][0])
    prefix = media.PROJECT_PREFIX + 'accepted-' + benchmark + '/uniform8/'
    request['oss_prefix'] = prefix
    archive_key = media.PROJECT_PREFIX + 'originals/tempcompass.zip'
    image = Image.new('RGB', (2, 2), (17, 23, 41))
    stream = io.BytesIO(); image.save(stream, format='PNG'); png = stream.getvalue()
    z = io.BytesIO()
    with zipfile.ZipFile(z, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for index in range(clips):
            bundle.writestr(f'videos/clip_{index:04d}.mp4', f'never-decode-original-{index}'.encode())
    archive_raw = z.getvalue()
    entries, rows, objects = [], [], {archive_key: archive_raw}
    for index in range(clips):
        ident = f'clip_{index:04d}'
        frames = []
        for slot in range(8):
            relative = f'frames/{ident}/{slot:02d}.png'
            objects[prefix + relative] = png
            frames.append({'relative_path': relative, 'source_frame_index': slot, 'size_bytes': len(png),
                           'sha256': media.digest(png), 'rgb_sha256': media.digest(image.tobytes()), 'width': 2, 'height': 2})
        original = f'never-decode-original-{index}'.encode()
        entries.append({'processed_media_identity': 'tempcompass:' + ident, 'video_id': ident,
            'underlying_source_video_ids': [f'underlying:{index}'], 'archive_key': archive_key,
            'archive_etag': 'accepted-object-version', 'archive_sha256': media.digest(archive_raw),
            'member_path': f'videos/{ident}.mp4', 'original_size_bytes': len(original),
            'original_sha256': media.digest(original), 'decode_complete': True, 'decoded_frame_count': 8,
            'full_decode_passes': 2, 'selected_indices': list(range(8)), 'frames': frames,
            'image_paths': [f['relative_path'] for f in frames]})
        if is_video:
            entries[-1].pop('processed_media_identity')
            entries[-1].update(source_video_id=f'youtube:{ident}', underlying_source_video_ids=[f'youtube:{ident}'])
    groups = sorted(set(range(1, 901)) - {490})
    for index in range(qa):
        entry = entries[index // 3 if is_video else index % clips]
        row = {'sample_id': f'q:{index}', 'source_video_id': f'component:{index % clips}',
            'processed_media_identity': entry.get('processed_media_identity'), 'video_id': entry['video_id'],
            'underlying_source_video_ids': entry['underlying_source_video_ids'], 'dataset': 'tempcompass',
            'split': 'test', 'question': 'Which direction?', 'choices': ['left', 'right'], 'answer': 'A',
            'original_answer': 'A. left', 'image_paths': entry['image_paths']}
        if is_video:
            group = groups[index // 3]
            row.pop('processed_media_identity')
            row.update(sample_id=f'videomme:{group:03d}-{index % 3 + 1}', dataset='videomme',
                       source_video_id=entry['source_video_id'], choices=['left', 'right', 'up', 'down'],
                       duration='short' if group <= 300 else 'medium' if group <= 600 else 'long')
        rows.append(row)
    manifest = root / 'legacy-inputs/manifest.jsonl'; manifest.parent.mkdir()
    manifest.write_bytes(b''.join(runner.canonical(row) + b'\n' for row in rows))
    receipt = {'complete': True, 'failures': [], 'num_frames': 8, 'sampling': media.SAMPLING, 'entries': entries}
    request['source_manifest'] = runner.artifact(manifest)
    request['accepted_decode_receipt'] = write(root / 'legacy-inputs/receipt.json', receipt)
    objects[prefix + 'manifest.jsonl'] = manifest.read_bytes()
    objects[prefix + 'FRAME_CACHE_RECEIPT.json'] = Path(request['accepted_decode_receipt']['path']).read_bytes()
    terminal = {'tempcompass_frame_cache_ready': True, 'archive_full_oss_get_sha256_revalidated': True,
        'qa_count': 1580, 'clip_count': 410, 'png_count': 3280,
        'objects_before_terminal': {k: {'bytes': len(v), 'sha256': media.digest(v)}
                                   for k, v in objects.items() if k.startswith(prefix)}}
    terminal_pin = write(root / 'legacy-inputs/TERMINAL.json', terminal)
    objects[prefix + 'TERMINAL.json'] = Path(terminal_pin['path']).read_bytes()
    accepted = {'status': 'TEMP_COMPASS_REMOTE_AUDIT_PASSED',
        'all_original_candidates_resolved_by_two_reviews': True, 'review': {'passed': True},
        'exact_oss_prefix': prefix, 'media_inventory': receipt,
        'local_and_oss_objects': sum(k.startswith(prefix) for k in objects),
        'local_and_oss_bytes': sum(len(v) for k, v in objects.items() if k.startswith(prefix))}
    if is_video:
        # VideoMME stores only PNGs under its exact frame prefix. Completion
        # and acceptance metadata are separately pinned outside that prefix.
        for name in ('manifest.jsonl', 'FRAME_CACHE_RECEIPT.json', 'TERMINAL.json'):
            objects.pop(prefix + name)
        frames = [{'relative_path': f['relative_path'], 'bucket_key': prefix + f['relative_path'],
                   'size_bytes': f['size_bytes'], 'sha256': f['sha256']} for entry in entries for f in entry['frames']]
        complete = {'status': 'VIDEOMME_BUCKET_MEDIA_COMPLETE', 'complete': True,
            'full_oss_get_sha256_verified': True, 'exact_frame_key_set_verified': True, 'frame_prefix': prefix,
            'source_count': 899, 'frame_count': 7192, 'frame_cache_receipt': request['accepted_decode_receipt'],
            'frames': frames, 'cache_bytes': sum(f['size_bytes'] for f in frames)}
        terminal_pin = write(root / 'legacy-inputs/VIDEOMME_BUCKET_MEDIA_COMPLETE.json', complete)
        accepted = {'status': 'accepted_media_only_not_formal_evaluation',
            'all_local_png_sha256_rgb_and_dimensions_rehashed': True, 'two_exact_oss_key_size_etag_lists_stable': True,
            'control_sha256': {'VIDEOMME_BUCKET_MEDIA_COMPLETE.json': terminal_pin['sha256'],
                              'FRAME_CACHE_RECEIPT.json': request['accepted_decode_receipt']['sha256']},
            'head_evidence': [{'key': f['bucket_key'], 'bytes': f['size_bytes'], 'sha256': f['sha256']} for f in frames],
            'png_bytes': complete['cache_bytes']}
    request['legacy_reuse'] = {'inventory': terminal_pin, 'acceptance': write(root / 'legacy-inputs/ACCEPTED.json', accepted)}
    request['max_png_bytes'] = 1_000_000
    archive_receipt = {'binding': {'bucket': media.BUCKET, 'key': archive_key, 'size_bytes': len(archive_raw),
        'sha256': media.digest(archive_raw)}, 'source_stream_sha256': media.digest(archive_raw),
        'source_bytes': len(archive_raw), 'oss_readback': {'bytes': len(archive_raw), 'sha256': media.digest(archive_raw),
        'verification': 'complete_OSS_GET_SHA256_and_size'}}
    objects[archive_key + '.CACHE_VERIFIED.json'] = media.encode(archive_receipt)
    store = FakeStore(objects, archive_key)
    store.list_metadata = lambda prefix: {k: {'size_bytes': len(v), 'etag': media.digest(v)}
                                         for k, v in store.objects.items() if k.startswith(prefix)}
    monkeypatch.setattr(media, 'OSSStore', lambda *a, **kw: store)
    monkeypatch.setattr(media, 'decode_selected', Mock(side_effect=AssertionError('Legacy reuse must not decode')))
    monkeypatch.setattr(media, 'extract_member', Mock(side_effect=AssertionError('Legacy reuse must not read video members')))
    declaration = progressive.make_declaration(env, request, previous['contract']['producer']['source_manifest'],
        previous['ready_path'], producer_source_root=producer)
    env['benchmark_preparations'][benchmark + ':f8'] = declaration
    return env, plan, declaration, store


def test_reuse_publishes_truthful_ready_and_hydrates_exact_old_pngs(old_cache, monkeypatch):
    env, plan, declaration, store = old_cache
    request = declaration['contract']['request']
    assert len(progressive.validate_declarations(env, plan)) == 6
    asset = legacy.prepare_asset(env, request, store=store)
    assert not store.puts and not Path(asset['media_root']).exists()
    assert asset['cache']['kind'] == 'legacy' and 'complete' not in asset['cache'] and 'plan' not in asset['cache']
    ready = progressive.publish_ready(env, declaration, asset)
    proof = runner.read(ready['ready_pin']['path'])['remote_verification']
    assert proof['cache_preparation'] == 'accepted_legacy_f8_reuse'
    assert proof['original_videos_decoded'] is False and proof['new_pngs_uploaded'] is False
    expected = 7192 if request['benchmark'] == 'videomme899' else 3280
    assert proof['exact_legacy_prefix_verified'] and proof['frame_objects'] == expected
    network_reads = len(store.gets)
    assert progressive.resolve_ready(env, request['benchmark'], 8) == ready
    assert len(store.gets) == network_reads
    monkeypatch.setattr('platform.system', lambda: 'Linux')
    Path(asset['media_root']).parent.mkdir(parents=True, exist_ok=True)
    fragment = media.hydrate_frame_cache(asset['cache'], media_root=asset['media_root'], max_bytes=asset['max_bytes'], store=store)
    assert len(fragment['media_files']) == expected
    assert not store.puts


@pytest.mark.parametrize('damage', ['extra', 'missing', 'corrupt_png', 'metadata'])
def test_legacy_remote_damage_never_publishes_ready(old_cache, damage):
    env, _, declaration, store = old_cache
    request = declaration['contract']['request']; asset = legacy.prepare_asset(env, request, store=store)
    png = next(k for k in store.objects if k.endswith('.png'))
    if damage == 'extra': store.objects[request['oss_prefix'] + 'unexpected.png'] = b'extra'
    elif damage == 'missing': del store.objects[png]
    elif damage == 'corrupt_png': store.objects[png] = b'!' * len(store.objects[png])
    else:
        key = request['oss_prefix'] + 'manifest.jsonl' if request['benchmark'] == 'tempcompass_mc' else png
        store.objects[key] = b'!' * len(store.objects[key])
    with pytest.raises(ValueError, match='inventory|SHA'):
        progressive.publish_ready(env, declaration, asset)
    assert not Path(declaration['ready_path']).exists()


@pytest.mark.parametrize('frames', [4, 16])
def test_legacy_reuse_cannot_claim_other_frame_counts(old_cache, frames):
    env, _, declaration, _ = old_cache
    request = deepcopy(declaration['contract']['request']); request['frames'] = frames
    with pytest.raises(ValueError, match='F8'):
        progressive.make_declaration(env, request, declaration['contract']['producer']['source_manifest'],
                                    declaration['ready_path'], declaration['contract']['producer']['source_root'])


def test_reuse_rejects_changed_original_frame_indices(old_cache):
    env, _, declaration, store = old_cache
    asset = legacy.prepare_asset(env, declaration['contract']['request'], store=store)
    pin = asset['cache']['receipt']; value = runner.read(pin['path'])
    value['entries'][0]['selected_indices'][1] = 7
    asset['cache']['receipt'] = write(Path(pin['path']), value)
    with pytest.raises(ValueError, match='receipt|sampling'):
        progressive.publish_ready(env, declaration, asset)


def test_legacy_publisher_needs_no_decoder_descriptor_and_reuses_ready(old_cache, monkeypatch):
    env, _, declaration, store = old_cache
    budget = Mock()
    budget.asset_lock.side_effect = lambda owner: nullcontext()
    budget.reserve.return_value = 'owned-legacy-token'
    monkeypatch.setattr(publisher, 'budget_for', lambda _: budget)
    monkeypatch.setattr(progressive, 'validate_declarations', lambda *args: {'legacy': declaration})
    monkeypatch.setattr(publisher, 'producer_outcome', lambda *args: {'successful': True, 'evidence': []})
    result = publisher.publish_available(env, {}, producer_start={'synthetic': True})
    assert result['status'] == 'SUITE_CACHE_READINESS_COMPLETE'
    assert not Path(declaration['ready_path']).with_name('PREPARED_ASSET.json').exists()
    assert not store.puts
    budget.reserve.assert_called_once()
    assert budget.reserve.call_args.args[2] == 0
    assert budget.release.call_args.args == ('owned-legacy-token',)
    again = publisher.publish_available(env, {}, producer_start={'synthetic': True})
    assert again['ready'] == result['ready'] and again['status'] == result['status']
    assert budget.reserve.call_count == 1


def test_decoder_exit_only_owns_nonlegacy_declarations(old_cache):
    env, _, declaration, _ = old_cache
    decoder = env['benchmark_preparations']['tempcompass_mc:f4']
    root = Path(env['storage']['roots'][0]) / 'cpu-attempt'; root.mkdir()
    producer = decoder['contract']['producer']
    candidate = write(root / 'CANDIDATE.json', {'environment': {'experiment_root': producer['source_root']},
        'benchmark_preparations': [decoder['contract']['request']]})
    driver = write(root / 'DRIVER.json', {'synthetic': True})
    start = write(root / 'START.json', {'status': 'CPU_CACHE_PREPARATION_STARTED', 'output_root': str(root),
        'source_manifest': producer['source_manifest'], 'candidate': candidate, 'driver': driver})
    write(root / 'EXIT.json', {'status': 'CPU_CACHE_PREPARATION_EXITED', 'returncode': 0, 'group_cleanup_verified': True})
    assert publisher.producer_outcome(start, {'f4': decoder, 'f8': declaration})['successful'] is True


def test_decoder_failure_does_not_mark_independent_legacy_reuse_failed(old_cache, monkeypatch):
    env, _, declaration, _ = old_cache
    budget = Mock(); budget.asset_lock.side_effect = lambda owner: nullcontext()
    budget.reserve.return_value = 'owned-token'
    monkeypatch.setattr(publisher, 'budget_for', lambda _: budget)
    decoder = env['benchmark_preparations']['tempcompass_mc:f4']
    monkeypatch.setattr(progressive, 'validate_declarations', lambda *args: {'f4': decoder, 'f8': declaration})
    monkeypatch.setattr(publisher, 'producer_outcome', lambda *args: {'successful': False, 'evidence': []})
    with pytest.raises(ValueError, match='unsuccessfully'):
        publisher.publish_available(env, {}, producer_start={'synthetic': True})
    assert Path(declaration['ready_path']).exists()
    assert not Path(declaration['ready_path']).with_name('READY.FAILED.json').exists()
    assert Path(decoder['ready_path']).with_name('READY.FAILED.json').exists()


def test_prefix_etag_drift_during_readback_prevents_ready(old_cache):
    env, _, declaration, store = old_cache
    asset = legacy.prepare_asset(env, declaration['contract']['request'], store=store)
    original = store.list_metadata
    calls = []
    def listing(prefix):
        value = original(prefix); calls.append(prefix)
        if len(calls) == 2:
            value[next(iter(value))]['etag'] = 'changed-with-same-size'
        return value
    store.list_metadata = listing
    with pytest.raises(ValueError, match='changed during full remote readback'):
        progressive.publish_ready(env, declaration, asset)
    assert not Path(declaration['ready_path']).exists()


@pytest.mark.private_evidence
@pytest.mark.parametrize('benchmark', ['videomme899', 'tempcompass_mc'])
def test_actual_historical_mapping_evidence_binds_current_f8_inputs(benchmark):
    evidence = Path(__file__).resolve().parents[1] / 'docs/operations/evidence'
    request = next(r for r in runner.read(evidence / 'suite-machine-readonly-20261120-v5/payload/CANDIDATE-1-normal.json')['benchmark_preparations']
                   if r['benchmark'] == benchmark and r['frames'] == 8)
    # Read-only local evidence check, not fresh remote acceptance.
    if benchmark == 'videomme899':
        base = evidence / 'benchmark-bucket-cache-20261013/videomme-oss-completion-a1'
        rec = base / 'completion-20261013T152044Z/FRAME_CACHE_RECEIPT.json'
        inv = base / 'completion-20261013T152044Z/VIDEOMME_BUCKET_MEDIA_COMPLETE.json'
        accepted = base / 'independent-final-20261013/REMOTE_AUDIT.json'
        request['oss_prefix'] = runner.read(inv)['frame_prefix']
        name, expected = 'videomme899_manifest.jsonl', 7192
    else:
        base = evidence / 'tempcompass-evaluation-preparation-20261021/runtime-v1/data-acceptance'
        rec, inv = base / 'evidence/inputs/FRAME_CACHE_RECEIPT.json', base / 'evidence/inputs/TERMINAL.json'
        accepted = base / 'accepted-data/REMOTE_AUDIT.json'
        request['oss_prefix'] = runner.read(accepted)['exact_oss_prefix']
        name, expected = 'tempcompass_mc_manifest.jsonl', 3280
    assert runner.artifact(rec)['sha256'] == request['accepted_decode_receipt']['sha256']
    request['accepted_decode_receipt'] = runner.artifact(rec)
    request['source_manifest'] = runner.artifact(evidence / 'suite-history-source-readonly-20261119-v1' / name)
    request['legacy_reuse'] = {'inventory': runner.artifact(inv), 'acceptance': runner.artifact(accepted)}
    mappings, _ = legacy.input_inventory(request)
    assert len(mappings) == expected
