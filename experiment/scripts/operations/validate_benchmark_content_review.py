#!/usr/bin/env python3
"""Validate a separate, SHA-bound review without rewriting scanner evidence.

The original sampled-frame policy, candidate list, and passed:false remain
immutable. Separate primary and independent visual reviews distinguish graphic
hash collisions from shared publisher packaging; neither implies pixel-level
disjointness. Confirmed story/action overlap or unresolved overlap blocks.
This is evidence validation, not an automatic visual classifier.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

SHA256 = re.compile(r'[0-9a-f]{64}')
RESOLUTIONS = {'distinct_content_hash_collision', 'shared_publisher_boilerplate_only'}
RESOLVED = 'resolved_no_substantive_overlap_in_reviewed_samples'


def require(value, message):
    if not value:
        raise ValueError(message)


def safe(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and
            not any(p.is_symlink() for p in (path, *path.parents)), 'Unsafe evidence path')
    return path


def sha(path):
    digest = hashlib.sha256()
    with safe(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(safe(path).read_text())


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def timestamp(value):
    require(isinstance(value, str), 'Review timestamp must be a string')
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(result.tzinfo is not None, 'Review time must have timezone')
    return result


def relative(root, value):
    require(isinstance(value, str) and value, 'Missing relative evidence path')
    path = Path(value)
    require(not path.is_absolute() and '..' not in path.parts and str(path) == value,
            'Unsafe relative evidence path')
    return safe(root / path)


def verify(path, expected):
    require(isinstance(expected, str) and SHA256.fullmatch(expected), 'Invalid evidence SHA256')
    path = safe(path)
    require(path.is_file() and sha(path) == expected, 'Evidence SHA mismatch: ' + str(path))
    return path


def audit_artifact(audit_path, item):
    path = Path(item['path'])
    if not path.is_absolute():
        path = relative(audit_path.parent, item['path'])
    return verify(path, item['sha256'])


def unique_rows(rows, key, label):
    require(isinstance(rows, list), label + ' must be a list')
    result = {}
    for row in rows:
        require(isinstance(row, dict) and key in row and row[key] not in result,
                label + ' has invalid or duplicate identity')
        result[row[key]] = row
    return result


def pair_evidence_names(index, candidate):
    train = candidate['training_source_video_id'].removeprefix('youtube:')
    benchmark = candidate['benchmark_source_video_id'].removeprefix('youtube:')
    require(re.fullmatch(r'[A-Za-z0-9_-]{11}', train) and
            re.fullmatch(r'[A-Za-z0-9_-]{11}', benchmark), 'Unsupported candidate source identity')
    matches = candidate['frame_matches']
    require(isinstance(matches, list) and matches, 'Candidate must retain all frame matches')
    return {f'context/train_{train}.jpg', f'context/benchmark_{benchmark}.jpg'} | {
        f'matches/pair_{index + 1:02d}_match_{slot + 1:02d}.jpg' for slot in range(len(matches))}


def validate_review(audit_path, audit, review_path, review_sha256, *,
                    independent_path=None, independent_sha256=None,
                    benchmark_inventory=None):
    """Return bound proof; throw on a changed/missing/unresolved review input."""
    audit_path, review_path = safe(audit_path), verify(review_path, review_sha256)
    review = read(review_path)
    require(isinstance(review, dict) and review.get('schema_version') == 1 and
            review.get('kind') == 'separate_visual_candidate_adjudication', 'Unsupported content-review schema')
    require(review.get('original_audit', {}).get('sha256') == sha(audit_path) and
            review['original_audit'].get('passed') is False, 'Review refers to a different original audit')
    for key in ('policy', 'binding', 'media_binding'):
        require(review.get(key) == audit.get(key) and key in audit, 'Review changed original ' + key)
    require(audit.get('passed') is False and audit.get('exact_duplicate_pairs') == [],
            'Review may only resolve original perceptual candidates, never exact duplicates')
    require(audit.get('scan_complete') is True and audit.get('errors') == [] and
            audit.get('uninformative_sources') == [], 'Review cannot repair incomplete scans')
    require(review.get('original_formal_ready_sha256') == audit.get('original_formal_ready_sha256') and
            isinstance(audit.get('original_formal_ready_sha256'), str) and
            SHA256.fullmatch(audit['original_formal_ready_sha256']), 'Original formal gate changed')
    reviewer = review.get('reviewer')
    require(isinstance(reviewer, str) and reviewer.strip(), 'Named primary reviewer required')
    stamp = timestamp(review['reviewed_utc'])
    for key in ('candidate_review_complete', 'all_original_candidates_resolved',
                'all_original_candidate_rows_preserved', 'original_audit_not_overwritten',
                'independent_review_required_before_use'):
        require(review.get(key) is True, 'Primary review incomplete: ' + key)
    for key in ('hash_thresholds_changed', 'qa_or_sources_removed', 'frames_removed_or_resampled',
                'training_or_benchmark_media_modified', 'gold_answers_or_model_predictions_used',
                'formal_ready_created', 'gpu_submission_performed'):
        require(review.get(key) is False, 'Primary review changed scope or data: ' + key)
    candidates = audit.get('perceptual_duplicate_pairs')
    require(isinstance(candidates, list) and candidates, 'Review requires original nonempty candidate list')
    decisions = review.get('pairs')
    require(isinstance(decisions, list) and len(decisions) == len(candidates), 'Incomplete candidate decisions')
    capture_pin = review['derivative_capture']
    capture_path = verify(relative(review_path.parent, capture_pin['path']), capture_pin['sha256'])
    capture = read(capture_path)
    require(capture.get('kind') == 'readonly_sha_bound_candidate_contact_sheets' and
            capture.get('audit_sha256') == sha(audit_path), 'Capture does not bind original audit')
    for key in ('policy_unchanged', 'original_sample_slots_unchanged', 'all_152_frame_fingerprints_equal_existing'):
        require(capture.get(key) is True, 'Capture failed original frame/policy verification: ' + key)
    require(capture.get('remote_files_written') == 0 and capture.get('gpu_used') is False,
            'Expected read-only CPU evidence capture')
    require(capture.get('scanner_sha256') == audit['environment']['scanner_sha256'] and
            capture.get('frame_cache_receipt_sha256') == audit['media_binding']['frame_cache_receipt_sha256'],
            'Capture scanner or frame cache changed')
    training_receipt_path = audit_artifact(audit_path, audit['artifacts']['training_fingerprints_receipt'])
    training_receipt = read(training_receipt_path)
    require(training_receipt.get('passed') is True and training_receipt.get('policy') == audit['policy'] and
            training_receipt.get('binding') == audit['binding'] and
            training_receipt.get('original_formal_ready_sha256') == audit['original_formal_ready_sha256'],
            'Training fingerprint receipt is incomplete or changed')
    training_path = verify(training_receipt_path.parent / 'training_fingerprints.jsonl',
                           training_receipt['fingerprints_sha256'])
    benchmark_path = audit_artifact(audit_path, audit['artifacts']['benchmark_fingerprints'])
    require(capture.get('training_receipt_sha256') == sha(training_receipt_path) and
            capture.get('training_fingerprints_sha256') == sha(training_path) and
            capture.get('benchmark_fingerprints_sha256') == sha(benchmark_path),
            'Capture fingerprints refer to different scans')
    training_rows = [json.loads(line) for line in training_path.read_text().splitlines() if line.strip()]
    training = unique_rows(training_rows, 'relative_path', 'Training fingerprints')
    benchmark = unique_rows(read(benchmark_path), 'source_video_id', 'Benchmark fingerprints')
    require(benchmark_inventory is not None and benchmark_inventory.get('complete') is True,
            'Fresh benchmark frame inventory required for review')
    benchmark_records = unique_rows(benchmark_inventory['records'], 'source_video_id', 'Current benchmark records')
    expected_sources = {('train', p['training_source_video_id']) for p in candidates} | {
        ('benchmark', p['benchmark_source_video_id']) for p in candidates}
    sources = capture.get('sources')
    require(isinstance(sources, list) and len(sources) == len(expected_sources), 'Capture source coverage differs')
    captured = {}
    for row in sources:
        key = (row['kind'], row['source_video_id'])
        require(key in expected_sources and key not in captured and row.get('fingerprints_equal_existing') is True,
                'Missing, extra, duplicate, or unverified capture source')
        captured[key] = row
    require(set(captured) == expected_sources, 'Capture source coverage incomplete')
    for candidate in candidates:
        t = training[candidate['training_path']]
        b = benchmark[candidate['benchmark_source_video_id']]
        ct, cb = captured[('train', candidate['training_source_video_id'])], captured[('benchmark', candidate['benchmark_source_video_id'])]
        require(t['source_video_id'] == candidate['training_source_video_id'], 'Candidate training source changed')
        require(ct.get('fingerprints') == t['frames'] and cb.get('fingerprints') == b['frames'] and
                ct.get('original_sha256') == t['sha256'] and ct.get('original_size_bytes') == t['size_bytes'] and
                ct.get('indices') == t['selected_indices'] and
                ct.get('decoded_frame_count') == t['decoded_frame_count'], 'Captured frames differ from original fingerprints')
        require(len(ct['fingerprints']) == len(cb['fingerprints']) == 8, 'All eight context frames required')
        record = benchmark_records[candidate['benchmark_source_video_id']]
        require(cb.get('original_sha256') == record['original_sha256'] and
                cb.get('original_size_bytes') == record['original_size_bytes'] and
                cb.get('indices') == record['selected_indices'], 'Benchmark original identity or sample indices changed')
        require(cb.get('cached_pngs') == [
            {'path': str(Path(benchmark_inventory['root']) / f['relative_path']),
             'sha256': f['sha256'], 'size_bytes': f['size_bytes']} for f in record['frames']],
             'Benchmark capture PNG binding differs from fresh complete inventory')
        seen_matches = set()
        for match in candidate['frame_matches']:
            require(type(match.get('training_slot')) is int and type(match.get('benchmark_slot')) is int and
                    0 <= match['training_slot'] < 8 and 0 <= match['benchmark_slot'] < 8,
                    'Candidate matched frame slot invalid')
            slots = match['training_slot'], match['benchmark_slot']
            require(slots not in seen_matches, 'Duplicate original matched frame slots')
            seen_matches.add(slots)
            tf, bf = t['frames'][slots[0]], b['frames'][slots[1]]
            pd = (int(tf['phash'], 16) ^ int(bf['phash'], 16)).bit_count()
            dd = (int(tf['dhash'], 16) ^ int(bf['dhash'], 16)).bit_count()
            require(match.get('phash_hamming') == pd and match.get('dhash_hamming') == dd and
                    match.get('same_decoded_rgb') is (tf['rgb_sha256'] == bf['rgb_sha256']) and
                    pd <= audit['policy']['phash_max_hamming'] and dd <= audit['policy']['dhash_max_hamming'] and
                    tf.get('informative') is True and bf.get('informative') is True,
                    'Candidate hash distances or RGB equality differ from attested fingerprints')
    require(capture.get('training_videos_redecoded') == len({p['training_path'] for p in candidates}) and
            capture.get('benchmark_existing_pngs_read') == 8 * len({p['benchmark_source_video_id'] for p in candidates}),
            'Capture frame counts incomplete')
    files = unique_rows(capture.get('files'), 'relative_path', 'Capture file inventory')
    expected_files = set().union(*(pair_evidence_names(i, p) for i, p in enumerate(candidates)))
    require(set(files) == expected_files, 'Contact sheet inventory differs from exact candidates/matches/context')
    artifacts = [{'path': str(audit_path), 'sha256': sha(audit_path)},
                 {'path': str(review_path), 'sha256': sha(review_path)},
                 {'path': str(capture_path), 'sha256': sha(capture_path)},
                 {'path': str(training_receipt_path), 'sha256': sha(training_receipt_path)},
                 {'path': str(training_path), 'sha256': sha(training_path)},
                 {'path': str(benchmark_path), 'sha256': sha(benchmark_path)}]
    metadata_pin = review['selected_metadata']
    metadata_path = verify(relative(review_path.parent, metadata_pin['path']), metadata_pin['sha256'])
    metadata = read(metadata_path)
    require(metadata.get('audit_sha256') == sha(audit_path) and metadata.get('pairs') == candidates,
            'Selected metadata candidate binding differs')
    artifacts.append({'path': str(metadata_path), 'sha256': sha(metadata_path)})
    for name, item in files.items():
        path = verify(relative(capture_path.parent, name), item['sha256'])
        require(path.stat().st_size == item['bytes'], 'Contact sheet byte count changed')
        artifacts.append({'path': str(path), 'sha256': item['sha256']})
    for index, (candidate, decision) in enumerate(zip(candidates, decisions)):
        require(type(decision.get('pair_index')) is int and decision['pair_index'] == index + 1,
                'Review candidate order changed')
        require({key: decision.get('original_classification' if key == 'classification' else key)
                 for key in candidate} == candidate, 'Review candidate content changed')
        require(decision.get('decision') == RESOLVED and decision.get('resolution_category') in RESOLUTIONS,
                'True or unresolved content overlap blocks evaluation')
        require(all(isinstance(decision.get(k), str) and decision[k].strip()
                    for k in ('matched_frame_observation', 'eight_frame_context_observation')) and
                decision.get('all_original_candidate_matches_visually_inspected') is True and
                decision.get('training_slots_inspected') == list(range(8)) and
                decision.get('benchmark_slots_inspected') == list(range(8)), 'Explicit complete per-pair review required')
        evidence = unique_rows(decision.get('evidence'), 'relative_path', 'Per-pair evidence')
        required_names = {'derivatives/' + n for n in pair_evidence_names(index, candidate)}
        require(set(evidence) == required_names, 'Per-pair review omits or adds matched/context evidence')
        for name, item in evidence.items():
            original = files[name.removeprefix('derivatives/')]
            require(item.get('sha256') == original['sha256'] and item.get('bytes') == original['bytes'],
                    'Per-pair evidence SHA or size differs')
    require(capture_path.parent == review_path.parent / 'derivatives', 'Unsupported derivative layout')
    expected_counts = {
        'candidate_pairs': len(candidates),
        'candidate_frame_matches': sum(len(p['frame_matches']) for p in candidates),
        'unique_training_sources': len({p['training_source_video_id'] for p in candidates}),
        'unique_benchmark_sources': len({p['benchmark_source_video_id'] for p in candidates}),
        'source_context_frames_visually_inspected': 8 * len(expected_sources),
        'unresolved_pairs': 0, 'substantive_overlapping_scene_pairs_observed': 0,
    }
    for category in RESOLUTIONS:
        expected_counts[category + '_pairs'] = sum(d['resolution_category'] == category for d in decisions)
        expected_counts[category + '_frame_matches'] = sum(len(d['frame_matches']) for d in decisions
                                                          if d['resolution_category'] == category)
    require(review.get('counts') == expected_counts, 'Primary review counts differ from preserved candidates')
    independent = validate_independent(independent_path, independent_sha256, audit_path,
                                       review_path, review, capture_path, files, stamp)
    artifacts.append({'path': str(safe(independent_path)), 'sha256': independent_sha256})
    return {'schema_version': 1, 'passed': True, 'review': {'path': str(review_path), 'sha256': sha(review_path)},
            'original_audit_passed': False, 'original_candidates_preserved': True,
            'candidate_count': len(candidates), 'matched_frame_count': sum(len(p['frame_matches']) for p in candidates),
            'decisions': decisions, 'reviewer': reviewer, 'reviewed_utc': review['reviewed_utc'],
            'independent_review': independent, 'counts': expected_counts,
            'perceptual_duplicate_pairs': candidates,
            'capture_receipt': {'path': str(capture_path), 'sha256': sha(capture_path)}, 'artifacts': artifacts,
            'limitation': 'Reviewed uniform8 candidates only; no content-disjointness claim for unsampled portions.'}


def validate_independent(path, digest, audit_path, review_path, primary, capture_path, files, primary_time):
    require(path is not None and digest is not None, 'A separate independent visual review is required')
    path = verify(path, digest)
    independent = read(path)
    require(independent.get('schema_version') == 1 and
            independent.get('kind') == 'independent_visual_candidate_review_v1', 'Unsupported independent-review schema')
    require(independent.get('audit_sha256') == sha(audit_path) and
            independent.get('primary_review_sha256') == sha(review_path) and
            independent.get('derivative_capture_sha256') == sha(capture_path), 'Independent review input binding differs')
    require(isinstance(independent.get('reviewer'), str) and independent['reviewer'].strip() and
            independent['reviewer'] != primary['reviewer'], 'A distinct independent reviewer is required')
    require(timestamp(independent.get('reviewed_utc')) >= primary_time, 'Independent review predates primary review')
    for field in ('independent_of_primary', 'passed', 'all_matched_frames_and_context_visually_inspected'):
        require(independent.get(field) is True, 'Independent review incomplete or failed: ' + field)
    pairs = independent.get('pairs')
    require(isinstance(pairs, list) and len(pairs) == len(primary['pairs']), 'Independent review omits candidate pairs')
    for index, (decision, reviewed) in enumerate(zip(pairs, primary['pairs']), 1):
        require(type(decision.get('pair_index')) is int and decision['pair_index'] == index and
                decision.get('primary_pair_sha256') == canonical_sha(reviewed), 'Independent pair binding differs')
        require(decision.get('decision') == 'agree' and isinstance(decision.get('reason'), str) and
                decision['reason'].strip(), 'Independent disagreement or unexplained decision blocks evaluation')
    evidence = unique_rows(independent.get('evidence_files'), 'relative_path', 'Independent review evidence')
    expected = {'derivatives/' + name: item for name, item in files.items()}
    require(set(evidence) == set(expected), 'Independent review evidence coverage differs')
    for name, item in evidence.items():
        require(item.get('sha256') == expected[name]['sha256'] and item.get('bytes') == expected[name]['bytes'],
                'Independent evidence SHA or size differs')
    return {'path': str(path), 'sha256': digest, 'reviewer': independent['reviewer'],
            'reviewed_utc': independent['reviewed_utc'], 'passed': True, 'pair_count': len(pairs)}


def validate_content_audit(path, required_binding, training_inventory=None, benchmark_inventory=None,
                           *, content_module=None, review_path=None, review_sha256=None,
                           independent_path=None, independent_sha256=None):
    """Legacy checks plus a separate, explicitly selected review admission path."""
    result = {'provided': path is not None, 'passed': False, 'errors': [], 'required_binding': required_binding}
    if path is None:
        result['errors'].append('--content-audit is required')
        return result
    try:
        path = safe(Path(path).absolute())
        audit = read(path)
        require(isinstance(audit, dict), 'Content audit JSON object required')
        result.update(path=str(path), sha256=sha(path))
        require(audit.get('schema_version') == 1, 'content audit schema_version must be 1')
        require(audit.get('binding') == required_binding, 'content audit binding does not match current annotations/manifests')
        require(isinstance(audit.get('method'), str) and audit['method'].strip(), 'content audit must describe its method')
        require(audit.get('exact_duplicate_pairs') == [], 'content audit exact_duplicate_pairs must be an explicit empty list')
        if review_path is None:
            require(review_sha256 is None and independent_path is None and independent_sha256 is None,
                    'Review or independent SHA/path provided without primary review')
            require(audit.get('perceptual_duplicate_pairs') == [], 'content audit perceptual_duplicate_pairs must be an explicit empty list')
            require(audit.get('passed') is True, 'content audit passed:true required')
        else:
            require(training_inventory is not None and benchmark_inventory is not None,
                    'Review requires fresh complete media inventories')
        if training_inventory is not None and benchmark_inventory is not None:
            require(content_module is not None, 'Content policy implementation required')
            require(training_inventory.get('complete') is True and benchmark_inventory.get('complete') is True,
                    'Fresh complete media inventories required')
            require(audit.get('media_binding') == content_module.media_binding(training_inventory, benchmark_inventory),
                    'content audit media SHA inventories do not match current bytes')
            require(audit.get('policy') == content_module.POLICY and audit.get('scope') == 'uniform8_sampled_frames',
                    'explicit supported sampled-frame audit policy required')
            require(audit.get('scan_complete') is True and audit.get('errors') == [] and audit.get('uninformative_sources') == [],
                    'content audit has missing, failed, or inconclusive scans')
            expected = benchmark_inventory.get('expected_source_videos', benchmark_inventory['expected_files'])
            require(audit.get('training_file_count') == training_inventory['expected_files'] and
                    audit.get('benchmark_source_count') == expected, 'content audit media counts mismatch')
            for name in ('training_fingerprints_receipt', 'benchmark_fingerprints'):
                audit_artifact(path, audit.get('artifacts', {})[name])
            result.update(scope='uniform8_sampled_frames', limitation=content_module.POLICY['limitation'])
        if review_path is not None:
            result['adjudication'] = validate_review(path, audit, review_path, review_sha256,
                independent_path=independent_path, independent_sha256=independent_sha256,
                benchmark_inventory=benchmark_inventory)
            result['admission_mode'] = 'original_scan_plus_bound_candidate_review_v1'
        else:
            result['admission_mode'] = 'original_scan_without_candidates_v1'
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        result['errors'].append(str(error))
    result['passed'] = not result['errors']
    return result
