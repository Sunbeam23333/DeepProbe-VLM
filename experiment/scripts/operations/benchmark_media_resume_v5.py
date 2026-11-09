#!/usr/bin/env python3
"""Resume exact Video-MME acquisition without replaying completed archives.

Default operation is verification only. --execute is an explicit action switch.
Startup JSON arrives on stdin: {"hf_token": "...", "prior_failure_sha256": "..."}.
The token is used only in memory for requests to https://huggingface.co:443.
Signed CDN URLs are passed to the unchanged, SHA-pinned v4 worker via stdin.
This program never removes the old coordinator failure marker or old run data.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import http.client
import json
from pathlib import Path
import re
import shlex
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
LOCAL_PLAN = ROOT / 'docs/operations/evidence/benchmark-media-acquisition-20261012/videomme_archive_directory.json'
PLAN_SHA = 'ca3c186c375286500e0576b80eac0012e6fcf61abe9dd8714be731fe88292ac0'
REGISTRY_SHA = 'a143d6a364221738b727d3944527039060315e04373599c5829ab3ea60eb0226'
WORKER_SHA = 'dd8993522994bc262d1b1c4b15bf528937acffb00a9fd0711a0ffa0be043eed9'
APPROVED_CACHE_IMPLEMENTATIONS = {WORKER_SHA, '52ccd0803d1078833686a9d71d669edecabcf5c39b0b20f74de67825e844a1c2'}
REVISION = 'ead1408f75b618502df9a1d8e0950166bf0a2a0b'
REPO = 'lmms-lab/Video-MME'
REMOTE_PYTHON = '/home/jinhaopeng/.cache/Paper/DeepProbe-VLM/env/bin/python'
MEDIA = Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/benchmark-media-20261012/videomme')
REGISTRY = MEDIA.parents[1] / 'formal-source-final-20261009-v1/benchmark_source_registry.json'
WORKER = MEDIA / 'benchmark_media_cache_v4.py'
CONTROL = MEDIA / 'resume-20261013-v5'
REMOTE_SELF = CONTROL / 'benchmark_media_resume_v5.py'
BASE_URL = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/'
SHA_RE = re.compile(r'[0-9a-f]{64}')
ARCHIVE_RE = re.compile(r'videos_chunked_(?:0[1-9]|1[0-9]|20)\.zip')
TRANSIENT_HTTP = {408, 425, 429, 500, 502, 503, 504}


class SafeError(Exception):
    """Carries only a fixed diagnostic code, never network exception text."""


class ProbeTransportError(Exception):
    pass


def require(condition, code):
    if not condition:
        raise SafeError(code)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def emit(event, **fields):
    print(json.dumps({'event': event, **fields}, sort_keys=True), flush=True)


def bounded_json(path, maximum=20_000_000):
    require(not path.is_symlink() and path.is_file(), 'receipt_missing_or_symlink')
    require(path.stat().st_size <= maximum, 'receipt_oversized')
    return json.loads(path.read_text())


def contained_file(media, relative):
    require(isinstance(relative, str), 'invalid_relative_path')
    path = Path(relative)
    require(not path.is_absolute() and '..' not in path.parts, 'unsafe_relative_path')
    current = media
    for component in path.parts:
        current = current / component
        require(not current.is_symlink(), 'symlink_in_media_path')
    require(current.is_file(), 'media_file_missing')
    require(current.resolve().is_relative_to(media.resolve()), 'media_path_escape')
    return current


def verify_source_entry(media, member, archive, entry, expected_entry_sha, entry_path):
    from PIL import Image
    identity = Path(member['path']).stem
    require(sha(entry_path) == expected_entry_sha, 'entry_receipt_sha_mismatch')
    expected_fields = {'source_video_id': 'youtube:' + identity, 'repo_id': REPO,
                       'revision': REVISION, 'archive_path': archive['path'],
                       'archive_sha256': archive['sha256'], 'member_path': member['path'],
                       'original_size_bytes': member['size_bytes'], 'decode_complete': True}
    for key, value in expected_fields.items():
        require(entry.get(key) == value, 'entry_provenance_mismatch')
    require(bool(SHA_RE.fullmatch(entry.get('original_sha256', ''))), 'invalid_source_sha')
    require(entry.get('cache_implementation_sha256') in APPROVED_CACHE_IMPLEMENTATIONS,
            'unapproved_cache_implementation')
    count = entry.get('decoded_frame_count')
    require(type(count) is int and count > 0, 'invalid_decoded_frame_count')
    indices = [round(i * (count - 1) / 7) for i in range(8)]
    require(entry.get('selected_indices') == indices, 'sampling_indices_mismatch')
    passes = entry.get('full_decode_passes', 2)
    require(passes in {1, 2}, 'invalid_decode_passes')
    if passes == 1:
        require(entry.get('container_declared_frame_count') == count and
                entry.get('second_pass_frame_count') is None, 'single_pass_count_mismatch')
    else:
        require(entry.get('second_pass_frame_count') == count, 'second_pass_count_mismatch')
    frames = entry.get('frames')
    expected_paths = [f'frames/{identity}/{slot:02d}.png' for slot in range(8)]
    require(isinstance(frames, list) and len(frames) == 8, 'wrong_frame_count')
    require(entry.get('image_paths') == expected_paths, 'image_order_mismatch')
    for slot, frame in enumerate(frames):
        require(frame.get('relative_path') == expected_paths[slot], 'frame_path_order_mismatch')
        require(frame.get('source_frame_index') == indices[slot], 'frame_index_mismatch')
        path = contained_file(media, expected_paths[slot])
        require(path.stat().st_size == frame.get('size_bytes') and
                sha(path) == frame.get('sha256'), 'png_size_or_sha_mismatch')
        with Image.open(path) as image:
            require(image.mode == 'RGB', 'png_mode_mismatch')
            require(image.size == (frame.get('width'), frame.get('height')), 'png_dimensions_mismatch')
            require(hashlib.sha256(image.tobytes()).hexdigest() == frame.get('rgb_sha256'),
                    'rgb_sha_mismatch')


def verify_archive_state(media, archive, source_ids):
    """Read-only verification of every completed source and its receipt chain."""
    require(bool(ARCHIVE_RE.fullmatch(archive['path'])), 'unknown_archive_name')
    name = archive['path']
    selected = [member for member in archive['members']
                if 'youtube:' + Path(member['path']).stem in source_ids]
    expected_ids = sorted('youtube:' + Path(member['path']).stem for member in selected)
    identifiers = [source.split(':')[1] for source in expected_ids]
    complete_path = media / 'archives' / (name + '.complete.json')
    verified_path = media / 'archives' / (name + '.verified.json')
    admission_path = media / 'archives' / (name + '.admission.json')
    original = media / 'temporary' / name
    partial = original.with_suffix('.partial')
    for path in [complete_path, verified_path, admission_path, original, partial]:
        require(not path.is_symlink(), 'archive_state_symlink')
    require(not partial.exists(), 'unresolved_partial_archive')
    for identity in identifiers:
        require(not (media / 'temporary' / (identity + '.mp4')).exists(), 'unresolved_temporary_source')
    if not complete_path.exists():
        if original.exists() and verified_path.exists():
            verified = bounded_json(verified_path)
            require(verified.get('verified') is True and verified.get('archive_path') == name and
                    verified.get('archive_sha256') == archive['sha256'] and
                    verified.get('size_bytes') == archive['size_bytes'] and
                    verified.get('repo_id') == REPO and verified.get('revision') == REVISION,
                    'verified_archive_receipt_mismatch')
            require(original.stat().st_size == archive['size_bytes'] and sha(original) == archive['sha256'],
                    'verified_archive_bytes_mismatch')
            # Source receipts may only be reused after v4 rechecks their PNG SHA.
            # Any orphan frame directory without its source receipt is unresolved.
            for identity in identifiers:
                entry = media / 'entries' / (identity + '.json')
                frames = media / 'frames' / identity
                require(not frames.exists() or entry.is_file(), 'orphan_partial_source_cache')
            return {'state': 'verified_archive_ready', 'archive': name, 'source_count': len(selected)}
        existing_entries = any((media / 'entries' / (identity + '.json')).exists() or
                               (media / 'frames' / identity).exists() for identity in identifiers)
        require(not any(path.exists() for path in [original, verified_path, admission_path]) and
                not existing_entries, 'incomplete_archive_requires_inspection')
        return {'state': 'not_started', 'archive': name, 'source_count': len(selected)}
    require(not original.exists(), 'complete_archive_has_temporary_payload')
    complete = bounded_json(complete_path)
    require(complete.get('archive_path') == name and complete.get('archive_sha256') == archive['sha256'],
            'archive_complete_identity_mismatch')
    require(complete.get('source_count') == len(selected) and complete.get('source_video_ids') == expected_ids,
            'archive_complete_source_set_mismatch')
    require(complete.get('temporary_archive_deleted_after_verified_cache') is True,
            'archive_cleanup_receipt_mismatch')
    receipt_shas = complete.get('source_receipt_sha256', {})
    require(set(receipt_shas) == set(identifiers), 'archive_entry_receipt_set_mismatch')
    verified = bounded_json(verified_path)
    require(verified.get('verified') is True and verified.get('archive_path') == name and
            verified.get('archive_sha256') == archive['sha256'] and
            verified.get('size_bytes') == archive['size_bytes'] and
            verified.get('repo_id') == REPO and verified.get('revision') == REVISION,
            'archive_verified_receipt_mismatch')
    for member in selected:
        identity = Path(member['path']).stem
        entry_path = media / 'entries' / (identity + '.json')
        entry = bounded_json(entry_path)
        verify_source_entry(media, member, archive, entry, receipt_shas[identity], entry_path)
    return {'state': 'complete_verified', 'archive': name, 'source_count': len(selected),
            'complete_receipt_sha256': sha(complete_path), 'verified_receipt_sha256': sha(verified_path)}


def remote_probe(payload):
    require(sha(WORKER) == WORKER_SHA, 'frozen_v4_worker_changed')
    require(sha(MEDIA / 'videomme_archive_directory.json') == PLAN_SHA, 'pinned_media_plan_changed')
    require(sha(REGISTRY) == REGISTRY_SHA, 'pinned_source_registry_changed')
    legacy_failure = MEDIA / 'COORDINATOR_FAILURE.json'
    acknowledged = payload.get('prior_failure_sha256')
    if legacy_failure.exists():
        require(acknowledged == sha(legacy_failure), 'prior_failure_acknowledgment_mismatch')
    else:
        require(acknowledged is None, 'acknowledged_prior_failure_missing')
    with (MEDIA / '.acquisition.lock').open('rb') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SafeError('acquisition_worker_is_active') from None
        if payload.get('final_probe'):
            final = MEDIA / 'FRAME_CACHE_RECEIPT.json'
            sentinel = MEDIA / 'FRAME_CACHE_COMPLETE.json'
            if not final.exists() and not sentinel.exists():
                return {'state': 'not_finalized'}
            require(final.is_file() and sentinel.is_file(), 'partial_final_publication')
            value = bounded_json(sentinel)
            require(value.get('complete') is True and value.get('source_count') == 899 and
                    value.get('frame_count') == 7192 and value.get('receipt_sha256') == sha(final),
                    'final_sentinel_mismatch')
            return {'state': 'finalized', 'receipt_sha256': sha(final)}
        plan = {entry['path']: entry for entry in bounded_json(MEDIA / 'videomme_archive_directory.json')}
        spec = payload['archive']
        require(plan.get(spec.get('path')) == spec, 'incoming_archive_plan_mismatch')
        registry = bounded_json(REGISTRY)
        source_ids = {entry['source_video_id'] for entry in registry['entries'] if entry['dataset'] == 'videomme'}
        require(len(source_ids) == 899, 'wrong_fixed_source_count')
        return verify_archive_state(MEDIA, spec, source_ids)


class CredentialSafeRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, token):
        super().__init__()
        self._token = token

    def redirect_request(self, request, response, code, message, headers, new_url):
        target = urllib.parse.urlsplit(urllib.parse.urljoin(request.full_url, new_url))
        require(target.scheme == 'https' and target.hostname and not target.username and
                not target.password, 'unsafe_hf_redirect')
        following = super().redirect_request(request, response, code, message, headers, new_url)
        if following is None:
            return None
        for collection in [following.headers, following.unredirected_hdrs]:
            for key in list(collection):
                if key.lower() in {'authorization', 'proxy-authorization', 'cookie'}:
                    del collection[key]
        if target.hostname == 'huggingface.co' and target.port in {None, 443} and self._token:
            following.add_unredirected_header('Authorization', 'Bearer ' + self._token)
        return following


def resolve_signed_url(spec, token, *, opener=None, sleeper=time.sleep, now=time.time, attempts=6, logger=emit):
    opener = opener or urllib.request.build_opener(CredentialSafeRedirect(token))
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(BASE_URL + spec['path'], headers={
            'Range': 'bytes=0-0', 'Accept-Encoding': 'identity', 'Cache-Control': 'no-cache',
            'User-Agent': 'DeepProbe-VLM-pinned-media-resume-v5'})
        if token:
            request.add_unredirected_header('Authorization', 'Bearer ' + token)
        retry_after = 0
        try:
            with opener.open(request, timeout=45) as response:
                require(response.status == 206, 'hf_range_response_mismatch')
                require(response.headers.get('Content-Range') == f'bytes 0-0/{spec["size_bytes"]}',
                        'hf_range_identity_mismatch')
                require(len(response.read(2)) == 1, 'hf_range_length_mismatch')
                signed = response.url
            target = urllib.parse.urlsplit(signed)
            require(target.scheme == 'https' and target.hostname and not target.username and
                    not target.password and target.port in {None, 443}, 'unsafe_signed_cdn_url')
            require(target.hostname == 'huggingface.co' or target.hostname.endswith('.hf.co') or
                    target.hostname.endswith('.huggingface.co'), 'unexpected_cdn_host')
            expiry = urllib.parse.parse_qs(target.query).get('Expires', [])
            require(len(expiry) == 1 and expiry[0].isdigit() and int(expiry[0]) - now() >= 300,
                    'signed_url_lifetime_insufficient')
            return signed
        except urllib.error.HTTPError as error:
            status = error.code
            if status not in TRANSIENT_HTTP:
                raise SafeError(f'hf_http_{status}') from None
            raw_retry = error.headers.get('Retry-After', '') if error.headers else ''
            retry_after = min(60, int(raw_retry)) if raw_retry.isdigit() else 0
            code = f'hf_http_{status}'
        except ssl.SSLCertVerificationError:
            raise SafeError('hf_tls_certificate_verification_failed') from None
        except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError,
                http.client.IncompleteRead, ssl.SSLError) as error:
            code = 'hf_transient_' + type(error).__name__
        if attempt == attempts:
            raise SafeError('hf_resolve_retry_budget_exhausted') from None
        delay = max(retry_after, min(45, 3 * (2 ** (attempt - 1))))
        logger('hf_resolve_retry', archive=spec['path'], attempt=attempt, maximum_attempts=attempts,
               error_code=code, delay_seconds=delay)
        sleeper(delay)
    raise SafeError('hf_resolve_unreachable')


def ssh_command(script, *arguments):
    return ['ssh', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30',
            '-o', 'ServerAliveCountMax=3', 'haopengjin2ali',
            shlex.join([REMOTE_PYTHON, '-u', str(script), *arguments])]


def probe(spec, prior_failure_sha, *, final=False):
    payload = {'archive': spec, 'prior_failure_sha256': prior_failure_sha, 'final_probe': final}
    for attempt in range(1, 4):
        try:
            result = subprocess.run(ssh_command(REMOTE_SELF, '--remote-probe'),
                                    input=json.dumps(payload), text=True, capture_output=True, timeout=240)
            lines = result.stdout.strip().splitlines()
            value = json.loads(lines[-1]) if lines else {}
            if value.get('state') == 'blocked':
                raise SafeError(value.get('error_code', 'remote_probe_blocked'))
            if result.returncode == 255 or not lines:
                raise ProbeTransportError()
            require(result.returncode == 0 and value.get('state'), 'remote_probe_protocol_failure')
            return value
        except (subprocess.TimeoutExpired, json.JSONDecodeError, ProbeTransportError):
            if attempt == 3:
                raise SafeError('remote_probe_transport_exhausted') from None
            emit('remote_probe_retry', attempt=attempt, archive=spec['path'] if spec else None)
            time.sleep(2 ** attempt)
    raise SafeError('remote_probe_unreachable')


def relay_worker_event(line):
    """Never print raw worker/SSH output, including a possible exception URL."""
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return
    allowed = {
        'download_progress': {'archive', 'bytes', 'elapsed_seconds'},
        'archive_verified': {'archive', 'bytes', 'elapsed_seconds'},
        'source_cached': {'source_video_id', 'decoded_frame_count', 'png_bytes', 'seconds'},
        'archive_complete': {'archive', 'sources', 'decode_seconds'},
        'complete': {'sources', 'frames', 'receipt_sha256'},
    }
    event = value.get('event')
    if event in allowed:
        emit(event, **{key: value[key] for key in allowed[event] if key in value})


def invoke_worker(mode, payload=None):
    process = subprocess.Popen(ssh_command(WORKER, mode), stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    process.stdin.write(json.dumps(payload) if payload is not None else '')
    process.stdin.close()
    for line in process.stdout:
        relay_worker_event(line)
    return process.wait()


def remote_marker(payload):
    attempt = payload['attempt_id']
    require(bool(re.fullmatch(r'[a-z0-9-]{1,64}', attempt)), 'invalid_attempt_id')
    require(payload['name'] in {'STARTED.json', 'FAILED.json', 'COMPLETE.json'}, 'invalid_marker_name')
    directory = CONTROL / attempt
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / payload['name']
    require(not path.exists() and not path.is_symlink(), 'attempt_marker_already_exists')
    temporary = path.with_suffix('.writing')
    with temporary.open('x') as stream:
        json.dump(payload['value'], stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        __import__('os').fsync(stream.fileno())
    temporary.rename(path)
    return {'state': 'marker_written', 'name': payload['name']}


def write_marker(attempt, name, value):
    result = subprocess.run(ssh_command(REMOTE_SELF, '--remote-marker'),
                            input=json.dumps({'attempt_id': attempt, 'name': name, 'value': value}),
                            text=True, capture_output=True, timeout=45)
    require(result.returncode == 0, 'remote_attempt_marker_write_failed')


def read_startup(stream):
    text = stream.read(8193)
    require(len(text) <= 8192, 'startup_input_oversized')
    value = json.loads(text) if text.strip() else {}
    require(isinstance(value, dict) and set(value) <= {'hf_token', 'prior_failure_sha256'},
            'invalid_startup_fields')
    token = value.get('hf_token', '')
    require(isinstance(token, str) and len(token) <= 4096 and
            all(32 < ord(char) < 127 for char in token), 'invalid_credential_format')
    prior = value.get('prior_failure_sha256')
    require(prior is None or isinstance(prior, str) and SHA_RE.fullmatch(prior), 'invalid_prior_failure_sha')
    return token, prior


def coordinate(args, startup):
    token, prior_failure_sha = startup
    require(sha(LOCAL_PLAN) == PLAN_SHA, 'local_pinned_plan_changed')
    specs = json.loads(LOCAL_PLAN.read_text())
    specs.sort(key=lambda spec: (spec['path'] != 'videos_chunked_20.zip', spec['path']))
    states = []
    for spec in specs:
        state = probe(spec, prior_failure_sha)
        emit('archive_verified_before_resume', **state)
        states.append(state)
    completed = sum(state['source_count'] for state in states if state['state'] == 'complete_verified')
    emit('resume_preflight_complete', completed_sources=completed,
         complete_archives=sum(state['state'] == 'complete_verified' for state in states), execute=args.execute)
    if not args.execute:
        return
    write_marker(args.attempt_id, 'STARTED.json', {'complete': False, 'prior_failure_sha256': prior_failure_sha,
                 'coordinator_sha256': sha(Path(__file__)), 'worker_sha256': WORKER_SHA,
                 'archive_directory_sha256': PLAN_SHA, 'completed_sources_at_start': completed})
    for spec, previous in zip(specs, states):
        if previous['state'] == 'complete_verified':
            emit('archive_skipped_after_full_verification', archive=spec['path'], sources=previous['source_count'])
            continue
        fresh = probe(spec, prior_failure_sha)
        if fresh['state'] == 'complete_verified':
            continue
        emit('resuming_archive', archive=spec['path'], previous_state=fresh['state'])
        signed = resolve_signed_url(spec, token) if fresh['state'] == 'not_started' else ''
        rc = invoke_worker('archive', {'archive': spec, 'signed_url': signed})
        signed = None
        # Resolve any ambiguous SSH/worker outcome by exact read-only archive state.
        # Never blindly rerun a mutation after a client failure.
        resulting = probe(spec, prior_failure_sha)
        require(resulting['state'] == 'complete_verified', 'worker_archive_not_complete')
        emit('archive_postrun_independently_verified', archive=spec['path'],
             sources=resulting['source_count'], observed_worker_returncode=rc)
    final = probe(None, prior_failure_sha, final=True)
    if final['state'] == 'not_finalized':
        rc = invoke_worker('finalize')
        final = probe(None, prior_failure_sha, final=True)
        require(final['state'] == 'finalized', 'worker_finalization_not_complete')
    write_marker(args.attempt_id, 'COMPLETE.json', {'complete': True, 'source_count': 899, 'frame_count': 7192,
                 'receipt_sha256': final['receipt_sha256'], 'coordinator_sha256': sha(Path(__file__))})
    emit('resume_complete', source_count=899, frame_count=7192, receipt_sha256=final['receipt_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--attempt-id', default='a1')
    parser.add_argument('--remote-probe', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--remote-marker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.remote_probe or args.remote_marker:
        try:
            payload = json.load(sys.stdin)
            value = remote_probe(payload) if args.remote_probe else remote_marker(payload)
            print(json.dumps(value), flush=True)
        except Exception as error:
            code = str(error) if isinstance(error, SafeError) else 'remote_validation_' + type(error).__name__
            print(json.dumps({'state': 'blocked', 'error_code': code}), flush=True)
            sys.exit(2)
        return
    require(bool(re.fullmatch(r'[a-z0-9-]{1,64}', args.attempt_id)), 'invalid_attempt_id')
    try:
        coordinate(args, read_startup(sys.stdin))
    except Exception as error:
        code = str(error) if isinstance(error, SafeError) else 'coordinator_' + type(error).__name__
        emit('resume_failed', error_code=code, mutations_enabled=args.execute)
        if args.execute:
            try:
                write_marker(args.attempt_id, 'FAILED.json', {'complete': False, 'error_code': code,
                             'action': 'inspect exact archive state before another attempt'})
            except Exception:
                emit('failure_marker_unavailable')
        sys.exit(1)


if __name__ == '__main__':
    main()
