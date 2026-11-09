#!/usr/bin/env python3
"""Cache one pinned TempCompass ZIP in OSS without local media payloads."""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE_SHA = 'c1dde8d109ff069519feaa01f35c3dfe9c1369e365dbe1433a3a55560f65870c'
PROOF_SHA = 'bc5d203c91738d7a9c6d5992c2735a6b05b4c80d23186d7968b3b9134b29e97f'
PLAN_SHA = 'a0d9c2e69cba3eef47ff9eee20a114262f5838db43e2a352d2bd57cb65a7de58'
PREFIX = 'jinhaopeng/Paper/DeepProbe-VLM/benchmark-cache/20261017-tempcompass-v1/hf-archives/'
CONTROL = Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/tempcompass-archive-20261017/control')
REVISION = 'c8a67d88a3bc6fd4b2f9ae2f9e112668fbe05722'
ASSET = {'asset_id': 'tempcompass_videos', 'repo_id': 'lmms-eval/TempCompass',
         'revision': REVISION, 'path': 'tempcompass_videos.zip',
         'size_bytes': 418421516,
         'sha256': 'fb7c8a2b742a3d10ae1bd1a6183fb602edb346a4c7fd91c631dba46b753e0f0b'}
LIBS = Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/benchmark-videomme-oss-decode-20261013/python-libs')


class SafeError(Exception):
    pass


def require(ok, reason):
    if not ok:
        raise SafeError(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_base(path):
    require(not path.is_symlink() and sha(path.read_bytes()) == BASE_SHA, 'base_worker_changed')
    spec = importlib.util.spec_from_file_location('tempcompass_frozen_stream_base', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PREFIX = PREFIX
    module.CONTROL = CONTROL
    return module


def validate_plan(raw, proof_raw):
    require(sha(raw) == PLAN_SHA, 'plan_sha_changed')
    require(sha(proof_raw) == PROOF_SHA, 'source_proof_sha_changed')
    plan = json.loads(raw)
    require(plan['assets'] == [ASSET], 'asset_identity_changed')
    require(plan['prefix'] == PREFIX and plan['bucket'] == 'ss-devbox-stream'
            and plan['endpoint'] == 'https://oss-cn-wulanchabu.aliyuncs.com'
            and plan['region'] == 'cn-wulanchabu', 'destination_changed')
    require(plan['archive_count'] == 1 and plan['total_bytes'] == ASSET['size_bytes']
            and plan['part_size_bytes'] == 16777216, 'plan_size_changed')
    require(plan['base_worker_sha256'] == BASE_SHA
            and plan['source_binding_sha256'] == PROOF_SHA, 'release_binding_changed')
    proof = json.loads(proof_raw)
    require(proof['revision'] == REVISION and proof['git_fsck_exit_code'] == 0
            and proof['archive']['sha256'] == ASSET['sha256']
            and proof['archive']['bytes'] == ASSET['size_bytes'], 'source_proof_invalid')
    return plan


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def resolve_anonymous(base, opener=None):
    """Resolve only the pinned mirror path, then accept an approved HF CDN."""
    allowed_paths = {'/datasets/' + repo + '/resolve/' + REVISION + '/' + ASSET['path']
                     for repo in ['lmms-eval/TempCompass', 'lmms-lab/TempCompass']}
    url = 'https://hf-mirror.com/datasets/lmms-eval/TempCompass/resolve/' + REVISION + '/' + ASSET['path']
    opener = opener or urllib.request.build_opener(NoRedirect(), urllib.request.ProxyHandler({}))
    seen = set()
    for _ in range(4):
        parsed = urllib.parse.urlsplit(url)
        require(parsed.scheme == 'https' and parsed.hostname == 'hf-mirror.com'
                and parsed.port in (None, 443) and not parsed.username and not parsed.password
                and not parsed.fragment and parsed.path in allowed_paths and url not in seen,
                'mirror_source_identity_changed')
        seen.add(url)
        request = urllib.request.Request(url, headers={'Range': 'bytes=0-0',
                  'Accept-Encoding': 'identity', 'Cache-Control': 'no-cache'})
        try:
            response = opener.open(request, timeout=30)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            require(response.status in {301, 302, 303, 307, 308}, 'mirror_non_redirect')
            target = response.headers.get('Location')
            require(isinstance(target, str) and 0 < len(target) <= 16384
                    and all(32 < ord(c) < 127 for c in target), 'redirect_location_invalid')
            target = urllib.parse.urljoin(url, target)
        if urllib.parse.urlsplit(target).hostname == 'hf-mirror.com':
            url = target
            continue
        return base.validate_url(target)
    raise SafeError('mirror_redirect_limit')


class AnonymousControl:
    def __init__(self, base):
        self.base = base
        self.refreshes = 0

    def check_alive(self):
        return None

    def refresh(self):
        require(self.refreshes < 3, 'anonymous_refresh_budget_exhausted')
        self.refreshes += 1
        return resolve_anonymous(self.base)


def publish(base, store, plan):
    expected = {PREFIX + 'CACHE_PLAN.json', PREFIX + 'ARCHIVE_CATALOG_COMPLETE.json',
                base.key_for(ASSET), base.receipt_key(ASSET)}
    existing = set(store.keys())
    require(existing <= expected, 'unexpected_prefix_object')
    if PREFIX + 'CACHE_PLAN.json' not in existing:
        require(not existing, 'prefix_not_empty_without_plan')
        require(not list(store.uploads(base.key_for(ASSET))), 'unowned_multipart')
    base.put_immutable(store, PREFIX + 'CACHE_PLAN.json', base.canonical({'plan_sha256': PLAN_SHA, 'plan': plan}))
    base.require_initialized(store, plan, PLAN_SHA)
    # The frozen worker checks the exact payload and bound multipart state.
    signed = None if store.exists(base.key_for(ASSET)) else resolve_anonymous(base)
    result = base.upload_asset(store, ASSET, PLAN_SHA, signed, source_control=AnonymousControl(base))
    require(result['status'] == 'CACHED_VERIFIED', 'archive_not_verified')
    receipt_raw = store.get_small(base.receipt_key(ASSET))
    receipt = json.loads(receipt_raw)
    require(receipt['binding'] == base.binding(ASSET, PLAN_SHA)
            and receipt['source_stream_sha256'] == ASSET['sha256']
            and receipt['source_bytes'] == ASSET['size_bytes']
            and receipt['oss_readback'] == {'bytes': ASSET['size_bytes'], 'sha256': ASSET['sha256'],
                                          'verification': 'complete_OSS_GET_SHA256_and_size'},
            'archive_receipt_invalid')
    catalog = {'schema_version': 1, 'plan_sha256': PLAN_SHA, 'source_binding_sha256': PROOF_SHA,
               'archive_count': 1, 'total_bytes': ASSET['size_bytes'],
               'entries': [{**base.binding(ASSET, PLAN_SHA), 'receipt_sha256': sha(receipt_raw)}],
               'all_pinned_zip_archives_cached': True, 'formal_evaluation_ready': False,
               'tempcompass_frame_cache_ready': False, 'all_benchmark_assets_ready': False}
    base.put_immutable(store, PREFIX + 'ARCHIVE_CATALOG_COMPLETE.json', base.canonical(catalog))
    require(set(store.keys()) == expected, 'final_prefix_key_set_mismatch')
    return {'status': 'TEMPCOMPASS_ARCHIVE_CACHED_VERIFIED', 'archive_bytes': ASSET['size_bytes'],
            'archive_sha256': ASSET['sha256'], 'catalog_sha256': sha(base.canonical(catalog)),
            'receipt_sha256': sha(receipt_raw), 'exact_prefix_object_count': 4,
            'new_local_zip_or_mp4_written': False, 'formal_evaluation_ready': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    base = load_base(args.release / 'hf_to_stream_oss_cache_v3.py')
    plan = validate_plan((args.release / 'TEMPCOMPASS_ARCHIVE_PLAN.json').read_bytes(),
                         (args.release / 'PINNED_GIT_SOURCE_BINDING.json').read_bytes())
    if not args.execute:
        print(json.dumps({'status': 'PREPARED', 'archive_count': 1, 'bytes': ASSET['size_bytes']}))
        return
    sys.path.insert(0, str(LIBS))
    with base.cache_lock():
        terminal_path = CONTROL / 'TERMINAL.json'
        require(not terminal_path.exists() and not (CONTROL / 'STARTED.json').exists(),
                'one_shot_already_started_requires_exact_inspection')
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        base.save_state(CONTROL / 'STARTED.json', {'started_utc': started,
            'worker_sha256': sha(Path(__file__).read_bytes()), 'plan_sha256': PLAN_SHA,
            'base_worker_sha256': BASE_SHA, 'gpu_used': False, 'process_attempts': 1})
        try:
            result = publish(base, base.OSSStore(), plan)
            result.update(exit_code=0, started_utc=started,
                          finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        except Exception as error:
            code = str(error) if isinstance(error, (SafeError, base.SafeError)) else type(error).__name__
            result = {'status': 'FAILED', 'error_code': code, 'exit_code': 2,
                      'started_utc': started, 'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        base.save_state(terminal_path, result)
        print(json.dumps(result, sort_keys=True), flush=True)
        if result['exit_code']:
            raise SystemExit(2)


if __name__ == '__main__':
    try:
        main()
    except SafeError as error:
        print(json.dumps({'status': 'FAILED', 'error_code': str(error)}), flush=True)
        raise SystemExit(2)
