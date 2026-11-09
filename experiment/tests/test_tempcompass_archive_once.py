import pytest
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'experiment/scripts/operations/cache_tempcompass_archive_once.py'
RELEASE = ROOT / 'experiment/docs/operations/evidence/tempcompass-acquisition-20261017/archive-release-v1'
spec = importlib.util.spec_from_file_location('tempcompass_archive_tested', SOURCE)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class Response:
    def __init__(self, status=302, location='https://cas-bridge.xethub.hf.co/payload?signed=yes'):
        self.status = status
        self.headers = {'Location': location}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class Opener:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        return next(self.responses)


class TempCompassArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = worker.load_base(ROOT / 'experiment/scripts/operations/hf_to_stream_oss_cache_v3.py')

    @pytest.mark.private_evidence
    def test_frozen_plan_and_git_proof_are_accepted(self):
        plan = worker.validate_plan((RELEASE / 'TEMPCOMPASS_ARCHIVE_PLAN.json').read_bytes(),
                                    (RELEASE / 'PINNED_GIT_SOURCE_BINDING.json').read_bytes())
        self.assertEqual(plan['assets'], [worker.ASSET])

    @pytest.mark.private_evidence
    def test_changed_plan_rejected_before_network(self):
        with self.assertRaisesRegex(worker.SafeError, 'plan_sha_changed'):
            worker.validate_plan(b'{}', (RELEASE / 'PINNED_GIT_SOURCE_BINDING.json').read_bytes())

    @pytest.mark.private_evidence
    def test_changed_git_proof_rejected_before_network(self):
        with self.assertRaisesRegex(worker.SafeError, 'source_proof_sha_changed'):
            worker.validate_plan((RELEASE / 'TEMPCOMPASS_ARCHIVE_PLAN.json').read_bytes(), b'{}')

    def test_anonymous_exact_revision_to_approved_cdn(self):
        opener = Opener([Response()])
        self.assertEqual(worker.resolve_anonymous(self.base, opener),
                         'https://cas-bridge.xethub.hf.co/payload?signed=yes')
        req = opener.requests[0]
        self.assertIn('/resolve/' + worker.REVISION + '/', req.full_url)
        self.assertIsNone(req.get_header('Authorization'))
        self.assertEqual(req.get_header('Range'), 'bytes=0-0')

    def test_pinned_repo_alias_redirect_allowed(self):
        url = 'https://hf-mirror.com/datasets/lmms-lab/TempCompass/resolve/' + worker.REVISION + '/tempcompass_videos.zip'
        opener = Opener([Response(location=url), Response()])
        worker.resolve_anonymous(self.base, opener)
        self.assertEqual(len(opener.requests), 2)

    def test_redirect_to_main_ref_rejected(self):
        url = 'https://hf-mirror.com/datasets/lmms-eval/TempCompass/resolve/main/tempcompass_videos.zip'
        with self.assertRaisesRegex(worker.SafeError, 'mirror_source_identity_changed'):
            worker.resolve_anonymous(self.base, Opener([Response(location=url)]))

    def test_unknown_cdn_rejected(self):
        with self.assertRaises(self.base.SafeError):
            worker.resolve_anonymous(self.base, Opener([Response(location='https://example.com/payload')]))

    def test_redirect_credentials_rejected(self):
        with self.assertRaises(self.base.SafeError):
            worker.resolve_anonymous(self.base, Opener([Response(location='https://user@cas-bridge.xethub.hf.co/payload')]))

    def test_nonredirect_media_body_not_read(self):
        with self.assertRaisesRegex(worker.SafeError, 'mirror_non_redirect'):
            worker.resolve_anonymous(self.base, Opener([Response(status=200)]))

    def test_no_local_source_and_exact_new_prefix(self):
        self.assertIsNone(self.base.approved_local_path(worker.ASSET))
        self.assertTrue(self.base.key_for(worker.ASSET).startswith(worker.PREFIX))
        self.assertNotIn('20261013-v1', self.base.key_for(worker.ASSET))

    def test_unexpected_old_prefix_object_blocks_before_writes(self):
        class Store:
            def keys(self):
                return [worker.PREFIX + 'unrecognized_old_object']
        with self.assertRaisesRegex(worker.SafeError, 'unexpected_prefix_object'):
            worker.publish(self.base, Store(), {})

    def test_refresh_budget_is_bounded(self):
        control = worker.AnonymousControl(self.base)
        control.refreshes = 3
        with self.assertRaisesRegex(worker.SafeError, 'anonymous_refresh_budget_exhausted'):
            control.refresh()


if __name__ == '__main__':
    unittest.main()
