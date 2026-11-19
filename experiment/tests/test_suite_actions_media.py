"""Owned frame cache lifecycle tests; no scheduler or subprocess is started."""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import patch

from deepprobe_vlm import suite_actions as actions, suite_media as media, suite_runner as runner
from deepprobe_vlm.suite_storage import SharedBudget


class CacheReaderLifecycleTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.frame = b'synthetic-local-frame'
        self.items = [{'path': 'frames/0.png', 'bytes': len(self.frame),
                       'sha256': hashlib.sha256(self.frame).hexdigest()}]
        self.cache_root = self.root / 'media' / 'shared-cache'
        self.data = {'cache': {'kind': 'synthetic-fixture'}, 'media_root': str(self.cache_root), 'max_bytes': 1024}
        self.configuration = {'schema_version': 2, 'roots': [str(self.root)],
            'ledger_root': str(self.root / 'ledger'), 'home_limit_bytes': 150_000_000_000,
            'oss_limit_bytes': None, 'home_safety_bytes': 0, 'oss_safety_bytes': 0,
            'free_guard_bytes': 0, 'oss_prefixes': ['oss://ss-devbox-stream/synthetic/']}
        self.budget = SharedBudget(self.configuration, lambda: self.fail('Unlimited OSS must not scan quota'))
        self.worker = self.make_worker('evaluation-one')
        self.task = self.worker.tasks['evaluation-one']
        self.addCleanup(patch.stopall)
        patch.object(media, '_bound_cache_inventory', side_effect=lambda cache:
            (None, None, deepcopy(self.items), None, None, None)).start()
        patch.object(media, 'OSSStore', return_value=object()).start()
        patch.object(media, 'hydrate_frame_cache', side_effect=self.hydrate).start()

    def make_worker(self, task_id):
        worker = object.__new__(actions.Actions)
        worker.root = self.root
        worker.state = self.root / task_id
        worker.tasks = {task_id: {'id': task_id, 'benchmark': 'videomme899', 'frames': 8}}
        worker.env = {'storage': self.configuration, 'runtime_environment': {},
            'tools': {'oss_config': '/synthetic/config'},
            'assets': {'benchmarks': {'videomme899': {'8': self.data}}}}
        worker._shared_budget = self.budget
        worker.budget = lambda *args, **kwargs: {'synthetic_reservation': True}
        return worker

    def hydrate(self, cache, *, media_root, max_bytes, store, resume):
        root = Path(media_root)
        root.mkdir(parents=True, exist_ok=True)
        owner = {'schema_version': 1, 'kind': 'suite_benchmark_media',
            'cache_binding_sha256': runner.digest(cache), 'media_root': str(root),
            'max_bytes': max_bytes, 'png_bytes': len(self.frame)}
        media._write_once(root / '.suite-media-owner.json', media.encode(owner))
        frame = root / self.items[0]['path']
        frame.parent.mkdir(exist_ok=True)
        media._write_once(frame, self.frame)
        metadata = root.parent / (root.name + '.acceptance')
        metadata.mkdir(exist_ok=True)
        media._write_once(metadata / 'READY.json', b'{"synthetic":true}')
        return {'media_root': str(root), 'acceptance': runner.artifact(metadata / 'READY.json')}

    def test_cache_released_after_reader_and_rehydrated_with_same_metadata(self):
        with self.worker.benchmark_asset(self.task) as first:
            self.assertTrue((self.cache_root / self.items[0]['path']).exists())
            original_acceptance = first['data']['acceptance']
        self.assertFalse((self.cache_root / self.items[0]['path']).exists())
        self.assertTrue((self.cache_root / '.suite-media-owner.json').exists())
        receipt = runner.read(first['release_receipt']['path'])
        self.assertEqual(receipt['status'], 'OWNED_LOCAL_BENCHMARK_PNGS_RELEASED')
        self.assertEqual(receipt['released_home_bytes'], len(self.frame))
        self.assertFalse(receipt['oss_deletion_performed'])
        with self.worker.benchmark_asset(self.task) as second:
            self.assertEqual(second['data']['acceptance'], original_acceptance)
            self.assertTrue((self.cache_root / self.items[0]['path']).exists())
        self.assertFalse((self.cache_root / self.items[0]['path']).exists())

    def test_same_cache_lock_covers_entire_reader_lifetime(self):
        worker_two = self.make_worker('evaluation-two')
        entered_one, release_one, attempted_two, entered_two = (threading.Event() for _ in range(4))
        failures = []
        def reader_one():
            try:
                with self.worker.benchmark_asset(self.task):
                    entered_one.set()
                    self.assertTrue(release_one.wait(5))
                    self.assertTrue((self.cache_root / self.items[0]['path']).exists())
            except BaseException as exc:
                failures.append(exc)
        def reader_two():
            attempted_two.set()
            try:
                with worker_two.benchmark_asset(worker_two.tasks['evaluation-two']):
                    entered_two.set()
                    self.assertTrue((self.cache_root / self.items[0]['path']).exists())
            except BaseException as exc:
                failures.append(exc)
        first = threading.Thread(target=reader_one)
        second = threading.Thread(target=reader_two)
        first.start()
        try:
            self.assertTrue(entered_one.wait(5))
            second.start()
            self.assertTrue(attempted_two.wait(5))
            self.assertFalse(entered_two.wait(0.1))
        finally:
            release_one.set()
            first.join(5)
            if second.ident is not None:
                second.join(5)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertFalse(failures, str(failures))
        self.assertTrue(entered_two.is_set())
        self.assertFalse((self.cache_root / self.items[0]['path']).exists())

    def test_validation_exception_releases_no_longer_used_cache(self):
        with self.assertRaisesRegex(ValueError, 'synthetic validation failed'):
            with self.worker.benchmark_asset(self.task):
                raise ValueError('synthetic validation failed before any subprocess')
        self.assertFalse((self.cache_root / self.items[0]['path']).exists())

    def test_unknown_reader_keeps_cache_and_blocks_another_reader(self):
        with self.assertRaisesRegex(ValueError, 'cache and reader lease preserved'):
            with self.worker.benchmark_asset(self.task):
                self.worker._benchmark_lease['readers_quiet'] = False
        self.assertTrue((self.cache_root / self.items[0]['path']).exists())
        with self.assertRaisesRegex(ValueError, 'Unreconciled benchmark reader lease'):
            with self.worker.benchmark_asset(self.task):
                self.fail('Unknown previous reader must not be adopted')

    def test_unowned_file_blocks_all_local_deletion(self):
        with self.assertRaisesRegex(ValueError, 'Unowned local cache file'):
            with self.worker.benchmark_asset(self.task):
                (self.cache_root / 'keep-me.txt').write_bytes(b'not-owned')
        self.assertTrue((self.cache_root / self.items[0]['path']).exists())
        self.assertEqual((self.cache_root / 'keep-me.txt').read_bytes(), b'not-owned')

    def test_changed_frame_sha_blocks_local_deletion(self):
        frame = self.cache_root / self.items[0]['path']
        with self.assertRaises(ValueError):
            with self.worker.benchmark_asset(self.task):
                frame.write_bytes(b'x' * len(self.frame))
        self.assertEqual(frame.read_bytes(), b'x' * len(self.frame))

    def test_evaluation_command_records_descendant_quiescence_before_release(self):
        with patch('deepprobe_vlm.suite_process.run_group', return_value=SimpleNamespace(
                returncode=0, group_cleanup_verified=True)) as process:
            with self.worker.benchmark_asset(self.task) as asset:
                self.worker.command(['/synthetic/evaluate'], self.worker.task_dir(self.task['id']), 'evaluate')
                self.assertTrue((self.cache_root / self.items[0]['path']).exists())
        self.assertEqual(process.call_count, 1)
        proof = runner.read(asset['release_receipt']['path'])
        self.assertTrue(runner.read(proof['command_exit']['path'])['group_cleanup_verified'])

    def test_interrupted_command_releases_only_if_process_tree_is_proven_quiet(self):
        for verified in (True, False):
            with self.subTest(verified=verified):
                failure = RuntimeError('synthetic command interruption')
                failure.group_cleanup_verified = verified
                failure.group_returncode = -15
                with patch('deepprobe_vlm.suite_process.run_group', side_effect=failure):
                    with self.assertRaisesRegex((RuntimeError, ValueError), 'interruption|lease preserved'):
                        with self.worker.benchmark_asset(self.task):
                            self.worker.command(['/synthetic/evaluate'], self.worker.task_dir(self.task['id']), 'evaluate')
                self.assertEqual((self.cache_root / self.items[0]['path']).exists(), not verified)


if __name__ == '__main__':
    unittest.main()
