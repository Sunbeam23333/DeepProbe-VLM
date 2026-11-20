"""Cross-video overlap, bounded pending payloads and immutable cache failures."""
from copy import deepcopy
import io
import json
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from unittest.mock import Mock
import zipfile

from deepprobe_vlm import suite_media as media
import test_suite_media as media_fixture
from test_suite_media import FakeStore, make_video


class ParallelCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.video = make_video()

    setUp = media_fixture.OriginalFrameCacheTest.setUp
    build = media_fixture.OriginalFrameCacheTest.build

    def clips(self, count=3):
        archive = self.binding['archives'][0]
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for index in range(count):
                bundle.writestr(f'videos/clip_{index}.mp4', self.video)
        raw = buffer.getvalue()
        archive.update(size_bytes=len(raw), sha256=media.digest(raw))
        receipt = json.loads(self.store.objects[archive['receipt']['key']])
        receipt['binding'].update(size_bytes=len(raw), sha256=media.digest(raw))
        receipt.update(source_bytes=len(raw), source_stream_sha256=media.digest(raw),
                       oss_readback={'bytes': len(raw), 'sha256': media.digest(raw),
                                     'verification': 'complete_OSS_GET_SHA256_and_size'})
        receipt_raw = media.encode(receipt)
        archive['receipt'].update(size_bytes=len(receipt_raw), sha256=media.digest(receipt_raw))
        rows, sources = [], []
        with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
            for index, info in enumerate(bundle.infolist()):
                source = deepcopy(self.binding['sources'][0])
                source['media_identity'] = f'tempcompass:clip_{index}'
                source['member'].update(path=info.filename, size_bytes=info.file_size,
                                        compressed_bytes=info.compress_size, crc32=info.CRC)
                sources.append(source)
                row = deepcopy(self.rows[0])
                row.update(sample_id=f'question:{index}', video_id=f'clip_{index}',
                           processed_media_identity=source['media_identity'])
                rows.append(row)
        manifest = b''.join((json.dumps(row) + '\n').encode() for row in rows)
        self.manifest.write_bytes(manifest)
        self.binding.update(manifest_sha256=media.digest(manifest), manifest_bytes=len(manifest),
                            expected_samples=count, expected_clips=count, sources=sources)
        self.initial_objects = {archive['key']: raw, archive['receipt']['key']: receipt_raw}
        self.store = FakeStore(self.initial_objects, archive['key'])

    def reset_output(self, name):
        self.out = self.root / name
        self.store = FakeStore(self.initial_objects, self.binding['archives'][0]['key'])

    def test_real_parallel_decode_matches_serial_bytes_and_two_pass_receipts(self):
        self.clips()
        serial = self.build()
        objects = deepcopy(self.store.objects)
        self.reset_output('parallel')
        barrier = threading.Barrier(3)
        def decode(original, frames, **bounds):
            barrier.wait(timeout=5)
            return media.decode_selected(original, frames, **bounds)
        writes = []
        def record_calls(function):
            def called(*args, **kwargs):
                writes.append(threading.current_thread().name)
                return function(*args, **kwargs)
            return called
        for name in ('get', 'head', 'range', 'keys', 'put_verified'):
            setattr(self.store, name, record_calls(getattr(self.store, name)))
        parallel = self.build(workers=3, decoder=decode)
        self.assertEqual(serial, parallel)
        self.assertEqual(objects, self.store.objects)
        self.assertEqual(set(writes), {threading.current_thread().name})
        receipt = json.loads((self.out / 'FRAME_CACHE_RECEIPT.json').read_bytes())
        self.assertTrue(all(entry['full_decode_passes'] == 2 for entry in receipt['entries']))

    def test_parallel_resume_revalidates_pngs_without_any_decode(self):
        self.clips()
        complete = self.build(workers=3)
        self.store.gets.clear()
        with patch.object(media, 'extract_member', side_effect=AssertionError('must reuse verified entries')):
            self.assertEqual(complete, self.build(workers=2, resume=True))
        self.assertEqual(len([key for key in self.store.gets if key.endswith('.png')]), 12)

    def test_parallel_global_png_limit_counts_all_clips_before_writes(self):
        self.clips()
        full = self.build()
        cap = full['png_bytes'] // 3 + 1
        self.reset_output('limited')
        with self.assertRaisesRegex(media.CacheError, 'byte cap'):
            self.build(workers=3, max_bytes=cap)
        pngs = [raw for key, raw in self.store.objects.items() if key.startswith(self.prefix) and key.endswith('.png')]
        self.assertEqual(len(pngs), 4)
        self.assertLessEqual(sum(map(len, pngs)), cap)
        self.assertFalse((self.out / 'CACHE_COMPLETE.json').exists())
        self.assertEqual(len([key for key in self.store.keys(self.prefix) if '/entries/' in key]), 1)

    def test_partial_parallel_write_resumes_identical_bytes(self):
        self.clips()
        put = self.store.put_verified
        def interrupt(key, raw):
            if key.endswith('01.png'):
                raise RuntimeError('simulated writer failure')
            put(key, raw)
        self.store.put_verified = interrupt
        with self.assertRaisesRegex(RuntimeError, 'writer failure'):
            self.build(workers=3)
        partial = deepcopy(self.store.objects)
        self.store.put_verified = put
        result = self.build(workers=3, resume=True)
        self.assertEqual(result['frame_count'], 12)
        self.assertTrue(all(self.store.objects[key] == raw for key, raw in partial.items()))

    def test_worker_failure_drains_readers_and_does_not_publish_completion(self):
        self.clips(2)
        actual = media._prepared_cache_entry
        barrier = threading.Barrier(2)
        active = set()
        lock = threading.Lock()
        def prepare(identity, *args, **kwargs):
            with lock:
                active.add(identity)
            try:
                barrier.wait(timeout=5)
                if identity.endswith('_0'):
                    raise RuntimeError('decoder failed')
                time.sleep(.05)
                return actual(identity, *args, **kwargs)
            finally:
                with lock:
                    active.remove(identity)
        with patch.object(media, '_prepared_cache_entry', side_effect=prepare):
            with self.assertRaisesRegex(RuntimeError, 'decoder failed'):
                self.build(workers=2)
        self.assertFalse(active)
        self.assertFalse(any(t.name.startswith('suite-frame-cache') for t in threading.enumerate()))
        self.assertEqual(self.store.keys(self.prefix), [self.prefix + 'BUILD_PLAN.json'])

    def test_invalid_workers_or_memory_budget_fails_before_output(self):
        self.clips()
        for workers in (0, 9, True, 1.5):
            with self.assertRaisesRegex(media.CacheError, 'workers'):
                self.build(workers=workers)
        with self.assertRaisesRegex(media.CacheError, 'memory budget'):
            self.build(workers=3, max_inflight_bytes=1)
        self.assertFalse(self.out.exists())
        self.assertFalse(self.store.puts)

    def test_memory_credit_limits_active_and_completed_uncommitted_jobs(self):
        started = []
        barrier = threading.Barrier(2)
        def prepare(job):
            started.append(job)
            if job < 2:
                barrier.wait(timeout=5)
            return bytearray(10)
        iterator = media._bounded_prepared_entries([(i, 10) for i in range(8)], prepare,
                                                    workers=8, max_inflight_bytes=20)
        first = next(iterator)
        # The coordinator has not committed the yielded first result, so no
        # third future may be admitted even if the second already completed.
        self.assertEqual(sorted(started), [0, 1])
        del first
        self.assertEqual(len(list(iterator)), 7)

    def test_workflow_forwards_explicit_parallel_limits_without_changing_request(self):
        from deepprobe_vlm import suite_workflow as workflow, suite_runner as runner, suite_storage
        self.clips(2)
        original = self.root / 'accepted-originals.json'
        original.write_text('{}')
        request = {'benchmark': 'tempcompass_mc', 'frames': 4, 'source_manifest': runner.artifact(self.manifest),
            'accepted_decode_receipt': runner.artifact(original), 'output_dir': str(self.out),
            'media_root': str(self.root / 'media'), 'oss_prefix': self.prefix, 'max_png_bytes': 1_000_000,
            'metadata_reserve_bytes': 128 * 1024**2}
        before = deepcopy(request)
        environment = {'suite_id': 'fixture', 'storage': {'roots': [str(self.root)]},
                       'tools': {'oss_config': str(self.root / 'unused-config')}}
        budget = Mock()
        budget.reserve.return_value = 'own-token'
        with patch.object(suite_storage, 'budget_for', return_value=budget), \
             patch.object(media, 'OSSStore', return_value=self.store), \
             patch.object(workflow, 'source_catalog', return_value=[]), \
             patch.object(workflow, 'bind_legacy_archive_identities', return_value={}), \
             patch.object(media, 'create_source_binding', return_value=deepcopy(self.binding)), \
             patch.object(media, '_bound_cache_inventory'), \
             patch.object(media, 'build_frame_cache', wraps=media.build_frame_cache) as builder:
            _, complete = workflow.prepare_benchmark(environment, request, workers=2, max_inflight_bytes=4 * 1024**3)
        self.assertEqual(complete['frame_count'], 8)
        self.assertEqual(builder.call_args.kwargs['workers'], 2)
        self.assertEqual(builder.call_args.kwargs['max_inflight_bytes'], 4 * 1024**3)
        self.assertEqual(before, request)
        budget.release.assert_called_once()


class ExactListMetadataTest(unittest.TestCase):
    prefix = media.PROJECT_PREFIX + 'exact/frames/'

    def store(self, pages):
        calls = []
        store = media.OSSStore.__new__(media.OSSStore)
        def list_objects(**kwargs):
            calls.append(kwargs)
            return pages[len(calls) - 1]
        store.bucket = SimpleNamespace(list_objects=list_objects)
        return store, calls

    def row(self, name, size=5, etag='"etag"'):
        return SimpleNamespace(key=self.prefix + name, size=size, etag=etag)

    def page(self, rows, truncated=False, marker=''):
        return SimpleNamespace(object_list=rows, is_truncated=truncated, next_marker=marker)

    def test_exact_paginated_mapping_strips_etags(self):
        first, second = self.row('a.png'), self.row('b.png')
        store, calls = self.store([self.page([first], True, first.key), self.page([second])])
        self.assertEqual(store.list_metadata(self.prefix), {row.key: {'size_bytes': 5, 'etag': 'etag'} for row in (first, second)})
        self.assertEqual(calls, [{'prefix': self.prefix, 'marker': '', 'max_keys': 1000},
                                 {'prefix': self.prefix, 'marker': first.key, 'max_keys': 1000}])

    def test_rejects_duplicate_escape_missing_metadata_and_bad_marker(self):
        row = self.row('a.png')
        cases = [
            [self.page([row], True, row.key), self.page([row])],
            [self.page([self.row('../escape')])],
            [self.page([SimpleNamespace(key=media.PROJECT_PREFIX + 'elsewhere/a', size=5, etag='x')])],
            [self.page([self.row('a', size=True)])],
            [self.page([self.row('a', etag='')])],
            [self.page([row], True, '')],
            [self.page([], True, self.prefix + 'unknown')],
            [self.page([row], True, self.prefix + 'skipped-key')],
        ]
        for pages in cases:
            store, _ = self.store(pages)
            with self.assertRaises(media.CacheError):
                store.list_metadata(self.prefix)
