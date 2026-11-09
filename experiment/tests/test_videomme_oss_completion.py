"""Offline bounded ZIP/range, immutable writes and real PyAV sampling tests."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings
import zipfile

HERE = Path(__file__).resolve()
WORKER = HERE.with_name('complete_videomme_from_oss.py')
if not WORKER.exists(): WORKER = HERE.parents[1] / 'scripts/operations/complete_videomme_from_oss.py'
SPEC = importlib.util.spec_from_file_location('vmoss', WORKER)
W = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(W)


class Store:
    def __init__(self, raw): self.raw, self.calls = raw, []
    def range(self, key, start, end, etag, total):
        assert key == 'object' and etag == 'fixed' and total == len(self.raw)
        assert 0 <= start <= end < total
        self.calls.append((start, end))
        return self.raw[start:end+1]


def fixture(payload=b'original video bytes', duplicates=False):
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_STORED) as z:
            z.writestr('data/abcdefghijk.mp4', payload)
            if duplicates: z.writestr('data/abcdefghijk.mp4', payload)
    raw = buf.getvalue()
    with zipfile.ZipFile(io.BytesIO(raw)) as z: info = z.infolist()[0]
    member = {'path': info.filename, 'size_bytes': info.file_size, 'compressed_bytes': info.compress_size, 'crc32': info.CRC}
    return raw, {'key': 'object', 'etag': 'fixed'}, {'size_bytes': len(raw)}, member


class BoundedZipTests(unittest.TestCase):
    def test_random_seek_and_bounded_ranges_reconstruct_exact_bytes(self):
        raw = bytes(range(128)); store = Store(raw)
        with patch.object(W, 'BLOCK', 7):
            reader = W.RangeReader(store, 'object', len(raw), 'fixed')
            self.assertEqual(reader.read(11), raw[:11]); reader.seek(48)
            self.assertEqual(reader.read(13), raw[48:61]); reader.seek(-8, 2)
            self.assertEqual(reader.read(), raw[-8:]); self.assertEqual(reader.read(), b'')
        self.assertTrue(all(end-start+1 <= 7 for start, end in store.calls))

    def test_invalid_seek_fails_without_range(self):
        store = Store(b'abc'); reader = W.RangeReader(store, 'object', 3, 'fixed')
        for position in [-1, 4]:
            with self.assertRaises(W.SafeError): reader.seek(position)
        self.assertFalse(store.calls)

    def test_unbounded_archive_read_is_rejected(self):
        store = Store(b'x' * 100)
        with patch.object(W, 'BLOCK', 4):
            with self.assertRaisesRegex(W.SafeError, 'unbounded_zip_read'):
                W.RangeReader(store, 'object', 100, 'fixed').read()
        self.assertFalse(store.calls)

    def test_member_crc_length_and_full_source_sha(self):
        raw, checked, spec, member = fixture()
        stream, sha = W.extract_member(Store(raw), checked, spec, member)
        with stream: self.assertEqual(stream.read(), b'original video bytes')
        self.assertEqual(sha, hashlib.sha256(b'original video bytes').hexdigest())

    def test_corrupt_member_crc_rejected(self):
        raw, checked, spec, member = fixture()
        changed = raw.replace(b'original video bytes', b'corruptx video bytes')
        self.assertEqual(len(raw), len(changed))
        with self.assertRaises(zipfile.BadZipFile): W.extract_member(Store(changed), checked, spec, member)

    def test_duplicate_member_rejected(self):
        raw, checked, spec, member = fixture(duplicates=True)
        with self.assertRaisesRegex(W.SafeError, 'zip_member_missing_or_duplicate'):
            W.extract_member(Store(raw), checked, spec, member)

    def test_each_pinned_member_metadata_field_is_enforced(self):
        raw, checked, spec, member = fixture()
        for key in ('size_bytes', 'compressed_bytes', 'crc32'):
            with self.subTest(key=key), self.assertRaisesRegex(W.SafeError, 'zip_member_metadata_changed'):
                W.extract_member(Store(raw), checked, spec, {**member, key: member[key]+1})

    def test_member_ram_limit_fails_before_network(self):
        raw, checked, spec, member = fixture(); store = Store(raw)
        with patch.object(W, 'MAX_MEMBER', 1), self.assertRaisesRegex(W.SafeError, 'member_memory_limit'):
            W.extract_member(store, checked, spec, member)
        self.assertFalse(store.calls)

    def test_oss_range_rejects_wrong_range_or_length(self):
        for status, header, raw in [(200, 'bytes 1-2/4', b'ab'), (206, 'bytes 0-1/4', b'ab'), (206, 'bytes 1-2/4', b'a')]:
            stream = io.BytesIO(raw); stream.status = status; stream.headers = {'Content-Range': header}
            store = W.OSSStore.__new__(W.OSSStore)
            store.bucket = SimpleNamespace(get_object=lambda *a, **k: stream)
            with self.assertRaises(W.SafeError): store.range('key', 1, 2, 'pinned', 4)

    def test_range_sends_if_match_and_returns_exact_body(self):
        calls = []; stream = io.BytesIO(b'ab'); stream.status = 206; stream.headers = {'Content-Range': 'bytes 1-2/4'}
        def get(*args, **kwargs): calls.append((args, kwargs)); return stream
        store = W.OSSStore.__new__(W.OSSStore); store.bucket = SimpleNamespace(get_object=get)
        self.assertEqual(store.range('key', 1, 2, 'pinned', 4), b'ab')
        self.assertEqual(calls[0][1]['headers'], {'If-Match': 'pinned'})


class FilesAndBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve(); (self.root/'frames').mkdir()
        self.addCleanup(patch.stopall); patch.object(W, 'MEDIA', self.root).start()

    def test_existing_bytes_are_never_overwritten(self):
        path = self.root/'proof'; W.put_file(path, b'original'); W.put_file(path, b'original')
        with self.assertRaisesRegex(W.SafeError, 'local_immutable_conflict'): W.put_file(path, b'changed')
        self.assertEqual(path.read_bytes(), b'original')

    def test_symlink_rejected(self):
        path = self.root/'proof'; path.write_bytes(b'keep'); alias = self.root/'alias'; alias.symlink_to(path)
        with self.assertRaisesRegex(W.SafeError, 'unsafe_local_path'): W.put_file(alias, b'keep')

    def test_external_file_aliases_each_charged_and_internal_alias_not_twice(self):
        outside = self.root/'external'; outside.write_bytes(b'0123456789')
        inside = self.root/'frames/local'; inside.write_bytes(b'123')
        (self.root/'frames/alias1').symlink_to(outside); (self.root/'frames/alias2').symlink_to(outside)
        (self.root/'frames/internal').symlink_to(inside)
        self.assertEqual(W.tree_bytes(self.root/'frames'), 23)

    def test_external_directory_symlink_fails_closed(self):
        outside = self.root/'outside'; outside.mkdir(); (self.root/'frames/alias').symlink_to(outside)
        with self.assertRaisesRegex(W.SafeError, 'unaccounted_external_directory_symlink'):
            W.tree_bytes(self.root/'frames')

    def test_reused_remaining_source_provenance_is_bound(self):
        spec={'path':'archive.zip','sha256':'a'*64}; member={'path':'data/abcdefghijk.mp4','size_bytes':10}
        checked={'etag':'etag','receipt_sha256':'b'*64}
        row={'source_video_id':'youtube:abcdefghijk','original_size_bytes':10,'archive_path':'archive.zip',
             'archive_sha256':'a'*64,'member_path':member['path'],'repo_id':'lmms-lab/Video-MME','revision':W.REVISION,
             'oss_archive_etag':'etag','oss_archive_receipt_sha256':'b'*64,'cache_implementation_sha256':W.sha(W.__file__),
             'original_sha256':'c'*64}
        W.validate_source_binding(row,spec,member,checked)
        for key in row:
            with self.subTest(key=key), self.assertRaisesRegex(W.SafeError,'existing_new_source_provenance_changed'):
                W.validate_source_binding({**row,key:'changed'},spec,member,checked)

    def test_png_cap_rejects_before_any_excess_write(self):
        budget = W.FrameBudget(5); first = self.root/'frames/first'
        budget.write(first, b'1234'); budget.write(first, b'1234')
        with self.assertRaisesRegex(W.SafeError, 'png_total_capacity_limit'):
            budget.write(self.root/'frames/second', b'12')
        self.assertEqual(budget.used, 4); self.assertFalse((self.root/'frames/second').exists())

    def test_source_boundary_capacity_is_checked(self):
        budget = W.FrameBudget(5)
        with patch.object(W, 'tree_bytes', return_value=50), patch.object(W, 'LIMIT', 99):
            with self.assertRaisesRegex(W.SafeError, 'source_boundary_root_capacity_failed'): budget.source_admit()

    def test_partial_cache_cannot_emit_completion(self):
        (self.root/'entries').mkdir()
        with patch.object(W, 'tree_bytes', return_value=0):
            with self.assertRaisesRegex(W.SafeError, 'incomplete_source_set'):
                W.finalize(None, [], set(), {'frame_cap_bytes': 10}, {})
        self.assertFalse((self.root/'FRAME_CACHE_COMPLETE.json').exists())
        self.assertFalse((self.root/'VIDEOMME_BUCKET_MEDIA_COMPLETE.json').exists())

    def test_conflicting_oss_object_is_not_overwritten(self):
        store = W.OSSStore.__new__(W.OSSStore); store.get = lambda key: b'original'
        store.bucket = SimpleNamespace(put_object=lambda *a, **k: self.fail('must not overwrite'))
        with self.assertRaisesRegex(W.SafeError, 'oss_immutable_conflict'): store.put_verified('key', b'changed')


try:
    import av
    from PIL import Image
    HAVE_AV = True
except ImportError: HAVE_AV = False


@unittest.skipUnless(HAVE_AV, 'real PyAV tests run on the isolated remote CPU runtime')
class RealDecodeTests(unittest.TestCase):
    @staticmethod
    def video(count=13):
        out = io.BytesIO()
        with av.open(out, mode='w', format='mp4') as container:
            stream = container.add_stream('mpeg4', rate=10); stream.width = 32; stream.height = 24; stream.pix_fmt = 'yuv420p'
            for i in range(count):
                frame = av.VideoFrame.from_image(Image.new('RGB', (32,24), ((i*17)%256,(i*31)%256,(i*47)%256)))
                for packet in stream.encode(frame): container.mux(packet)
            for packet in stream.encode(): container.mux(packet)
        return out.getvalue()

    def test_in_memory_fast_path_is_full_sequential_uniform8(self):
        raw = self.video(); selected, count, indices, declared, passes = W.decode_selected(io.BytesIO(raw))
        self.assertEqual((count, declared, passes), (13,13,1)); self.assertEqual(indices, [round(i*12/7) for i in range(8)])
        with av.open(io.BytesIO(raw)) as container:
            originals = [f.to_image().convert('RGB').tobytes() for f in container.decode(video=0)]
        self.assertEqual([selected[i].tobytes() for i in indices], [originals[i] for i in indices])

    def test_wrong_or_missing_declared_count_runs_two_complete_passes(self):
        raw = self.video(); expected = W.decode_selected(io.BytesIO(raw))
        for declared in (0, 5, 99):
            with self.subTest(declared=declared):
                actual = W.decode_selected(io.BytesIO(raw), declared_count_override=declared)
                self.assertEqual(actual[1:3], expected[1:3]); self.assertEqual(actual[4], 2)
                self.assertEqual([actual[0][i].tobytes() for i in actual[2]], [expected[0][i].tobytes() for i in expected[2]])

    def test_short_video_repeated_indices_keep_eight_slots(self):
        _, count, indices, _, _ = W.decode_selected(io.BytesIO(self.video(3)))
        self.assertEqual(count,3); self.assertEqual(len(indices),8); self.assertEqual(len(set(indices)),3)

    def test_same_frames_as_frozen_v4_disk_decoder(self):
        old = Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/benchmark-media-20261012/videomme/benchmark_media_cache_v4.py')
        if not old.exists(): self.skipTest('remote frozen original available only on Ali')
        self.assertEqual(W.sha(old), 'dd8993522994bc262d1b1c4b15bf528937acffb00a9fd0711a0ffa0be043eed9')
        spec = importlib.util.spec_from_file_location('frozen_v4', old); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        raw = self.video()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'synthetic.mp4'; path.write_bytes(raw); expected = module.decode_selected(path)
        actual = W.decode_selected(io.BytesIO(raw))
        self.assertEqual(actual[1:], expected[1:])
        self.assertEqual([actual[0][i].tobytes() for i in actual[2]], [expected[0][i].tobytes() for i in expected[2]])


if __name__ == '__main__': unittest.main()
