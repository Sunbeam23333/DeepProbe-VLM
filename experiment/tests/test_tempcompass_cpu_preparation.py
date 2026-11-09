"""Local-only source/QA binding, memory decode, archive and output fail-closed checks."""
import pytest
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'scripts/operations/prepare_tempcompass_cpu.py'
SPEC = importlib.util.spec_from_file_location('tempcompass_prepare', PATH)
W = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(W)
EVIDENCE = ROOT / 'docs/operations/evidence'
SOURCE = EVIDENCE / 'tempcompass-acquisition-20261017/source-freeze-v1'


def fixture_inputs():
    paths = {'rows': SOURCE / 'GITHUB_MC_ROWS_CANONICAL.json', 'source_map': SOURCE / 'OFFICIAL_CLIP_SOURCE_MAP.json',
             'meta': SOURCE / 'GITHUB_meta_info.json', 'git_binding': SOURCE / 'PINNED_GIT_SOURCE_BINDING.json',
             'archive_audit': EVIDENCE / 'tempcompass-acquisition-20261017/completion-check-20261021/ARCHIVE_MEMBER_AUDIT.json',
             'archive_terminal': EVIDENCE / 'tempcompass-acquisition-20261017/completion-check-20261021/control_TERMINAL.json'}
    for key, path in paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == W.PINS[key]
    return {k: json.loads(p.read_text()) for k, p in paths.items()}


class SourceAndQuestionTests(unittest.TestCase):
    @pytest.mark.private_evidence
    def test_all_real_frozen_questions_roundtrip_and_source_counts(self):
        inputs = fixture_inputs()
        rows = W.validate_bundle(inputs)
        self.assertEqual(len(rows), 1580)
        self.assertEqual(len({r['video_id'] for r in rows}), 410)
        self.assertEqual(len({s for r in rows for s in r['underlying_source_video_ids']}), 207)
        self.assertEqual(len({r['sample_id'] for r in rows}), 1580)
        for row, raw in zip(rows, inputs['rows']):
            self.assertEqual(row['original_question'], raw['question'])
            self.assertEqual(row['original_answer'], raw['answer'])
            self.assertEqual(len(row['image_paths']), 8)

    def test_composite_source_graph_joins_transitive_dependencies(self):
        entries = [{'video_id': 'a', 'underlying_source_video_ids': ['shutterstock:1']},
                   {'video_id': 'ab', 'underlying_source_video_ids': ['shutterstock:1', 'shutterstock:2']},
                   {'video_id': 'bc', 'underlying_source_video_ids': ['shutterstock:2', 'shutterstock:3']},
                   {'video_id': 'd', 'underlying_source_video_ids': ['shutterstock:4']}]
        result = W.source_components(entries)
        self.assertEqual(result['a'], result['ab'])
        self.assertEqual(result['a'], result['bc'])
        self.assertNotEqual(result['a'], result['d'])
        self.assertEqual(result, W.source_components(list(reversed(entries))))

    def test_unknown_namespace_and_duplicate_clip_fail(self):
        for entries in [[{'video_id': 'a', 'underlying_source_video_ids': ['youtube:123']}],
                        [{'video_id': 'a', 'underlying_source_video_ids': ['shutterstock:1']}] * 2]:
            with self.assertRaises(ValueError):
                W.source_components(entries)

    @pytest.mark.private_evidence
    def test_gold_and_choice_reordering_fail_before_media(self):
        for mutation in ('answer', 'question'):
            inputs = fixture_inputs()
            if mutation == 'answer':
                inputs['rows'][0]['answer'] = 'D. invented'
            else:
                inputs['rows'][0]['question'] = inputs['rows'][0]['question'].replace('A. ', 'B. ', 1)
            with self.assertRaises(ValueError):
                W.validate_bundle(inputs)

    @pytest.mark.private_evidence
    def test_missing_qa_dimension_count_or_member_fail(self):
        for mutation in ('qa', 'member', 'identity'):
            inputs = fixture_inputs()
            if mutation == 'qa':
                inputs['rows'].pop()
            elif mutation == 'member':
                inputs['archive_audit']['members'].pop()
            else:
                inputs['archive_audit']['members'][0]['name'] = 'videos/incorrect.mp4'
            with self.assertRaises(ValueError):
                W.validate_bundle(inputs)

    @pytest.mark.private_evidence
    def test_failed_archive_receipt_is_never_accepted(self):
        inputs = fixture_inputs()
        inputs['archive_audit']['full_oss_get_sha256_passed'] = False
        with self.assertRaisesRegex(ValueError, 'archive_audit_not_accepted'):
            W.validate_bundle(inputs)


class FakeImage:
    def __init__(self, index):
        self.index = index

    def convert(self, mode):
        assert mode == 'RGB'
        return self


class FakeFrame:
    width, height = 16, 8

    def __init__(self, index):
        self.index = index

    def to_image(self):
        return FakeImage(self.index)


class FakeContainer:
    def __init__(self, frames, late_error=False):
        self.frames, self.late_error = frames, late_error
        self.streams = SimpleNamespace(video=[SimpleNamespace(width=16, height=8)])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def decode(self, video):
        for i in range(self.frames):
            yield FakeFrame(i)
        if self.late_error:
            raise RuntimeError('late decode failure after last sampled frame')


class DecodeTests(unittest.TestCase):
    def setUp(self):
        # Synthetic decoding tests are portable; Linux RSS behavior is tested
        # separately instead of treating macOS ru_maxrss units as Linux units.
        guard = patch.object(W, 'memory_guard', lambda: None)
        guard.start()
        self.addCleanup(guard.stop)

    def test_two_full_passes_use_observed_count_and_exact_uniform_indices(self):
        calls = []
        def open_video(*args, **kwargs):
            calls.append(kwargs)
            return FakeContainer(10)
        images, count, indices, dims = W.decode_uniform8(io.BytesIO(b'video'), SimpleNamespace(open=open_video))
        self.assertEqual(len(calls), 2)
        self.assertEqual(indices, [0, 1, 3, 4, 5, 6, 8, 9])
        self.assertEqual([images[i].index for i in indices], indices)
        self.assertEqual((count, dims), (10, (16, 8)))

    def test_single_frame_still_has_eight_slots(self):
        images, count, indices, _ = W.decode_uniform8(io.BytesIO(b'video'), SimpleNamespace(open=lambda *a, **k: FakeContainer(1)))
        self.assertEqual(indices, [0] * 8)
        self.assertEqual(len(images), 1)

    def test_late_decode_failure_is_not_hidden_by_sampling(self):
        with self.assertRaisesRegex(RuntimeError, 'late decode failure'):
            W.decode_uniform8(io.BytesIO(b'video'), SimpleNamespace(open=lambda *a, **k: FakeContainer(10, True)))

    def test_mismatched_second_pass_and_empty_video_fail(self):
        containers = iter([FakeContainer(10), FakeContainer(11)])
        with self.assertRaisesRegex(ValueError, 'complete_decode_count_mismatch'):
            W.decode_uniform8(io.BytesIO(b'video'), SimpleNamespace(open=lambda *a, **k: next(containers)))
        with self.assertRaisesRegex(ValueError, 'invalid_frame_count'):
            W.decode_uniform8(io.BytesIO(b'video'), SimpleNamespace(open=lambda *a, **k: FakeContainer(0)))

    def test_pixel_and_stream_ambiguity_limits_fail_before_iteration(self):
        container = FakeContainer(5)
        container.streams.video[0].width = W.MAX_PIXELS
        with self.assertRaisesRegex(ValueError, 'declared_pixel_cap'):
            W.decode_uniform8(io.BytesIO(b'video'), SimpleNamespace(open=lambda *a, **k: container))
        container.streams.video.append(container.streams.video[0])
        with self.assertRaisesRegex(ValueError, 'ambiguous_video_streams'):
            W.decode_uniform8(io.BytesIO(b'video'), SimpleNamespace(open=lambda *a, **k: container))


def zip_fixture(names=('videos/123.mp4',)):
    memory = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with zipfile.ZipFile(memory, 'w', zipfile.ZIP_STORED) as z:
            for name in names:
                z.writestr(name, b'video fixture')
    memory.seek(0)
    z = zipfile.ZipFile(memory)
    members = [{'name': i.filename, 'bytes': i.file_size, 'compressed_bytes': i.compress_size,
                'crc32': i.CRC, 'header_offset': i.header_offset, 'compression': i.compress_type,
                'sha256': hashlib.sha256(b'video fixture').hexdigest()} for i in z.infolist()]
    return z, members


class ArchiveTests(unittest.TestCase):
    def test_exact_member_metadata_sha_and_crc(self):
        z, members = zip_fixture()
        with z:
            infos = W.checked_zip_infos(z, members)
            with W.read_member(z, infos[members[0]['name']], members[0]) as source:
                self.assertEqual(source.read(), b'video fixture')
            changed = {**members[0], 'sha256': '0' * 64}
            with self.assertRaisesRegex(ValueError, 'member_sha_changed'):
                W.read_member(z, infos[members[0]['name']], changed)

    def test_duplicate_and_path_escape_rejected(self):
        for names in [('videos/123.mp4', 'videos/123.mp4'), ('../123.mp4',), ('/123.mp4',), ('videos\\123.mp4',)]:
            z, members = zip_fixture(names)
            with z, self.assertRaises(ValueError):
                W.checked_zip_infos(z, members)

    def test_member_crc_size_header_and_compression_are_pinned(self):
        for field in ('bytes', 'compressed_bytes', 'crc32', 'header_offset', 'compression'):
            z, members = zip_fixture()
            members[0][field] += 1
            with z, self.assertRaisesRegex(ValueError, 'zip_member_metadata_changed'):
                W.checked_zip_infos(z, members)

    def test_symlink_and_special_file_rejected(self):
        for mode in (stat.S_IFLNK, stat.S_IFIFO):
            z, members = zip_fixture()
            z.infolist()[0].external_attr = (mode | 0o644) << 16
            with z, self.assertRaises(ValueError):
                W.checked_zip_infos(z, members)


class OutputTests(unittest.TestCase):
    def test_existing_file_not_overwritten_and_png_cap_precedes_write(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(W, 'OUTPUT', Path(directory).resolve()):
            store = SimpleNamespace(put_verified=lambda *a: None)
            out = W.Artifacts(store, None)
            out.put('proof', b'original')
            with self.assertRaisesRegex(ValueError, 'output_collision'):
                out.put('proof', b'changed')
            with patch.object(W, 'FRAME_CAP', 2), self.assertRaisesRegex(ValueError, 'png_capacity_failed'):
                out.put('frames/123/00.png', b'long', frame=True)
            self.assertFalse((W.OUTPUT / 'frames').exists())
            self.assertEqual((W.OUTPUT / 'proof').read_bytes(), b'original')

    def test_parent_symlink_and_relative_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            (base / 'real').mkdir()
            (base / 'alias').symlink_to(base / 'real')
            for path in (base / 'alias/x', base / '../escape', Path('relative')):
                with self.assertRaises(ValueError):
                    W.safe(path)

    def test_failed_attempt_never_writes_to_preexisting_output(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            output = base / 'old'
            output.mkdir()
            config = base / 'config.json'
            config.write_text('{}')
            with patch.object(W, 'P1', base), patch.object(W, 'OUTPUT', output), patch.object(W, 'OWN_OUTPUT', False), \
                 patch.object(W, 'prepare', side_effect=W.PreparationError('new_output_root_exists')), \
                 patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': ''}):
                self.assertEqual(W.main(['--config', str(config), '--config-sha256', W.sha(config)]), 1)
            self.assertEqual(list(output.iterdir()), [])


class BoundedResourceTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(W, 'memory_guard', lambda: None)
        guard.start()
        self.addCleanup(guard.stop)

    @staticmethod
    def frames(count=8):
        return [{'informative': True, 'phash': '1234567890abcdef',
                 'dhash': '4567890abcdef123', 'rgb_sha256': '0' * 64} for _ in range(count)]

    def inputs(self, train_count=3, benchmark_count=2, frames=8):
        train = [{'relative_path': f'{i}.mp4', 'source_video_id': f'train:{i}', 'frames': self.frames(frames)}
                 for i in range(train_count)]
        benchmark = [{'source_video_id': f'test:{i}', 'frames': self.frames(frames)}
                     for i in range(benchmark_count)]
        return train, benchmark

    def test_decode_checks_memory_inside_first_complete_pass(self):
        calls = []
        def guard():
            calls.append(True)
            if len(calls) == 3:
                raise W.PreparationError('rss_cap_exceeded')
        with patch.object(W, 'memory_guard', guard), self.assertRaisesRegex(ValueError, 'rss_cap_exceeded'):
            W.decode_uniform8(io.BytesIO(b'video'), SimpleNamespace(open=lambda *a, **k: FakeContainer(1000)))
        self.assertEqual(len(calls), 3)

    def test_content_pair_cap_raises_instead_of_truncating(self):
        train, benchmark = self.inputs(frames=1)
        policy = {'phash_max_hamming': 6, 'dhash_max_hamming': 10}
        with patch.object(W, 'CONTENT_PAIR_CAP', 5), self.assertRaisesRegex(ValueError, 'content_pair_cap_exceeded'):
            W.bounded_candidate_pairs(train, benchmark, policy)
        with patch.object(W, 'CONTENT_PAIR_CAP', 6):
            self.assertEqual(len(W.bounded_candidate_pairs(train, benchmark, policy)), 6)

    def test_content_match_cap_raises_instead_of_truncating(self):
        train, benchmark = self.inputs(train_count=1, benchmark_count=1)
        policy = {'phash_max_hamming': 6, 'dhash_max_hamming': 10}
        with patch.object(W, 'CONTENT_MATCH_CAP', 63), self.assertRaisesRegex(ValueError, 'content_frame_match_cap_exceeded'):
            W.bounded_candidate_pairs(train, benchmark, policy)
        with patch.object(W, 'CONTENT_MATCH_CAP', 64):
            pairs = W.bounded_candidate_pairs(train, benchmark, policy)
            self.assertEqual(len(pairs[0]['frame_matches']), 64)

    def test_content_search_keeps_pinned_exact_results_and_checks_memory(self):
        spec = importlib.util.spec_from_file_location('pinned_scanner', ROOT / 'scripts/operations/audit_benchmark_content.py')
        scanner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scanner)
        self.assertEqual(hashlib.sha256(Path(scanner.__file__).read_bytes()).hexdigest(), W.PINS['content_helper'])
        train, benchmark = self.inputs()
        benchmark[0]['frames'][0]['phash'] = 'ffffffffffffffff'
        benchmark[0]['frames'][1]['informative'] = False
        with patch.object(W, 'memory_guard') as guard:
            bounded = W.bounded_candidate_pairs(train, benchmark, scanner.POLICY)
        self.assertEqual(bounded, scanner.candidate_pairs(train, benchmark))
        self.assertGreater(guard.call_count, 16)

    def test_bounded_json_exact_bytes_unicode_and_limit(self):
        value = {'z': ['中文', '😀', '\\', 'x' * 40000], 'a': True}
        raw = W.encode(value)
        self.assertEqual(W.bounded_encode(value, len(raw)), raw)
        with self.assertRaisesRegex(ValueError, 'control_capacity_failed'):
            W.bounded_encode(value, len(raw) - 1)

    def test_control_cap_prevents_any_local_or_oss_write(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(W, 'OUTPUT', Path(directory).resolve()):
            store = SimpleNamespace(put_verified=lambda *a: self.fail('OSS write before cap'))
            out = W.Artifacts(store, None)
            with patch.object(W, 'CONTROL_OBJECT_CAP', 100), self.assertRaisesRegex(ValueError, 'control_capacity_failed'):
                out.json('too-large.json', {'content': 'x' * 1000})
            self.assertEqual(list(W.OUTPUT.iterdir()), [])

    def test_search_cap_failure_has_failed_receipt_and_no_ready_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            output = base / 'new'
            config = base / 'config.json'
            config.write_text('{}')
            def limited(*args):
                output.mkdir()
                W.OWN_OUTPUT = True
                train, benchmark = self.inputs()
                W.bounded_candidate_pairs(train, benchmark, {'phash_max_hamming': 6, 'dhash_max_hamming': 10})
            with patch.object(W, 'P1', base), patch.object(W, 'OUTPUT', output), \
                 patch.object(W, 'prepare', side_effect=limited), patch.object(W, 'CONTENT_MATCH_CAP', 1), \
                 patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': ''}):
                self.assertEqual(W.main(['--config', str(config), '--config-sha256', W.sha(config)]), 1)
            self.assertEqual({p.name for p in output.iterdir()}, {'FAILED.json'})
            self.assertEqual(json.loads((output / 'FAILED.json').read_text())['code'], 'content_frame_match_cap_exceeded')


if __name__ == '__main__':
    unittest.main()
