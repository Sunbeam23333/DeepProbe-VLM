"""CPU-only data safety and conversion tests; all fixtures are synthetic."""
import importlib.util
import io
import json
import stat
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from deepprobe_vlm.manifest import (
    ManifestError, load_manifest, resolve_media_path, safe_relative_path,
    validate_disjoint, validate_records,
)

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location("data_" + name, ROOT / "scripts/data" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONVERT = script("convert")
SPLIT = script("split")
EXTRACT = script("safe_extract")
DOWNLOAD = script("download_hf")
TOKEN_BUDGET = script("check_token_budget")


def record(**kwargs):
    return {"sample_id": "synthetic:q1", "source_video_id": "synthetic:video1",
            "video_path": "video.mp4", "question": "Which object?", "answer": "A",
            "choices": ["Circle", "Square"], "split": "train", **kwargs}


class ManifestTests(unittest.TestCase):
    def test_valid_media_and_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "video.mp4").write_bytes(b"synthetic-placeholder-not-video")
            (root / "manifest.jsonl").write_text(json.dumps(record()) + "\n")
            self.assertEqual(len(load_manifest(root / "manifest.jsonl", root)), 1)

    def test_missing_media_is_hard_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ManifestError):
                validate_records([record()], d)

    def test_root_required(self):
        with self.assertRaises(ManifestError):
            validate_records([record()])

    def test_whitespace_cannot_bypass_source_check(self):
        with self.assertRaises(ManifestError):
            validate_records([record(source_video_id=" source:1")], check_media=False)

    def test_path_traversal_absolute_windows_url(self):
        for value in ("../x", "/etc/passwd", "a/../../x", "C:\\temp\\x", "C:/x", "https://x/a", "~/.x", "", "."):
            with self.subTest(value=value), self.assertRaises(ManifestError):
                safe_relative_path(value)

    def test_symlink_escape(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as outside:
            root = Path(d)
            (root / "link").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ManifestError):
                resolve_media_path("link/test.mp4", root)

    def test_duplicate_ids(self):
        with self.assertRaisesRegex(ManifestError, "duplicate"):
            validate_records([record(), record()], check_media=False)

    def test_multiple_questions_share_source_within_split(self):
        rows = [record(), record(sample_id="synthetic:q2")]
        self.assertEqual(len(validate_records(rows, check_media=False)), 2)

    def test_source_leakage(self):
        with self.assertRaisesRegex(ManifestError, "source-video leakage"):
            validate_disjoint([record()], [record(sample_id="q2", split="test", video_path="other.mp4")])

    def test_media_leakage_under_different_id(self):
        with self.assertRaisesRegex(ManifestError, "same media"):
            validate_disjoint([record()], [record(sample_id="q2", source_video_id="bad:new", split="test")])

    def test_media_exclusive_and_mc_canonical(self):
        for item in (record(image_paths=["frame.jpg"]), record(answer="Circle"), record(choices=[])):
            with self.assertRaises(ManifestError):
                validate_records([item], check_media=False)

    def test_invalid_bounds(self):
        for bounds in ({"start_seconds": 1}, {"start_seconds": 5, "end_seconds": 3},
                       {"start_seconds": 0, "end_seconds": float("nan")}):
            with self.assertRaises(ManifestError):
                validate_records([record(**bounds)], check_media=False)

    def test_conversation_expansion_preserves_source(self):
        row = {"id": "ABCDE", "video": "academic_source/Charades/ABCDE.mp4", "conversations": [
            {"from": "human", "value": "<image>\nFirst?"}, {"from": "gpt", "value": "Circle"},
            {"from": "human", "value": "Second?"}, {"from": "gpt", "value": "Square"}]}
        rows = list(CONVERT.convert_llava(row, SimpleNamespace(input=Path("ann.json"), split="train"), {}))
        self.assertEqual({r["source_video_id"] for r in rows}, {"charades:ABCDE"})
        self.assertNotEqual(rows[0]["sample_id"], rows[1]["sample_id"])
        self.assertEqual(rows[0]["question"], "First?")

    def test_youtube_normalization(self):
        self.assertEqual(CONVERT.source_id("youtube/ytb_4V1vnSvwqvA.mp4"), "youtube:4V1vnSvwqvA")
        self.assertEqual(CONVERT.source_id("youtube/ytb_4V1vnSvwqvA_12_30.mp4"), "youtube:4V1vnSvwqvA")
        with self.assertRaises(ValueError):
            CONVERT.source_id("unknown/question_001.mp4")

    def test_videomme_uses_youtube_not_numeric_video_id(self):
        row = {"video_id": "001", "videoID": "abcdefghijk", "question_id": "001-1", "question": "Q?",
               "options": ["A. First", "B. Second"], "answer": "B", "duration": "short", "task_type": "count", "domain": "test"}
        out = list(CONVERT.convert_videomme(row, SimpleNamespace(video_prefix="videos"), {}))[0]
        self.assertEqual(out["source_video_id"], "youtube:abcdefghijk")
        self.assertEqual(out["choices"], ["First", "Second"])
        self.assertNotIn("B", out["question"])

    def test_source_group_split_and_exclusion(self):
        rows = [record(sample_id=f"q{i}-{q}", source_video_id=f"source:{i}", video_path=f"video{i}.mp4")
                for i in range(100) for q in range(2)]
        train, val, omitted = SPLIT.group_split(rows, .2, 42, {"source:0"})
        self.assertTrue(train and val)
        self.assertEqual(len(omitted), 2)
        self.assertFalse({r["source_video_id"] for r in train} & {r["source_video_id"] for r in val})
        self.assertEqual((train, val, omitted), SPLIT.group_split(rows, .2, 42, {"source:0"}))

    def test_no_repartitioning_benchmark(self):
        with self.assertRaisesRegex(ValueError, "never use evaluation"):
            SPLIT.group_split([record(split="test")])

    def test_unknown_download_size_and_unmatched_pattern(self):
        item = SimpleNamespace(rfilename="x.json", size=None, blob_id="a", lfs=None)
        with self.assertRaisesRegex(ValueError, "missing file size"):
            DOWNLOAD.choose_files([item], ["*.json"])
        with self.assertRaisesRegex(ValueError, "matched no files"):
            DOWNLOAD.choose_files([item], ["*.mp4"])

    def test_archive_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.zip"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("../escape.txt", "bad")
            with self.assertRaises(ManifestError):
                EXTRACT.extract_archive(path, Path(d) / "out", 100, True)

    def test_archive_symlink_and_cap(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.tar"
            with tarfile.open(path, "w") as t:
                entry = tarfile.TarInfo("link")
                entry.type = tarfile.SYMTYPE
                entry.linkname = "/tmp/outside"
                t.addfile(entry)
            with self.assertRaisesRegex(ValueError, "links/special"):
                EXTRACT.extract_archive(path, Path(d) / "out", 100, True)
            path = Path(d) / "large.zip"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("large", "123456")
            with self.assertRaisesRegex(ValueError, "exceeds cap"):
                EXTRACT.extract_archive(path, Path(d) / "out", 5, True)

    def test_archive_dry_run_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            path, output = Path(d) / "valid.zip", Path(d) / "out"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("sub/file", "data")
            EXTRACT.extract_archive(path, output, 100)
            self.assertFalse(output.exists())
            EXTRACT.extract_archive(path, output, 100, True)
            self.assertEqual((output / "sub/file").read_text(), "data")
            with self.assertRaises(FileExistsError):
                EXTRACT.extract_archive(path, output, 100, True)

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Optional Pillow dependency not installed")
    def test_frame_cache_preserves_source_and_reuses_decode(self):
        from PIL import Image
        cache = script("cache_frames")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for index in range(2):
                Image.new("RGB", (4, 4), (index * 100, 0, 0)).save(root / f"frame{index}.png")
            first = record()
            first.pop("video_path")
            first["image_paths"] = ["frame0.png", "frame1.png"]
            first.update(start_seconds=1.0, end_seconds=3.0, bounds_applied_to_cached_frames=True)
            second = {**first, "sample_id": "synthetic:q2"}
            output = root / "cache"
            summary = cache.build_cache([first, second], root, output, 2, 1000000)
            self.assertEqual(summary["unique_decodes"], 1)
            rows = load_manifest(output / "manifest.jsonl", output)
            self.assertEqual(rows[0]["source_video_id"], first["source_video_id"])
            self.assertEqual(rows[0]["start_seconds"], 1.0)
            self.assertEqual(rows[0]["image_paths"], rows[1]["image_paths"])
            self.assertEqual(len(rows[0]["frame_cache_fingerprint"]), 64)
            with self.assertRaises(FileExistsError):
                cache.build_cache([first], root, output, 2, 1000000)

    def test_token_budget_uses_exact_prompt_answer_and_special_tokens(self):
        from deepprobe_vlm.data import MultimodalCollator

        class TinyTokenizer:
            eos_token_id, bos_token_id, pad_token_id = 2, 1, 0

            def encode(self, text, add_special_tokens=False):
                return [ord(c) + 3 for c in text]

        collator = MultimodalCollator(TinyTokenizer(), num_visual_tokens=8, max_text_tokens=10000)
        item = record()
        prefix, suffix, answer = collator.tokenize_record(item)
        prompt = "Video frames in chronological order:\n\nQuestion: Which object?\nA. Circle\nB. Square\nAnswer with the option letter only.\nAnswer:"
        expected = 1 + len(prompt) + len(" A") + 1  # BOS + exact prompt + answer + EOS.
        self.assertEqual(sum(map(len, (prefix, suffix, answer))), expected)
        self.assertEqual(answer, TinyTokenizer().encode(" A") + [2])
        collator.max_text_tokens = expected
        passing = TOKEN_BUDGET.check_records([item], collator)
        self.assertTrue(passing["passed"])
        self.assertEqual(passing["observed_max_visual_plus_text_tokens"], expected + 8)
        collator.max_text_tokens = expected - 1
        failing = TOKEN_BUDGET.check_records([item, {**item, "sample_id": "synthetic:q2"}], collator)
        self.assertFalse(failing["passed"])
        self.assertEqual(failing["overflow_count"], 2)
        self.assertEqual(failing["overflow_samples"][0]["over_budget_by"], 1)
        self.assertEqual([x["sample_id"] for x in failing["overflow_samples"]], ["synthetic:q1", "synthetic:q2"])
        with self.assertRaisesRegex(ValueError, "no silent truncation"):
            collator.tokenize_record(item)

    def test_inference_token_precheck_excludes_gold_answer(self):
        from deepprobe_vlm.data import MultimodalCollator

        class TinyTokenizer:
            eos_token_id, bos_token_id, pad_token_id = 2, None, None

            def encode(self, text, add_special_tokens=False):
                return [ord(c) for c in text]

        collator = MultimodalCollator(TinyTokenizer(), num_visual_tokens=2, max_text_tokens=10000,
                                     include_answer=False)
        item = record()
        first = TOKEN_BUDGET.check_records([item], collator)
        second = TOKEN_BUDGET.check_records([{**item, "answer": "SECRET ANSWER" * 100}], collator)
        self.assertEqual(first["observed_text_tokens"], second["observed_text_tokens"])
        self.assertEqual(collator.tokenize_record(item)[2], [])
        self.assertEqual(collator.tokenizer.pad_token_id, 2)


if __name__ == "__main__":
    unittest.main()
