"""CPU-only MVBench layout selection tests using synthetic placeholder files."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from deepprobe_vlm.manifest import ManifestError, validate_records


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mvbench_path_convert", ROOT / "scripts/data/convert.py")
CONVERT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERT)


class MVBenchMediaPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = SimpleNamespace(task="action_antonym", video_prefix="video", media_root=self.root)
        self.row = {"video": "12345.webm", "question": "Which action?",
                    "candidates": ["Opening", "Closing"], "answer": "Closing"}

    def media(self, relative_path):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-placeholder-not-video")
        return path

    def convert(self, row=None):
        return list(CONVERT.convert_mvbench(self.row if row is None else row, self.args, {}))[0]

    def test_official_and_legacy_layout_preserve_identity_and_qa(self):
        for folder in ("ssv2_video", "ssv2_video_mp4"):
            with self.subTest(folder=folder):
                relative_path = f"video/{folder}/12345.webm"
                path = self.media(relative_path)
                converted = self.convert()
                self.assertEqual(converted, {
                    "sample_id": "mvbench:action_antonym:42ecbb484712df28e8278cf6",
                    "source_video_id": "ssv2:12345", "video_path": relative_path,
                    "question": "Which action?", "choices": ["Opening", "Closing"],
                    "answer": "B", "split": "test", "dataset": "mvbench", "task_type": "action_antonym",
                })
                self.assertEqual(validate_records([converted], self.root), [converted])
                path.unlink()

    def test_both_layouts_present_are_ambiguous(self):
        for folder in ("ssv2_video", "ssv2_video_mp4"):
            self.media(f"video/{folder}/12345.webm")
        with self.assertRaisesRegex(ManifestError, "ambiguous MVBench SSV2"):
            self.convert()

    def test_missing_exact_filename_fails_without_extension_substitution(self):
        for folder in ("ssv2_video", "ssv2_video_mp4"):
            self.media(f"video/{folder}/12345.mp4")
        converted = self.convert()
        with self.assertRaisesRegex(ManifestError, "missing media file"):
            validate_records([converted], self.root)

    def test_directory_is_not_a_media_candidate(self):
        (self.root / "video/ssv2_video/12345.webm").mkdir(parents=True)
        legacy = self.media("video/ssv2_video_mp4/12345.webm")
        self.assertEqual(self.convert()["video_path"], "video/ssv2_video_mp4/12345.webm")
        legacy.unlink()
        with self.assertRaisesRegex(ManifestError, "missing media file"):
            validate_records([self.convert()], self.root)

    def test_path_traversal_is_rejected(self):
        with self.assertRaises(ManifestError):
            self.convert({**self.row, "video": "../12345.webm"})

    def test_symlink_escape_is_rejected_for_each_layout(self):
        with tempfile.TemporaryDirectory() as outside:
            for folder in ("ssv2_video", "ssv2_video_mp4"):
                with self.subTest(folder=folder):
                    link = self.root / "video" / folder
                    link.parent.mkdir(exist_ok=True)
                    link.symlink_to(outside, target_is_directory=True)
                    with self.assertRaisesRegex(ManifestError, "escapes media_root"):
                        self.convert()
                    link.unlink()

    def test_other_task_retains_raw_video_and_official_bounds(self):
        self.args.task = "action_sequence"
        row = {**self.row, "video": "ABCDE.mp4", "start": 2, "end": 4}
        converted = self.convert(row)
        self.assertEqual(converted["video_path"], "video/star/Charades_v1_480/ABCDE.mp4")
        self.assertEqual(converted["source_video_id"], "charades:ABCDE")
        self.assertEqual((converted["start_seconds"], converted["end_seconds"]), (2.0, 4.0))


if __name__ == "__main__":
    unittest.main()
