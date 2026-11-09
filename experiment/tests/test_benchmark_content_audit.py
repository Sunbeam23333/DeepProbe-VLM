"""Overlap audit tests use generated pixels/media, never benchmark claims."""
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("content_audit", ROOT / "scripts/operations/audit_benchmark_content.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ContentAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy
            from PIL import Image
        except ImportError:
            raise unittest.SkipTest("NumPy/Pillow required for image content tests")

    def image(self, seed=1):
        import numpy as np
        from PIL import Image
        array = np.random.default_rng(seed).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        return Image.fromarray(array, "RGB")

    def frame(self, seed=1):
        return audit.fingerprint_image(self.image(seed))

    def test_same_and_lossy_reencode_detected_unrelated_not_detected(self):
        from PIL import Image
        import io
        original = self.image()
        stream = io.BytesIO()
        original.save(stream, format="JPEG", quality=85)
        stream.seek(0)
        with Image.open(stream) as decoded:
            reencoded = audit.fingerprint_image(decoded)
        a = audit.fingerprint_image(original)
        self.assertTrue(a["informative"])
        train = [{"source_video_id": "train", "relative_path": "train.mp4", "frames": [a]}]
        benchmark = [{"source_video_id": "benchmark", "frames": [reencoded]}]
        pairs = audit.candidate_pairs(train, benchmark)
        self.assertEqual(len(pairs), 1)
        self.assertFalse(pairs[0]["frame_matches"][0]["same_decoded_rgb"])
        benchmark[0]["frames"] = [self.frame(99)]
        self.assertEqual(audit.candidate_pairs(train, benchmark), [])

    def test_index_does_not_miss_distance_six_and_dhash_filters(self):
        rng = random.Random(1042)
        for _ in range(50):
            base = rng.getrandbits(63)
            altered = base
            for bit in rng.sample(range(63), 6):
                altered ^= 1 << bit
            frame = {"phash": f"{base:016x}", "dhash": "0" * 16, "rgb_sha256": "a" * 64, "informative": True}
            other = dict(frame, phash=f"{altered:016x}")
            train = [{"source_video_id": "train", "relative_path": "train.mp4", "frames": [frame]}]
            benchmark = [{"source_video_id": "benchmark", "frames": [other]}]
            self.assertEqual(len(audit.candidate_pairs(train, benchmark)), 1)
            other["dhash"] = "f" * 16
            self.assertEqual(audit.candidate_pairs(train, benchmark), [])

    def test_low_information_frames_are_explicitly_uninformative(self):
        from PIL import Image
        result = audit.fingerprint_image(Image.new("RGB", (64, 64), "black"))
        self.assertFalse(result["informative"])
        self.assertEqual(result["grayscale_std"], 0)

    def make_cache(self, root):
        source = "youtube:12345678901"
        frames = []
        for slot in range(8):
            relative = f"frames/{source[8:]}/{slot}.png"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            image = self.image(slot + 1)
            image.save(path)
            frames.append({"relative_path": relative, "size_bytes": path.stat().st_size,
                           "sha256": audit.sha256_file(path), "width": image.width, "height": image.height,
                           "rgb_sha256": audit.hashlib.sha256(image.tobytes()).hexdigest(),
                           "source_frame_index": audit.uniform_indices(11, 8)[slot]})
        archive = {"path": "videos_chunked_01.zip", "sha256": "c" * 64, "size_bytes": 100,
                   "members": [{"path": "data/" + source[8:] + ".mp4", "size_bytes": 100}]}
        (root / "videomme_archive_directory.json").write_text(json.dumps([archive]))
        archive_digest = audit.sha256_file(root / "videomme_archive_directory.json")
        (root / "archives").mkdir()
        (root / "archives" / (archive["path"] + ".verified.json")).write_text(json.dumps({
            "verified": True, "archive_sha256": archive["sha256"], "size_bytes": 100,
            "repo_id": "lmms-lab/Video-MME", "revision": audit.VIDEOMME_REVISION}))
        entry = {"archive_path": archive["path"], "archive_sha256": archive["sha256"],
                 "repo_id": "lmms-lab/Video-MME", "revision": audit.VIDEOMME_REVISION,
                 "member_path": archive["members"][0]["path"], "source_video_id": source, "original_sha256": "a" * 64, "original_size_bytes": 100,
                 "decode_complete": True, "decoded_frame_count": 11, "second_pass_frame_count": 11,
                 "selected_indices": audit.uniform_indices(11, 8), "frames": frames,
                 "image_paths": [frame["relative_path"] for frame in frames]}
        receipt = {"schema_version": 1, "kind": "videomme_uniform8_frame_cache", "complete": True,
                   "num_frames": 8, "sampling": audit.POLICY["sampling"], "failures": [], "entries": [entry],
                   "archive_directory_sha256": archive_digest}
        (root / "entries").mkdir()
        entry_path = root / "entries" / (source[8:] + ".json")
        entry_path.write_text(json.dumps(entry))
        (root / "archives" / (archive["path"] + ".complete.json")).write_text(json.dumps({
            "archive_sha256": archive["sha256"], "source_video_ids": [source],
            "source_receipt_sha256": {source[8:]: audit.sha256_file(entry_path)}}))
        path = root / "CACHE.json"
        path.write_text(json.dumps(receipt))
        return path, receipt, source

    def test_cache_requires_exact_all_sources_sampling_and_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, receipt, source = self.make_cache(root)
            patcher = mock.patch.object(audit, "ARCHIVE_DIRECTORY_SHA256", receipt["archive_directory_sha256"])
            patcher.start()
            self.addCleanup(patcher.stop)
            result = audit.frame_cache_inventory(path, root, {source})
            self.assertTrue(result["complete"])
            self.assertEqual(len(result["entries"]), 8)
            self.assertFalse(audit.frame_cache_inventory(path, root, {source, "youtube:other000000"})["complete"])
            receipt["entries"][0]["selected_indices"][0] = 1
            path.write_text(json.dumps(receipt))
            self.assertFalse(audit.frame_cache_inventory(path, root, {source})["complete"])
            receipt["entries"][0]["selected_indices"][0] = 0
            path.write_text(json.dumps(receipt))
            (root / receipt["entries"][0]["image_paths"][0]).write_bytes(b"changed")
            self.assertFalse(audit.frame_cache_inventory(path, root, {source})["complete"])

    def test_cache_checks_frame_order_second_pass_and_decoded_rgb_metadata(self):
        import copy
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, baseline, source = self.make_cache(root)
            with mock.patch.object(audit, "ARCHIVE_DIRECTORY_SHA256", baseline["archive_directory_sha256"]):
                for change in ("second_count", "slot", "width", "rgb"):
                    receipt = copy.deepcopy(baseline)
                    entry = receipt["entries"][0]
                    if change == "second_count":
                        entry["second_pass_frame_count"] += 1
                    elif change == "slot":
                        entry["frames"][0]["source_frame_index"] += 1
                    elif change == "width":
                        entry["frames"][0]["width"] += 1
                    else:
                        entry["frames"][0]["rgb_sha256"] = "0" * 64
                    source_path = root / "entries" / (source[8:] + ".json")
                    source_path.write_text(json.dumps(entry))
                    complete_path = root / "archives" / (entry["archive_path"] + ".complete.json")
                    complete = json.loads(complete_path.read_text())
                    complete["source_receipt_sha256"][source[8:]] = audit.sha256_file(source_path)
                    complete_path.write_text(json.dumps(complete))
                    path.write_text(json.dumps(receipt))
                    self.assertFalse(audit.frame_cache_inventory(path, root, {source})["complete"], change)

    def test_count_verified_single_complete_pass_preserves_same_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, receipt, source = self.make_cache(root)
            entry = receipt["entries"][0]
            entry.update(full_decode_passes=1, container_declared_frame_count=11, second_pass_frame_count=None)
            def persist():
                source_path = root / "entries" / (source[8:] + ".json")
                source_path.write_text(json.dumps(entry))
                complete_path = root / "archives" / (entry["archive_path"] + ".complete.json")
                complete = json.loads(complete_path.read_text())
                complete["source_receipt_sha256"][source[8:]] = audit.sha256_file(source_path)
                complete_path.write_text(json.dumps(complete))
                path.write_text(json.dumps(receipt))
            with mock.patch.object(audit, "ARCHIVE_DIRECTORY_SHA256", receipt["archive_directory_sha256"]):
                persist()
                self.assertTrue(audit.frame_cache_inventory(path, root, {source})["complete"])
                entry["container_declared_frame_count"] = 12
                persist()
                self.assertFalse(audit.frame_cache_inventory(path, root, {source})["complete"])

    def test_source_gate_rejects_unbound_or_incomplete_content_audit(self):
        gate_spec = importlib.util.spec_from_file_location("test_bound_source_gate", ROOT / "scripts/operations/prepare_formal_source_gate.py")
        gate = importlib.util.module_from_spec(gate_spec)
        gate_spec.loader.exec_module(gate)
        training = {"expected_files": 1, "entries": [{"sha256": "a" * 64}]}
        benchmark = {"expected_files": 8, "expected_source_videos": 1,
                     "entries": [{"sha256": "b" * 64}], "original_entries": [{"sha256": "c" * 64}],
                     "receipt_sha256": "d" * 64}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "evidence.json"
            artifact.write_text("{}")
            report = {"schema_version": 1, "binding": {"test": True}, "method": "tested method",
                      "passed": True, "exact_duplicate_pairs": [], "perceptual_duplicate_pairs": [],
                      "media_binding": audit.media_binding(training, benchmark), "policy": audit.POLICY,
                      "scope": "uniform8_sampled_frames", "scan_complete": True, "errors": [],
                      "uninformative_sources": [], "training_file_count": 1, "benchmark_source_count": 1,
                      "artifacts": {key: {"path": artifact.name, "sha256": audit.sha256_file(artifact)}
                                    for key in ("training_fingerprints_receipt", "benchmark_fingerprints")}}
            path = root / "CONTENT_AUDIT.json"
            path.write_text(json.dumps(report))
            self.assertTrue(gate.validate_content_audit(path, {"test": True}, training, benchmark)["passed"])
            benchmark["entries"][0]["sha256"] = "e" * 64
            self.assertFalse(gate.validate_content_audit(path, {"test": True}, training, benchmark)["passed"])
            benchmark["entries"][0]["sha256"] = "b" * 64
            report["scan_complete"] = False
            path.write_text(json.dumps(report))
            self.assertFalse(gate.validate_content_audit(path, {"test": True}, training, benchmark)["passed"])

    def test_exact_overlap_uses_original_video_hashes_with_frame_cache(self):
        gate_spec = importlib.util.spec_from_file_location("test_exact_source_gate", ROOT / "scripts/operations/prepare_formal_source_gate.py")
        gate = importlib.util.module_from_spec(gate_spec)
        gate_spec.loader.exec_module(gate)
        training = {"complete": True, "entries": [{"relative_path": "train.mp4", "sha256": "a" * 64}]}
        benchmark = {"complete": True, "entries": [{"relative_path": "frame.png", "sha256": "b" * 64}],
                     "original_entries": [{"relative_path": "original.mp4", "sha256": "a" * 64}]}
        result = gate.exact_overlap_report(training, benchmark)
        self.assertFalse(result["passed"])
        self.assertEqual(result["duplicate_pairs"][0]["benchmark_path"], "original.mp4")

    def test_complete_decode_matches_reference_and_corrupt_media_fails(self):
        try:
            import av
        except ImportError:
            self.skipTest("PyAV unavailable")
        import subprocess
        import shutil
        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg unavailable")
        from deepprobe_vlm.data import decode_video
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "fixture.mp4"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                            "testsrc=size=96x64:rate=8", "-frames:v", "17", "-threads", "1",
                            "-pix_fmt", "yuv420p", str(video)], check=True, capture_output=True)
            frames, count, indices = audit.decode_uniform_images(video)
            self.assertEqual(count, 17)
            self.assertEqual(indices, audit.uniform_indices(17, 8))
            self.assertEqual([frame.tobytes() for frame in frames],
                             [frame.tobytes() for frame in decode_video(video, 8)])
            video.write_bytes(b"not an mp4")
            entry = {"relative_path": video.name, "source_video_id": "fixture",
                     "sha256": audit.sha256_file(video), "size_bytes": video.stat().st_size}
            self.assertFalse(audit.fingerprint_training_entry(entry, root)["decode_complete"])

    def test_missing_media_is_retained_as_decode_failure(self):
        entry = {"relative_path": "missing.mp4", "source_video_id": "youtube:12345678901", "sha256": "a" * 64, "size_bytes": 1}
        result = audit.fingerprint_training_entry(entry, Path("/nonexistent"))
        self.assertFalse(result["decode_complete"])
        self.assertIn("error", result)

    def test_changed_media_during_decode_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "video.mp4"
            video.write_bytes(b"original bytes")
            entry = {"relative_path": video.name, "source_video_id": "youtube:12345678901", "sha256": audit.sha256_file(video), "size_bytes": video.stat().st_size}
            def decode(path):
                path.write_bytes(b"changed bytes")
                return [self.image()] * 8, 10, audit.uniform_indices(10, 8)
            with mock.patch.object(audit, "decode_uniform_images", side_effect=decode):
                result = audit.fingerprint_training_entry(entry, root)
            self.assertFalse(result["decode_complete"])
            self.assertIn("changed while decoding", result["error"])


if __name__ == "__main__":
    unittest.main()
