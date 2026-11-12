"""Original-video sampling, immutable cache resume and bounded OSS access."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from deepprobe_vlm import suite_media as media


class FakeStore:
    def __init__(self, objects, archive_key):
        self.objects = deepcopy(objects)
        self.archive_key = archive_key
        self.etag = "accepted-object-version"
        self.ranges, self.puts, self.gets = [], [], []

    def head(self, key):
        return {"size_bytes": len(self.objects[key]), "etag": self.etag}

    def get(self, key, maximum=media.MAX_METADATA_BYTES):
        self.gets.append(key)
        raw = self.objects.get(key)
        if raw is not None:
            media.require(len(raw) <= maximum, "fake object exceeds byte bound")
        return raw

    def range(self, key, start, end, etag, total):
        media.require(etag == self.etag and total == len(self.objects[key]), "fake range identity changed")
        self.ranges.append((key, start, end, etag, total))
        return self.objects[key][start:end + 1]

    def put_verified(self, key, raw):
        media.require(key not in self.objects or self.objects[key] == raw, "fake immutable object differs")
        self.puts.append(key)
        self.objects[key] = raw
        assert self.get(key, len(raw)) == raw

    def keys(self, prefix):
        return sorted(key for key in self.objects if key.startswith(prefix))


def make_video():
    import av
    from PIL import Image
    output = io.BytesIO()
    with av.open(output, "w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
        for index in range(10):
            frame = av.VideoFrame.from_image(Image.new("RGB", (32, 24), (index * 23, 0, 0)))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


class OriginalFrameCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.video = make_video()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        key = media.PROJECT_PREFIX + "fixture/hf-archives/tempcompass.zip"
        raw = io.BytesIO()
        with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr("videos/clip_reverse.mp4", self.video)
        archive_raw = raw.getvalue()
        with zipfile.ZipFile(io.BytesIO(archive_raw)) as bundle:
            info = bundle.getinfo("videos/clip_reverse.mp4")
        archive = {"key": key, "size_bytes": len(archive_raw), "sha256": media.digest(archive_raw),
                   "etag": "accepted-object-version"}
        receipt = {"binding": {"bucket": media.BUCKET, "key": key, "size_bytes": len(archive_raw),
                                "sha256": archive["sha256"]},
                   "source_stream_sha256": archive["sha256"], "source_bytes": len(archive_raw),
                   "oss_readback": {"bytes": len(archive_raw), "sha256": archive["sha256"],
                                    "verification": "complete_OSS_GET_SHA256_and_size"}}
        receipt_raw = media.encode(receipt)
        archive["receipt"] = {"key": key + ".CACHE_VERIFIED.json", "size_bytes": len(receipt_raw),
                              "sha256": media.digest(receipt_raw)}
        self.rows = [{"sample_id": f"question:{index}", "source_video_id": "component:clip",
                      "dataset": "tempcompass", "processed_media_identity": "tempcompass:clip_reverse",
                      "video_id": "clip_reverse", "original_answer": "A. left",
                      "underlying_source_video_ids": ["shutterstock:clip"], "split": "test",
                      "question": "Which direction?", "choices": ["left", "right"], "answer": "A",
                      "image_paths": [f"prior-uniform8/clip/{slot:02d}.png" for slot in range(8)]}
                     for index in range(2)]
        self.manifest_raw = b"".join(media.encode(row).replace(b"\n", b"") + b"\n" for row in self.rows)
        self.manifest = self.root / "source-manifest.jsonl"
        self.manifest.write_bytes(self.manifest_raw)
        self.binding = {"schema_version": 1, "bucket": media.BUCKET, "dataset": "tempcompass",
                        "manifest_sha256": media.digest(self.manifest_raw), "manifest_bytes": len(self.manifest_raw),
                        "expected_samples": 2, "expected_clips": 1, "archives": [archive],
                        "sources": [{"media_identity": "tempcompass:clip_reverse", "archive_key": key,
                                     "underlying_source_video_ids": ["shutterstock:clip"],
                                     "member": {"path": info.filename, "size_bytes": info.file_size,
                                                "compressed_bytes": info.compress_size, "crc32": info.CRC,
                                                "sha256": media.digest(self.video)}}]}
        self.store = FakeStore({key: archive_raw, archive["receipt"]["key"]: receipt_raw}, key)
        self.prefix = media.PROJECT_PREFIX + "fixture/shape-ablation/tempcompass/uniform4/"
        self.out = self.root / "cache-metadata"

    def build(self, **changes):
        args = dict(frames=4, output_dir_new=self.out, oss_prefix_new=self.prefix,
                    max_bytes=1_000_000, store=self.store)
        args.update(changes)
        return media.build_frame_cache(self.manifest, self.binding, **args)

    def test_real_original_decode_uses_exact_uniform_indices_and_preserves_qa(self):
        result = self.build()
        self.assertEqual(result["status"], "FRAME_CACHE_COMPLETE")
        self.assertEqual(result["frame_count"], 4)
        self.assertFalse(result["formal_evaluation_completed"])
        self.assertFalse(result["local_video_or_png_payloads"])
        receipt = json.loads((self.out / "FRAME_CACHE_RECEIPT.json").read_text())
        entry = receipt["entries"][0]
        self.assertEqual(entry["decoded_frame_count"], 10)
        self.assertEqual(entry["full_decode_passes"], 2)
        self.assertEqual(entry["selected_indices"], [0, 3, 6, 9])
        self.assertEqual([f["source_frame_index"] for f in entry["frames"]], [0, 3, 6, 9])
        transformed = [json.loads(line) for line in (self.out / "manifest.jsonl").read_text().splitlines()]
        for old, new in zip(self.rows, transformed):
            self.assertEqual({k: v for k, v in old.items() if k != "image_paths"},
                             {k: v for k, v in new.items() if k != "image_paths"})
            self.assertEqual(len(new["image_paths"]), 4)
        self.assertEqual({path.name for path in self.out.iterdir()},
                         {"BUILD_PLAN.json", "manifest.jsonl", "FRAME_CACHE_RECEIPT.json", "CACHE_COMPLETE.json"})
        self.assertEqual(len(self.store.keys(self.prefix)), result["exact_prefix_object_count"])
        self.assertTrue(self.store.ranges)
        self.assertTrue(all(row[3] == "accepted-object-version" for row in self.store.ranges))

    def test_f16_repeats_only_real_original_indices_when_video_is_short(self):
        result = self.build(frames=16, oss_prefix_new=self.prefix.replace("uniform4", "uniform16"))
        entry = json.loads((self.out / "FRAME_CACHE_RECEIPT.json").read_text())["entries"][0]
        self.assertEqual(result["frame_count"], 16)
        self.assertEqual(entry["selected_indices"], media.uniform_indices(10, 16))
        self.assertEqual(set(entry["selected_indices"]), set(range(10)))

    def test_temporal_bounds_are_applied_to_original_timestamps(self):
        selected = media.decode_selected(io.BytesIO(self.video), 4, start_seconds=.2, end_seconds=.7)
        self.assertEqual(selected["decoded_frame_count"], 5)
        self.assertEqual(selected["selected_indices"], [0, 1, 3, 4])
        self.assertEqual([selected["selected"][i]["source_frame_index"] for i in selected["selected_indices"]],
                         [2, 3, 5, 6])
        self.assertEqual(selected["original_stream_frame_count"], 10)

    def test_changed_archive_etag_stops_before_any_write(self):
        self.store.etag = "different-version"
        with self.assertRaisesRegex(media.CacheError, "ETag changed"):
            self.build()
        self.assertFalse(self.out.exists())
        self.assertFalse(self.store.puts)

    def test_original_member_sha_mismatch_cannot_create_completed_cache(self):
        self.binding["sources"][0]["member"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(media.CacheError, "member SHA"):
            self.build()
        self.assertFalse((self.out / "CACHE_COMPLETE.json").exists())
        self.assertEqual(self.store.keys(self.prefix), [self.prefix + "BUILD_PLAN.json"])

    def test_cache_bytes_cap_stops_before_first_png_write(self):
        with self.assertRaisesRegex(media.CacheError, "byte cap"):
            self.build(max_bytes=1)
        self.assertEqual(self.store.keys(self.prefix), [self.prefix + "BUILD_PLAN.json"])

    def test_complete_resume_revalidates_every_frame_without_decoding(self):
        first = self.build()
        self.store.gets.clear()
        with patch.object(media, "extract_member", side_effect=AssertionError("must reuse verified frame cache")):
            second = self.build(resume=True)
        self.assertEqual(first, second)
        self.assertEqual(len([key for key in self.store.gets if key.endswith(".png")]), 4)

    def test_resume_rejects_corruption_extra_keys_and_changed_plan(self):
        self.build()
        png = next(key for key in self.store.keys(self.prefix) if key.endswith(".png"))
        saved = self.store.objects[png]
        self.store.objects[png] = saved[:-1] + bytes([saved[-1] ^ 1])
        with self.assertRaisesRegex(media.CacheError, "resumed frame full GET SHA"):
            self.build(resume=True)
        self.store.objects[png] = saved
        self.store.objects[self.prefix + "unowned.bin"] = b"bad"
        with self.assertRaisesRegex(media.CacheError, "inventory differs"):
            self.build(resume=True)
        del self.store.objects[self.prefix + "unowned.bin"]
        with self.assertRaisesRegex(media.CacheError, "exact existing local build plan"):
            self.build(resume=True, max_bytes=2_000_000)

    def test_partial_source_can_resume_only_with_identical_png_bytes(self):
        original_put = self.store.put_verified
        def interrupted(key, raw):
            if key.endswith("01.png"):
                raise RuntimeError("simulated preemption")
            original_put(key, raw)
        self.store.put_verified = interrupted
        with self.assertRaisesRegex(RuntimeError, "preemption"):
            self.build()
        self.store.put_verified = original_put
        result = self.build(resume=True)
        self.assertEqual(result["frame_count"], 4)
        self.assertEqual(result["status"], "FRAME_CACHE_COMPLETE")

    def test_manifest_coverage_and_processed_clip_identity_are_mandatory(self):
        for mutate in (
            lambda b: b.update(expected_samples=1),
            lambda b: b["sources"][0]["member"].update(path="frames/cached.png"),
            lambda b: b["sources"][0]["member"].update(path="videos/clip.mp4"),
            lambda b: b["sources"][0].update(underlying_source_video_ids=["wrong-source"]),
            lambda b: b["sources"].append(deepcopy(b["sources"][0])),
        ):
            altered = deepcopy(self.binding)
            mutate(altered)
            with self.assertRaises((media.CacheError, ValueError)):
                media.validate_binding(altered, self.manifest_raw)

    def test_existing_destination_is_never_adopted_without_explicit_resume(self):
        self.store.objects[self.prefix + "unowned.bin"] = b"unrelated data"
        with self.assertRaisesRegex(media.CacheError, "empty exact OSS prefix"):
            self.build()
        self.assertFalse(self.out.exists())

    def test_source_binding_preparation_reads_only_archive_metadata(self):
        source, archive = self.binding["sources"][0], self.binding["archives"][0]
        accepted = {"complete": True, "entries": [{
            "processed_media_identity": source["media_identity"], "decode_complete": True,
            "underlying_source_video_ids": source["underlying_source_video_ids"],
            "archive_key": archive["key"], "archive_sha256": archive["sha256"], "archive_etag": archive["etag"],
            "member_path": source["member"]["path"], "original_size_bytes": source["member"]["size_bytes"],
            "original_sha256": source["member"]["sha256"]}]}
        with patch.object(zipfile.ZipFile, "open", side_effect=AssertionError("must not extract video members")):
            binding = media.create_source_binding(self.manifest_raw, media.encode(accepted),
                                                  self.binding["archives"], store=self.store)
        self.assertEqual(binding["sources"], self.binding["sources"])
        self.assertFalse(self.store.puts)
        self.assertTrue(self.store.ranges)

    def _full_scope_cache(self, frames=4):
        """A real ZIP and full-size 1580-question/410-clip synthetic test corpus."""
        archive = self.binding["archives"][0]
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for index in range(410):
                bundle.writestr(f"videos/clip_{index:04d}.mp4", self.video)
        zip_raw = zip_buffer.getvalue()
        archive.update(size_bytes=len(zip_raw), sha256=media.digest(zip_raw))
        receipt_raw = media.encode({"binding": {"bucket": media.BUCKET, "key": archive["key"],
                "size_bytes": archive["size_bytes"], "sha256": archive["sha256"]},
            "source_stream_sha256": archive["sha256"], "source_bytes": archive["size_bytes"],
            "oss_readback": {"bytes": archive["size_bytes"], "sha256": archive["sha256"],
                             "verification": "complete_OSS_GET_SHA256_and_size"}})
        archive["receipt"].update(size_bytes=len(receipt_raw), sha256=media.digest(receipt_raw))
        sources = []
        with zipfile.ZipFile(io.BytesIO(zip_raw)) as bundle:
            for index, info in enumerate(bundle.infolist()):
                sources.append({"media_identity": f"tempcompass:clip_{index:04d}", "archive_key": archive["key"],
                    "underlying_source_video_ids": [f"shutterstock:source_{index:04d}"],
                    "member": {"path": info.filename, "size_bytes": info.file_size, "compressed_bytes": info.compress_size,
                               "crc32": info.CRC, "sha256": media.digest(self.video)}})
        rows = []
        for index in range(1580):
            clip = index % 410
            row = deepcopy(self.rows[0])
            row.update(sample_id=f"question:{index}", video_id=f"clip_{clip:04d}",
                       processed_media_identity=f"tempcompass:clip_{clip:04d}", source_video_id=f"component:{clip}",
                       underlying_source_video_ids=[f"shutterstock:source_{clip:04d}"],
                       image_paths=[f"old8/clip_{clip:04d}/{slot:02d}.png" for slot in range(8)])
            rows.append(row)
        manifest_raw = b"".join((json.dumps(row) + "\n").encode() for row in rows)
        self.manifest.write_bytes(manifest_raw)
        self.binding.update(manifest_sha256=media.digest(manifest_raw), manifest_bytes=len(manifest_raw),
                            expected_samples=1580, expected_clips=410, sources=sources)
        self.store = FakeStore({archive["key"]: zip_raw, archive["receipt"]["key"]: receipt_raw}, archive["key"])
        decoded = media.decode_selected(io.BytesIO(self.video), frames)
        def decoder(original, frame_count, **bounds):
            self.assertEqual(media.digest(original.getvalue()), media.digest(self.video))
            self.assertEqual(frame_count, frames)
            return deepcopy(decoded)
        self.prefix = self.prefix.replace("uniform4", f"uniform{frames}")
        self.build(frames=frames, max_bytes=5_000_000, decoder=decoder)
        binding_path = self.root / "source-binding.json"
        binding_path.write_bytes(media.encode(self.binding))
        return media.cache_asset_binding(self.out, source_manifest=self.manifest, source_binding=binding_path,
                                        media_root=self.root / "node-media", max_bytes=5_000_000)

    def test_full_scope_hydration_exports_action_compatible_acceptance(self):
        from deepprobe_vlm.suite_actions import validate_benchmark
        asset = self._full_scope_cache()
        with patch("platform.system", return_value="Linux"):
            fragment = media.hydrate_frame_cache(asset["cache"], media_root=asset["media_root"],
                                                  max_bytes=asset["max_bytes"], store=self.store)
        records = validate_benchmark(fragment, {"benchmark": "tempcompass", "frames": 4})
        self.assertEqual(len(records), 1580)
        self.assertEqual(len(fragment["media_files"]), 1640)
        ready = json.loads(Path(fragment["acceptance"]["path"]).read_text())
        self.assertEqual(ready["status"], "SUITE_BENCHMARK_READY")
        self.assertFalse(ready["formal_evaluation_completed"])
        with patch("platform.system", return_value="Linux"):
            again = media.hydrate_frame_cache(asset["cache"], media_root=asset["media_root"],
                                              max_bytes=asset["max_bytes"], store=self.store, resume=True)
        self.assertEqual(fragment, again)

    def test_hydration_fails_capacity_gate_before_creating_media_directory(self):
        asset = self._full_scope_cache()
        with patch("platform.system", return_value="Linux"):
            with self.assertRaisesRegex(media.CacheError, "capacity cap"):
                media.hydrate_frame_cache(asset["cache"], media_root=asset["media_root"], max_bytes=1, store=self.store)
        self.assertFalse(Path(asset["media_root"]).exists())

    def test_legacy_f8_adapter_requires_exact_original_decode_receipt(self):
        from deepprobe_vlm.checkpoint_retention import file_pin
        asset = self._full_scope_cache(frames=8)
        suite_receipt = json.loads(Path(asset["cache"]["receipt"]["path"]).read_text())
        entries, mappings = [], []
        for item in suite_receipt["entries"]:
            source = item["source_binding"]
            entries.append({"processed_media_identity": item["media_identity"], "decode_complete": True,
                            "decoded_frame_count": item["decoded_frame_count"], "selected_indices": item["selected_indices"],
                            "image_paths": item["image_paths"], "frames": item["frames"],
                            "member_path": source["member"]["path"], "original_sha256": source["member"]["sha256"],
                            "original_size_bytes": source["member"]["size_bytes"],
                            "archive_sha256": self.binding["archives"][0]["sha256"]})
            mappings.extend({"relative_path": row["relative_path"], "key": self.prefix + row["relative_path"],
                             "size_bytes": row["size_bytes"], "sha256": row["sha256"]} for row in item["frames"])
        legacy = {"complete": True, "num_frames": 8, "sampling": media.SAMPLING, "entries": entries}
        legacy_path = self.root / "legacy-receipt.json"
        legacy_path.write_bytes(media.encode(legacy))
        source_binding = deepcopy(self.binding)
        source_binding["accepted_original_decode_receipt_sha256"] = media.digest(legacy_path.read_bytes())
        binding_path = self.root / "legacy-source-binding.json"
        binding_path.write_bytes(media.encode(source_binding))
        cache = {"kind": "legacy", "manifest": asset["cache"]["manifest"], "source_manifest": asset["cache"]["source_manifest"],
                 "source_binding": file_pin(binding_path), "receipt": file_pin(legacy_path), "frame_objects": mappings}
        bound = media._bound_cache_inventory(cache)
        self.assertEqual(len(bound[0]), 1580)
        self.assertEqual(len(bound[2]), 3280)
        self.assertEqual(bound[3], 8)
        source_binding["accepted_original_decode_receipt_sha256"] = "0" * 64
        binding_path.write_bytes(media.encode(source_binding))
        cache["source_binding"] = file_pin(binding_path)
        with self.assertRaisesRegex(media.CacheError, "exact accepted original decode receipt"):
            media._bound_cache_inventory(cache)

    def test_real_sdk_execution_is_rejected_on_the_development_mac(self):
        with patch("platform.system", return_value="Darwin"):
            with self.assertRaisesRegex(media.CacheError, "remote Linux"):
                media.OSSStore(self.root / "not-a-config", allowed_prefix=self.prefix)


if __name__ == "__main__":
    unittest.main()
