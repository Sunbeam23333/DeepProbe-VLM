"""Dependency-free tests of latency boundaries, selection, provenance and failures."""
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import ModuleType, SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deepprobe_vlm import benchmark_latency as latency


class LatencyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.media = self.root / "media"
        self.media.mkdir()
        (self.media / "frame.bin").write_bytes(b"byte-hash fixture; never decoded")
        self.records = [{"sample_id": f"dev:{i:03d}", "source_video_id": f"source:{i}",
                         "question": "What happens?", "answer": "A", "choices": ["one", "two"],
                         "split": "validation", "dataset": "local_development_fixture",
                         "image_paths": ["frame.bin"]} for i in range(64)]
        self.manifest = self.root / "validation.jsonl"
        self.manifest.write_bytes(b"".join(latency.canonical(row)+b"\n" for row in self.records))
        self.meta = self.root / "source.json"
        self.meta.write_text(json.dumps({"scope": "development", "manifest_sha256": latency.sha(self.manifest),
            "quality_score_based_selection": False, "source_description": "synthetic CPU test data"}))
        self.checkpoint = self.root / "checkpoint"
        for name in ("language_model/config.json", "vision_model/config.json", "projector.safetensors"):
            target = self.checkpoint / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"placeholder bytes, never loaded as weights")
        (self.checkpoint / "deepprobe_config.json").write_text(json.dumps({"num_frames": 8, "pool_grid": 4, "num_loops": 1}))
        (self.checkpoint / "training_provenance.json").write_text(json.dumps({"training_sources": [], "tuning_sources": []}))
        self.plan = latency.freeze_plan(self.manifest, self.checkpoint, self.media, self.meta)

    def tearDown(self):
        self.temp.cleanup()

    def test_selection_is_stable_and_independent_of_labels_scores_and_input_order(self):
        expected = [r["sample_id"] for r in latency.select_records(self.records)]
        changed = [dict(r, answer="B", score=i, prediction="anything") for i, r in enumerate(reversed(self.records))]
        self.assertEqual([r["sample_id"] for r in latency.select_records(changed)], expected)
        self.assertEqual(len(expected), 32)

    def test_reject_test_benchmark_duplicate_and_small_manifests(self):
        for rows in (self.records[:31], self.records + [self.records[0]],
                     [dict(r, split="test") for r in self.records],
                     [dict(r, dataset="TempCompass") for r in self.records]):
            with self.subTest(rows=rows[0].get("dataset")), self.assertRaises(ValueError):
                latency.select_records(rows)

    def test_cache_pair_shares_only_exact_comparison_inputs(self):
        other = copy.deepcopy(self.plan)
        other["cache"] = False
        latency.validate_plan(other)
        self.assertEqual(other["comparison_identity"], self.plan["comparison_identity"])
        other["model_config"]["num_loops"] = 2
        with self.assertRaisesRegex(ValueError, "comparison identity"):
            latency.validate_plan(other)
        other["comparison_identity"] = latency.comparison_identity(other)
        self.assertNotEqual(other["comparison_identity"], self.plan["comparison_identity"])

    def test_protocol_rejects_changed_settings_even_with_recomputed_hash(self):
        for key, value in (("warmup_requests_per_sample", 0), ("measured_requests_per_sample", 1),
                           ("max_new_tokens", 1), ("include_answer", True), ("batch_size", True)):
            changed = copy.deepcopy(self.plan)
            changed["settings"][key] = value
            changed["comparison_identity"] = latency.comparison_identity(changed)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "settings"):
                latency.validate_plan(changed)

    def test_full_input_verification_detects_media_checkpoint_and_source_changes(self):
        self.assertEqual([r["sample_id"] for r in latency.verify_inputs(self.plan)],
                         [r["sample_id"] for r in self.plan["samples"]])
        (self.media / "frame.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "pinned bytes"):
            latency.verify_inputs(self.plan)
        (self.media / "frame.bin").write_bytes(b"byte-hash fixture; never decoded")
        (self.checkpoint / "extra.bin").write_bytes(b"unexpected")
        with self.assertRaisesRegex(ValueError, "checkpoint inventory"):
            latency.verify_inputs(self.plan)
        changed = copy.deepcopy(self.plan)
        changed["source_files"][0]["sha256"] = "0" * 64
        changed["comparison_identity"] = latency.comparison_identity(changed)
        with self.assertRaisesRegex(ValueError, "pinned bytes"):
            latency.verify_inputs(changed)

    def test_metadata_must_bind_development_manifest(self):
        self.meta.write_text(json.dumps({"scope": "test", "manifest_sha256": latency.sha(self.manifest),
            "quality_score_based_selection": False, "source_description": "wrong scope"}))
        changed = copy.deepcopy(self.plan)
        changed["source_metadata"] = latency.pin(self.meta)
        changed["comparison_identity"] = latency.comparison_identity(changed)
        with self.assertRaisesRegex(ValueError, "development source metadata"):
            latency.verify_inputs(changed)

    def test_inventory_coverage_and_traversal_cannot_be_weakened(self):
        changed = copy.deepcopy(self.plan)
        changed["media_files"] = [{"path": "../outside", "bytes": 0, "sha256": "0" * 64}]
        changed["comparison_identity"] = latency.comparison_identity(changed)
        with self.assertRaisesRegex(ValueError, "unsafe inventory"):
            latency.validate_plan(changed)
        changed = copy.deepcopy(self.plan)
        changed["source_files"] = changed["source_files"][:-1]
        changed["comparison_identity"] = latency.comparison_identity(changed)
        with self.assertRaisesRegex(ValueError, "dependency inventory"):
            latency.validate_plan(changed)

    def test_schema_validation_loads_no_gpu_library_or_weights(self):
        path = self.root / "plan.json"
        path.write_bytes(latency.canonical(self.plan))
        source = str(Path(__file__).resolve().parents[1] / "src")
        code = "import sys;sys.path.insert(0,sys.argv[1]);from deepprobe_vlm import benchmark_latency as m;m.main(['--plan',sys.argv[2],'--plan-sha256',sys.argv[3]]);assert 'torch' not in sys.modules;assert 'transformers' not in sys.modules"
        result = subprocess.run([sys.executable, "-B", "-c", code, source, str(path), latency.sha(path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "PLAN_SCHEMA_VALID_NO_EXECUTION")

    def test_request_boundary_order_excludes_scoring_logging_and_model_load(self):
        events = []
        moments = iter([10., 12., 17., 18.])
        def clock():
            events.append("clock")
            return next(moments)
        def mem():
            events.append("memory")
            return {"allocated": 100, "reserved": 200, "peak_allocated": 300, "peak_reserved": 400}
        def prep(): events.append("prepare"); return "batch"
        def gen(batch): self.assertEqual(batch, "batch"); events.append("generate"); return "tokens"
        def decode(tokens):
            self.assertEqual(tokens, "tokens"); events.append("decode")
            return {"generated_token_ids": [51, 2], "prediction": "A"}
        row = latency.measure_request(prepare=prep, generate=gen, decode=decode,
            synchronize=lambda: events.append("sync"), reset_peak=lambda: events.append("reset"), memory=mem, clock=clock)
        self.assertEqual(events, ["sync", "memory", "reset", "clock", "prepare", "sync", "clock", "generate",
                                  "sync", "clock", "decode", "sync", "clock", "memory"])
        self.assertEqual([row[k] for k in latency.TIMINGS], [2., 5., 1., 8.])
        self.assertEqual(row["generated_tokens"], 2)
        self.assertEqual(row["generated_token_ids"], [51, 2])  # EOS counts; no retokenization.
        self.assertEqual(row["memory"]["peak_allocated_bytes"], 300)

    def test_invalid_token_ids_and_nonmonotonic_timer_fail(self):
        def measure(ids, times):
            return latency.measure_request(prepare=lambda: None, generate=lambda b: None,
                decode=lambda t: {"generated_token_ids": ids, "prediction": "A"},
                synchronize=lambda: None, reset_peak=lambda: None,
                memory=lambda: {"allocated": 0, "reserved": 0, "peak_allocated": 0, "peak_reserved": 0},
                clock=iter(times).__next__)
        for ids in ([], [True], [-1], [1]*17):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "actual generated"):
                measure(ids, [0., 1., 2., 3.])
        with self.assertRaisesRegex(ValueError, "monotonic"):
            measure([1], [0., 2., 1., 3.])

    def trials(self):
        rows = []
        for sid in ("a", "b"):
            for phase, count in (("warmup", 3), ("measured", 10)):
                for i in range(count):
                    value = 999. if phase == "warmup" else float(i+1)
                    rows.append({"sample_id": sid, "phase": phase, "repeat_index": i,
                        **{key: value for key in latency.TIMINGS}, "generated_tokens": 2,
                        "generated_token_ids": [i, 2], "prediction": f"repeat {i}", "answer": "A",
                        "quality": {"correct": i == 0}, "memory": {"peak_allocated_bytes": 99, "peak_reserved_bytes": 123}})
        return rows

    def test_summary_excludes_warmups_and_keeps_output_variation_and_first_quality(self):
        summary = latency.summarize(self.trials(), ["a", "b"])
        self.assertEqual(summary["measured_requests"], 20)
        self.assertEqual(summary["warmup_requests_excluded"], 6)
        self.assertEqual(summary["timings"]["model_seconds"]["mean"], 5.5)
        self.assertEqual(summary["timings"]["model_seconds"]["median"], 5.5)
        self.assertEqual(summary["timings"]["model_seconds"]["p95"], 10.)
        self.assertEqual(summary["total_measured_generated_tokens"], 40)
        self.assertEqual(summary["first_measured_quality"][0]["prediction"], "repeat 0")
        self.assertEqual(summary["per_sample"][0]["distinct_measured_token_sequences"], 10)

    def test_missing_duplicate_or_extra_trial_cannot_produce_complete_summary(self):
        rows = self.trials()
        for altered in (rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]],
                        rows + [dict(rows[0], sample_id="extra")]):
            with self.assertRaises(ValueError):
                latency.summarize(altered, ["a", "b"])

    def test_p95_uses_predeclared_linear_interpolation(self):
        self.assertAlmostEqual(latency.stats([1., 2., 3., 4.])["p95"], 3.85)
        self.assertEqual(latency.stats([7.])["p95"], 7.)

    def test_failed_input_verification_preserves_attempt_without_gpu_import(self):
        (self.media / "frame.bin").write_bytes(b"changed")
        output = self.root / "failed-run"
        with self.assertRaisesRegex(ValueError, "pinned bytes"):
            latency.execute(self.plan, latency.digest(self.plan), output)
        failure = json.loads((output / "FAILURE.json").read_text())
        self.assertEqual(latency.sha(output / "PLAN.json"), latency.digest(self.plan))
        self.assertEqual(failure["completed_requests"], 0)
        self.assertFalse(failure["automatic_retry"])
        self.assertFalse((output / "SUMMARY.json").exists())
        with self.assertRaises(FileExistsError):
            latency.execute(self.plan, latency.digest(self.plan), output)

    def test_execution_rejects_wrong_plan_bytes_before_creating_output(self):
        output = self.root / "mismatched-run"
        with self.assertRaisesRegex(ValueError, "execution plan bytes"):
            latency.execute(self.plan, "0" * 64, output)
        self.assertFalse(output.exists())

    def fake_runtime(self, fail_after=None):
        """Full runner adapters; no torch import, model weights or media decode."""
        calls = {"generate": [], "model_load": 0, "prepare": 0, "sync": 0}
        class Tensor:
            def __init__(self, shape, values=None, dtype="int64"):
                self.shape, self.values, self.dtype = shape, values, dtype
                self.ndim = len(shape)
            def to(self, device): return self
            def detach(self): return self
            def cpu(self): return self
            def tolist(self): return self.values
            def __getitem__(self, index): return Tensor(self.shape[1:], self.values[index], self.dtype)
        class Model:
            deep_config = SimpleNamespace(num_frames=8, num_visual_tokens=128, max_text_tokens=512)
            @classmethod
            def from_pretrained(cls, checkpoint, *, torch_dtype):
                self.assertEqual(checkpoint, self.checkpoint)
                self.assertEqual(torch_dtype, "fake-bfloat16")
                calls["model_load"] += 1
                return cls()
            def to(self, device): return self
            def eval(self): return self
            def parameters(self): return iter([SimpleNamespace(dtype="fake-bfloat16")])
            def greedy_generate(self, **kwargs):
                if fail_after is not None and len(calls["generate"]) == fail_after:
                    raise RuntimeError("injected generation failure")
                calls["generate"].append(kwargs)
                return Tensor((1, 3), [[101, 0, 2]])
        class Tokenizer:
            eos_token_id = 2
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                self.assertEqual(kwargs, {"local_files_only": True, "trust_remote_code": False})
                return cls()
            def decode(self, ids, *, skip_special_tokens):
                if ids != [101, 0, 2] or skip_special_tokens is not True:
                    raise AssertionError("actual token IDs not passed to decoder")
                return "A"
        class Processor:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                self.assertEqual(kwargs, {"local_files_only": True, "use_fast": False})
                return cls()
        class Dataset:
            def __init__(self, records, processor, frames, media_root):
                self.records = records
            def __getitem__(self, index):
                calls["prepare"] += 1
                return {"record": self.records[index], "pixel_values": Tensor((8, 3, 2, 2), dtype="float32")}
        class Collator:
            def __init__(self, tokenizer, **kwargs):
                if kwargs != {"num_visual_tokens": 128, "max_text_tokens": 512, "include_answer": False}:
                    raise AssertionError("prompt contract changed")
            def tokenize_record(self, record): return [7], [8, 9], []
            def __call__(self, examples):
                return {"input_ids": Tensor((1, 131)), "attention_mask": Tensor((1, 131)),
                        "visual_mask": Tensor((1, 131), dtype="bool"), "pixel_values": Tensor((1, 8, 3, 2, 2), dtype="float32")}
        def sync(device): calls["sync"] += 1
        torch = ModuleType("torch")
        torch.cuda = SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True,
            set_device=lambda d: None, manual_seed_all=lambda s: None, device_count=lambda: 8,
            synchronize=sync, reset_peak_memory_stats=lambda d: None,
            memory_allocated=lambda d: 100, memory_reserved=lambda d: 200,
            max_memory_allocated=lambda d: 150, max_memory_reserved=lambda d: 250,
            get_device_properties=lambda d: SimpleNamespace(name="fake test GPU", total_memory=1000, major=9, minor=0, uuid="fake"))
        torch.device = lambda name: name
        torch.manual_seed = lambda seed: None
        torch.bfloat16 = "fake-bfloat16"
        torch.inference_mode = nullcontext
        torch.get_num_threads = lambda: 1
        torch.version = SimpleNamespace(cuda="fake")
        torch.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True, benchmark=True, version=lambda: 0))
        modules = {"torch": torch}
        for name, members in {
            "transformers": {"AutoTokenizer": Tokenizer, "AutoImageProcessor": Processor},
            "deepprobe_vlm.data": {"MultimodalCollator": Collator, "VideoManifestDataset": Dataset},
            "deepprobe_vlm.modeling": {"DeepProbeVLM": Model},
            "deepprobe_vlm.evaluate": {"score_answer": lambda prediction, answer, choices: {
                "metric": "fixture_exact_match", "correct": int(prediction == answer)}}}.items():
            module = ModuleType(name)
            module.__dict__.update(members)
            modules[name] = module
        return calls, modules

    def test_full_execute_wires_both_cache_modes_and_all_416_requests(self):
        for cache in (True, False):
            with self.subTest(cache=cache):
                plan = copy.deepcopy(self.plan)
                plan["cache"] = cache
                payload = json.dumps(plan, indent=2).encode() + b"\n"
                import hashlib
                plan_sha = hashlib.sha256(payload).hexdigest()
                calls, modules = self.fake_runtime()
                output = self.root / ("cache-on" if cache else "cache-off")
                with patch.dict(sys.modules, modules), patch.dict("os.environ", {"WORLD_SIZE": "1"}):
                    summary = latency.execute(plan, plan_sha, output, plan_bytes=payload)
                self.assertEqual(calls["model_load"], 1)
                self.assertEqual(calls["prepare"], 32 * (3+10))
                self.assertEqual(calls["sync"], 4 * 416)
                self.assertEqual(len(calls["generate"]), 416)
                for call in calls["generate"]:
                    self.assertEqual(set(call), {"input_ids", "attention_mask", "visual_mask", "pixel_values",
                                                 "max_new_tokens", "eos_token_id", "use_cache"})
                    self.assertEqual((call["max_new_tokens"], call["eos_token_id"], call["use_cache"]), (16, 2, cache))
                rows = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), 416)
                self.assertTrue(all(r["generated_token_ids"] == [101, 0, 2] and r["generated_tokens"] == 3 for r in rows))
                self.assertEqual(summary["measured_requests"], 320)
                self.assertEqual(summary["warmup_requests_excluded"], 96)
                self.assertEqual(summary["total_measured_generated_tokens"], 960)
                self.assertEqual(summary["requests_sha256"], latency.sha(output / "requests.jsonl"))
                self.assertEqual(summary["runtime_sha256"], latency.sha(output / "RUNTIME.json"))
                self.assertEqual(latency.sha(output / "PLAN.json"), plan_sha)
                self.assertFalse(summary["measurement_scope"]["serving_or_eight_gpu_throughput_measured"])
                self.assertFalse((output / "FAILURE.json").exists())
                self.assertEqual(json.loads((output / "SUMMARY.json").read_text()), summary)

    def test_full_execute_failure_retains_durable_prefix_and_no_success(self):
        calls, modules = self.fake_runtime(fail_after=5)
        output = self.root / "incomplete-model-run"
        with patch.dict(sys.modules, modules), patch.dict("os.environ", {"WORLD_SIZE": "1"}):
            with self.assertRaisesRegex(RuntimeError, "injected generation failure"):
                latency.execute(self.plan, latency.digest(self.plan), output)
        rows = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 5)
        self.assertEqual([r["phase"] for r in rows], ["warmup"] * 3 + ["measured"] * 2)
        failure = json.loads((output / "FAILURE.json").read_text())
        self.assertEqual(failure["completed_requests"], 5)
        self.assertFalse(failure["automatic_retry"])
        self.assertFalse((output / "SUMMARY.json").exists())
        self.assertTrue((output / "RUNTIME.json").exists())

    def test_receipt_fsync_failure_never_publishes_final_name(self):
        target = self.root / "SUMMARY.json"
        with patch.object(latency.os, "fsync", side_effect=OSError("injected fsync failure")):
            with self.assertRaisesRegex(OSError, "fsync failure"):
                latency.write_json(target, {"status": "complete"})
        self.assertFalse(target.exists())
        pending = list(self.root.glob(".SUMMARY.json.pending-*"))
        self.assertEqual(len(pending), 1)
        self.assertEqual(json.loads(pending[0].read_text()), {"status": "complete"})

    def test_receipt_publication_failure_and_existing_target_never_overwrite(self):
        target = self.root / "SUMMARY.json"
        with patch.object(latency.os, "link", side_effect=OSError("injected publication failure")):
            with self.assertRaisesRegex(OSError, "publication failure"):
                latency.write_json(target, {"status": "complete"})
        self.assertFalse(target.exists())
        target.write_bytes(b"existing immutable receipt")
        with self.assertRaises(FileExistsError):
            latency.write_json(target, {"status": "replacement"})
        self.assertEqual(target.read_bytes(), b"existing immutable receipt")

    def test_full_execute_summary_fsync_and_publication_failures_preserve_requests(self):
        original_write = latency.write_json
        for failure_point in ("fsync", "link"):
            with self.subTest(failure_point=failure_point):
                calls, modules = self.fake_runtime()
                output = self.root / ("failed-summary-" + failure_point)
                def write_with_failure(path, value):
                    if Path(path).name == "SUMMARY.json":
                        with patch.object(latency.os, failure_point, side_effect=OSError("injected summary persistence failure")):
                            return original_write(path, value)
                    return original_write(path, value)
                with patch.dict(sys.modules, modules), patch.dict("os.environ", {"WORLD_SIZE": "1"}), \
                        patch.object(latency, "write_json", side_effect=write_with_failure):
                    with self.assertRaisesRegex(OSError, "summary persistence failure"):
                        latency.execute(self.plan, latency.digest(self.plan), output)
                rows = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), 416)
                self.assertEqual(len(calls["generate"]), 416)
                self.assertFalse((output / "SUMMARY.json").exists())
                failure = json.loads((output / "FAILURE.json").read_text())
                self.assertEqual(failure["completed_requests"], 416)
                self.assertFalse(failure["complete_summary_written"])
                self.assertFalse(failure["automatic_retry"])

    def test_full_execute_cleanup_failure_cannot_reverse_summary_publication(self):
        original_write = latency.write_json
        calls, modules = self.fake_runtime()
        output = self.root / "summary-published-cleanup-failed"
        def write_with_cleanup_failure(path, value):
            if Path(path).name == "SUMMARY.json":
                with patch.object(Path, "unlink", side_effect=OSError("injected cleanup failure")):
                    return original_write(path, value)
            return original_write(path, value)
        with patch.dict(sys.modules, modules), patch.dict("os.environ", {"WORLD_SIZE": "1"}), \
                patch.object(latency, "write_json", side_effect=write_with_cleanup_failure):
            summary = latency.execute(self.plan, latency.digest(self.plan), output)
        self.assertEqual(len(calls["generate"]), 416)
        self.assertEqual(json.loads((output / "SUMMARY.json").read_text()), summary)
        self.assertFalse((output / "FAILURE.json").exists())
        pending = list(output.glob(".SUMMARY.json.pending-*"))
        self.assertEqual(len(pending), 1)
        self.assertTrue(pending[0].samefile(output / "SUMMARY.json"))


if __name__ == "__main__":
    unittest.main()
