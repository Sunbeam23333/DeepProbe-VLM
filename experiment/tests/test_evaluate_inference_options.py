"""CPU protocol tests for opt-in inference paths; no weights or media downloads."""
from contextlib import ExitStack, contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from deepprobe_vlm import evaluate


class FakeEvaluationModel:
    def __init__(self):
        self.deep_config = SimpleNamespace(
            num_frames=1, num_visual_tokens=1, max_text_tokens=16,
            language_model="tiny-cpu-backbone", num_loops=2,
        )
        self.greedy_generate = Mock(return_value=torch.tensor([[2]]))

    def to(self, device):
        if device.type != "cpu":
            raise AssertionError("This test must not allocate a GPU")
        return self

    def eval(self):
        return self


class EvaluationInferenceOptionsTest(unittest.TestCase):
    @contextmanager
    def evaluation_fixture(self):
        """Run the real CLI/protocol writer with isolated CPU input adapters."""
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            config = {"num_frames": 1, "pool_grid": 1, "num_loops": 2}
            (checkpoint / "deepprobe_config.json").write_text(json.dumps(config))
            (checkpoint / "training_provenance.json").write_text(json.dumps({
                "training_sources": ["training-only"], "tuning_sources": [],
            }))
            record = {
                "sample_id": "test:1", "source_video_id": "held-out-video",
                "split": "test", "video_path": "unused-video.mp4",
                "question": "What happens?", "choices": ["one", "two"], "answer": "B",
            }
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(record) + "\n")
            model = FakeEvaluationModel()
            tokenizer = SimpleNamespace(eos_token_id=0, decode=Mock(return_value="B"))
            batch = {
                "input_ids": torch.tensor([[1, 0, 2]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
                "visual_mask": torch.tensor([[False, True, False]]),
                "pixel_values": torch.zeros(1, 1, 3, 2, 2),
            }
            model_loader = stack.enter_context(patch.object(
                evaluate.DeepProbeVLM, "from_pretrained", return_value=model))
            manifest_loader = stack.enter_context(patch.object(
                evaluate, "load_manifest", return_value=[record]))
            stack.enter_context(patch.object(evaluate.AutoTokenizer, "from_pretrained", return_value=tokenizer))
            stack.enter_context(patch.object(evaluate.AutoImageProcessor, "from_pretrained", return_value=object()))
            stack.enter_context(patch.object(evaluate, "VideoManifestDataset", return_value=[{
                "pixel_values": batch["pixel_values"][0],
            }]))
            stack.enter_context(patch.object(evaluate, "MultimodalCollator", return_value=Mock(return_value=batch)))
            stack.enter_context(patch.object(torch.cuda, "is_available", return_value=False))
            stack.enter_context(patch.object(evaluate, "synchronise"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            yield SimpleNamespace(
                root=root, checkpoint=checkpoint, manifest=manifest, config=config,
                model=model, model_loader=model_loader, manifest_loader=manifest_loader,
                tokenizer=tokenizer, batch=batch, record=record,
            )

    def run_evaluation(self, fixture, name="predictions", options=()):
        output = fixture.root / (name + ".jsonl")
        result = evaluate.main([
            "--checkpoint", str(fixture.checkpoint), "--manifest", str(fixture.manifest),
            "--output", str(output), "--max-new-tokens", "4", *options,
        ])
        self.assertEqual(result, 0)
        protocol = json.loads(output.with_suffix(".protocol.json").read_text())
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["prediction"], "B")
        self.assertEqual(rows[0]["correct"], 1)
        self.assertEqual(rows[0]["protocol_sha256"], evaluate.canonical_sha256(protocol))
        return output, protocol, rows[0]

    def test_defaults_preserve_protocol_and_generator_arguments(self):
        with self.evaluation_fixture() as fixture:
            _, protocol, row = self.run_evaluation(fixture)
            self.assertNotIn("inference_options", protocol)
            self.assertNotIn("inference_options", row)
            self.assertEqual(row["timing_status"], "raw_cold_or_mixed_no_warmup")
            kwargs = fixture.model.greedy_generate.call_args.kwargs
            self.assertEqual(set(kwargs), set(fixture.batch) | {
                "max_new_tokens", "eos_token_id", "use_cache",
            })
            self.assertEqual(kwargs["max_new_tokens"], 4)
            self.assertEqual(kwargs["eos_token_id"], 0)
            self.assertIs(kwargs["use_cache"], True)
            for key, value in fixture.batch.items():
                torch.testing.assert_close(kwargs[key], value)

            # Spelling the default backend explicitly must not create a new
            # scientific protocol or inject even a default-valued model kwarg.
            _, explicit_protocol, explicit_row = self.run_evaluation(
                fixture, "explicit-reference", ("--pool-backend", "reference"))
            self.assertEqual(explicit_protocol, protocol)
            self.assertEqual(explicit_row["protocol_sha256"], row["protocol_sha256"])
            self.assertNotIn("pool_backend", fixture.model.greedy_generate.call_args.kwargs)

    def test_nondefault_options_are_recorded_and_passed_to_generation(self):
        cases = [
            (("--last-token-logits",), {"last_token_logits": True}),
            (("--pool-backend", "triton"), {"pool_backend": "triton"}),
            (("--profile-ranges",), {"profile_ranges": True}),
            (("--last-token-logits", "--pool-backend", "triton", "--profile-ranges"),
             {"last_token_logits": True, "pool_backend": "triton", "profile_ranges": True}),
        ]
        for options, expected in cases:
            with self.subTest(options=options), self.evaluation_fixture() as fixture:
                _, base_protocol, _ = self.run_evaluation(fixture, "reference")
                _, protocol, row = self.run_evaluation(fixture, "optimized", options)
                self.assertEqual(protocol["inference_options"], expected)
                self.assertEqual(row["inference_options"], expected)
                self.assertNotEqual(evaluate.canonical_sha256(base_protocol), row["protocol_sha256"])
                kwargs = fixture.model.greedy_generate.call_args.kwargs
                for key in ("last_token_logits", "pool_backend", "profile_ranges"):
                    if key in expected:
                        self.assertEqual(kwargs[key], expected[key])
                    else:
                        self.assertNotIn(key, kwargs)
                expected_timing = ("diagnostic_ranges_not_benchmark" if expected.get("profile_ranges")
                                   else "raw_cold_or_mixed_no_warmup")
                self.assertEqual(row["timing_status"], expected_timing)

    def test_frozen_videomme_rejects_each_option_before_loading_inputs(self):
        for options in (("--last-token-logits",), ("--pool-backend", "triton"), ("--profile-ranges",)):
            with self.subTest(options=options), patch.object(evaluate, "load_manifest") as load:
                with patch.object(evaluate.DeepProbeVLM, "from_pretrained") as model_load:
                    with self.assertRaisesRegex(ValueError, "Frozen Video-MME protocol forbids"):
                        evaluate.main([
                            "--checkpoint", "/nonexistent/checkpoint", "--manifest", "/nonexistent/manifest",
                            "--output", "/nonexistent/predictions.jsonl", "--protocol", evaluate.VIDEOMME_PROTOCOL,
                            "--max-new-tokens", "16", *options,
                        ])
                    load.assert_not_called()
                    model_load.assert_not_called()

    def test_resume_changed_options_rejects_before_model_load_or_generation(self):
        cases = [
            ((), ("--last-token-logits",)),
            ((), ("--pool-backend", "triton")),
            ((), ("--profile-ranges",)),
            (("--last-token-logits",), ()),
            (("--pool-backend", "triton"), ()),
            (("--profile-ranges",), ()),
        ]
        for initial, resumed in cases:
            with self.subTest(initial=initial, resumed=resumed), self.evaluation_fixture() as fixture:
                output, _, _ = self.run_evaluation(fixture, options=initial)
                paths = [output, output.with_suffix(".protocol.json"), output.with_suffix(".summary.json")]
                original = {path: path.read_bytes() for path in paths}
                fixture.model_loader.reset_mock()
                fixture.model.greedy_generate.reset_mock()
                with self.assertRaisesRegex(ValueError, "Existing receipt differs"):
                    self.run_evaluation(fixture, options=("--resume", *resumed))
                fixture.model_loader.assert_not_called()
                fixture.model.greedy_generate.assert_not_called()
                for path, payload in original.items():
                    self.assertEqual(path.read_bytes(), payload)

    def test_identical_optin_resume_keeps_durable_predictions(self):
        with self.evaluation_fixture() as fixture:
            options = ("--last-token-logits", "--profile-ranges")
            output, protocol, row = self.run_evaluation(fixture, options=options)
            payload = output.read_bytes()
            fixture.model.greedy_generate.reset_mock()
            _, resumed_protocol, resumed_row = self.run_evaluation(fixture, options=("--resume", *options))
            self.assertEqual(resumed_protocol, protocol)
            self.assertEqual(resumed_row, row)
            self.assertEqual(output.read_bytes(), payload)
            fixture.model.greedy_generate.assert_not_called()

    def test_profiled_branch_preserves_options_and_marks_diagnostic_timing(self):
        with self.evaluation_fixture() as fixture, patch.object(torch.profiler, "profile") as profile:
            trace = profile.return_value.__enter__.return_value
            output, protocol, row = self.run_evaluation(fixture, options=(
                "--profile", "--last-token-logits", "--pool-backend", "triton", "--profile-ranges",
            ))
            expected = {"last_token_logits": True, "pool_backend": "triton", "profile_ranges": True}
            self.assertEqual(protocol["inference_options"], expected)
            self.assertEqual(row["inference_options"], expected)
            self.assertEqual(row["timing_status"], "diagnostic_ranges_not_benchmark")
            kwargs = fixture.model.greedy_generate.call_args.kwargs
            self.assertEqual({key: kwargs[key] for key in expected}, expected)
            trace.export_chrome_trace.assert_called_once_with(str(output.with_suffix(".trace.json")))
            self.assertEqual(profile.call_args.kwargs["activities"], [torch.profiler.ProfilerActivity.CPU])


if __name__ == "__main__":
    unittest.main()
