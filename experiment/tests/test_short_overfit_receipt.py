"""CPU-only checks that failed short-overfit measurements leave JSON evidence."""
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/operations/verify_short_overfit.py"
SPEC = importlib.util.spec_from_file_location("verify_short_overfit_receipt", SCRIPT)
overfit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(overfit)


class ShortOverfitFailureReceiptTest(unittest.TestCase):
    @staticmethod
    def metrics(mean_nll, accuracy):
        return {
            "samples": [{"sample_id": "tiny-1", "mean_nll": mean_nll, "finite": True}],
            "all_finite": True, "supervised_tokens": 2,
            "token_weighted_mean_nll": mean_nll,
            "teacher_forced_token_accuracy": accuracy,
        }

    def test_nonfinite_and_malformed_metrics_fail_without_invalid_json(self):
        before = self.metrics(2.0, 0.0)
        for final_nll, final_accuracy in ((math.nan, 1.0), (math.inf, 1.0),
                                           (0.4, math.nan), (None, 1.0),
                                           (0.4, math.inf)):
            with self.subTest(final_nll=final_nll, final_accuracy=final_accuracy):
                after = self.metrics(final_nll, final_accuracy)
                result = overfit.evaluate_acceptance(
                    before, after, max_loss_ratio=0.5, min_final_token_accuracy=0.95)
                self.assertEqual(result["status"], "failed")
                self.assertFalse(result["checks"]["all_metrics_finite"])
                json.dumps(result, allow_nan=False)

    def test_ratio_overflow_fails_without_infinity_in_receipt(self):
        result = overfit.evaluate_acceptance(
            self.metrics(1e-320, 0.0), self.metrics(1.0, 1.0),
            max_loss_ratio=0.5, min_final_token_accuracy=0.95)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["final_to_initial_loss_ratio"])
        json.dumps(result, allow_nan=False)

    def test_main_writes_failed_json_if_metric_contains_nan_or_inf(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "short_overfit.json"

            def inject_nonfinite_metrics(_args, report):
                report.update({
                    "status": "short_overfit_engineering_gate_passed",
                    "short_overfit_engineering_gate_passed": True,
                    "before": {"token_weighted_mean_nll": math.nan},
                    "after": {"samples": [{"mean_nll": math.inf}]},
                    "acceptance": {"status": "passed", "checks": {"all_metrics_finite": True}},
                })

            args = ["--config", "config.json", "--train-manifest", "tiny.jsonl",
                    "--media-root", ".", "--initial-checkpoint", "initial",
                    "--final-checkpoint", "final", "--training-output", "training",
                    "--expected-final-step", "10", "--output", str(output)]
            with mock.patch.object(overfit, "run", side_effect=inject_nonfinite_metrics), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = overfit.main(args)
            self.assertEqual(code, 1)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "short_overfit_engineering_gate_failed")
            self.assertFalse(report["short_overfit_engineering_gate_passed"])
            self.assertIsNone(report["before"]["token_weighted_mean_nll"])
            self.assertIsNone(report["after"]["samples"][0]["mean_nll"])
            self.assertEqual(report["acceptance"]["status"], "failed")
            self.assertFalse(report["acceptance"]["checks"]["all_metrics_finite"])
            self.assertIn("$.before.token_weighted_mean_nll", report["errors"][0])

    def test_main_writes_error_json_for_metric_validation_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "short_overfit.json"
            args = ["--config", "config.json", "--train-manifest", "tiny.jsonl",
                    "--media-root", ".", "--initial-checkpoint", "initial",
                    "--final-checkpoint", "final", "--training-output", "training",
                    "--expected-final-step", "10", "--output", str(output)]
            with mock.patch.object(overfit, "run", side_effect=ValueError("invalid metric")), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = overfit.main(args)
            self.assertEqual(code, 1)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "error")
            self.assertFalse(report["short_overfit_engineering_gate_passed"])
            self.assertIn("invalid metric", report["errors"][0])

    def test_main_preserves_partial_nonfinite_measurement_on_later_error(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "short_overfit.json"

            def fail_after_initial_measurement(_args, report):
                report["before"] = {"samples": [{"sample_id": "tiny-1", "mean_nll": math.inf}]}
                raise RuntimeError("final measurement failed")

            args = ["--config", "config.json", "--train-manifest", "tiny.jsonl",
                    "--media-root", ".", "--initial-checkpoint", "initial",
                    "--final-checkpoint", "final", "--training-output", "training",
                    "--expected-final-step", "10", "--output", str(output)]
            with mock.patch.object(overfit, "run", side_effect=fail_after_initial_measurement), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(overfit.main(args), 1)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "error")
            self.assertIsNone(report["before"]["samples"][0]["mean_nll"])
            self.assertTrue(any("final measurement failed" in error for error in report["errors"]))
            self.assertTrue(any("non-finite metric" in error for error in report["errors"]))

    def test_measure_skips_loss_and_accuracy_for_nonfinite_logits(self):
        class Scalar:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

            def all(self):
                return self

        class Labels:
            def to(self, _device):
                return self

            def __getitem__(self, _index):
                return self

            def __ne__(self, _value):
                return SimpleNamespace(sum=lambda: Scalar(2))

        class Logits:
            def __getitem__(self, _index):
                return self

            def float(self):
                return self

        class Model:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

            def to(self, _device):
                return self

            def set_stage(self, _stage):
                pass

            def eval(self):
                pass

            def __call__(self, **_batch):
                return SimpleNamespace(logits=Logits())

        class Dataset:
            def __init__(self, *_args):
                pass

            def __getitem__(self, _index):
                return None

        class Collator:
            def __init__(self, *_args, **_kwargs):
                pass

            def __call__(self, _items):
                return {"labels": Labels()}

        torch = ModuleType("torch")
        torch.float32 = "float32"
        torch.no_grad = contextlib.nullcontext
        torch.isfinite = lambda _value: Scalar(False)
        functional = ModuleType("torch.nn.functional")
        functional.cross_entropy = mock.Mock(side_effect=AssertionError("loss must be skipped"))
        nn = ModuleType("torch.nn")
        nn.functional = functional
        package = ModuleType("deepprobe_vlm")
        package.__path__ = []
        data = ModuleType("deepprobe_vlm.data")
        data.VideoManifestDataset = Dataset
        data.MultimodalCollator = Collator
        modeling = ModuleType("deepprobe_vlm.modeling")
        modeling.DeepProbeVLM = Model
        modules = {"torch": torch, "torch.nn": nn, "torch.nn.functional": functional,
                   "deepprobe_vlm": package, "deepprobe_vlm.data": data,
                   "deepprobe_vlm.modeling": modeling}
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, modules):
            result = overfit.measure(
                Path(directory), config=SimpleNamespace(num_frames=1, num_visual_tokens=1,
                                                        max_text_tokens=2),
                records=[{"sample_id": "tiny-1", "source_video_id": "source-1"}],
                tokenizer=None, processor=None, media_root=Path(directory),
                device=SimpleNamespace(type="cpu"), bf16=False)
        self.assertFalse(result["all_finite"])
        self.assertEqual(result["samples"][0]["nonfinite_reason"], "logits")
        self.assertIsNone(result["samples"][0]["mean_nll"])
        self.assertIsNone(result["teacher_forced_token_accuracy"])
        functional.cross_entropy.assert_not_called()
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
