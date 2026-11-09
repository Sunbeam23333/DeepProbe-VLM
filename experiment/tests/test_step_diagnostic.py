"""CPU evidence that the opt-in step trace observes without changing updates."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import TrainingArguments

from deepprobe_vlm.step_diagnostic import (
    DiagnosticCollator, FirstStepTrace, SAMPLE_ID_KEY, _full_tensor_sha256, _optimizer_snapshot,
    _rng_sha256, ddp_bucket_snapshot, summarize_tensors,
)
from deepprobe_vlm.train import DeepProbeTrainer, DiagnosticDeepProbeTrainer


COMPARATOR_PATH = Path(__file__).resolve().parents[1] / "scripts/operations/compare_step_diagnostic.py"
COMPARATOR_SPEC = importlib.util.spec_from_file_location("compare_step_diagnostic", COMPARATOR_PATH)
COMPARATOR = importlib.util.module_from_spec(COMPARATOR_SPEC)
COMPARATOR_SPEC.loader.exec_module(COMPARATOR)


class TinyRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, input_ids, labels):
        return {"loss": (self.linear(input_ids.float()) - labels.float()).square().mean()}

    def save_pretrained(self, output_dir):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")

    def load_checkpoint_weights(self, checkpoint):
        self.load_state_dict(torch.load(Path(checkpoint) / "model.pt", weights_only=True))


class TinyLanguageWithEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 2)

    def get_input_embeddings(self):
        return self.embedding


class TinyFrozenEmbeddingModel(nn.Module):
    """Projector-only training with a DeepProbe-style frozen LM boundary."""

    def __init__(self, stage):
        super().__init__()
        self.stage = stage
        self.language_model = TinyLanguageWithEmbedding()
        self.language_model.requires_grad_(False)
        self.projector = nn.Linear(2, 1)

    def prepare_embeddings(self, input_ids):
        return self.language_model.get_input_embeddings()(input_ids)

    def forward(self, input_ids, labels):
        prediction = self.projector(self.prepare_embeddings(input_ids)).mean(dim=1)
        return {"loss": (prediction - labels).square().mean()}


def collate_records(examples):
    return {"input_ids": torch.stack([item["input_ids"] for item in examples]),
            "labels": torch.stack([item["labels"] for item in examples])}


class StepDiagnosticTest(unittest.TestCase):
    def test_ddp_bucket_receipt_reports_active_layout_without_becoming_a_gate(self):
        class FakeDDP:
            def __init__(self, rebuilt):
                self._has_rebuilt_buckets = rebuilt

            def _get_ddp_logging_data(self):
                return {"bucket_sizes": "156420", "rebuilt_bucket_sizes": "25348, 131072",
                        "has_rebuilt_buckets": int(self._has_rebuilt_buckets),
                        "find_unused_parameters": 0, "unrelated_private_value": object()}

        self.assertEqual(ddp_bucket_snapshot(nn.Linear(1, 1)), {"status": "not_ddp"})
        with patch("deepprobe_vlm.step_diagnostic.torch.nn.parallel.DistributedDataParallel", FakeDDP):
            initial = ddp_bucket_snapshot(FakeDDP(False))
            rebuilt = ddp_bucket_snapshot(FakeDDP(True))
        self.assertEqual(initial["active_bucket_sizes"], "156420")
        self.assertEqual(rebuilt["active_bucket_sizes"], "25348, 131072")
        self.assertNotIn("unrelated_private_value", rebuilt)
        json.dumps(rebuilt)

    def test_frozen_alignment_embedding_skips_tap_without_changing_update(self):
        records = [{"record": {"sample_id": "alignment-sample"},
                    "input_ids": torch.tensor([0, 1]), "labels": torch.tensor([0.5])}]
        torch.manual_seed(71)
        original = TinyFrozenEmbeddingModel(stage="alignment")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = dict(max_steps=1, per_device_train_batch_size=1,
                          gradient_accumulation_steps=1, learning_rate=0.01,
                          optim="adamw_torch", use_cpu=True, report_to="none",
                          save_strategy="no", eval_strategy="no", logging_strategy="no",
                          disable_tqdm=True, remove_unused_columns=False,
                          dataloader_num_workers=0)
            baseline = DeepProbeTrainer(
                model=deepcopy(original),
                args=TrainingArguments(output_dir=str(root / "baseline"), **common),
                train_dataset=records, data_collator=collate_records)
            torch.manual_seed(73)
            baseline.train()
            trace = FirstStepTrace(root / "alignment", target_step=1, rank=0,
                                   world_size=1, resume_checkpoint=None)
            diagnostic = DiagnosticDeepProbeTrainer(
                model=deepcopy(original),
                args=TrainingArguments(output_dir=str(root / "alignment"), **common),
                train_dataset=records, data_collator=DiagnosticCollator(collate_records),
                step_trace=trace)
            torch.manual_seed(73)
            diagnostic.train()
            for name, reference in baseline.model.state_dict().items():
                torch.testing.assert_close(diagnostic.model.state_dict()[name], reference,
                                           atol=0, rtol=0)
            torch.testing.assert_close(diagnostic.optimizer.state_dict(),
                                       baseline.optimizer.state_dict(), atol=0, rtol=0)
            receipt = json.loads(trace.path.read_text())
            self.assertEqual(receipt["status"], "complete")
            self.assertNotIn("embedding_gradient_boundary", receipt["microbatches"][0])
            self.assertEqual(receipt["microbatches"][0]["ddp_bucket_before_forward"],
                             {"status": "not_ddp"})
            self.assertEqual(receipt["microbatches"][0]["ddp_bucket_after_backward"],
                             {"status": "not_ddp"})
            self.assertIn("after_backward_before_clip", receipt["events"])

    def test_frozen_sft_embedding_diagnostic_fails_closed(self):
        records = [{"record": {"sample_id": "invalid-sft"},
                    "input_ids": torch.tensor([0, 1]), "labels": torch.tensor([0.5])}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = FirstStepTrace(root, target_step=1, rank=0, world_size=1,
                                   resume_checkpoint=None)
            trainer = DiagnosticDeepProbeTrainer(
                model=TinyFrozenEmbeddingModel(stage="sft"),
                args=TrainingArguments(output_dir=str(root), max_steps=1,
                                       per_device_train_batch_size=1, use_cpu=True,
                                       report_to="none", save_strategy="no", eval_strategy="no",
                                       logging_strategy="no", disable_tqdm=True,
                                       remove_unused_columns=False, dataloader_num_workers=0),
                train_dataset=records, data_collator=DiagnosticCollator(collate_records),
                step_trace=trace)
            with self.assertRaisesRegex(RuntimeError, "SFT requires a trainable input embedding"):
                trainer.train()

    def test_tensor_fingerprints_are_bounded_sensitive_and_rng_neutral(self):
        torch.manual_seed(29)
        values = torch.arange(10000, dtype=torch.float32)
        before_rng = _rng_sha256()
        first = summarize_tensors([("values", values), ("empty", torch.empty(0))])
        self.assertEqual(_rng_sha256(), before_rng)
        self.assertEqual(first["tensors"]["values"]["sample_policy"], "first_middle_last_8")
        self.assertEqual(first["tensors"]["values"]["sampled_elements"], 24)
        self.assertEqual(first["tensors"]["empty"]["sample_policy"], "complete")
        self.assertEqual(first["tensors"]["empty"]["sampled_elements"], 0)
        self.assertEqual(first, summarize_tensors([("values", values), ("empty", torch.empty(0))]))
        values[5000] += 0.25
        second = summarize_tensors([("values", values)])
        self.assertNotEqual(first["tensors"]["values"]["sample_sha256"],
                            second["tensors"]["values"]["sample_sha256"])
        self.assertNotEqual(first["tensors"]["values"]["sum"],
                            second["tensors"]["values"]["sum"])
        self.assertEqual(_rng_sha256(), before_rng)

    def test_optimizer_state_is_mapped_to_model_parameter_names(self):
        torch.manual_seed(13)
        model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        model(torch.ones(1, 3)).sum().backward()
        optimizer.step()
        reference = _optimizer_snapshot(model, optimizer)
        self.assertEqual(reference["tensor_states"]["tensor_count"], 12)
        self.assertEqual(reference["param_groups"][0]["parameter_names"],
                         ["0.weight", "0.bias", "1.weight", "1.bias"])
        same_model = deepcopy(model)
        same_optimizer = torch.optim.AdamW(same_model.parameters(), lr=0.01)
        same_optimizer.load_state_dict(optimizer.state_dict())
        self.assertEqual(reference, _optimizer_snapshot(same_model, same_optimizer))
        wrong_model = deepcopy(model)
        wrong_optimizer = torch.optim.AdamW(reversed(list(wrong_model.parameters())), lr=0.01)
        wrong_optimizer.load_state_dict(optimizer.state_dict())
        self.assertNotEqual(reference["param_groups"],
                            _optimizer_snapshot(wrong_model, wrong_optimizer)["param_groups"])

    def test_traced_trainer_keeps_model_inputs_and_update_unchanged(self):
        records = [{"record": {"sample_id": "actual-id-1"},
                    "input_ids": torch.tensor([1., 2.]), "labels": torch.tensor([0.5])}]
        wrapped = DiagnosticCollator(collate_records)
        batch = wrapped(records)
        self.assertEqual(batch[SAMPLE_ID_KEY], ["actual-id-1"])
        torch.manual_seed(17)
        original = TinyRegression()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common_args = dict(max_steps=1, per_device_train_batch_size=1,
                               gradient_accumulation_steps=1, learning_rate=0.01,
                               optim="adamw_torch", use_cpu=True, report_to="none",
                               save_strategy="no", eval_strategy="no",
                               disable_tqdm=True, remove_unused_columns=False,
                               dataloader_num_workers=0)
            baseline = DeepProbeTrainer(
                model=deepcopy(original),
                args=TrainingArguments(output_dir=str(root / "baseline"), **common_args),
                train_dataset=records, data_collator=collate_records)
            torch.manual_seed(23)
            baseline.train()
            trace = FirstStepTrace(root / "traced", target_step=1, rank=0,
                                   world_size=1, resume_checkpoint=None)
            observed = DiagnosticDeepProbeTrainer(
                model=deepcopy(original),
                args=TrainingArguments(output_dir=str(root / "traced"), **common_args),
                train_dataset=records, data_collator=wrapped, step_trace=trace)
            torch.manual_seed(23)
            observed.train()
            for name, value in baseline.model.state_dict().items():
                torch.testing.assert_close(value, observed.model.state_dict()[name], atol=0, rtol=0)
            torch.testing.assert_close(baseline.optimizer.state_dict(),
                                       observed.optimizer.state_dict(), atol=0, rtol=0)
            report = json.loads(trace.path.read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["microbatches"][0]["sample_ids"], ["actual-id-1"])
            self.assertEqual(set(report["microbatches"][0]["input_tensors"]),
                             {"input_ids", "labels"})
            self.assertIn("raw_forward_loss", report["microbatches"][0]["loss"])
            self.assertIn("after_backward_before_clip", report["events"])
            self.assertIn("after_clip_before_optimizer", report["events"])
            self.assertEqual(report["events"]["before_forward"]["optimizer"]
                             ["tensor_states"]["tensor_count"], 0)
            self.assertGreater(report["events"]["after_optimizer_before_scheduler"]
                               ["optimizer"]["tensor_states"]["tensor_count"], 0)

    def test_resume_trace_starts_after_optimizer_load_at_same_step_as_control(self):
        records = [
            {"record": {"sample_id": f"actual-id-{index}"},
             "input_ids": torch.tensor([float(index), float(index + 1)]),
             "labels": torch.tensor([float(index) / 3])}
            for index in range(4)
        ]
        torch.manual_seed(7)
        initial = TinyRegression()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common_args = dict(max_steps=2, per_device_train_batch_size=1,
                               gradient_accumulation_steps=1, learning_rate=0.01,
                               optim="adamw_torch", use_cpu=True, report_to="none",
                               save_strategy="steps", save_steps=1, eval_strategy="no",
                               disable_tqdm=True, remove_unused_columns=False,
                               dataloader_num_workers=0)

            def make_trainer(branch):
                output = root / branch
                trace = FirstStepTrace(output, target_step=2, rank=0, world_size=1,
                                       resume_checkpoint=str(root / "control/checkpoint-1")
                                       if branch == "resumed" else None)
                trainer = DiagnosticDeepProbeTrainer(
                    model=deepcopy(initial),
                    args=TrainingArguments(output_dir=str(output), **common_args),
                    train_dataset=records, data_collator=DiagnosticCollator(collate_records),
                    step_trace=trace)
                return trainer, trace

            control, control_trace = make_trainer("control")
            torch.manual_seed(31)
            control.train()
            self.assertTrue((root / "control/checkpoint-1/optimizer.pt").is_file())
            resumed, resumed_trace = make_trainer("resumed")
            torch.manual_seed(31)
            resumed.train(resume_from_checkpoint=str(root / "control/checkpoint-1"))
            first = json.loads(control_trace.path.read_text())
            second = json.loads(resumed_trace.path.read_text())
            self.assertEqual(first["status"], "complete")
            self.assertEqual(second["status"], "complete")
            self.assertEqual(first["events"]["before_forward"]["model"],
                             second["events"]["before_forward"]["model"])
            self.assertEqual(first["events"]["before_forward"]["optimizer"],
                             second["events"]["before_forward"]["optimizer"])
            self.assertEqual(first["events"]["before_forward"]["rng_sha256_before_forward"],
                             second["events"]["before_forward"]["rng_sha256_before_forward"])
            self.assertGreater(first["events"]["before_forward"]["optimizer"]
                               ["tensor_states"]["tensor_count"], 0)
            self.assertEqual(first["microbatches"][0]["sample_ids"],
                             second["microbatches"][0]["sample_ids"])
            self.assertEqual(first["microbatches"][0]["input_batch_sha256"],
                             second["microbatches"][0]["input_batch_sha256"])
            selected = _full_tensor_sha256(torch.tensor([0, 1], dtype=torch.int64))
            activation = _full_tensor_sha256(torch.ones(1, 2, 2))
            rows = _full_tensor_sha256(torch.ones(2, 2))
            boundary = {
                "schema_version": 1,
                "purpose": "P0_05_EMBEDDING_GRADIENT_BOUNDARY_NOT_ACCEPTANCE",
                "selected_token_ids": selected,
                "selected_token_count": 2,
                "gradients": {
                    "multimodal_output": deepcopy(activation),
                    "lookup_output": deepcopy(activation),
                    "local_embedding_weight_rows": deepcopy(rows),
                },
                "post_backward_embedding_weight_rows": deepcopy(rows),
                "post_backward_ddp_synchronized": True,
            }
            first["microbatches"][0]["embedding_gradient_boundary"] = deepcopy(boundary)
            second["microbatches"][0]["embedding_gradient_boundary"] = deepcopy(boundary)
            control_trace.path.write_text(json.dumps(first))
            resumed_trace.path.write_text(json.dumps(second))
            comparison = COMPARATOR.compare(root / "control", root / "resumed", step=2, world_size=1)
            self.assertEqual(comparison["status"], "no_observed_difference")
            valid_first, valid_second = deepcopy(first), deepcopy(second)
            second["microbatches"][0]["embedding_gradient_boundary"]["gradients"][
                "multimodal_output"]["sha256"] = "0" * 64
            resumed_trace.path.write_text(json.dumps(second))
            difference = COMPARATOR.compare(root / "control", root / "resumed", step=2, world_size=1)
            self.assertEqual(difference["earliest_observed_difference"]["first_difference"]["phase"],
                             "multimodal_embedding_gradient")
            second["microbatches"][0]["sample_ids"] = ["changed-id"]
            resumed_trace.path.write_text(json.dumps(second))
            difference = COMPARATOR.compare(root / "control", root / "resumed", step=2, world_size=1)
            self.assertEqual(difference["status"], "observed_difference")
            self.assertEqual(difference["earliest_observed_difference"]["first_difference"]["phase"],
                             "actual_sample_ids")

            first["microbatches"][0].pop("embedding_gradient_boundary")
            second["microbatches"][0].pop("embedding_gradient_boundary")
            control_trace.path.write_text(json.dumps(first))
            resumed_trace.path.write_text(json.dumps(second))
            with self.assertRaisesRegex(ValueError, "missing a valid embedding gradient boundary"):
                COMPARATOR.compare(root / "control", root / "resumed", step=2, world_size=1)

            first["microbatches"][0]["embedding_gradient_boundary"] = deepcopy(boundary)
            second["microbatches"][0]["embedding_gradient_boundary"] = deepcopy(boundary)
            second["microbatches"][0]["embedding_gradient_boundary"][
                "post_backward_ddp_synchronized"] = False
            control_trace.path.write_text(json.dumps(first))
            resumed_trace.path.write_text(json.dumps(second))
            with self.assertRaisesRegex(ValueError, "was not DDP synchronized"):
                COMPARATOR.compare(root / "control", root / "resumed", step=2, world_size=1)

            second["microbatches"][0]["embedding_gradient_boundary"][
                "post_backward_ddp_synchronized"] = True
            second["microbatches"].append(deepcopy(second["microbatches"][0]))
            second["microbatches"][1].pop("embedding_gradient_boundary")
            resumed_trace.path.write_text(json.dumps(second))
            with self.assertRaisesRegex(ValueError, "microbatch 1 is missing"):
                COMPARATOR.compare(root / "control", root / "resumed", step=2, world_size=1)

            first["microbatches"] = []
            second["microbatches"] = []
            control_trace.path.write_text(json.dumps(first))
            resumed_trace.path.write_text(json.dumps(second))
            with self.assertRaisesRegex(ValueError, "no diagnostic microbatches"):
                COMPARATOR.compare(root / "control", root / "resumed", step=2, world_size=1)

            first, second = deepcopy(valid_first), deepcopy(valid_second)
            for receipt in (first, second):
                earlier = deepcopy(receipt["microbatches"][0])
                earlier["embedding_gradient_boundary"]["post_backward_ddp_synchronized"] = False
                receipt["microbatches"].insert(0, earlier)
            control_trace.path.write_text(json.dumps(first))
            resumed_trace.path.write_text(json.dumps(second))
            comparison = COMPARATOR.compare(root / "control", root / "resumed", step=2, world_size=1)
            self.assertEqual(comparison["status"], "no_observed_difference")

            second["microbatches"][0]["embedding_gradient_boundary"][
                "post_backward_ddp_synchronized"] = True
            resumed_trace.path.write_text(json.dumps(second))
            difference = COMPARATOR.compare(root / "control", root / "resumed", step=2, world_size=1)
            self.assertEqual(difference["earliest_observed_difference"]["first_difference"]["phase"],
                             "embedding_ddp_sync")

            second["microbatches"][0]["embedding_gradient_boundary"][
                "post_backward_ddp_synchronized"] = 1
            resumed_trace.path.write_text(json.dumps(second))
            with self.assertRaisesRegex(ValueError, "no Boolean DDP synchronization flag"):
                COMPARATOR.compare(root / "control", root / "resumed", step=2, world_size=1)


if __name__ == "__main__":
    unittest.main()
