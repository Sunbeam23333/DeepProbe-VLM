"""Optional pinned upstream-code contract test (tiny random weights, no weights download).

Run with DEEPPROBE_TEST_NATIVE_OURO=1 to fetch reviewed, pinned Python model code.
"""
import os
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import AutoConfig, AutoModelForCausalLM, SiglipVisionConfig, SiglipVisionModel, TrainingArguments

from deepprobe_vlm.modeling import DeepProbeConfig, DeepProbeVLM, OURO_REVISION
from deepprobe_vlm.step_diagnostic import DiagnosticCollator, FirstStepTrace
from deepprobe_vlm.train import DeepProbeTrainer, DiagnosticDeepProbeTrainer


COMPARATOR_PATH = Path(__file__).resolve().parents[1] / "scripts/operations/compare_step_diagnostic.py"
COMPARATOR_SPEC = importlib.util.spec_from_file_location("compare_step_diagnostic_native", COMPARATOR_PATH)
COMPARATOR = importlib.util.module_from_spec(COMPARATOR_SPEC)
COMPARATOR_SPEC.loader.exec_module(COMPARATOR)


@unittest.skipUnless(os.environ.get("DEEPPROBE_TEST_NATIVE_OURO") == "1", "opt-in pinned remote code test")
class NativeOuroTest(unittest.TestCase):
    def test_native_eager_full_bptt_checkpointing_and_per_loop_cache(self):
        self._run_native_contract("eager")

    def test_native_sdpa_full_bptt_checkpointing_and_per_loop_cache(self):
        self._run_native_contract("sdpa")

    def test_diagnostic_trainer_captures_real_ouro_gradient_boundary(self):
        torch.set_num_threads(1)
        torch.manual_seed(37)
        config = AutoConfig.from_pretrained("ByteDance/Ouro-1.4B", revision=OURO_REVISION, trust_remote_code=True)
        config.hidden_size = 32
        config.intermediate_size = 64
        config.num_hidden_layers = 2
        config.num_attention_heads = 4
        config.num_key_value_heads = 4
        config.head_dim = 8
        config.vocab_size = 64
        config.max_position_embeddings = 128
        config.layer_types = ["full_attention"] * 2
        config.total_ut_steps = 2
        config._attn_implementation = "eager"
        language = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, torch_dtype=torch.float32, code_revision=OURO_REVISION)
        vision = SiglipVisionModel(SiglipVisionConfig(
            hidden_size=8, intermediate_size=16, num_hidden_layers=1,
            num_attention_heads=2, image_size=8, patch_size=4))
        model = DeepProbeVLM(DeepProbeConfig(num_frames=1, pool_grid=1, num_loops=2,
                                             freeze_vision=False, attn_implementation="eager"),
                             language, vision)
        model.config.pad_token_id = 0
        self.assertIsNone(model.language_model.get_input_embeddings().padding_idx)
        record = {
            "record": {"sample_id": "native-ouro-bos-zero"},
            "input_ids": torch.tensor([0, 0, 2, 3, 4]),
            "attention_mask": torch.ones(5, dtype=torch.long),
            "visual_mask": torch.tensor([False, True, False, False, False]),
            "pixel_values": torch.randn(1, 3, 8, 8),
            "labels": torch.tensor([-100, -100, -100, 3, 4]),
        }

        def collate(examples):
            return {key: torch.stack([example[key] for example in examples])
                    for key in ("input_ids", "attention_mask", "visual_mask", "pixel_values", "labels")}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = dict(max_steps=1, per_device_train_batch_size=1,
                          gradient_accumulation_steps=1, learning_rate=0.001,
                          optim="adamw_torch", use_cpu=True, report_to="none",
                          save_strategy="no", eval_strategy="no", logging_strategy="no",
                          disable_tqdm=True, remove_unused_columns=False,
                          dataloader_num_workers=0)
            baseline = DeepProbeTrainer(
                model=deepcopy(model),
                args=TrainingArguments(output_dir=str(root / "baseline"), **common),
                train_dataset=[record], data_collator=collate)
            torch.manual_seed(53)
            baseline.train()

            trace = FirstStepTrace(root / "diagnostic", target_step=1, rank=0,
                                   world_size=1, resume_checkpoint=None)
            diagnostic = DiagnosticDeepProbeTrainer(
                model=deepcopy(model),
                args=TrainingArguments(output_dir=str(root / "diagnostic"), **common),
                train_dataset=[record], data_collator=DiagnosticCollator(collate),
                step_trace=trace)
            torch.manual_seed(53)
            diagnostic.train()

            for name, reference in baseline.model.state_dict().items():
                torch.testing.assert_close(diagnostic.model.state_dict()[name], reference,
                                           atol=0, rtol=0)
            torch.testing.assert_close(diagnostic.optimizer.state_dict(),
                                       baseline.optimizer.state_dict(), atol=0, rtol=0)
            receipt = json.loads(trace.path.read_text())
            self.assertEqual(receipt["status"], "complete")
            self.assertEqual(receipt["target_optimizer_step"], 1)
            self.assertEqual(len(receipt["microbatches"]), 1)
            self.assertEqual(receipt["microbatches"][0]["sample_ids"], ["native-ouro-bos-zero"])
            boundary = receipt["microbatches"][0]["embedding_gradient_boundary"]
            self.assertEqual(boundary["purpose"], "P0_05_EMBEDDING_GRADIENT_BOUNDARY_NOT_ACCEPTANCE")
            self.assertEqual(boundary["selected_token_count"], 4)
            self.assertIs(boundary["post_backward_ddp_synchronized"], True)
            self.assertEqual(set(boundary["gradients"]), {
                "lookup_output", "multimodal_output", "local_embedding_weight_rows"})
            self.assertEqual(boundary["gradients"]["local_embedding_weight_rows"],
                             boundary["post_backward_embedding_weight_rows"])
            comparison = COMPARATOR.compare(root / "diagnostic", root / "diagnostic",
                                            step=1, world_size=1)
            self.assertEqual(comparison["status"], "no_observed_difference")

    def _run_native_contract(self, attention):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        config = AutoConfig.from_pretrained("ByteDance/Ouro-1.4B", revision=OURO_REVISION, trust_remote_code=True)
        config.hidden_size = 32
        config.intermediate_size = 64
        config.num_hidden_layers = 2
        config.num_attention_heads = 4
        config.num_key_value_heads = 4
        config.head_dim = 8
        config.vocab_size = 64
        config.max_position_embeddings = 128
        config.layer_types = ["full_attention"] * 2
        config.total_ut_steps = 3
        config._attn_implementation = attention
        language = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.float32,
                                                  code_revision=OURO_REVISION)
        vision = SiglipVisionModel(SiglipVisionConfig(hidden_size=8, intermediate_size=16,
                                                     num_hidden_layers=1, num_attention_heads=2,
                                                     image_size=8, patch_size=4))
        deep_config = DeepProbeConfig(num_frames=1, pool_grid=1, num_loops=3,
                                      freeze_vision=False, attn_implementation=attention)
        model = DeepProbeVLM(deep_config, language, vision)
        value = dict(input_ids=torch.tensor([[0, 0, 2, 3, 4]]), attention_mask=torch.ones(1, 5, dtype=torch.long),
                     visual_mask=torch.tensor([[False, True, False, False, False]]),
                     pixel_values=torch.randn(1, 1, 3, 8, 8), labels=torch.tensor([[-100, -100, -100, 3, 4]]))
        self.assertEqual(value["input_ids"][0, 0].item(), 0)
        self.assertFalse(value["visual_mask"][0, 0].item())
        model.gradient_checkpointing_enable({"use_reentrant": False})
        result = model(**value)
        result.loss.backward()
        self.assertTrue(torch.isfinite(result.loss))
        self.assertGreater(language.model.layers[0].self_attn.q_proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.projector.projection[0].weight.grad.abs().sum().item(), 0)
        self.assertEqual([name for name, p in model.named_parameters() if p.requires_grad and p.grad is None], [])
        language.gradient_checkpointing_disable()
        model.eval()
        supervised_value = dict(value)
        value.pop("labels")
        # Exercise the real pinned Ouro signature and head slicing, not only a
        # stub that accepts logits_to_keep. The native depth caches remain the
        # same while the opted-in head sees just the selected final position.
        generated = []
        for use_cache in (False, True):
            for last_token_logits in (False, True):
                head_lengths = []
                hook = language.lm_head.register_forward_pre_hook(
                    lambda _module, args: head_lengths.append(args[0].shape[1]))
                try:
                    generated.append(model.greedy_generate(
                        **value, max_new_tokens=5, use_cache=use_cache,
                        last_token_logits=last_token_logits))
                finally:
                    hook.remove()
                expected_lengths = [5, 1, 1, 1, 1] if use_cache else [5, 6, 7, 8, 9]
                self.assertEqual(head_lengths, [1] * 5 if last_token_logits else expected_lengths)
        for actual in generated[1:]:
            torch.testing.assert_close(actual, generated[0], atol=0, rtol=0)
        embeds = model.prepare_embeddings(**{k: value[k] for k in ("input_ids", "visual_mask", "pixel_values")})
        cached = model._language_forward(inputs_embeds=embeds, attention_mask=value["attention_mask"], use_cache=True)
        self.assertEqual(len(cached.past_key_values.key_cache), 6)
        # Trainer stores tokenizer PAD=0 in the shared Ouro config. Ouro also
        # uses token 0 as BOS/EOS, so checkpoint reload must retain its gradient.
        model.config.pad_token_id = 0
        model.train()
        model.zero_grad(set_to_none=True)
        rng_before_forward = torch.get_rng_state().clone()
        fresh_loss = model(**supervised_value).loss
        fresh_loss.backward()
        self.assertEqual([name for name, parameter in model.named_parameters()
                          if parameter.requires_grad and parameter.grad is None], [])
        fresh_gradients = {name: parameter.grad.detach().clone()
                           for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertGreater(model.language_model.get_input_embeddings().weight.grad[0].abs().sum().item(), 0)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = DeepProbeVLM.from_pretrained(directory).eval()
            in_place_language = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, torch_dtype=torch.float32, code_revision=OURO_REVISION)
            self.assertEqual(in_place_language.get_input_embeddings().padding_idx, 0)
            reloaded_in_place = DeepProbeVLM(
                deep_config, in_place_language, SiglipVisionModel(vision.config)).eval()
            reloaded_in_place.load_checkpoint_weights(directory)
            model.eval()
            expected = model(**value).logits
            torch.testing.assert_close(restored(**value).logits, expected, atol=1e-6, rtol=1e-6)
            for candidate in (restored, reloaded_in_place):
                self.assertEqual(candidate.config.pad_token_id, 0)
                embedding = candidate.language_model.get_input_embeddings()
                self.assertIsNone(embedding.padding_idx)
                self.assertIsNone(candidate.language_model.model.padding_idx)
                candidate.train()
                candidate.zero_grad(set_to_none=True)
                torch.set_rng_state(rng_before_forward.clone())
                candidate_loss = candidate(**supervised_value).loss
                torch.testing.assert_close(candidate_loss, fresh_loss, atol=0, rtol=0)
                candidate_loss.backward()
                self.assertGreater(embedding.weight.grad[0].abs().sum().item(), 0)
                candidate_gradients = {name: parameter.grad for name, parameter in candidate.named_parameters()
                                       if parameter.requires_grad}
                self.assertEqual(set(candidate_gradients), set(fresh_gradients))
                for name, reference in fresh_gradients.items():
                    self.assertIsNotNone(candidate_gradients[name], name)
                    torch.testing.assert_close(candidate_gradients[name], reference, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
