"""Small real Torch tests; no pretrained weights or network required."""
import json
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM, SiglipVisionConfig, SiglipVisionModel

from deepprobe_vlm.modeling import DeepProbeConfig, DeepProbeVLM
from deepprobe_vlm.step_diagnostic import EmbeddingGradientTap


class TinyNativeLoop(nn.Module):
    """Causal recurrent-depth stand-in with separate per-loop cache states."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, max_position_embeddings=128, total_ut_steps=2)
        self.model = nn.Module()
        self.model.total_ut_steps = 2
        self.model.early_exit_gate = nn.Linear(8, 1)
        self.embedding = nn.Embedding(32, 8)
        self.shared = nn.Linear(8, 8)
        self.head = nn.Linear(8, 32)
        self.calls = 0
        self.loop_outputs = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, labels, use_cache, past_key_values,
                return_dict, exit_at_step, use_weighted_exit):
        assert labels is None and not use_weighted_exit
        assert exit_at_step == self.model.total_ut_steps-1
        self.calls += 1
        hidden = inputs_embeds
        caches = []
        self.loop_outputs = []
        for index in range(self.model.total_ut_steps):
            summed = hidden.cumsum(1)
            if past_key_values is not None:
                summed = summed + past_key_values[index]
            caches.append(summed[:, -1:])
            hidden = torch.tanh(self.shared(summed))
            if hidden.requires_grad:
                hidden.retain_grad()
            self.loop_outputs.append(hidden)
        return SimpleNamespace(logits=self.head(hidden), past_key_values=caches if use_cache else None)


class TinyCheckpointOuro(TinyNativeLoop):
    """Mimic Ouro's config-driven embedding construction for a CPU resume test."""

    def __init__(self, pad_token_id=None):
        super().__init__()
        self.config.pad_token_id = pad_token_id
        self.config.bos_token_id = 0
        self.config.eos_token_id = 0
        self.embedding = nn.Embedding(32, 8, padding_idx=pad_token_id)

    def save_pretrained(self, output_dir, **kwargs):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps({
            "pad_token_id": self.config.pad_token_id,
            "bos_token_id": self.config.bos_token_id,
            "eos_token_id": self.config.eos_token_id,
        }))
        torch.save(self.state_dict(), output / "weights.pt")

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        root = Path(path)
        saved = json.loads((root / "config.json").read_text())
        model = cls(pad_token_id=saved["pad_token_id"])
        model.load_state_dict(torch.load(root / "weights.pt", weights_only=True))
        return model


def vision():
    return SiglipVisionModel(SiglipVisionConfig(hidden_size=8, intermediate_size=16,
                                               num_hidden_layers=1, num_attention_heads=2,
                                               image_size=8, patch_size=4))


def batch():
    return dict(input_ids=torch.tensor([[1, 0, 2, 3, 4]]), attention_mask=torch.ones(1, 5, dtype=torch.long),
                visual_mask=torch.tensor([[False, True, False, False, False]]),
                pixel_values=torch.randn(1, 1, 3, 8, 8),
                labels=torch.tensor([[-100, -100, -100, 3, 4]]))


class ModelContractTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_final_loop_loss_has_full_bptt_and_single_native_call(self):
        model = DeepProbeVLM(DeepProbeConfig(num_frames=1, pool_grid=1, num_loops=2,
                                           freeze_vision=False), TinyNativeLoop(), vision())
        value = batch()
        result = model(**value)
        result.loss.backward()
        self.assertTrue(torch.isfinite(result.loss))
        self.assertEqual(model.language_model.calls, 1)
        for state in model.language_model.loop_outputs:
            self.assertIsNotNone(state.grad)
            self.assertGreater(state.grad.abs().sum().item(), 0)
        for module in (model.language_model.shared, model.projector, model.vision_model):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
        self.assertEqual([name for name, p in model.named_parameters() if p.requires_grad and p.grad is None], [])
        expected = torch.nn.functional.cross_entropy(result.logits[:, :-1].float().reshape(-1, 32), value["labels"][:, 1:].reshape(-1))
        torch.testing.assert_close(result.loss, expected)

    def test_embedding_gradient_tap_preserves_training_gradients(self):
        model = DeepProbeVLM(DeepProbeConfig(num_frames=1, pool_grid=1, num_loops=2,
                                           freeze_vision=False), TinyNativeLoop(), vision())
        reference = deepcopy(model)
        value = batch()
        reference(**value).loss.backward()
        tap = EmbeddingGradientTap(model, value["input_ids"])
        tap.install()
        try:
            model(**value).loss.backward()
            rng_before_summary = torch.get_rng_state().clone()
            receipt = tap.summary(sync_gradients=False)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng_before_summary))
        finally:
            tap.close()
        for (name, actual), (expected_name, expected) in zip(
                model.named_parameters(), reference.named_parameters(), strict=True):
            self.assertEqual(name, expected_name)
            if expected.grad is not None:
                torch.testing.assert_close(actual.grad, expected.grad, atol=0, rtol=0)
            else:
                self.assertIsNone(actual.grad)
        self.assertEqual(set(receipt["gradients"]), {
            "lookup_output", "multimodal_output", "local_embedding_weight_rows"})
        self.assertEqual(receipt["gradients"]["lookup_output"]["shape"], [1, 5, 8])
        self.assertEqual(receipt["gradients"]["multimodal_output"]["shape"], [1, 5, 8])
        self.assertEqual(receipt["selected_token_count"], 5)
        self.assertEqual(receipt["gradients"]["local_embedding_weight_rows"],
                         receipt["post_backward_embedding_weight_rows"])
        self.assertLess(len(json.dumps(receipt)), 3000)
        self.assertFalse(hasattr(model, "_deepprobe_step_embedding_gradient_tap"))

    def test_ouro_checkpoint_keeps_token_zero_trainable_after_reload(self):
        config = DeepProbeConfig(num_frames=1, pool_grid=1, num_loops=2)
        model = DeepProbeVLM(config, TinyCheckpointOuro(), vision()).eval()
        value = batch()
        value["input_ids"] = torch.tensor([[0, 1, 2, 3, 4]])
        # Trainer writes tokenizer.pad_token_id=0 into the shared HF config
        # before saving; it must not turn Ouro's BOS/EOS into an embedding pad.
        model.config.pad_token_id = 0
        self.assertIsNone(model.language_model.embedding.padding_idx)
        reference_loss = model(**value).loss
        reference_loss.backward()
        reference_gradient = model.language_model.embedding.weight.grad[0].clone()
        self.assertGreater(reference_gradient.abs().sum().item(), 0)

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            saved_config = json.loads((Path(directory) / "language_model/config.json").read_text())
            self.assertEqual(saved_config["pad_token_id"], 0)
            loaded_padding_indices = []

            def load_language(path, **kwargs):
                language = TinyCheckpointOuro.from_pretrained(path)
                loaded_padding_indices.append(language.embedding.padding_idx)
                return language

            with patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=load_language):
                restored = DeepProbeVLM.from_pretrained(directory).eval()
                reloaded_in_place = DeepProbeVLM(config, TinyCheckpointOuro(), vision()).eval()
                reloaded_in_place.config.pad_token_id = 0
                reloaded_in_place.load_checkpoint_weights(directory)

            self.assertEqual(loaded_padding_indices, [0, 0])
            for candidate in (restored, reloaded_in_place):
                self.assertEqual(candidate.config.pad_token_id, 0)
                self.assertIsNone(candidate.language_model.embedding.padding_idx)
                self.assertIsNone(candidate.language_model.model.padding_idx)
                candidate_loss = candidate(**value).loss
                torch.testing.assert_close(candidate_loss, reference_loss, atol=0, rtol=0)
                candidate_loss.backward()
                torch.testing.assert_close(candidate.language_model.embedding.weight.grad[0],
                                           reference_gradient, atol=0, rtol=0)

    def test_cached_matches_uncached_native_loop_generation(self):
        model = DeepProbeVLM(DeepProbeConfig(num_frames=1, pool_grid=1, num_loops=3), TinyNativeLoop(), vision()).eval()
        value = batch()
        value.pop("labels")
        cached = model.greedy_generate(**value, max_new_tokens=6, use_cache=True)
        uncached = model.greedy_generate(**value, max_new_tokens=6, use_cache=False)
        torch.testing.assert_close(cached, uncached)

    def test_alignment_updates_only_projector(self):
        model = DeepProbeVLM(DeepProbeConfig(num_frames=1, pool_grid=1, num_loops=2), TinyNativeLoop(), vision())
        model.set_stage("alignment")
        model(**batch()).loss.backward()
        self.assertTrue(all(p.grad is None for p in model.language_model.parameters()))
        self.assertTrue(all(p.grad is None for p in model.vision_model.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.projector.parameters()))

    def test_round_trip_real_hf_components(self):
        config = DeepProbeConfig(language_backend="causal_lm", num_loops=1, num_frames=1, pool_grid=1)
        language = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
                                               num_hidden_layers=1, num_attention_heads=2,
                                               num_key_value_heads=2, max_position_embeddings=128))
        model = DeepProbeVLM(config, language, vision()).eval()
        value = batch()
        expected = model(**value).logits
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = DeepProbeVLM.from_pretrained(directory).eval()
            torch.testing.assert_close(expected, restored(**value).logits, atol=1e-6, rtol=1e-6)
            value.pop("labels")
            torch.testing.assert_close(restored.greedy_generate(**value, max_new_tokens=4, use_cache=True),
                                       restored.greedy_generate(**value, max_new_tokens=4, use_cache=False))

    def test_config_rejects_fake_recurrence_and_mutable_revision(self):
        with self.assertRaises(ValueError):
            DeepProbeConfig(language_revision="main")
        with self.assertRaises(ValueError):
            DeepProbeConfig(language_backend="causal_lm", num_loops=4)


if __name__ == "__main__":
    unittest.main()
