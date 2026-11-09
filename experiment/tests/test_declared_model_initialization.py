"""Real CPU constructor/gradient checks with fixed tiny pretrained backbones.

Only the two pretrained loaders are mocked. The VLM constructor, connector,
SigLIP forward, recurrent language forward, loss and optimizer all run normally.
"""
from copy import deepcopy
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from transformers import SiglipVisionConfig, SiglipVisionModel, set_seed

from deepprobe_vlm.modeling import DeepProbeConfig, DeepProbeVLM


class FixedTinyNativeLoop(nn.Module):
    """Small causal recurrent backbone with Ouro's fixed-depth interface."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, max_position_embeddings=256)
        self.model = nn.Module()
        self.model.total_ut_steps = 4
        self.model.early_exit_gate = nn.Linear(8, 1)
        self.embedding = nn.Embedding(32, 8)
        self.shared = nn.Linear(8, 8)
        self.head = nn.Linear(8, 32)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, labels, use_cache,
                past_key_values, return_dict, exit_at_step, use_weighted_exit):
        assert labels is None and not use_weighted_exit
        assert exit_at_step == self.model.total_ut_steps - 1
        assert not use_cache and past_key_values is None
        hidden = inputs_embeds
        positions = torch.arange(1, hidden.shape[1] + 1, dtype=hidden.dtype,
                                 device=hidden.device).reshape(1, -1, 1)
        for _ in range(self.model.total_ut_steps):
            hidden = torch.tanh(self.shared(hidden.cumsum(1) / positions))
        return SimpleNamespace(logits=self.head(hidden), past_key_values=None)


class DeclaredModelInitializationTest(unittest.TestCase):
    def setUp(self):
        torch_rng = torch.get_rng_state()
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        thread_count = torch.get_num_threads()
        self.addCleanup(torch.set_rng_state, torch_rng)
        self.addCleanup(random.setstate, python_rng)
        self.addCleanup(np.random.set_state, numpy_rng)
        self.addCleanup(torch.set_num_threads, thread_count)
        torch.set_num_threads(1)
        # These fixed templates represent pretrained inputs. Deepcopy in the
        # loaders preserves their weights without consuming constructor RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261021)
            self.language = FixedTinyNativeLoop()
            self.vision = SiglipVisionModel(SiglipVisionConfig(
                hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                num_attention_heads=2, image_size=16, patch_size=4))
        self.config = DeepProbeConfig(num_loops=4, num_frames=8, pool_grid=4,
                                      freeze_vision=False, max_text_tokens=512)

    def build(self, seed):
        set_seed(seed)
        with patch("transformers.AutoModelForCausalLM.from_pretrained",
                   side_effect=lambda *args, **kwargs: deepcopy(self.language)) as language_loader, \
             patch("transformers.SiglipVisionModel.from_pretrained",
                   side_effect=lambda *args, **kwargs: deepcopy(self.vision)) as vision_loader:
            model = DeepProbeVLM.from_backbones(self.config, torch_dtype=torch.float32)
        language_loader.assert_called_once_with(
            self.config.language_model, revision=self.config.language_revision,
            trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="sdpa")
        vision_loader.assert_called_once_with(
            self.config.vision_model, revision=self.config.vision_revision,
            torch_dtype=torch.float32)
        self.assertTrue(all(parameter.device.type == "cpu" for parameter in model.parameters()))
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in model.parameters()))
        return model

    def assert_same_state(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for name in actual:
            with self.subTest(tensor=name):
                torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)

    def assert_fixed_backbones(self, model):
        self.assert_same_state(model.language_model.state_dict(), self.language.state_dict())
        self.assert_same_state(model.vision_model.state_dict(), self.vision.state_dict())

    def test_same_seed_recreates_connector_with_unchanged_pretrained_weights(self):
        for seed in (17, 29, 43):
            with self.subTest(seed=seed):
                first, repeated = self.build(seed), self.build(seed)
                self.assert_same_state(first.projector.state_dict(), repeated.projector.state_dict())
                self.assert_fixed_backbones(first)
                self.assert_fixed_backbones(repeated)

    def test_independent_seeds_change_connector_but_not_backbone_weights(self):
        models = {seed: self.build(seed) for seed in (17, 29, 43)}
        for model in models.values():
            self.assert_fixed_backbones(model)
        for left, right in ((17, 29), (17, 43), (29, 43)):
            with self.subTest(seeds=(left, right)):
                a, b = models[left].projector.state_dict(), models[right].projector.state_dict()
                self.assertTrue(any(not torch.equal(a[name], b[name]) for name in a))

    def test_fresh_no_alignment_sft_backpropagates_and_updates_required_groups(self):
        model = self.build(17)
        self.assert_fixed_backbones(model)
        model.set_stage("sft")
        model.train()
        frozen = ("language_model.model.early_exit_gate.", "vision_model.vision_model.head.")
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, not name.startswith(frozen), name)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        count = self.config.num_visual_tokens
        input_ids = torch.tensor([[0] * count + [1, 2, 3, 4]])
        visual_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        visual_mask[:, :count] = True
        labels = torch.full_like(input_ids, -100)
        labels[:, -2:] = input_ids[:, -2:]
        pixels = torch.linspace(-1, 1, 8 * 3 * 16 * 16).reshape(1, 8, 3, 16, 16)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.01, weight_decay=0.0)
        result = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                       visual_mask=visual_mask, pixel_values=pixels, labels=labels)
        self.assertTrue(torch.isfinite(result.loss).item())
        result.loss.backward()
        groups = (model.language_model, model.vision_model, model.projector)
        for group in groups:
            trainable = [parameter for parameter in group.parameters() if parameter.requires_grad]
            self.assertTrue(trainable)
            self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                for parameter in trainable))
            self.assertTrue(any(torch.count_nonzero(parameter.grad).item() for parameter in trainable))
        for name, parameter in model.named_parameters():
            if name.startswith(frozen):
                self.assertIsNone(parameter.grad, name)
        optimizer.step()
        after = dict(model.named_parameters())
        for prefix in ("language_model.", "vision_model.", "projector."):
            self.assertTrue(any(not torch.equal(before[name], parameter)
                                for name, parameter in after.items()
                                if name.startswith(prefix) and parameter.requires_grad), prefix)
        for name, parameter in after.items():
            if name.startswith(frozen):
                torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
        # Updating a constructed model must not mutate the fixed loader inputs.
        for prefix, template in (("language_model.", self.language), ("vision_model.", self.vision)):
            original = {name.removeprefix(prefix): value for name, value in before.items()
                        if name.startswith(prefix)}
            self.assert_same_state(template.state_dict(), original)


if __name__ == "__main__":
    unittest.main()
