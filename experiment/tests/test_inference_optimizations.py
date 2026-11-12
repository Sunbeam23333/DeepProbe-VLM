"""CPU contracts for opt-in inference paths; no weights, network or CUDA needed."""
from contextlib import nullcontext
from copy import deepcopy
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from deepprobe_vlm.modeling import DeepProbeConfig, DeepProbeVLM, SpatialPoolProjector


_UNSPECIFIED = object()


class CountingEmbedding(nn.Embedding):
    def __init__(self):
        super().__init__(32, 8)
        self.lengths = []

    def forward(self, input_ids):
        self.lengths.append(input_ids.shape[1])
        return super().forward(input_ids)


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.patch = nn.Linear(3, 8)

    def forward(self, pixel_values, return_dict):
        return SimpleNamespace(last_hidden_state=self.patch(pixel_values.flatten(2).transpose(1, 2)))


class SlicingNativeLoop(nn.Module):
    """Causal native recurrence with independent depth caches and a sliced head."""
    def __init__(self, scripted_tokens=None, missing_cache=False):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, max_position_embeddings=128, total_ut_steps=3)
        self.model = nn.Module()
        self.model.total_ut_steps = 3
        self.embedding = CountingEmbedding()
        self.shared = nn.Linear(8, 8)
        self.head = nn.Linear(8, 32)
        self.calls = []
        self.scripted_tokens = scripted_tokens
        self.missing_cache = missing_cache

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, labels, use_cache, past_key_values,
                return_dict, exit_at_step, use_weighted_exit, logits_to_keep=_UNSPECIFIED):
        assert labels is None and not use_weighted_exit
        assert exit_at_step == self.model.total_ut_steps - 1
        hidden = inputs_embeds
        caches = []
        for depth in range(self.model.total_ut_steps):
            summed = hidden.cumsum(1)
            if past_key_values is not None:
                summed = summed + past_key_values[depth]
            caches.append(summed[:, -1:])
            hidden = torch.tanh(self.shared(summed))
        head_hidden = hidden if logits_to_keep is _UNSPECIFIED else hidden[:, -logits_to_keep:]
        logits = self.head(head_hidden)
        if self.scripted_tokens is not None:
            logits = torch.zeros_like(logits)
            logits[..., self.scripted_tokens[len(self.calls)]] = 1
        self.calls.append({"input_length": inputs_embeds.shape[1], "head_length": head_hidden.shape[1],
                           "logits_option_omitted": logits_to_keep is _UNSPECIFIED,
                           "mask_length": attention_mask.shape[1], "past": past_key_values,
                           "returned_cache": caches})
        return SimpleNamespace(logits=logits, past_key_values=caches if use_cache and not self.missing_cache else None)


class UnsupportedLoop(SlicingNativeLoop):
    def forward(self, inputs_embeds, attention_mask, labels, use_cache, past_key_values,
                return_dict, exit_at_step, use_weighted_exit):
        return super().forward(inputs_embeds, attention_mask, labels, use_cache, past_key_values,
                               return_dict, exit_at_step, use_weighted_exit)


class KwargsOnlyLoop(SlicingNativeLoop):
    def forward(self, **kwargs):
        return super().forward(**kwargs)


class PositionalOnlyLoop(SlicingNativeLoop):
    def forward(self, logits_to_keep=0, /, **kwargs):
        return super().forward(**kwargs)


class InternallyBrokenLoop(SlicingNativeLoop):
    def forward(self, *, logits_to_keep=0, **kwargs):
        self.calls.append("called")
        raise TypeError("backend internal failure")


def make_model(language=None):
    config = DeepProbeConfig(num_frames=1, pool_grid=1, num_loops=3, freeze_vision=False)
    return DeepProbeVLM(config, language if language is not None else SlicingNativeLoop(), TinyVision())


def example():
    return {"input_ids": torch.tensor([[1, 0, 2, 3]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "visual_mask": torch.tensor([[False, True, False, False]]),
            "pixel_values": torch.randn(1, 1, 3, 2, 2)}


class InferenceOptimizationsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_last_token_logits_matches_full_projection_and_keeps_depth_caches(self):
        model = make_model().eval()
        value = example()
        outputs = []
        for use_cache in (False, True):
            for last_token_logits in (False, True):
                with self.subTest(cache=use_cache, sliced=last_token_logits):
                    model.language_model.calls.clear()
                    actual = model.greedy_generate(**value, max_new_tokens=5, use_cache=use_cache,
                                                   last_token_logits=last_token_logits)
                    outputs.append(actual)
                    calls = model.language_model.calls
                    expected_lengths = [4, 1, 1, 1, 1] if use_cache else [4, 5, 6, 7, 8]
                    self.assertEqual([call["input_length"] for call in calls], expected_lengths)
                    self.assertEqual([call["mask_length"] for call in calls], [4, 5, 6, 7, 8])
                    self.assertEqual([call["head_length"] for call in calls],
                                     [1] * 5 if last_token_logits else expected_lengths)
                    self.assertTrue(all(call["logits_option_omitted"] == (not last_token_logits) for call in calls))
                    for previous, current in zip(calls, calls[1:]):
                        if use_cache:
                            self.assertIs(current["past"], previous["returned_cache"])
                            self.assertEqual(len(current["past"]), 3)
                            self.assertEqual(len({tensor.data_ptr() for tensor in current["past"]}), 3)
                        else:
                            self.assertIsNone(current["past"])
        for actual in outputs[1:]:
            torch.testing.assert_close(actual, outputs[0], rtol=0, atol=0)

    def test_training_still_uses_full_logits_with_identical_loss_and_gradients(self):
        model = make_model().train()
        reference = deepcopy(model)
        value = example()
        labels = torch.tensor([[-100, -100, 2, 3]])
        actual = model(**value, labels=labels)
        actual.loss.backward()
        # Explicit pre-optimization backbone call and loss form an independent
        # oracle: no inference option may leak into the training call.
        embeds = reference.prepare_embeddings(value["input_ids"], value["visual_mask"], value["pixel_values"])
        expected = reference.language_model(inputs_embeds=embeds, attention_mask=value["attention_mask"],
                                            labels=None, use_cache=False, past_key_values=None,
                                            return_dict=True, exit_at_step=2, use_weighted_exit=False)
        loss = F.cross_entropy(expected.logits[:, :-1].float().reshape(-1, 32), labels[:, 1:].reshape(-1))
        loss.backward()
        self.assertEqual(actual.logits.shape, (1, 4, 32))
        self.assertTrue(model.language_model.calls[0]["logits_option_omitted"])
        torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
        torch.testing.assert_close(actual.loss, loss, atol=0, rtol=0)
        for (name, parameter), (expected_name, expected_parameter) in zip(
                model.named_parameters(), reference.named_parameters(), strict=True):
            self.assertEqual(name, expected_name)
            self.assertIsNotNone(parameter.grad, name)
            torch.testing.assert_close(parameter.grad, expected_parameter.grad, atol=0, rtol=0)

    def test_opt_in_requires_explicit_keyword_support_without_executing_backbone(self):
        for language in (UnsupportedLoop(), KwargsOnlyLoop(), PositionalOnlyLoop()):
            with self.subTest(backbone=type(language).__name__):
                model = make_model(language)
                with self.assertRaisesRegex(ValueError, "explicit backbone logits_to_keep"):
                    model.greedy_generate(**example(), max_new_tokens=1, last_token_logits=True)
                self.assertEqual(language.calls, [])
                self.assertEqual(language.embedding.lengths, [])

    def test_uninspectable_signature_fails_closed(self):
        model = make_model()
        with patch("deepprobe_vlm.modeling.inspect.signature", side_effect=ValueError("opaque callable")):
            with self.assertRaisesRegex(ValueError, "Cannot verify explicit"):
                model.greedy_generate(**example(), max_new_tokens=1, last_token_logits=True)
        self.assertEqual(model.language_model.calls, [])

    def test_backbone_typeerror_is_not_retried_or_fallback(self):
        language = InternallyBrokenLoop()
        with self.assertRaisesRegex(TypeError, "backend internal failure"):
            make_model(language).greedy_generate(**example(), max_new_tokens=1, last_token_logits=True)
        self.assertEqual(language.calls, ["called"])

    def test_default_accepts_backbone_without_slicing_option(self):
        language = UnsupportedLoop(scripted_tokens=[2])
        actual = make_model(language).greedy_generate(**example(), max_new_tokens=1)
        self.assertEqual(actual.tolist(), [[2]])
        self.assertTrue(language.calls[0]["logits_option_omitted"])

    def test_eos_and_token_budget_skip_all_unused_tail_preparation(self):
        for use_cache in (False, True):
            for last_token_logits in (False, True):
                for scripted, budget, eos, expected in (([7], 4, 7, [7]),
                                                        ([2, 7], 4, 7, [2, 7]),
                                                        ([2, 3, 7], 3, 7, [2, 3, 7]),
                                                        ([2], 1, None, [2]),
                                                        ([2, 3, 4], 3, None, [2, 3, 4])):
                    with self.subTest(cache=use_cache, sliced=last_token_logits, expected=expected):
                        language = SlicingNativeLoop(scripted_tokens=scripted)
                        with patch("deepprobe_vlm.modeling.torch.cat", wraps=torch.cat) as cat:
                            actual = make_model(language).greedy_generate(
                                **example(), max_new_tokens=budget, eos_token_id=eos,
                                use_cache=use_cache, last_token_logits=last_token_logits)
                        self.assertEqual(actual.tolist(), [expected])
                        self.assertEqual(language.embedding.lengths, [4] + [1] * (len(expected) - 1))
                        self.assertEqual(len(language.calls), len(expected))
                        # One final token concat, then only necessary mask and
                        # (when uncached) growing-prefix concatenations.
                        self.assertEqual(cat.call_count, 1 + (len(expected) - 1) * (1 if use_cache else 2))

    def test_missing_native_cache_still_fails_at_final_budget_token(self):
        for sliced in (False, True):
            language = SlicingNativeLoop(scripted_tokens=[2], missing_cache=True)
            with self.assertRaisesRegex(RuntimeError, "native KV cache"):
                make_model(language).greedy_generate(**example(), max_new_tokens=1, last_token_logits=sliced)
            self.assertEqual(language.embedding.lengths, [4])

    def test_immediate_eos_without_cache_preserves_early_exit(self):
        language = SlicingNativeLoop(scripted_tokens=[7], missing_cache=True)
        actual = make_model(language).greedy_generate(**example(), eos_token_id=7, last_token_logits=True)
        self.assertEqual(actual.tolist(), [[7]])
        self.assertEqual(language.embedding.lengths, [4])

    def test_invalid_opt_in_values_fail_before_model_execution(self):
        for option in ({"last_token_logits": 1}, {"last_token_logits": "true"},
                       {"profile_ranges": None}, {"profile_ranges": 1}, {"pool_backend": "auto"}):
            with self.subTest(option=option):
                model = make_model()
                with self.assertRaises(ValueError):
                    model.greedy_generate(**example(), max_new_tokens=1, **option)
                self.assertEqual(model.language_model.calls, [])

    def test_profiler_ranges_are_opt_in_and_preserve_outputs(self):
        model = make_model().eval()
        value = example()
        for use_cache in (False, True):
            with self.subTest(cache=use_cache):
                with patch("deepprobe_vlm.modeling.torch.profiler.record_function",
                           side_effect=AssertionError("default must not enter ranges")):
                    expected = model.greedy_generate(**value, max_new_tokens=3, use_cache=use_cache)
                with patch("deepprobe_vlm.modeling.torch.profiler.record_function",
                           side_effect=lambda name: nullcontext()) as ranges:
                    actual = model.greedy_generate(**value, max_new_tokens=3, use_cache=use_cache,
                                                   profile_ranges=True)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual([call.args[0] for call in ranges.call_args_list],
                                 ["deepprobe.vision", "deepprobe.pool_projector", "deepprobe.prefill"]
                                 + ["deepprobe.decode" if use_cache else "deepprobe.prefill"] * 2)

    def test_triton_projector_rejects_training_and_enabled_gradients(self):
        projector = SpatialPoolProjector(8, 8, 1)
        fake_kernel = ModuleType("deepprobe_vlm.cuda_pool")
        fake_kernel.weighted_spatial_pool = lambda *args: self.fail("kernel must not run")
        with patch.dict("sys.modules", {"deepprobe_vlm.cuda_pool": fake_kernel}):
            with torch.no_grad(), self.assertRaisesRegex(ValueError, "eval mode"):
                projector(torch.randn(1, 4, 8), backend="triton")
            projector.eval()
            with self.assertRaisesRegex(ValueError, "gradients disabled"):
                projector(torch.randn(1, 4, 8), backend="triton")

    def test_triton_projector_passes_existing_fp32_gate_and_projection_dtype(self):
        for clamp_gate in (False, True):
            with self.subTest(clamp_gate=clamp_gate):
                projector = SpatialPoolProjector(8, 6, 1).to(torch.bfloat16).eval()
                if clamp_gate:
                    with torch.no_grad():
                        projector.gate.weight.zero_()
                        projector.gate.bias.fill_(-100)
                states = torch.randn(2, 4, 8, dtype=torch.bfloat16)
                seen = []
                fake_kernel = ModuleType("deepprobe_vlm.cuda_pool")

                def kernel(actual_states, weights, grid):
                    self.assertIs(actual_states, states)
                    self.assertEqual(grid, 1)
                    self.assertEqual(weights.dtype, torch.float32)
                    self.assertEqual(weights.shape, (2, 4, 1))
                    expected_weights = torch.sigmoid(projector.gate(states)).float().clamp_min(1e-6)
                    torch.testing.assert_close(weights, expected_weights, atol=0, rtol=0)
                    if clamp_gate:
                        self.assertTrue(torch.all(weights == 1e-6))
                    seen.append(weights)
                    return (states.float() * weights).sum(1, keepdim=True) / weights.sum(1, keepdim=True)

                fake_kernel.weighted_spatial_pool = kernel
                with torch.no_grad(), patch.dict("sys.modules", {"deepprobe_vlm.cuda_pool": fake_kernel}):
                    actual = projector(states, backend="triton")
                    pooled = (states.float() * seen[0]).sum(1, keepdim=True) / seen[0].sum(1, keepdim=True)
                    expected = projector.projection(pooled.to(torch.bfloat16))
                self.assertEqual(len(seen), 1)
                self.assertEqual(actual.dtype, torch.bfloat16)
                self.assertEqual(actual.shape, (2, 1, 6))
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_triton_kernel_failure_is_not_silently_replaced_with_reference(self):
        projector = SpatialPoolProjector(8, 6, 1).eval()
        fake_kernel = ModuleType("deepprobe_vlm.cuda_pool")

        def broken_kernel(*args):
            raise RuntimeError("kernel execution failed")

        fake_kernel.weighted_spatial_pool = broken_kernel
        with torch.no_grad(), patch.dict("sys.modules", {"deepprobe_vlm.cuda_pool": fake_kernel}):
            with self.assertRaisesRegex(RuntimeError, "kernel execution failed"):
                projector(torch.randn(1, 4, 8), backend="triton")


if __name__ == "__main__":
    unittest.main()
