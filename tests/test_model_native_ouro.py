"""Optional pinned upstream-code contract test (tiny random weights, no weights download).

Run with DEEPPROBE_TEST_NATIVE_OURO=1 to fetch reviewed, pinned Python model code.
"""
import os
import tempfile
import unittest

import torch
from transformers import AutoConfig, AutoModelForCausalLM, SiglipVisionConfig, SiglipVisionModel

from deepprobe_vlm.modeling import DeepProbeConfig, DeepProbeVLM, OURO_REVISION


@unittest.skipUnless(os.environ.get("DEEPPROBE_TEST_NATIVE_OURO") == "1", "opt-in pinned remote code test")
class NativeOuroTest(unittest.TestCase):
    def test_native_eager_full_bptt_checkpointing_and_per_loop_cache(self):
        self._run_native_contract("eager")

    def test_native_sdpa_full_bptt_checkpointing_and_per_loop_cache(self):
        self._run_native_contract("sdpa")

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
        model = DeepProbeVLM(DeepProbeConfig(num_frames=1, pool_grid=1, num_loops=3,
                                           freeze_vision=False, attn_implementation=attention), language, vision)
        value = dict(input_ids=torch.tensor([[1, 0, 2, 3, 4]]), attention_mask=torch.ones(1, 5, dtype=torch.long),
                     visual_mask=torch.tensor([[False, True, False, False, False]]),
                     pixel_values=torch.randn(1, 1, 3, 8, 8), labels=torch.tensor([[-100, -100, -100, 3, 4]]))
        model.gradient_checkpointing_enable({"use_reentrant": False})
        result = model(**value)
        result.loss.backward()
        self.assertTrue(torch.isfinite(result.loss))
        self.assertGreater(language.model.layers[0].self_attn.q_proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.projector.projection[0].weight.grad.abs().sum().item(), 0)
        self.assertEqual([name for name, p in model.named_parameters() if p.requires_grad and p.grad is None], [])
        language.gradient_checkpointing_disable()
        model.eval()
        value.pop("labels")
        torch.testing.assert_close(model.greedy_generate(**value, max_new_tokens=5, use_cache=True),
                                   model.greedy_generate(**value, max_new_tokens=5, use_cache=False))
        embeds = model.prepare_embeddings(**{k: value[k] for k in ("input_ids", "visual_mask", "pixel_values")})
        cached = model._language_forward(inputs_embeds=embeds, attention_mask=value["attention_mask"], use_cache=True)
        self.assertEqual(len(cached.past_key_values.key_cache), 6)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = DeepProbeVLM.from_pretrained(directory).eval()
            expected = model(**value).logits
            torch.testing.assert_close(restored(**value).logits, expected, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
