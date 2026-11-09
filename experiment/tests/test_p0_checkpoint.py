"""P0 diagnostic bounds plus real tiny CPU model checks (no remote assets)."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p0_checkpoint", ROOT / "scripts/operations/verify_p0_checkpoint.py")
P0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P0)
HAS_MODEL_DEPS = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "tokenizers", "PIL"))


class DiagnosticBoundsTest(unittest.TestCase):
    def arguments(self, output, *extra):
        return P0.parser().parse_args(["--config", "config.json", "--train-manifest", "train.jsonl",
                                       "--media-root", "media", "--output", str(output), *extra])

    def test_existing_report_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            output.write_text("previous result")
            with self.assertRaises(FileExistsError):
                P0.validate_arguments(self.arguments(output))
            self.assertEqual(output.read_text(), "previous result")

    def test_generation_bound_requires_real_cache_use(self):
        for count in (0, 1, 17, 1000):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "2–16"):
                P0.validate_arguments(self.arguments("unused-p0-report.json", "--max-new-tokens", str(count)))

    def test_ddp_is_rejected(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "8"}), self.assertRaisesRegex(ValueError, "single process"):
            P0.validate_arguments(self.arguments("unused-p0-report.json"))


@unittest.skipUnless(HAS_MODEL_DEPS, "requires installed torch/transformers/tokenizers/Pillow")
class TinyModelDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def create_inputs(self, root, *, frozen_vision=False):
        from PIL import Image
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import (LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast,
                                  SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel)
        lm, vision = root / "tiny_lm", root / "tiny_vision"
        LlamaForCausalLM(LlamaConfig(vocab_size=24, hidden_size=8, intermediate_size=16,
                                    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                    max_position_embeddings=256)).save_pretrained(lm)
        vocabulary = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3, "yes": 4, "no": 5,
                      "Question": 6, "Answer": 7, "Video": 8, "frames": 9, "in": 10,
                      "chronological": 11, "order": 12, ":": 13, "red": 14, "?": 15}
        backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]",
                                           eos_token="[EOS]", unk_token="[UNK]")
        tokenizer.save_pretrained(lm)
        SiglipVisionModel(SiglipVisionConfig(hidden_size=8, intermediate_size=16,
                                             num_hidden_layers=1, num_attention_heads=2,
                                             image_size=8, patch_size=4)).save_pretrained(vision)
        processor = SiglipImageProcessor(size={"height": 8, "width": 8})
        processor.save_pretrained(vision)
        Image.new("RGB", (8, 8), "red").save(root / "train.png")
        record = {"sample_id": "tiny-train", "source_video_id": "tiny-train-source",
                  "image_paths": ["train.png"], "question": "red?", "answer": "yes", "split": "train"}
        (root / "train.jsonl").write_text(json.dumps(record) + "\n")
        config = {"model": {"language_model": str(lm), "language_revision": "0" * 40,
                            "language_backend": "causal_lm", "vision_model": str(vision),
                            "vision_revision": "0" * 40, "num_loops": 1, "num_frames": 1,
                            "pool_grid": 1, "freeze_vision": frozen_vision, "max_text_tokens": 128},
                  "training": {"stage": "sft", "bf16": False, "gradient_checkpointing": True, "seed": 17}}
        (root / "config.json").write_text(json.dumps(config))
        return config, tokenizer, processor

    def run_diagnostic(self, root, *extra):
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            status = P0.main(["--config", str(root / "config.json"), "--train-manifest", str(root / "train.jsonl"),
                              "--media-root", str(root), "--output", str(root / "report.json"),
                              "--device", "cpu", *extra])
        return status, json.loads((root / "report.json").read_text())

    def test_real_tiny_sft_gradients_and_cache_without_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.create_inputs(root)
            status, report = self.run_diagnostic(root)
            self.assertEqual(status, 0, report["errors"])
            self.assertEqual(report["optimizer_steps"], 0)
            self.assertFalse(report["quality_metrics_computed"])
            self.assertFalse(report["input"]["generation_prompt_includes_gold_answer"])
            self.assertEqual(report["checks"]["cache_equivalence"]["status"], "passed")
            self.assertEqual(len(report["checks"]["cache_equivalence"]["cached_token_ids"][0]), 4)
            for group in report["checks"]["gradient_groups"].values():
                self.assertEqual(group["status"], "passed")
                self.assertEqual(group["missing_gradient_names"], [])
                self.assertGreater(group["nonzero_gradient_tensors"], 0)

    def test_existing_checkpoint_reload_writes_no_extra_weights(self):
        from deepprobe_vlm.modeling import DeepProbeConfig, DeepProbeVLM
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, tokenizer, processor = self.create_inputs(root, frozen_vision=True)
            checkpoint = root / "checkpoint"
            model = DeepProbeVLM.from_backbones(DeepProbeConfig.from_dict(config))
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint / "tokenizer")
            processor.save_pretrained(checkpoint / "image_processor")
            before = {path.relative_to(checkpoint): P0.sha256_file(path)
                      for path in checkpoint.rglob("*") if path.is_file()}
            del model
            status, report = self.run_diagnostic(root, "--checkpoint", str(checkpoint))
            self.assertEqual(status, 0, report["errors"])
            self.assertEqual(report["checks"]["gradient_groups"]["vision"]["status"], "expected_frozen")
            self.assertEqual(report["checkpoint_reload"]["status"], "passed")
            self.assertFalse(report["checkpoint_reload"]["fresh_save_roundtrip_verified"])
            self.assertEqual(before, {path.relative_to(checkpoint): P0.sha256_file(path)
                                      for path in checkpoint.rglob("*") if path.is_file()})

    def test_validation_manifest_is_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.create_inputs(root)
            manifest = root / "train.jsonl"
            record = json.loads(manifest.read_text())
            record["split"] = "validation"
            manifest.write_text(json.dumps(record) + "\n")
            status, report = self.run_diagnostic(root)
            self.assertEqual(status, 1)
            self.assertTrue(any("Only a train manifest" in error for error in report["errors"]))

    def test_missing_nonfinite_and_zero_group_gradients_fail(self):
        import torch
        from torch import nn
        model = SimpleNamespace(language_model=nn.Linear(2, 1), vision_model=nn.Linear(2, 1),
                                projector=nn.Linear(2, 1), deep_config=SimpleNamespace(freeze_vision=False))
        for module in (model.language_model, model.vision_model, model.projector):
            for parameter in module.parameters():
                parameter.grad = torch.ones_like(parameter)
        model.language_model.bias.grad = None
        model.vision_model.weight.grad.fill_(float("nan"))
        for parameter in model.projector.parameters():
            parameter.grad.zero_()
        groups = P0.gradient_report(model)
        self.assertEqual(groups["language"]["missing_gradient_names"], ["bias"])
        self.assertEqual(groups["vision"]["nonfinite_gradient_names"], ["weight"])
        self.assertEqual(groups["pool_projector"]["nonzero_gradient_tensors"], 0)
        self.assertTrue(all(group["status"] == "failed" for group in groups.values()))


if __name__ == "__main__":
    unittest.main()
