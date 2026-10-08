"""Actual tiny HF training, stage initialization, optimizer resume and eval."""
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast,
                          SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel)

from deepprobe_vlm.train import main as train_main
from deepprobe_vlm.evaluate import main as eval_main
from deepprobe_vlm.modeling import DeepProbeVLM


class TrainingCLITest(unittest.TestCase):
    def test_alignment_sft_resume_and_eval(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lm, vision = root/"tiny_lm", root/"tiny_vision"
            LlamaForCausalLM(LlamaConfig(vocab_size=24, hidden_size=8, intermediate_size=16,
                                        num_hidden_layers=1, num_attention_heads=2,
                                        num_key_value_heads=2, max_position_embeddings=256)).save_pretrained(lm)
            vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
                     "yes": 4, "no": 5, "Question": 6, "Answer": 7, "Video": 8,
                     "frames": 9, "in": 10, "chronological": 11, "order": 12,
                     ":": 13, "red": 14, "?": 15}
            backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]",
                                   eos_token="[EOS]", unk_token="[UNK]").save_pretrained(lm)
            SiglipVisionModel(SiglipVisionConfig(hidden_size=8, intermediate_size=16,
                                                 num_hidden_layers=1, num_attention_heads=2,
                                                 image_size=8, patch_size=4)).save_pretrained(vision)
            SiglipImageProcessor(size={"height": 8, "width": 8}).save_pretrained(vision)
            for name, colour in (("train", "red"), ("val", "blue"), ("test", "green")):
                Image.new("RGB", (8, 8), colour).save(root/f"{name}.png")
                record = dict(sample_id=name, source_video_id=name, image_paths=[f"{name}.png"],
                              question="red?", answer="yes", split="validation" if name == "val" else name)
                (root/f"{name}.jsonl").write_text(json.dumps(record)+"\n")
            config = {
                "model": dict(language_model=str(lm), language_revision="0"*40,
                              language_backend="causal_lm", vision_model=str(vision),
                              vision_revision="0"*40, num_loops=1, num_frames=1, pool_grid=1,
                              freeze_vision=True, max_text_tokens=128),
                "training": dict(stage="alignment", max_steps=1, bf16=False, use_cpu=True,
                                 gradient_checkpointing=False, gradient_accumulation_steps=1,
                                 save_steps=1, logging_steps=1, report_to="none", disable_tqdm=True)}
            path = root/"config.json"
            path.write_text(json.dumps(config))
            common = ["--config", str(path), "--train-manifest", str(root/"train.jsonl"),
                      "--eval-manifest", str(root/"val.jsonl"), "--media-root", str(root)]
            train_main(common+["--output-dir", str(root/"alignment")])
            self.assertTrue((root/"alignment/final/projector.safetensors").is_file())
            config["model"]["freeze_vision"] = False
            config["training"]["stage"] = "sft"
            config["training"]["max_steps"] = 2
            path.write_text(json.dumps(config))
            train_main(common+["--output-dir", str(root/"sft"),
                               "--init-checkpoint", str(root/"alignment/final")])
            self.assertTrue((root/"sft/checkpoint-1/optimizer.pt").is_file())
            uninterrupted = {k: v.clone() for k, v in DeepProbeVLM.from_pretrained(root/"sft/final").state_dict().items()}
            train_main(common+["--output-dir", str(root/"sft"), "--resume", str(root/"sft/checkpoint-1")])
            resumed = DeepProbeVLM.from_pretrained(root/"sft/final").state_dict()
            for key in uninterrupted:
                torch.testing.assert_close(resumed[key], uninterrupted[key], atol=0, rtol=0)
            state = json.loads((root/"sft/trainer_state.json").read_text())
            self.assertEqual(state["global_step"], 2)
            with self.assertRaises(FileExistsError):
                train_main(common+["--output-dir", str(root/"sft")])
            config["training"]["gradient_accumulation_steps"] = 2
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "training contract changed"):
                train_main(common+["--output-dir", str(root/"sft"), "--resume", str(root/"sft/checkpoint-1")])
            eval_main(["--checkpoint", str(root/"sft/final"), "--manifest", str(root/"test.jsonl"),
                       "--media-root", str(root), "--output", str(root/"predictions.jsonl"),
                       "--max-new-tokens", "2"])
            prediction = json.loads((root/"predictions.jsonl").read_text())
            self.assertEqual(prediction["source_video_id"], "test")
            self.assertEqual(prediction["contamination_check"], "passed_recorded_training_source_check")
            with self.assertRaises(ValueError):
                eval_main(["--checkpoint", str(root/"sft/final"), "--manifest", str(root/"train.jsonl"),
                           "--media-root", str(root), "--output", str(root/"leaky.jsonl")])


if __name__ == "__main__":
    unittest.main()
