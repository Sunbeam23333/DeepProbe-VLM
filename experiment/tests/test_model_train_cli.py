"""Actual tiny HF training, stage initialization, optimizer resume and eval."""
import json
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast,
                          SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel)

from deepprobe_vlm.train import (main as train_main, verify_checkpoint_config,
                                 verify_formal_ready,
                                 validate_p0_resumed_final_export_option)
from deepprobe_vlm.evaluate import main as eval_main
from deepprobe_vlm.modeling import DeepProbeConfig, DeepProbeVLM


class TrainingCLITest(unittest.TestCase):
    def test_p0_final_omission_requires_nonformal_exact_resume_and_final_step_save(self):
        with tempfile.TemporaryDirectory() as directory:
            cli = SimpleNamespace(p0_skip_resumed_final_export=True, resume="checkpoint-10",
                                  init_checkpoint=None, output_dir=directory)
            training = {"max_steps": 20, "save_steps": 10, "save_total_limit": 2}
            validate_p0_resumed_final_export_option(cli, training, formal_training=False)
            with self.assertRaisesRegex(ValueError, "Formal training cannot skip"):
                validate_p0_resumed_final_export_option(cli, training, formal_training=True)
            cli.resume = None
            with self.assertRaisesRegex(ValueError, "exact --resume"):
                validate_p0_resumed_final_export_option(cli, training, formal_training=False)
            cli.resume = "checkpoint-10"
            with self.assertRaisesRegex(ValueError, "checkpoint at the final step"):
                validate_p0_resumed_final_export_option(
                    cli, {**training, "save_steps": 7}, formal_training=False)
            with self.assertRaisesRegex(ValueError, "retain the fork and final"):
                validate_p0_resumed_final_export_option(
                    cli, {**training, "save_total_limit": 1}, formal_training=False)
            with self.assertRaisesRegex(ValueError, "optimizer/RNG"):
                validate_p0_resumed_final_export_option(
                    cli, {**training, "save_only_model": True}, formal_training=False)
            (Path(directory) / "final").mkdir()
            with self.assertRaisesRegex(ValueError, "preexisting final"):
                validate_p0_resumed_final_export_option(cli, training, formal_training=False)

    def test_alignment_initializer_must_match_loop_depth(self):
        base = {
            "language_model": "ByteDance/Ouro-1.4B", "language_revision": "0" * 40,
            "language_backend": "ouro", "vision_model": "google/siglip",
            "vision_revision": "1" * 40, "num_loops": 1, "num_frames": 8,
            "pool_grid": 4, "freeze_vision": True, "max_text_tokens": 512,
            "attn_implementation": "sdpa",
        }
        saved = DeepProbeConfig.from_dict(base)
        current = DeepProbeConfig.from_dict({**base, "num_loops": 2})
        with self.assertRaisesRegex(ValueError, "Checkpoint/config mismatch: num_loops"):
            verify_checkpoint_config(saved, current, exact_resume=False)

    def test_formal_ready_is_bound_to_manifests_and_gate_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, formal = root / "data", root / "formal"
            data.mkdir(); formal.mkdir()
            train, validation = data / "train.jsonl", data / "validation.jsonl"
            train.write_text("train\n")
            validation.write_text("validation\n")
            coverage = data / "coverage.jsonl"
            coverage.write_text("coverage\n")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            entries = [{"relative_path": "clip.mp4", "source_video_id": "youtube:abcdefghijk",
                        "size_bytes": 5, "sha256": "a" * 64}]
            entries_sha = hashlib.sha256(json.dumps(
                entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
            media_inventory = data / "training_media_inventory.json"
            media_inventory.write_text(json.dumps({
                "schema_version": 1, "kind": "formal_training_media_sha256_inventory",
                "media_root": "/formal/media", "file_count": 1, "total_bytes": 5,
                "entries_sha256": entries_sha, "entries": entries,
            }))
            outputs = {path.name: digest(path) for path in
                       (train, validation, coverage, media_inventory)}
            preparation = data / "preparation_report.json"
            preparation.write_text(json.dumps({
                "schema_version": 1, "status": "formal_training_data_ready",
                "formal_data_ready": True, "output_sha256": outputs,
                "training_media_inventory": {
                    "relative_path": media_inventory.name, "sha256": outputs[media_inventory.name],
                    "entries_sha256": entries_sha, "file_count": 1, "total_bytes": 5,
                },
            }))
            data_ready = data / "READY.json"
            data_ready.write_text(json.dumps({
                "schema_version": 1, "purpose": "FORMAL_TRAINING_DATA", "formal_data_ready": True,
                "report_sha256": digest(preparation),
                "manifest_sha256": {"train.jsonl": digest(train),
                                    "validation.jsonl": digest(validation)},
                "coverage_sha256": digest(coverage),
                "training_media_inventory_sha256": digest(media_inventory),
                "training_media_entries_sha256": entries_sha,
                "training_media_file_count": 1, "training_media_total_bytes": 5,
            }))
            gates = {name: {"status": "passed"} for name in (
                "pinned_annotations", "videomme_fixed_subset", "benchmark_source_identity",
                "training_manifests", "training_candidate_coverage", "training_data_attestation",
                "training_media_integrity", "declared_source_leakage", "artifact_integrity")}
            (formal / "source_policy.json").write_text(json.dumps({
                "schema_version": 1, "policy_id": "deepprobe-formal-benchmark-source-v1"}))
            (formal / "training_media_inventory.json").write_text(json.dumps({
                "checked": True, "complete": True, "root": "/formal/media", "entries": entries}))
            report = formal / "source_gate_report.json"
            report.write_text(json.dumps({
                "schema_version": 1, "status": "ready_for_training",
                "formal_training_ready": True, "formal_evaluation_ready": False,
                "benchmark_source_identity_gate": "passed_on_declared_subset",
                "training_readiness_gates": gates,
                "training_manifests": {"valid": True,
                    "sha256": {"train": digest(train), "validation": digest(validation)}},
                "training_candidate_coverage": {"valid": True, "sha256": digest(coverage)},
                "training_data_attestation": {
                    "provided": True, "valid": True, "report_sha256": digest(preparation),
                    "ready_sha256": digest(data_ready), "media_inventory": {
                        "sha256": digest(media_inventory), "entries_sha256": entries_sha,
                        "file_count": 1, "total_bytes": 5}},
                "training_media_attestation": {"checked": True, "passed": True,
                    "entries_sha256": entries_sha, "file_count": 1, "total_bytes": 5},
            }))
            artifact_paths = [formal / "source_policy.json", report,
                              formal / "training_media_inventory.json"]
            ready = formal / "FORMAL_READY.json"
            receipt = {
                "schema_version": 1, "policy_id": "deepprobe-formal-benchmark-source-v1",
                "status": "ready_for_training",
                "formal_training_ready": True,
                "formal_evaluation_ready": False,
                "benchmark_source_identity_gate": "passed_on_declared_subset",
                "training_readiness_gates": gates,
                "immutable_input_sha256": {
                    "training_train": digest(train),
                    "training_validation": digest(validation),
                    "training_coverage": digest(coverage),
                    "training_media_inventory": digest(media_inventory),
                },
                "artifact_sha256": {path.name: digest(path) for path in artifact_paths},
            }
            ready.write_text(json.dumps(receipt))
            verified = verify_formal_ready(
                ready, digest(train), digest(validation), data, "/formal/media")
            self.assertEqual(verified["policy_id"], "deepprobe-formal-benchmark-source-v1")
            self.assertFalse(verified["formal_evaluation_ready"])
            receipt["policy_id"] = "tampered-policy"
            ready.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "not ready for training"):
                verify_formal_ready(ready, digest(train), digest(validation), data, "/formal/media")
            receipt["policy_id"] = "deepprobe-formal-benchmark-source-v1"
            receipt["training_readiness_gates"]["training_media_integrity"]["status"] = "failed"
            ready.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "complete passed training gate"):
                verify_formal_ready(ready, digest(train), digest(validation), data, "/formal/media")
            receipt["training_readiness_gates"]["training_media_integrity"]["status"] = "passed"
            ready.write_text(json.dumps(receipt))
            (formal / "source_policy.json").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "artifact missing or changed"):
                verify_formal_ready(ready, digest(train), digest(validation), data, "/formal/media")

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
                                 save_steps=10, logging_steps=1, report_to="none", disable_tqdm=True,
                                 require_final_resume_checkpoint=True,
                                 ddp_find_unused_parameters=False)}
            path = root/"config.json"
            path.write_text(json.dumps(config))
            common = ["--config", str(path), "--train-manifest", str(root/"train.jsonl"),
                      "--eval-manifest", str(root/"val.jsonl"), "--media-root", str(root)]
            train_main(common+["--output-dir", str(root/"alignment"), "--diagnostic-step", "1"])
            self.assertTrue((root/"alignment/final/projector.safetensors").is_file())
            alignment_manifest = json.loads((root/"alignment/run_manifest.json").read_text())
            self.assertIs(alignment_manifest["training"]["ddp_find_unused_parameters"], True)
            alignment_contract = alignment_manifest["training_contract"]["training"]
            self.assertEqual(alignment_contract["ddp_find_unused_parameters"], "True")
            self.assertEqual(alignment_contract["ddp_bucket_cap_mb"], "None")
            self.assertEqual(alignment_contract["ddp_broadcast_buffers"], "None")
            diagnostic = json.loads((root/"alignment/diagnostics/optimizer-step-1/rank-0.json").read_text())
            self.assertEqual(diagnostic["status"], "complete")
            self.assertEqual(diagnostic["microbatches"][0]["sample_ids"], ["train"])
            self.assertEqual(diagnostic["microbatches"][0]["ddp_bucket_before_forward"],
                             {"status": "not_ddp"})
            self.assertEqual(diagnostic["microbatches"][0]["ddp_bucket_after_backward"],
                             {"status": "not_ddp"})
            self.assertEqual(set(diagnostic["microbatches"][0]["input_tensors"]),
                             {"input_ids", "attention_mask", "visual_mask", "pixel_values", "labels"})
            self.assertNotIn("embedding_gradient_boundary", diagnostic["microbatches"][0])
            self.assertIn("after_backward_before_clip", diagnostic["events"])
            # max_steps=1 did not reach save_steps=10; the explicit final-step
            # save must still create a complete native resume point.
            self.assertTrue((root/"alignment/checkpoint-1/optimizer.pt").is_file())
            self.assertTrue((root/"alignment/checkpoint-1/scheduler.pt").is_file())
            self.assertTrue((root/"alignment/checkpoint-1/rng_state.pth").is_file())
            config["model"]["freeze_vision"] = False
            config["training"]["stage"] = "sft"
            config["training"]["max_steps"] = 2
            config["training"]["save_steps"] = 1
            config["model"]["num_frames"] = 2
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "Checkpoint/config mismatch: num_frames"):
                train_main(common+["--output-dir", str(root/"wrong-depth"),
                                   "--init-checkpoint", str(root/"alignment/final")])
            config["model"]["num_frames"] = 1
            path.write_text(json.dumps(config))
            train_main(common+["--output-dir", str(root/"sft"),
                               "--init-checkpoint", str(root/"alignment/final")])
            self.assertTrue((root/"sft/checkpoint-1/optimizer.pt").is_file())
            sft_manifest = json.loads((root/"sft/run_manifest.json").read_text())
            self.assertIs(sft_manifest["training"]["ddp_find_unused_parameters"], True)
            sft_contract = sft_manifest["training_contract"]["training"]
            self.assertEqual(sft_contract["ddp_find_unused_parameters"], "True")
            self.assertEqual(sft_contract["ddp_bucket_cap_mb"], "None")
            self.assertEqual(sft_contract["ddp_broadcast_buffers"], "None")
            uninterrupted = {k: v.clone() for k, v in DeepProbeVLM.from_pretrained(root/"sft/final").state_dict().items()}
            train_main(common+["--output-dir", str(root/"sft"), "--resume", str(root/"sft/checkpoint-1")])
            resumed = DeepProbeVLM.from_pretrained(root/"sft/final").state_dict()
            for key in uninterrupted:
                torch.testing.assert_close(resumed[key], uninterrupted[key], atol=0, rtol=0)
            train_main(common+["--output-dir", str(root/"sft-skip-final"),
                               "--resume", str(root/"sft/checkpoint-1"),
                               "--p0-skip-resumed-final-export"])
            self.assertFalse((root/"sft-skip-final/final").exists())
            resumed_checkpoint = root/"sft-skip-final/checkpoint-2"
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth",
                         "projector.safetensors", "trainer_state.json"):
                self.assertTrue((resumed_checkpoint/name).is_file(), name)
            checkpoint_model = DeepProbeVLM.from_pretrained(resumed_checkpoint).state_dict()
            for key in uninterrupted:
                torch.testing.assert_close(checkpoint_model[key], uninterrupted[key], atol=0, rtol=0)
            state = json.loads((root/"sft/trainer_state.json").read_text())
            self.assertEqual(state["global_step"], 2)
            provenance_path = root/"sft/checkpoint-1/training_provenance.json"
            provenance_original = provenance_path.read_text()
            provenance_tampered = json.loads(provenance_original)
            provenance_tampered["training_contract"]["training"]["ddp_find_unused_parameters"] = "False"
            provenance_path.write_text(json.dumps(provenance_tampered))
            try:
                with self.assertRaisesRegex(ValueError, "training contract changed"):
                    train_main(common+["--output-dir", str(root/"sft"),
                                       "--resume", str(root/"sft/checkpoint-1")])
            finally:
                provenance_path.write_text(provenance_original)
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
