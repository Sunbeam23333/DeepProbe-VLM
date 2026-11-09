"""Live formal media/source gates, including direct CLI and rank failure paths."""
from __future__ import annotations

from datetime import timedelta
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch.distributed as dist

from deepprobe_vlm.formal_integrity import (
    REQUIRED_P1_SOURCE, SOURCE_DIRS, SOURCE_FILES, canonical_sha256,
    source_inventory, verify_attested_media_metadata,
    verify_attested_training_media, verify_frozen_source_manifest,
)
from deepprobe_vlm.train import main as train_main, rank_zero_integrity_check


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def distributed_failure_worker(rank: int, rendezvous: str, queue) -> None:
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        def check():
            if rank != 0:
                raise AssertionError("nonzero rank must never hash media")
            raise ValueError("rank-zero media changed")

        try:
            rank_zero_integrity_check("formal admission", check,
                                      expected_rank=rank, expected_world_size=2)
        except ValueError as exc:
            queue.put((rank, str(exc)))
        else:
            queue.put((rank, "UNEXPECTED_SUCCESS"))
    finally:
        dist.destroy_process_group()


class FakeModel:
    def set_stage(self, _stage):
        pass

    def parameter_counts(self):
        return {"trainable": 1, "total": 1}


class FakeTrainer:
    media_to_mutate: Path | None = None

    def __init__(self, *, args, **_kwargs):
        self.output = Path(args.output_dir)

    def is_world_process_zero(self):
        return True

    def train(self, *, resume_from_checkpoint):
        assert resume_from_checkpoint is None
        if self.media_to_mutate is not None:
            self.media_to_mutate.write_bytes(b"changed!")
        return SimpleNamespace(metrics={"train_loss": 0.5})

    def save_final_resume_checkpoint(self):
        pass

    def save_model(self, path):
        final = Path(path)
        final.mkdir(parents=True)
        write_json(final / "deepprobe_config.json", {})

    def save_state(self):
        write_json(self.output / "trainer_state.json", {"global_step": 1})

    def log_metrics(self, *_args):
        pass

    def save_metrics(self, prefix, metrics):
        write_json(self.output / f"{prefix}_results.json", metrics)

    def evaluate(self):
        return {"eval_loss": 0.4}


class FormalIntegrityTests(unittest.TestCase):
    def fixture(self, root: Path) -> dict:
        source, data, formal, media = (root / name for name in
                                       ("experiment", "data", "formal", "media"))
        for directory in (source, data, formal, media):
            directory.mkdir()
        for name in SOURCE_DIRS:
            (source / name).mkdir()
        for name in SOURCE_FILES:
            (source / name).write_text(name + "\n", encoding="utf-8")
        for relative in REQUIRED_P1_SOURCE:
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative + "\n", encoding="utf-8")
        source_manifest = root / "SOURCE_MANIFEST.json"
        write_json(source_manifest, {
            "schema_version": 1, "purpose": "DEEPPROBE_FROZEN_P1_SOURCE",
            "files": source_inventory(source),
        })

        train_media = media / "train.mp4"
        validation_media = media / "validation.mp4"
        train_media.write_bytes(b"original")
        validation_media.write_bytes(b"validate")
        train_record = {"sample_id": "train-1", "source_video_id": "source-train",
                        "video_path": train_media.name, "question": "Question?",
                        "answer": "yes", "split": "train"}
        validation_record = {"sample_id": "validation-1",
                             "source_video_id": "source-validation",
                             "video_path": validation_media.name,
                             "question": "Question?", "answer": "yes",
                             "split": "validation"}
        (data / "train.jsonl").write_text(json.dumps(train_record) + "\n")
        (data / "validation.jsonl").write_text(json.dumps(validation_record) + "\n")
        (data / "coverage.jsonl").write_text("coverage\n")
        entries = [{"relative_path": path.name, "source_video_id": source_id,
                    "size_bytes": path.stat().st_size, "sha256": sha(path)}
                   for path, source_id in ((train_media, "source-train"),
                                           (validation_media, "source-validation"))]
        inventory = {"schema_version": 1,
                     "kind": "formal_training_media_sha256_inventory",
                     "media_root": str(media.resolve()),
                     "file_count": len(entries),
                     "total_bytes": sum(row["size_bytes"] for row in entries),
                     "entries_sha256": canonical_sha256(entries), "entries": entries}
        write_json(data / "training_media_inventory.json", inventory)
        outputs = {name: sha(data / name) for name in
                   ("train.jsonl", "validation.jsonl", "coverage.jsonl",
                    "training_media_inventory.json")}
        write_json(data / "preparation_report.json", {
            "schema_version": 1, "status": "formal_training_data_ready",
            "formal_data_ready": True, "output_sha256": outputs,
            "training_media_inventory": {
                "relative_path": "training_media_inventory.json",
                "sha256": outputs["training_media_inventory.json"],
                "entries_sha256": inventory["entries_sha256"],
                "file_count": len(entries), "total_bytes": inventory["total_bytes"],
            },
        })
        write_json(data / "READY.json", {
            "schema_version": 1, "purpose": "FORMAL_TRAINING_DATA",
            "formal_data_ready": True,
            "report_sha256": sha(data / "preparation_report.json"),
            "manifest_sha256": {"train.jsonl": outputs["train.jsonl"],
                                "validation.jsonl": outputs["validation.jsonl"]},
            "coverage_sha256": outputs["coverage.jsonl"],
            "training_media_inventory_sha256": outputs["training_media_inventory.json"],
            "training_media_entries_sha256": inventory["entries_sha256"],
            "training_media_file_count": len(entries),
            "training_media_total_bytes": inventory["total_bytes"],
        })
        from deepprobe_vlm.train import FORMAL_TRAINING_GATES

        gates = {name: {"status": "passed"} for name in FORMAL_TRAINING_GATES}
        write_json(formal / "source_policy.json", {
            "schema_version": 1, "policy_id": "deepprobe-formal-benchmark-source-v1"})
        write_json(formal / "training_media_inventory.json", {
            "complete": True, "root": str(media.resolve()), "entries": entries})
        write_json(formal / "source_gate_report.json", {
            "schema_version": 1, "status": "ready_for_training",
            "formal_training_ready": True, "formal_evaluation_ready": False,
            "benchmark_source_identity_gate": "passed_on_declared_subset",
            "training_readiness_gates": gates,
            "training_manifests": {"valid": True,
                                   "sha256": {"train": outputs["train.jsonl"],
                                              "validation": outputs["validation.jsonl"]}},
            "training_candidate_coverage": {"valid": True,
                                            "sha256": outputs["coverage.jsonl"]},
            "training_data_attestation": {
                "provided": True, "valid": True,
                "report_sha256": sha(data / "preparation_report.json"),
                "ready_sha256": sha(data / "READY.json"),
                "media_inventory": {
                    "sha256": outputs["training_media_inventory.json"],
                    "entries_sha256": inventory["entries_sha256"],
                    "file_count": len(entries), "total_bytes": inventory["total_bytes"],
                },
            },
            "training_media_attestation": {
                "checked": True, "passed": True,
                "entries_sha256": inventory["entries_sha256"],
                "file_count": len(entries), "total_bytes": inventory["total_bytes"],
            },
        })
        artifact_hashes = {name: sha(formal / name) for name in
                           ("source_policy.json", "source_gate_report.json",
                            "training_media_inventory.json")}
        write_json(formal / "FORMAL_READY.json", {
            "schema_version": 1, "status": "ready_for_training",
            "formal_training_ready": True, "formal_evaluation_ready": False,
            "policy_id": "deepprobe-formal-benchmark-source-v1",
            "benchmark_source_identity_gate": "passed_on_declared_subset",
            "training_readiness_gates": gates,
            "immutable_input_sha256": {
                "training_train": outputs["train.jsonl"],
                "training_validation": outputs["validation.jsonl"],
                "training_coverage": outputs["coverage.jsonl"],
                "training_media_inventory": outputs["training_media_inventory.json"],
            },
            "artifact_sha256": artifact_hashes,
        })
        config = root / "config.json"
        write_json(config, {
            "model": {
                "language_model": "unused-language-model", "language_revision": "0" * 40,
                "language_backend": "causal_lm", "vision_model": "unused-vision-model",
                "vision_revision": "0" * 40, "num_loops": 1, "num_frames": 1,
                "pool_grid": 1, "freeze_vision": True, "max_text_tokens": 128,
                "attn_implementation": "sdpa",
            },
            "training": {
                "stage": "alignment", "formal_training": True,
                "require_final_resume_checkpoint": True,
                "max_steps": 1, "bf16": False, "use_cpu": True,
                "gradient_checkpointing": False, "save_steps": 1,
                "logging_steps": 1, "report_to": "none", "disable_tqdm": True,
            },
        })
        return {"source": source, "source_manifest": source_manifest,
                "data": data, "formal_ready": formal / "FORMAL_READY.json",
                "media": media, "train_media": train_media, "config": config}

    @staticmethod
    def cli(fixture: dict, output: Path) -> list[str]:
        return ["--config", str(fixture["config"]),
                "--train-manifest", str(fixture["data"] / "train.jsonl"),
                "--eval-manifest", str(fixture["data"] / "validation.jsonl"),
                "--output-dir", str(output),
                "--media-root", str(fixture["media"]),
                "--formal-ready", str(fixture["formal_ready"]),
                "--formal-training-data-dir", str(fixture["data"]),
                "--frozen-source-manifest", str(fixture["source_manifest"]),
                "--frozen-source-manifest-sha256", sha(fixture["source_manifest"])]

    def test_current_media_bytes_and_metadata_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            item = self.fixture(Path(directory))
            inventory = item["data"] / "training_media_inventory.json"
            expected = {"train.mp4": "source-train", "validation.mp4": "source-validation"}
            before = verify_attested_training_media(inventory, item["media"], sha(inventory), expected)
            self.assertEqual(before["file_count"], 2)
            self.assertEqual(before, verify_attested_media_metadata(
                inventory, item["media"], sha(inventory), expected))
            item["train_media"].write_bytes(b"changed!")  # same length
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                verify_attested_training_media(inventory, item["media"], sha(inventory), expected)
            with self.assertRaisesRegex(ValueError, "exactly cover"):
                verify_attested_training_media(
                    inventory, item["media"], sha(inventory), {"train.mp4": "source-train"})

    def test_claimed_source_sha_must_match_the_actual_frozen_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            item = self.fixture(Path(directory))
            manifest = item["source_manifest"]
            result = verify_frozen_source_manifest(item["source"], manifest, sha(manifest))
            self.assertEqual(result["manifest_sha256"], sha(manifest))
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                verify_frozen_source_manifest(item["source"], manifest, "f" * 64)
            (item["source"] / "src/deepprobe_vlm/train.py").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "tree differs"):
                verify_frozen_source_manifest(item["source"], manifest, sha(manifest))

    def _run_fake_direct_training(self, item: dict, output: Path,
                                  *, mutate_during_training: bool = False,
                                  mutate_during_model_load: bool = False):
        FakeTrainer.media_to_mutate = item["train_media"] if mutate_during_training else None
        tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0)

        def build_model(*_args, **_kwargs):
            if mutate_during_model_load:
                item["train_media"].write_bytes(b"changed!")
            return FakeModel()

        with patch("deepprobe_vlm.train._formal_source_root", return_value=item["source"]), \
             patch("deepprobe_vlm.train.DeepProbeVLM.from_backbones", side_effect=build_model), \
             patch("deepprobe_vlm.train.DeepProbeTrainer", FakeTrainer), \
             patch("deepprobe_vlm.train.AutoTokenizer.from_pretrained", return_value=tokenizer), \
             patch("deepprobe_vlm.train.AutoImageProcessor.from_pretrained", return_value=object()), \
             patch("deepprobe_vlm.train.importlib.metadata.version", return_value="test"):
            return train_main(self.cli(item, output))

    def test_direct_cli_checks_media_before_model_load_and_real_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            item = self.fixture(Path(directory))
            command = self.cli(item, Path(directory) / "run")
            with patch("deepprobe_vlm.train._formal_source_root", return_value=item["source"]), \
                 patch("deepprobe_vlm.train.DeepProbeVLM.from_backbones") as load_model:
                without_source_path = command[:-4] + command[-2:]
                with self.assertRaisesRegex(ValueError, "requires --frozen-source-manifest"):
                    train_main(without_source_path)
                load_model.assert_not_called()
                item["train_media"].write_bytes(b"changed!")
                with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                    train_main(command)
                load_model.assert_not_called()
                item["train_media"].write_bytes(b"original")
                command[-1] = "f" * 64
                with self.assertRaisesRegex(ValueError, "Frozen source manifest SHA-256 differs"):
                    train_main(command)
                load_model.assert_not_called()

    def test_media_change_during_model_load_stops_before_train(self):
        with tempfile.TemporaryDirectory() as directory:
            item = self.fixture(Path(directory))
            output = Path(directory) / "run"
            with patch.object(FakeTrainer, "train", wraps=FakeTrainer.train) as train_call:
                with self.assertRaisesRegex(ValueError, "Formal source/media changed"):
                    self._run_fake_direct_training(
                        item, output, mutate_during_model_load=True)
                train_call.assert_not_called()
            self.assertFalse((output / "FORMAL_TRAIN_COMPLETE.json").exists())

    def test_direct_cli_postcheck_controls_completion_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            item = self.fixture(Path(directory))
            valid_out = Path(directory) / "valid"
            self.assertEqual(self._run_fake_direct_training(item, valid_out), 0)
            receipt = json.loads((valid_out / "FORMAL_TRAIN_COMPLETE.json").read_text())
            self.assertEqual(receipt["status"], "formal_training_complete")
            self.assertEqual(receipt["media_before"], receipt["media_after"])
            failed_out = Path(directory) / "failed"
            with self.assertRaisesRegex(ValueError, "Formal training media SHA-256 differs"):
                self._run_fake_direct_training(item, failed_out, mutate_during_training=True)
            self.assertFalse((failed_out / "FORMAL_TRAIN_COMPLETE.json").exists())

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(),
                         "two-rank CPU Gloo process group unavailable")
    def test_two_ranks_receive_rank_zero_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = str(Path(directory) / "rendezvous")
            context = multiprocessing.get_context("spawn")
            queue = context.Queue()
            processes = [context.Process(target=distributed_failure_worker,
                                         args=(rank, rendezvous, queue)) for rank in (0, 1)]
            try:
                for process in processes:
                    process.start()
                for process in processes:
                    process.join(timeout=40)
                for process in processes:
                    self.assertEqual(process.exitcode, 0)
                messages = [queue.get(timeout=3) for _ in processes]
                self.assertEqual({rank for rank, _ in messages}, {0, 1})
                self.assertTrue(all("rank-zero media changed" in message
                                    for _, message in messages))
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
