"""Incomplete exact-resume state must fail before model loading."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from deepprobe_vlm.resume_checkpoint import verify_resume_checkpoint


def checkpoint(root, *, world_size=1, scaler=False):
    path = root / "checkpoint-3"
    path.mkdir(parents=True)
    (path / "trainer_state.json").write_text(json.dumps({"global_step": 3, "max_steps": 8}))
    names = ["optimizer.pt", "scheduler.pt"]
    names += (["rng_state.pth"] if world_size == 1 else
              [f"rng_state_{rank}.pth" for rank in range(world_size)])
    if scaler:
        names.append("scaler.pt")
    # This validator deliberately does not deserialize or authenticate tensors.
    for name in names:
        (path / name).write_bytes(b"unparsed synthetic state")
    return path


class ResumeCheckpointTests(unittest.TestCase):
    def test_single_and_eight_rank_layouts_have_explicit_structural_scope(self):
        for world_size in (1, 8):
            with self.subTest(world_size=world_size), tempfile.TemporaryDirectory() as directory:
                path = checkpoint(Path(directory), world_size=world_size)
                checked = verify_resume_checkpoint(path, world_size=world_size, expected_max_steps=8)
                self.assertEqual(checked["optimizer_step"], 3)
                self.assertEqual(checked["world_size"], world_size)
                self.assertEqual(checked["validation_scope"],
                                 "file_presence_and_trainer_state_structure_only")

    def test_missing_empty_or_nonregular_state_fails(self):
        for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "scaler.pt"):
            for failure in ("missing", "empty", "directory", "symlink"):
                with self.subTest(name=name, failure=failure), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    path = checkpoint(root, scaler=True)
                    item = path / name
                    content = item.read_bytes()
                    item.unlink()
                    if failure == "empty":
                        item.touch()
                    elif failure == "directory":
                        item.mkdir()
                    elif failure == "symlink":
                        target = root / "external-state"
                        target.write_bytes(content)
                        item.symlink_to(target)
                    with self.assertRaisesRegex(ValueError, "Exact resume"):
                        verify_resume_checkpoint(path, world_size=1, require_scaler=True)

    def test_fp16_requires_scaler_but_bf16_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            path = checkpoint(Path(directory))
            verify_resume_checkpoint(path, world_size=1)
            with self.assertRaisesRegex(ValueError, "missing scaler.pt"):
                verify_resume_checkpoint(path, world_size=1, require_scaler=True)
            (path / "scaler.pt").write_bytes(b"unparsed scaler")
            verify_resume_checkpoint(path, world_size=1, require_scaler=True)

    def test_rng_layout_cannot_silently_change_world_size_or_omit_a_rank(self):
        for failure in ("missing rank", "extra rank", "single-rank file", "wrong world size"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                path = checkpoint(Path(directory), world_size=8)
                requested_world_size = 8
                if failure == "missing rank":
                    (path / "rng_state_7.pth").unlink()
                elif failure == "extra rank":
                    (path / "rng_state_8.pth").write_bytes(b"extra")
                elif failure == "single-rank file":
                    (path / "rng_state.pth").write_bytes(b"extra")
                else:
                    requested_world_size = 1
                with self.assertRaisesRegex(ValueError, "complete RNG state"):
                    verify_resume_checkpoint(path, world_size=requested_world_size)

    def test_invalid_or_inconsistent_trainer_state_fails(self):
        states = [[], {"global_step": True, "max_steps": 8},
                  {"global_step": 2, "max_steps": 8}, {"global_step": 3, "max_steps": 2},
                  {"global_step": 3, "max_steps": 8.0}, {"global_step": 3}]
        for state in states:
            with self.subTest(state=state), tempfile.TemporaryDirectory() as directory:
                path = checkpoint(Path(directory))
                (path / "trainer_state.json").write_text(json.dumps(state))
                with self.assertRaisesRegex(ValueError, "step mismatch"):
                    verify_resume_checkpoint(path, world_size=1)
        with tempfile.TemporaryDirectory() as directory:
            path = checkpoint(Path(directory))
            with self.assertRaisesRegex(ValueError, "max_steps differs"):
                verify_resume_checkpoint(path, world_size=1, expected_max_steps=9)
            for content in (b"not json", b"\xff"):
                (path / "trainer_state.json").write_bytes(content)
                with self.assertRaisesRegex(ValueError, "valid trainer_state.json"):
                    verify_resume_checkpoint(path, world_size=1)

    def test_non_native_directory_and_invalid_options_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = checkpoint(root)
            for kwargs in ({"world_size": True}, {"world_size": 0},
                           {"world_size": 1, "require_scaler": 1},
                           {"world_size": 1, "expected_max_steps": False}):
                with self.assertRaises(ValueError):
                    verify_resume_checkpoint(path, **kwargs)
            alias = root / "alias"
            alias.symlink_to(path, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "regular checkpoint directory"):
                verify_resume_checkpoint(alias, world_size=1)
            renamed = root / "final"
            path.rename(renamed)
            with self.assertRaisesRegex(ValueError, "checkpoint-<positive-step>"):
                verify_resume_checkpoint(renamed, world_size=1)


class TrainingResumeAdmissionTests(unittest.TestCase):
    def test_both_entry_points_reject_missing_state_before_model_loading(self):
        from deepprobe_vlm import declared_training, train, train_declared

        fixture = Path(__file__).parent / "fixtures/training_configs/ouro_r4_alignment_a2_resume_s17.json"
        for module in (train, train_declared):
            for missing in ("optimizer.pt", "scheduler.pt", "rng_state_7.pth"):
                with self.subTest(entry=module.__name__, missing=missing), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    path = checkpoint(root, world_size=8)
                    (path / missing).unlink()
                    raw = json.loads(fixture.read_text())
                    raw["experiment_protocol"] = declared_training.declaration(
                        "two_stage", 17, "ByteDance/Ouro-1.4B")
                    (root / "config.json").write_text(json.dumps(raw))
                    for label in ("train", "validation"):
                        (root / f"{label}.jsonl").write_text("{}\n")
                    argv = ["--config", str(root / "config.json"),
                            "--train-manifest", str(root / "train.jsonl"),
                            "--eval-manifest", str(root / "validation.jsonl"),
                            "--output-dir", str(root / "output"), "--resume", str(path),
                            "--formal-ready", str(root / "ready.json"),
                            "--formal-training-data-dir", str(root),
                            "--frozen-source-manifest", str(root / "source.json"),
                            "--frozen-source-manifest-sha256", "a" * 64]
                    # Data/hardware admission is covered elsewhere; no CUDA or
                    # model operations run in this test of the real CLI path.
                    with ExitStack() as stack:
                        stack.enter_context(patch.dict(os.environ, {"WORLD_SIZE": "8"}))
                        stack.enter_context(patch.object(module, "set_seed"))
                        stack.enter_context(patch.object(module, "load_manifest", side_effect=[
                            [{"source_video_id": "train", "split": "train"}] * declared_training.TRAIN_EXAMPLES,
                            [{"source_video_id": "validation", "split": "validation"}]]))
                        stack.enter_context(patch.object(module, "validate_disjoint"))
                        stack.enter_context(patch.object(module, "manifest_media", return_value={}))
                        stack.enter_context(patch.object(module, "rank_zero_integrity_check",
                                                        side_effect=lambda phase, verify, **kwargs: verify()))
                        stack.enter_context(patch.object(module, "_verify_live_formal_inputs",
                                                        return_value={"receipt": {}}))
                        stack.enter_context(patch.object(module.torch.cuda, "is_available", return_value=True))
                        stack.enter_context(patch.object(module.torch.cuda, "is_bf16_supported", return_value=True))
                        stack.enter_context(patch.object(module, "TrainingArguments", return_value=SimpleNamespace(
                            device=SimpleNamespace(type="cpu"), world_size=8, process_index=0,
                            max_steps=-1, fp16=False)))
                        load_model = stack.enter_context(patch.object(module.DeepProbeVLM, "from_backbones"))
                        with self.assertRaisesRegex(ValueError, "Exact resume"):
                            module.main(argv)
                        load_model.assert_not_called()
                    self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
