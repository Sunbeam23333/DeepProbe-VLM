"""Exercise declared-runner admission/completion with real CPU checkpoint trees."""
import copy
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "declared_runner_gate_v9", ROOT / "scripts/operations/run_p1_stage_v9.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
d = runner.declared
CONFIGS = ROOT / "tests/fixtures/training_configs"
SOURCE_SHA = "f" * 64
CONFIG_SHA = "0" * 64
FORMAL = {
    "formal_ready_sha256": "a" * 64,
    "data_sha256": {"train.jsonl": "b" * 64, "validation.jsonl": "c" * 64},
    "training_media": {"inventory_sha256": "d" * 64, "entries_sha256": "e" * 64,
                       "file_count": 1, "total_bytes": 8},
}


def config(method="two_stage", seed=29, stage="sft"):
    name = "ouro_r4_sft_s17_save50_limit1.json" if stage == "sft" else "ouro_r4_alignment_a2_resume_s17.json"
    raw = json.loads((CONFIGS / name).read_text())
    raw["training"]["seed"] = seed
    raw["experiment_protocol"] = d.declaration(method, seed, raw["model"]["language_model"])
    return raw


def fresh_initialization(raw):
    language = raw["experiment_protocol"]["language_id"]
    backbones = {"language_id": language, "language_revision": d.LANGUAGE_REVISIONS[language],
                 "vision_id": d.VISION_ID, "vision_revision": d.VISION_REVISION}
    for label, identifier, revision in (
            ("language", language, d.LANGUAGE_REVISIONS[language]),
            ("vision", d.VISION_ID, d.VISION_REVISION)):
        files = {"config.json": {"bytes": 2, "sha256": "1" * 64},
                 "model.safetensors": {"bytes": 8, "sha256": "2" * 64}}
        backbones[label] = {
            "path": "/home/jinhaopeng/hf/models--" + identifier.replace("/", "--") + "/snapshots/" + revision,
            "files": files, "files_sha256": d.canonical(files),
        }
    return d.make_initialization(raw, "pretrained_backbones_random_connector",
                                 backbones=backbones, projector_sha256="3" * 64)


def contract(raw):
    initial = fresh_initialization(raw)
    if raw["training"]["stage"] == "sft" and raw["experiment_protocol"]["method"] == "two_stage":
        parent = {"stage": "alignment", "training": {"seed": str(raw["training"]["seed"])},
                  "declared_protocol": raw["experiment_protocol"], "initialization": initial}
        initial = d.make_initialization(raw, "matching_alignment_final", previous=parent,
                                        projector_sha256="4" * 64,
                                        checkpoint_provenance_sha256="5" * 64)
    return {
        "stage": raw["training"]["stage"], "training": {"seed": str(raw["training"]["seed"])},
        "declared_protocol": raw["experiment_protocol"], "initialization": initial,
        "formal_training": True, "require_final_resume_checkpoint": True,
        "config_sha256": CONFIG_SHA, "frozen_source_manifest_sha256": SOURCE_SHA,
        "formal_ready_sha256": FORMAL["formal_ready_sha256"],
        "train_manifest_sha256": FORMAL["data_sha256"]["train.jsonl"],
        "eval_manifest_sha256": FORMAL["data_sha256"]["validation.jsonl"],
        "world_size": 8, "global_batch_size": 128,
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def model_tree(path, raw, saved_contract):
    for component, dimension in (("language_model", 3), ("vision_model", 2)):
        write_json(path / component / "config.json", {"hidden_size": dimension})
        save_file({"weight": torch.ones(dimension, dtype=torch.float32)},
                  str(path / component / "model.safetensors"))
    write_json(path / "tokenizer/tokenizer_config.json", {})
    write_json(path / "image_processor/preprocessor_config.json", {})
    save_file({
        "gate.weight": torch.ones(1, 2), "gate.bias": torch.zeros(1),
        "projection.0.weight": torch.ones(3, 2), "projection.0.bias": torch.zeros(3),
        "projection.2.weight": torch.eye(3), "projection.2.bias": torch.zeros(3),
    }, str(path / "projector.safetensors"))
    torch.save(raw["training"], path / "training_args.bin")
    write_json(path / "deepprobe_config.json", raw["model"])
    write_json(path / "training_provenance.json", {"training_contract": saved_contract})


def native_checkpoint(path, raw, saved_contract, step=50):
    model_tree(path, raw, saved_contract)
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    parameter.sum().backward()
    optimizer.step()
    scheduler.step()
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    torch.save(scheduler.state_dict(), path / "scheduler.pt")
    for rank in range(8):
        torch.save({"cpu": torch.get_rng_state()}, path / f"rng_state_{rank}.pth")
    write_json(path / "trainer_state.json", {
        "global_step": step, "max_steps": 718, "log_history": [{"loss": 1.0}],
    })


def completed_stage(path, raw):
    saved_contract = contract(raw)
    model_tree(path / "final", raw, saved_contract)
    native_checkpoint(path / "checkpoint-718", raw, saved_contract, 718)
    write_json(path / "run_manifest.json", {
        "stage": raw["training"]["stage"], "world_size": 8, "training_contract": saved_contract,
    })
    write_json(path / "trainer_state.json", {
        "global_step": 718, "max_steps": 718, "log_history": [{"loss": 1.0}, {"eval_loss": 1.1}],
    })
    write_json(path / "train_results.json", {"train_loss": 1.0})
    write_json(path / "eval_results.json", {"eval_loss": 1.1})
    media = dict(FORMAL["training_media"], metadata_sha256="6" * 64)
    write_json(path / "FORMAL_TRAIN_COMPLETE.json", {
        "schema_version": 1, "status": "formal_training_complete", "stage": raw["training"]["stage"],
        "world_size": 8, "optimizer_step": 718,
        "frozen_source_manifest_sha256": SOURCE_SHA,
        "formal_ready_sha256": FORMAL["formal_ready_sha256"],
        "train_manifest_sha256": FORMAL["data_sha256"]["train.jsonl"],
        "eval_manifest_sha256": FORMAL["data_sha256"]["validation.jsonl"],
        "run_manifest_sha256": runner.sha256_file(path / "run_manifest.json"),
        "trainer_state_sha256": runner.sha256_file(path / "trainer_state.json"),
        "media_before": media, "media_after": media,
        **d.completion_identity(saved_contract, runner.read_json(path / "trainer_state.json")),
    })
    return saved_contract


class DeclaredRunnerGateTests(unittest.TestCase):
    def resume(self, path, raw):
        return runner.verify_resume_checkpoint(path, raw, CONFIG_SHA, FORMAL, SOURCE_SHA,
                                               raw["training"]["stage"])

    def outputs(self, path, raw):
        return runner.verify_training_outputs(path, raw["training"]["stage"], FORMAL,
                                              SOURCE_SHA, CONFIG_SHA, raw)

    def test_exact_resume_preserves_seed_method_and_native_state(self):
        for method, seed, stage in (("two_stage", 17, "alignment"), ("two_stage", 29, "sft"),
                                    ("two_stage", 43, "sft"), ("no_alignment", 17, "sft")):
            with self.subTest(method=method, seed=seed, stage=stage), tempfile.TemporaryDirectory() as name:
                raw = config(method, seed, stage)
                path = Path(name) / "checkpoint-50"
                saved_contract = contract(raw)
                native_checkpoint(path, raw, saved_contract)
                accepted = self.resume(path, raw)
                self.assertEqual(accepted["optimizer_step"], 50)
                self.assertEqual(accepted["training_contract"], saved_contract)
                self.assertIn("optimizer.pt", accepted["files"])
                self.assertIn("rng_state_7.pth", accepted["files"])

    def test_resume_rejects_changed_seed_method_source_and_model(self):
        with tempfile.TemporaryDirectory() as name:
            raw = config(seed=17)
            path = Path(name) / "checkpoint-50"
            native_checkpoint(path, raw, contract(raw))
            self.resume(path, raw)
            for changed in (config(seed=29), config("no_alignment", 17)):
                with self.subTest(protocol=changed["experiment_protocol"]), self.assertRaises(ValueError):
                    self.resume(path, changed)
            with self.assertRaisesRegex(ValueError, "frozen_source_manifest_sha256"):
                runner.verify_resume_checkpoint(path, raw, CONFIG_SHA, FORMAL, "7" * 64, "sft")
            changed = copy.deepcopy(raw)
            changed["model"]["num_loops"] = 2
            with self.assertRaisesRegex(ValueError, "model configuration"):
                self.resume(path, changed)

    def test_resume_rejects_missing_and_corrupt_initialization_before_launch(self):
        with tempfile.TemporaryDirectory() as name:
            raw = config("no_alignment", 17)
            path = Path(name) / "checkpoint-50"
            saved = contract(raw)
            native_checkpoint(path, raw, saved)
            self.resume(path, raw)
            cases = (("missing", None), ("empty", {}),
                     ("connector digest", dict(saved["initialization"], initial_connector_state_sha256="corrupt")),
                     ("seed", dict(saved["initialization"], seed=43)))
            for label, initial in cases:
                changed = dict(saved, initialization=initial)
                write_json(path / "training_provenance.json", {"training_contract": changed})
                with self.subTest(initial=label), self.assertRaises(ValueError):
                    self.resume(path, raw)
            for label in ("language", "vision"):
                changed = copy.deepcopy(saved)
                backbones = changed["initialization"]["pretrained_backbones"]
                backbones[label]["path"] = backbones[label + "_id"]
                write_json(path / "training_provenance.json", {"training_contract": changed})
                with self.subTest(hub_id_instead_of_snapshot=label), self.assertRaises(ValueError):
                    self.resume(path, raw)

    def test_two_stage_sft_resume_and_completion_require_alignment_ancestry(self):
        with tempfile.TemporaryDirectory() as name:
            raw = config()
            path = Path(name) / "checkpoint-50"
            saved = contract(raw)
            native_checkpoint(path, raw, saved)
            self.resume(path, raw)
            saved["initialization"] = fresh_initialization(raw)
            write_json(path / "training_provenance.json", {"training_contract": saved})
            with self.assertRaises(ValueError):
                self.resume(path, raw)
            with self.assertRaises(ValueError):
                d.completion_identity(saved, {"global_step": 718, "max_steps": 718})

    def test_resume_requires_frozen_max_steps_and_does_not_admit_overrun(self):
        with tempfile.TemporaryDirectory() as name:
            raw = config()
            path = Path(name) / "checkpoint-50"
            native_checkpoint(path, raw, contract(raw))
            state = runner.read_json(path / "trainer_state.json")
            self.resume(path, raw)
            for target in (None, 50, 1436, "718", 718.0):
                changed = dict(state, max_steps=target)
                if target is None:
                    changed.pop("max_steps")
                write_json(path / "trainer_state.json", changed)
                with self.subTest(max_steps=target), self.assertRaises(ValueError):
                    self.resume(path, raw)
            overrun = Path(name) / "checkpoint-719"
            path.rename(overrun)
            write_json(overrun / "trainer_state.json", dict(state, global_step=719))
            with self.assertRaises(ValueError):
                self.resume(overrun, raw)

    def test_native_resume_requires_optimizer_scheduler_all_ranks_and_weights(self):
        with tempfile.TemporaryDirectory() as name:
            raw = config()
            path = Path(name) / "checkpoint-50"
            native_checkpoint(path, raw, contract(raw))
            self.resume(path, raw)
            for relative in ("optimizer.pt", "scheduler.pt", "rng_state_7.pth",
                             "language_model/model.safetensors", "projector.safetensors"):
                item = path / relative
                original = item.read_bytes()
                item.unlink()
                with self.subTest(missing=relative), self.assertRaises(ValueError):
                    self.resume(path, raw)
                item.write_bytes(original)
            torch.save({"cpu": torch.get_rng_state()}, path / "rng_state_8.pth")
            with self.assertRaisesRegex(ValueError, "rank0..7"):
                self.resume(path, raw)
            (path / "rng_state_8.pth").unlink()
            state = runner.read_json(path / "trainer_state.json")
            write_json(path / "trainer_state.json", dict(state, global_step=49))
            with self.assertRaisesRegex(ValueError, "step mismatch"):
                self.resume(path, raw)

    def test_alignment_requires_completed_same_seed_source_and_ancestry(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "alignment"
            completed_stage(path, config(stage="alignment"))
            sft = config()
            accepted = runner.verify_alignment_final(path / "final", sft, FORMAL, SOURCE_SHA)
            self.assertEqual(accepted["rfg"], {"num_loops": 4, "num_frames": 8, "pool_grid": 4})
            with self.assertRaises(ValueError):
                runner.verify_alignment_final(path / "final", config(seed=43), FORMAL, SOURCE_SHA)
            with self.assertRaisesRegex(ValueError, "different formal"):
                runner.verify_alignment_final(path / "final", sft, FORMAL, "7" * 64)
            receipt = runner.read_json(path / "FORMAL_TRAIN_COMPLETE.json")
            write_json(path / "FORMAL_TRAIN_COMPLETE.json", dict(receipt, status="failed"))
            with self.assertRaisesRegex(ValueError, "completed declared-stage"):
                runner.verify_alignment_final(path / "final", sft, FORMAL, SOURCE_SHA)

    def test_native_cli_checks_alignment_completion_before_model_loading(self):
        from deepprobe_vlm import train_declared as train

        for case in ("complete", "missing receipt", "short training", "partial checkpoint"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as name:
                path = Path(name)
                alignment = path / "alignment"
                completed_stage(alignment, config(stage="alignment"))
                checkpoint = alignment / ("checkpoint-718" if case == "partial checkpoint" else "final")
                provenance = runner.read_json(checkpoint / "training_provenance.json")
                provenance.update(training_sources=["train"], tuning_sources=[])
                write_json(checkpoint / "training_provenance.json", provenance)
                if case == "missing receipt":
                    (alignment / "FORMAL_TRAIN_COMPLETE.json").unlink()
                elif case == "short training":
                    state = runner.read_json(alignment / "trainer_state.json")
                    write_json(alignment / "trainer_state.json", dict(state, global_step=717))
                write_json(path / "config.json", config())
                for label in ("train", "validation"):
                    (path / (label + ".jsonl")).write_text("{}\n")
                argv = ["--config", str(path / "config.json"),
                        "--train-manifest", str(path / "train.jsonl"),
                        "--eval-manifest", str(path / "validation.jsonl"),
                        "--output-dir", str(path / "output"), "--init-checkpoint", str(checkpoint),
                        "--formal-ready", str(path / "unused-ready.json"),
                        "--formal-training-data-dir", str(path),
                        "--frozen-source-manifest", str(path / "unused-source.json"),
                        "--frozen-source-manifest-sha256", SOURCE_SHA]
                # Isolate checkpoint admission from the already-tested data and
                # distributed admission. No CUDA function or model loader runs.
                with ExitStack() as stack:
                    stack.enter_context(patch.dict(os.environ, {"WORLD_SIZE": "1"}))
                    stack.enter_context(patch.object(train, "set_seed"))
                    stack.enter_context(patch.object(train, "load_manifest", side_effect=[
                        [{"source_video_id": "train", "split": "train"}] * d.TRAIN_EXAMPLES,
                        [{"source_video_id": "validation", "split": "validation"}],
                    ]))
                    stack.enter_context(patch.object(train, "validate_disjoint"))
                    stack.enter_context(patch.object(train, "manifest_media", return_value={}))
                    stack.enter_context(patch.object(train, "_verify_live_formal_inputs", return_value={"receipt": {}}))
                    stack.enter_context(patch.object(train.torch.cuda, "is_available", return_value=True))
                    stack.enter_context(patch.object(train.torch.cuda, "is_bf16_supported", return_value=True))
                    stack.enter_context(patch.object(train, "TrainingArguments", return_value=SimpleNamespace(
                        device=torch.device("cpu"), world_size=1, process_index=0)))
                    load_model = stack.enter_context(patch.object(
                        train.DeepProbeVLM, "from_backbones", side_effect=RuntimeError("model loading boundary")))
                    gate = stack.enter_context(patch.object(d, "verify_alignment_completion", wraps=d.verify_alignment_completion))
                    if case == "complete":
                        with self.assertRaisesRegex(RuntimeError, "model loading boundary"):
                            train.main(argv)
                        load_model.assert_called_once()
                    else:
                        with self.assertRaises(ValueError):
                            train.main(argv)
                        load_model.assert_not_called()
                    gate.assert_called_once_with(str(checkpoint), provenance["training_contract"])

    def test_completed718_binds_method_initialization_and_final_native_checkpoint(self):
        for method, seed in (("two_stage", 29), ("no_alignment", 17)):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as name:
                raw = config(method, seed)
                path = Path(name)
                completed_stage(path, raw)
                accepted = self.outputs(path, raw)
                self.assertEqual(accepted["optimizer_steps"], 718)
                self.assertEqual(accepted["checkpoints"], [{"name": "checkpoint-718", "step": 718}])
                original = runner.read_json(path / "final/training_provenance.json")
                write_json(path / "final/training_provenance.json", {"training_contract": {}})
                with self.assertRaisesRegex(ValueError, "Final model lost"):
                    self.outputs(path, raw)
                write_json(path / "final/training_provenance.json", original)
                write_json(path / "checkpoint-718/training_provenance.json", {"training_contract": {}})
                with self.assertRaisesRegex(ValueError, "Resume checkpoint lost"):
                    self.outputs(path, raw)

    def test_short_step_missing_completion_and_nonfinite_loss_never_pass(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)
            raw = config("no_alignment", 17)
            completed_stage(path, raw)
            receipt = runner.read_json(path / "FORMAL_TRAIN_COMPLETE.json")
            state = runner.read_json(path / "trainer_state.json")
            for changed_state in (dict(state, global_step=717), dict(state, max_steps=1436)):
                write_json(path / "trainer_state.json", changed_state)
                write_json(path / "FORMAL_TRAIN_COMPLETE.json", dict(
                    receipt, optimizer_step=changed_state["global_step"],
                    trainer_state_sha256=runner.sha256_file(path / "trainer_state.json")))
                with self.subTest(state=changed_state), self.assertRaisesRegex(ValueError, "optimizer-step target"):
                    self.outputs(path, raw)
            write_json(path / "trainer_state.json", state)
            write_json(path / "FORMAL_TRAIN_COMPLETE.json", receipt)
            write_json(path / "eval_results.json", {"eval_loss": float("nan")})
            with self.assertRaisesRegex(ValueError, "Nonfinite loss"):
                self.outputs(path, raw)
            (path / "FORMAL_TRAIN_COMPLETE.json").unlink()
            with self.assertRaisesRegex(ValueError, "completion receipt"):
                self.outputs(path, raw)

    def test_failed_attempt_archives_without_success_marker_or_weight_cleanup(self):
        for archive_fails in (False, True):
            with self.subTest(archive_fails=archive_fails), tempfile.TemporaryDirectory() as name:
                out = Path(name)
                job = runner.Runner.__new__(runner.Runner)
                job.args = SimpleNamespace(run_id="cpu-failure", stage="sft")
                job.experiment, job.out, job.source = ROOT, out, out / "source"
                job.uri = "test-only-unused-uri"
                job.stages = [{"name": "formal-train-sft", "returncode": 1}]
                job.log = (out / "stdout.log").open("w")
                job.log.write("fixture training failure\n")
                job.stop, job.emit, job.cleanup_weights = Mock(), Mock(), Mock()

                def archive(name, command, timeout, **kwargs):
                    self.assertEqual(name, "archive")
                    self.assertIn("--archive-helper", command)
                    self.assertFalse(kwargs["monitor_storage"])
                    self.assertFalse((out / "P1_STAGE_COMPLETE.json").exists())
                    if archive_fails:
                        raise RuntimeError("synthetic archive failure")
                    write_json(out / "ARCHIVED.json", {"status": "archive_verified"})

                job.run_stage = Mock(side_effect=archive)
                with patch.object(runner.subprocess, "run", side_effect=AssertionError("external process forbidden")):
                    self.assertEqual(job.finalize(1), 1)
                job.cleanup_weights.assert_not_called()
                job.run_stage.assert_called_once()
                self.assertEqual(runner.read_json(out / "EXIT_STATUS.json")["exit_code"], 1)
                self.assertFalse((out / "P1_STAGE_COMPLETE.json").exists())
                self.assertTrue((out / "source/scripts/data/download_hf.py").is_file())
                self.assertTrue((out / "stdout.snapshot.log").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
