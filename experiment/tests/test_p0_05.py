"""Pure planning/acceptance checks for the finite P0-05 GPU runner."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts/operations" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_script("run_p0_05.py")
OVERFIT = load_script("verify_short_overfit.py")


class P005PlanTest(unittest.TestCase):
    def base_config(self):
        return {
            "model": {"language_model": "unused", "language_revision": "0" * 40,
                      "language_backend": "causal_lm", "vision_model": "unused",
                      "vision_revision": "1" * 40, "num_loops": 1, "num_frames": 1,
                      "pool_grid": 1, "freeze_vision": False, "max_text_tokens": 32},
            "training": {"stage": "sft", "max_steps": 20, "save_steps": 10,
                         "save_total_limit": 2, "dataloader_num_workers": 2,
                         "learning_rate": 1e-5, "gradient_accumulation_steps": 1,
                         "seed": 17, "eval_steps": 10, "bf16": True}}

    def test_build_configs_preserves_strict_schedule_and_bounds_overfit(self):
        original = self.base_config()
        strict, overfit = RUNNER.build_configs(original, overfit_samples=4,
                                               overfit_steps=60, overfit_learning_rate=1e-5,
                                               overfit_warmup_steps=5)
        self.assertEqual(original["training"]["dataloader_num_workers"], 2)
        self.assertEqual(strict["training"]["max_steps"], 20)
        self.assertEqual(strict["training"]["save_steps"], 10)
        self.assertEqual(strict["training"]["dataloader_num_workers"], 0)
        restored = json.loads(json.dumps(strict))
        restored["training"]["dataloader_num_workers"] = 2
        self.assertEqual(restored, original)
        self.assertEqual(overfit["training"]["max_steps"], 60)
        self.assertEqual(overfit["training"]["learning_rate"], 1e-5)
        self.assertEqual(overfit["training"]["gradient_accumulation_steps"], 4)
        self.assertEqual(overfit["training"]["warmup_steps"], 5)
        self.assertEqual(overfit["training"]["lr_scheduler_type"], "constant_with_warmup")
        self.assertEqual(overfit["training"]["save_strategy"], "no")
        self.assertEqual(overfit["training"]["eval_strategy"], "no")
        self.assertNotIn("save_steps", overfit["training"])
        self.assertNotIn("eval_steps", overfit["training"])

    def test_invalid_resume_schedule_is_rejected(self):
        for final, split, limit in ((20, 20, 2), (20, 7, 2), (20, 10, 1)):
            config = self.base_config()
            config["training"].update(max_steps=final, save_steps=split, save_total_limit=limit)
            with self.subTest(final=final, split=split, limit=limit), self.assertRaises(ValueError):
                RUNNER.build_configs(config, overfit_samples=4, overfit_steps=60,
                                     overfit_learning_rate=1e-5, overfit_warmup_steps=5)

    def test_strict_gate_rejects_non_bf16_recipe(self):
        for update in ({"bf16": False}, {"fp16": True}):
            config = self.base_config()
            config["training"].update(update)
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, "bf16=true"):
                RUNNER.build_configs(config, overfit_samples=4, overfit_steps=60,
                                     overfit_learning_rate=1e-5, overfit_warmup_steps=5)

    def test_overfit_optimizer_update_covers_fixed_subset(self):
        for sample_count in (1, 4, 8):
            with self.subTest(sample_count=sample_count):
                _, overfit = RUNNER.build_configs(self.base_config(),
                    overfit_samples=sample_count, overfit_steps=60,
                    overfit_learning_rate=1e-5, overfit_warmup_steps=5)
                self.assertEqual(overfit["training"]["gradient_accumulation_steps"], sample_count)
        with self.assertRaisesRegex(ValueError, "warmup steps"):
            RUNNER.build_configs(self.base_config(), overfit_samples=4,
                                 overfit_steps=5, overfit_learning_rate=1e-5,
                                 overfit_warmup_steps=5)

    def test_step_diagnostic_is_symmetric_and_does_not_change_training_config(self):
        strict, _ = RUNNER.build_configs(self.base_config(), overfit_samples=4,
                                         overfit_steps=60, overfit_learning_rate=1e-5,
                                         overfit_warmup_steps=5)
        self.assertEqual(strict["training"]["max_steps"], 20)
        self.assertEqual(strict["training"]["save_steps"], 10)
        args = RUNNER.parser().parse_args([
            "--run-id", "example", "--config", "config.json", "--train-manifest", "train.jsonl",
            "--eval-manifest", "validation.jsonl", "--media-root", ".",
            "--output-dir", "out", "--master-port", "29500", "--diagnostic-step", "11"])
        args.frozen_source_manifest_sha256 = "a" * 64
        root = Path("out")
        paths = {"strict_config": root / "inputs/strict_config.json",
                 "overfit_config": root / "inputs/overfit_config.json",
                 "control": root / "strict/control", "resumed": root / "strict/resumed",
                 "overfit": root / "overfit/training", "overfit_manifest": root / "inputs/overfit.jsonl",
                 "control_copy_report": root / "evidence/control-copy.json",
                 "resumed_copy_report": root / "evidence/resumed-copy.json",
                 "strict_report": root / "evidence/strict.json",
                 "diagnostic_report": root / "evidence/step-diagnostic.json",
                 "overfit_report": root / "evidence/overfit.json",
                 "source_manifest": root / "inputs/source.json"}
        plan = RUNNER.build_plan(args, paths, split_step=10, final_step=20)
        for stage in ("control", "resume"):
            self.assertEqual(plan[stage][-2:], ["--diagnostic-step", "11"])
        self.assertIn("--p0-skip-resumed-final-export", plan["resume"])
        self.assertNotIn("--p0-skip-resumed-final-export", plan["control"])
        self.assertNotIn("--p0-skip-resumed-final-export", plan["overfit"])
        self.assertNotIn("resumed_copy_verify", plan)
        args.keep_duplicates = True
        forensic = RUNNER.build_plan(args, paths, split_step=10, final_step=20)
        self.assertNotIn("--p0-skip-resumed-final-export", forensic["resume"])
        self.assertIn("resumed_copy_verify", forensic)
        args.keep_duplicates = False
        self.assertNotIn("--diagnostic-step", plan["overfit"])
        self.assertEqual(plan["diagnostic_compare"][-2:],
                         ["--output", str(paths["diagnostic_report"])])
        args.diagnostic_step = None
        normal = RUNNER.build_plan(args, paths, split_step=10, final_step=20)
        self.assertNotIn("diagnostic_compare", normal)
        self.assertNotIn("--diagnostic-step", normal["control"])
        self.assertNotIn("--diagnostic-step", normal["resume"])

    def test_omitted_resumed_final_requires_complete_checkpoint_and_no_final_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "resumed"
            checkpoint = run / "checkpoint-20"
            (checkpoint / "language_model").mkdir(parents=True)
            (checkpoint / "vision_model").mkdir()
            for name in ("deepprobe_config.json", "optimizer.pt", "scheduler.pt",
                         "projector.safetensors", "language_model/model.safetensors",
                         "vision_model/model.safetensors"):
                (checkpoint / name).write_bytes(b"state")
            (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 20}))
            (checkpoint / "training_provenance.json").write_text(json.dumps({
                "training_contract": {"world_size": 8}}))
            for rank in range(8):
                (checkpoint / f"rng_state_{rank}.pth").write_bytes(b"rng")
            receipt = RUNNER.verify_resumed_final_export_omission(run, 20, 8)
            self.assertEqual(receipt["status"], "p0_resumed_final_export_intentionally_omitted")
            self.assertEqual(len(receipt["files"]), 16)
            self.assertTrue(receipt["direct_value_comparison_required"])
            (run / "final").mkdir()
            with self.assertRaisesRegex(RuntimeError, "not omitted"):
                RUNNER.verify_resumed_final_export_omission(run, 20, 8)
            (run / "final").rmdir()
            (checkpoint / "optimizer.pt").unlink()
            with self.assertRaisesRegex(RuntimeError, "missing or unsafe"):
                RUNNER.verify_resumed_final_export_omission(run, 20, 8)
            (checkpoint / "optimizer.pt").write_bytes(b"state")
            (checkpoint / "rng_state_7.pth").unlink()
            with self.assertRaisesRegex(RuntimeError, "RNG ranks differ"):
                RUNNER.verify_resumed_final_export_omission(run, 20, 8)
            (checkpoint / "rng_state_7.pth").write_bytes(b"rng")
            (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 19}))
            with self.assertRaisesRegex(RuntimeError, "wrong final optimizer step"):
                RUNNER.verify_resumed_final_export_omission(run, 20, 8)

    def test_fixed_subset_uses_first_record_from_distinct_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            rows = [
                {"sample_id": "a1", "source_video_id": "a", "split": "train"},
                {"sample_id": "a2", "source_video_id": "a", "split": "train"},
                {"sample_id": "b1", "source_video_id": "b", "split": "train"},
                {"sample_id": "c1", "source_video_id": "c", "split": "train"},
                {"sample_id": "d1", "source_video_id": "d", "split": "train"},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            selected = RUNNER.select_fixed_sources(path, 4)
            self.assertEqual([row["sample_id"] for row in selected], ["a1", "b1", "c1", "d1"])
            self.assertEqual(len({row["source_video_id"] for row in selected}), 4)

    def test_cleanup_requires_verified_manifest_and_removes_only_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight = root / "strict/control/checkpoint-20/model.safetensors"
            metadata = root / "strict/control/checkpoint-20/trainer_state.json"
            weight.parent.mkdir(parents=True)
            weight.write_bytes(b"weights")
            metadata.write_bytes(b"{}")
            files = [{"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                      "sha256": RUNNER.sha256_file(path)} for path in (weight, metadata)]
            manifest = root / "ARCHIVE_MANIFEST.json"
            RUNNER.write_new_json(manifest, {"files": files})
            invalid = {"status": "archive_verified",
                       "remote_manifest_and_completion_sha256_readback": False,
                       "archive_manifest_sha256": RUNNER.sha256_file(manifest)}
            with self.assertRaises(ValueError):
                RUNNER.cleanup_verified_archived_weights(root, invalid)
            self.assertTrue(weight.exists())
            receipt = {**invalid, "remote_manifest_and_completion_sha256_readback": True}
            result = RUNNER.cleanup_verified_archived_weights(root, receipt)
            self.assertEqual(result["status"], "only_verified_archived_weights_removed")
            self.assertFalse(weight.exists())
            self.assertTrue(metadata.exists())

    def test_incremental_archive_manifest_keeps_pruned_and_retained_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runtime").mkdir()
            pruned = root / "strict/resumed/checkpoint-20/model.safetensors"
            retained = root / "strict/control/checkpoint-10/optimizer.pt"
            pruned.parent.mkdir(parents=True)
            retained.parent.mkdir(parents=True)
            pruned.write_bytes(b"duplicate")
            retained.write_bytes(b"fork")
            uploaded = {}

            def upload(path, uri):
                uploaded[uri] = path.stat().st_size

            with mock.patch.object(RUNNER.shutil, "which", return_value="/bin/ossutil"), \
                    mock.patch.object(RUNNER.subprocess, "run",
                                      return_value=SimpleNamespace(stdout="")), \
                    mock.patch.object(RUNNER, "_upload_file", side_effect=upload), \
                    mock.patch.object(RUNNER, "_remote_object_length",
                                      side_effect=lambda uri: uploaded[uri]), \
                    mock.patch.object(RUNNER.VerifiedArchive, "_readback_sha"):
                archive = RUNNER.VerifiedArchive(root, "fixture")
                archive.archive_tree_before_prune("resumed-checkpoint", pruned.parent)
                shutil.rmtree(pruned.parent)
                receipt = archive.finalize(exit_code=0)
            manifest = json.loads((root / "ARCHIVE_MANIFEST.json").read_text())
            paths = {item["path"] for item in manifest["files"]}
            self.assertIn("strict/resumed/checkpoint-20/model.safetensors", paths)
            self.assertIn("strict/control/checkpoint-10/optimizer.pt", paths)
            self.assertIn("evidence/archive-fragments/resumed-checkpoint.json", paths)
            self.assertEqual(receipt["status"], "archive_verified")


class ShortOverfitAcceptanceTest(unittest.TestCase):
    def metrics(self, means, accuracy):
        rows = [{"sample_id": f"s{i}", "mean_nll": value, "finite": True}
                for i, value in enumerate(means)]
        return {"samples": rows, "all_finite": True, "supervised_tokens": 8,
                "token_weighted_mean_nll": sum(means) / len(means),
                "teacher_forced_token_accuracy": accuracy}

    def test_gate_requires_ratio_accuracy_and_every_sample(self):
        before = self.metrics([4.0, 3.0, 2.0, 1.0], 0.0)
        after = self.metrics([1.0, 0.8, 0.6, 0.4], 1.0)
        passed = OVERFIT.evaluate_acceptance(before, after, max_loss_ratio=0.5,
                                             min_final_token_accuracy=0.95)
        self.assertEqual(passed["status"], "passed")
        after["samples"][3]["mean_nll"] = 1.1
        failed = OVERFIT.evaluate_acceptance(before, after, max_loss_ratio=0.5,
                                             min_final_token_accuracy=0.95)
        self.assertEqual(failed["status"], "failed")
        self.assertFalse(failed["checks"]["all_samples_strictly_improved"])

    def test_nonfinite_overfit_metrics_are_json_serializable_failure(self):
        before = self.metrics([4.0, 3.0, 2.0, 1.0], 0.0)
        after = self.metrics([1.0, 0.8, 0.6, 0.4], 1.0)
        after["all_finite"] = False
        after["samples"][2].update(finite=False, mean_nll=None)
        after["token_weighted_mean_nll"] = None
        result = OVERFIT.evaluate_acceptance(before, after, max_loss_ratio=0.5,
                                             min_final_token_accuracy=0.95)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["final_to_initial_loss_ratio"])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
