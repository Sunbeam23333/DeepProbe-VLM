"""Local v9 CPU-preparation gates; every external process is intercepted.

The release/config/source/decode/token/capacity gates execute normally against
temporary files. Remote formal/media and pretrained inventory walks are doubles;
the sole allowed process boundary synthesizes a token report without a tokenizer,
model, CUDA, scheduler, network or OSS operation.
"""
import pytest
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "docs/operations/evidence/declared-training-v9-preparation-20261021/CONFIG_MATRIX.json"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


r = load("v9_cpu_fixture_runner", ROOT / "scripts/operations/run_p1_stage_v9.py")


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.private_evidence
class PrepareDeclaredV9CPUTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.base, self.p1 = self.root / "base", self.root / "base-p1"
        self.base.mkdir(); self.p1.mkdir()
        self.release = self.p1 / "releases/frozen-v9"
        self.experiment = self.release / "experiment"
        for name in r.SOURCE_DIRS:
            (self.experiment / name).mkdir(parents=True, exist_ok=True)
        for relative in (
                "scripts/operations/run_p1_stage_v9.py", "scripts/operations/run_p1_stage_v9.sh",
                "scripts/operations/stage_assets.py", "scripts/preflight.py",
                "src/deepprobe_vlm/train.py", "src/deepprobe_vlm/train_declared.py",
                "src/deepprobe_vlm/declared_training.py", "src/deepprobe_vlm/formal_integrity.py",
                "src/deepprobe_vlm/data.py", "scripts/data/check_token_budget.py", *r.SOURCE_FILES):
            path = self.experiment / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Nonexecuted CPU fixture: " + relative + "\n")
        script = self.experiment / "scripts/operations/prepare_declared_v9_noalign_cpu.py"
        shutil.copyfile(ROOT / "scripts/operations/prepare_declared_v9_noalign_cpu.py", script)
        self.m = load("v9_cpu_prepare_fixture", script)
        m = self.m
        m.BASE, m.P1, m.RELEASE, m.EXPERIMENT = self.base, self.p1, self.release, self.experiment
        m.OUTPUT = self.p1 / "attempt"
        m.DATA, m.FORMAL_READY = self.p1 / "data", self.p1 / "formal/FORMAL_READY.json"
        m.MEDIA, m.DECODE = self.base / "media", self.p1 / "decode.json"
        m.PYTHON, m.HUB = self.base / "env/bin/python", self.base / "hf/hub"
        bindings = []
        for index, row in enumerate(json.loads(MATRIX.read_text())["configs"]):
            relative = "configs/train/declared_v9/" + Path(row["path"]).name
            config, baseline = self.experiment / relative, self.release / f"baselines/{index}.json"
            config.parent.mkdir(parents=True, exist_ok=True)
            baseline.parent.mkdir(parents=True, exist_ok=True)
            config.write_bytes(Path(row["path"]).read_bytes())
            baseline.write_bytes(Path(row["baseline"]["path"]).read_bytes())
            bindings.append({"config": relative, "baseline": f"baselines/{index}.json",
                             "sha256": digest(config), "baseline_sha256": digest(baseline),
                             **{key: row[key] for key in ("method", "seed", "stage")}})
        put(self.release / "CONFIG_BINDINGS.json", {"configs": bindings})
        self.raw = json.loads((self.experiment / m.CONFIG_RELATIVE).read_text())
        self.runtime = copy.deepcopy(self.raw)
        self.offline = {}
        for label, model_key, revision_key in (("language", "language_model", "language_revision"),
                                               ("vision", "vision_model", "vision_revision")):
            identifier, revision = self.raw["model"][model_key], self.raw["model"][revision_key]
            location = "/home/jinhaopeng/.cache/Paper/DeepProbe-VLM/hf/hub/models--" + identifier.replace("/", "--") + "/snapshots/" + revision
            self.runtime["model"][model_key] = location
            self.offline[label] = {"configured_id": identifier, "revision": revision, "resolved_path": location,
                                   "files": {"config.json": {"bytes": 2, "sha256": "a" * 64},
                                             "model.safetensors": {"bytes": 8, "sha256": "b" * 64}},
                                   "total_bytes": 10}
        put(self.release / "PINNED_OFFLINE_MODEL_MAP.json", {"offline": self.offline})
        self.formal = {"formal_ready_sha256": "a" * 64,
                       "data_sha256": {"train.jsonl": "b" * 64, "validation.jsonl": "c" * 64},
                       "training_media": {"file_count": 3979, "total_bytes": 5_168_858_321}}
        put(m.FORMAL_READY, {"fixture": "formal receipt"})
        m.FIXED_FORMAL_SHA = digest(m.FORMAL_READY)
        put(m.DATA / "READY.json", {"counts": {"selected_qa": 96298,
            "train": {"qa": 91826, "source_videos": 3794}, "validation": {"qa": 4472, "source_videos": 185}}})
        self.decode = {"status": "all_selected_media_decoded_and_processed", "device": "cpu",
                       "gpu_validated": False, "unique_clips": 3979,
                       "manifest_sha256": dict(self.formal["data_sha256"]),
                       "clips": [{"video_path": f"clip-{i}.mp4", "source_video_id": f"source-{i}",
                                  "decoded_frames": 8, "pixel_shape": [8, 3, 384, 384], "finite": True}
                                 for i in range(3979)]}
        put(m.DECODE, self.decode)
        m.FIXED_DECODE_SHA = digest(m.DECODE)
        put(m.DATA / "training_media_inventory.json", {"entries": [
            {"relative_path": c["video_path"], "source_video_id": c["source_video_id"]} for c in self.decode["clips"]]})
        put(self.release / "P1_SOURCE_MANIFEST.json", {"files": r.source_inventory(self.experiment)})
        self.freeze_release()
        self.addCleanup(self.thaw_release)
        self.fault = None
        self.module = SimpleNamespace(
            declared=r.declared, validate_recipe=r.validate_recipe, verify_source_manifest=r.verify_source_manifest,
            verify_formal_inputs=Mock(side_effect=lambda *_: copy.deepcopy(self.formal)),
            prepare_offline_config=Mock(side_effect=self.prepare_offline),
            estimate_stage_storage=Mock(side_effect=lambda raw, stage: r.stage_storage_plan(
                stage, raw["training"]["save_total_limit"], 7_480_748_488, 27_235_396)),
            require_storage_admission=r.require_storage_admission)
        self.addCleanup(patch.stopall)
        patch.object(m, "load_runner", return_value=self.module).start()
        self.process = patch.object(subprocess, "run", side_effect=AssertionError("External process forbidden")).start()
        patch.object(subprocess, "Popen", side_effect=AssertionError("External process forbidden")).start()
        patch.object(socket, "create_connection", side_effect=AssertionError("Network forbidden")).start()
        patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}).start()
        patch.object(m.shutil, "disk_usage", return_value=SimpleNamespace(free=200_000_000_000)).start()

    def thaw_release(self):
        for path in (self.release, *self.release.rglob("*")):
            if not path.is_symlink():
                path.chmod(0o755 if path.is_dir() else 0o644)

    def freeze_release(self):
        path = self.release / "RELEASE_MANIFEST.json"
        if path.exists():
            self.thaw_release()
        rows = [{"path": p.relative_to(self.release).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
                for p in sorted(self.release.rglob("*")) if p.is_file() and p != path]
        put(path, {"status": "complete_source_inventory", "variant": "declared-v9", "files": rows,
                   "file_count": len(rows), "total_bytes": sum(row["bytes"] for row in rows),
                   "p1_source_manifest_sha256": digest(self.release / "P1_SOURCE_MANIFEST.json")})
        self.manifest_sha = digest(path)
        for item in self.release.rglob("*"):
            item.chmod(0o555 if item.is_dir() else 0o444)
        self.release.chmod(0o555)

    def replace_json(self, path, value):
        path.chmod(0o644)
        put(path, value)
        path.chmod(0o444)

    def prepare_offline(self, config, runtime_path, hub, stage):
        self.assertEqual((config, hub, stage), (self.experiment / self.m.CONFIG_RELATIVE, self.m.HUB, "sft"))
        put(runtime_path, self.runtime)
        offline = copy.deepcopy(self.offline)
        if self.fault == "offline":
            offline["language"]["files"]["model.safetensors"]["sha256"] = "c" * 64
        return r.validate_recipe(self.runtime, "sft"), offline

    def token_report(self, runtime_sha):
        return {"passed": True, "samples": 96298, "splits": {"train": 91826, "validation": 4472},
                "overflow_count": 0, "overflow_samples": [], "config_sha256": runtime_sha,
                "mode": "training", "include_answer": True, "max_text_tokens": 512, "num_visual_tokens": 128,
                "manifest_sha256": {str(self.m.DATA / k): v for k, v in self.formal["data_sha256"].items()},
                "language_model": self.runtime["model"]["language_model"],
                "language_revision": self.runtime["model"]["language_revision"],
                "tokenizer_source": self.runtime["model"]["language_model"],
                "prompt_source_sha256": digest(self.experiment / "src/deepprobe_vlm/data.py"),
                "media_decoded": False, "model_weights_loaded": False}

    def token_process(self, argv, **kwargs):
        m = self.m
        self.assertEqual(argv, [str(m.PYTHON), "-B", str(self.experiment / "scripts/data/check_token_budget.py"),
                               str(m.DATA / "train.jsonl"), str(m.DATA / "validation.jsonl"),
                               "--config", str(m.OUTPUT / "config.runtime.json"), "--mode", "training",
                               "--local-files-only", "--report", str(m.OUTPUT / "TOKEN_BUDGET.json")])
        self.assertEqual(kwargs["timeout"], 1200)
        for key, value in {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
                           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}.items():
            self.assertEqual(kwargs["env"][key], value)
        report = self.token_report(digest(m.OUTPUT / "config.runtime.json"))
        if self.fault == "token":
            report.update(passed=False, overflow_count=1, overflow_samples=[{"sample_id": "over-budget"}])
        put(m.OUTPUT / "TOKEN_BUDGET.json", report)
        return subprocess.CompletedProcess(argv, 0, "fixture token report\n", "")

    def run_main(self):
        self.process.side_effect = self.token_process
        return self.m.main(["--release-manifest-sha256", self.manifest_sha])

    def test_complete_readonly_release_source_and_all17_config_bindings(self):
        identity = self.m.verify_release(self.manifest_sha)
        source = r.verify_source_manifest(self.experiment, self.release / "P1_SOURCE_MANIFEST.json")
        self.assertEqual(identity["source_manifest_sha256"], source["manifest_sha256"])
        configs = self.m.verify_configs(self.module)
        self.assertEqual(len(configs), 17)
        self.assertTrue(all(c["cpu_inputs_prepared"] is False and c["ready_for_gpu_submission"] is False for c in configs))
        self.process.assert_not_called()

    def test_release_rejects_changed_bytes_extra_files_and_writable_entries(self):
        path = self.experiment / "README.md"
        original = path.read_bytes()
        path.chmod(0o644); path.write_bytes(b"tampered"); path.chmod(0o444)
        with self.assertRaisesRegex(ValueError, "byte mismatch"):
            self.m.verify_release(self.manifest_sha)
        path.chmod(0o644); path.write_bytes(original); path.chmod(0o444)
        path.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "read only"):
            self.m.verify_release(self.manifest_sha)
        path.chmod(0o444)
        self.release.chmod(0o755)
        extra = self.release / "unexpected"; extra.write_text("extra"); extra.chmod(0o444)
        self.release.chmod(0o555)
        with self.assertRaisesRegex(ValueError, "unexpected files"):
            self.m.verify_release(self.manifest_sha)
        self.process.assert_not_called()

    def test_bindings_cannot_read_config_or_baseline_outside_frozen_release(self):
        path = self.release / "CONFIG_BINDINGS.json"
        original = json.loads(path.read_text())
        for key, absolute in (("config", True), ("config", False), ("baseline", True), ("baseline", False)):
            changed = copy.deepcopy(original)
            row = changed["configs"][0]
            source = (self.experiment if key == "config" else self.release) / row[key]
            outside = self.root / (key + ".json")
            outside.write_bytes(source.read_bytes())
            row[key] = str(outside) if absolute else ("../../../../" if key == "config" else "../../../") + outside.name
            self.replace_json(path, changed)
            with self.subTest(key=key, absolute=absolute), self.assertRaises(ValueError):
                self.m.verify_configs(self.module)

    def test_decode_requires_exact_clip_source_mapping_and_full_finite_frames(self):
        self.assertEqual(self.m.verify_decode(self.formal)["clips"], 3979)
        for fault in ("mapping", "duplicate", "frames", "manifest"):
            changed = copy.deepcopy(self.decode)
            if fault == "mapping":
                a, b = changed["clips"][:2]
                a["source_video_id"], b["source_video_id"] = b["source_video_id"], a["source_video_id"]
            elif fault == "duplicate":
                changed["clips"][0] = changed["clips"][1]
            elif fault == "frames":
                changed["clips"][0]["decoded_frames"] = 7
            else:
                changed["manifest_sha256"]["train.jsonl"] = "d" * 64
            put(self.m.DECODE, changed)
            self.m.FIXED_DECODE_SHA = digest(self.m.DECODE)
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                self.m.verify_decode(self.formal)

    def test_tokens_reject_overflow_stale_scope_config_and_checkpoint_tokenizer(self):
        original = self.token_report("d" * 64)
        self.m.verify_tokens(original, "d" * 64, self.formal, self.runtime)
        for key, value in (("overflow_count", 1), ("overflow_samples", [{}]), ("samples", 91826),
                           ("config_sha256", "e" * 64), ("include_answer", False), ("num_visual_tokens", 64),
                           ("prompt_source_sha256", "e" * 64), ("model_weights_loaded", True),
                           ("tokenizer_source", "/unrelated/checkpoint/tokenizer"), ("checkpoint_tokenizer_sha256", {})):
            changed = dict(original, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.m.verify_tokens(changed, "d" * 64, self.formal, self.runtime)

    def test_capacity_uses_sft_peak_and_reserves_both_other_workers(self):
        self.m.OUTPUT.mkdir()
        result = self.m.capacity(self.module, self.runtime)
        self.module.estimate_stage_storage.assert_called_once_with(self.runtime, "sft")
        self.assertEqual(result["plan"]["required_additional_bytes"], 47_884_490_928)
        self.assertEqual(result["projected_bytes"], result["combined_accounted_bytes"] + 6_000_000_000 + 1_000_000_000 + 47_884_490_928)
        self.assertEqual(result["admission"]["current_project_bytes"], result["combined_accounted_bytes"] + 7_000_000_000)

    def test_capacity_does_not_substitute_alignment_plan_or_admit99gb(self):
        self.m.OUTPUT.mkdir()
        alignment = copy.deepcopy(self.runtime); alignment["training"]["stage"] = "alignment"
        with self.assertRaisesRegex(ValueError, "SFT storage peak"):
            self.m.capacity(self.module, alignment)
        with patch.object(r.declared, "staging_bytes", return_value={str(self.base): 44_115_509_072, str(self.p1): 0}):
            with self.assertRaises(RuntimeError):
                self.m.capacity(self.module, self.runtime)
        self.assertEqual(json.loads((self.m.OUTPUT / "CAPACITY_SNAPSHOT.json").read_text())["projected_bytes"], 99_000_000_000)

    def test_main_prepares_fresh_noalign_and_only_runs_the_token_checker(self):
        self.assertEqual(self.run_main(), 0)
        self.process.assert_called_once()
        self.assertEqual(self.module.verify_formal_inputs.call_count, 2)
        receipt = json.loads((self.m.OUTPUT / "CPU_PREPARED_PENDING_SCHEDULER_GATE.json").read_text())
        self.assertTrue(receipt["cpu_preparation_passed"])
        for key in ("ready_for_gpu_submission", "gpu_job_submitted", "scheduler_gate_checked",
                    "scheduler_dry_run_executed", "oss_checked", "oss_mutated", "total_training_budget_matches_two_stage"):
            self.assertIs(receipt[key], False)
        init = receipt["initialization"]
        self.assertEqual((init["kind"], init["seed"], init["method"]), ("pretrained_backbones_random_connector", 17, "no_alignment"))
        self.assertIsNone(init["alignment_final"])
        self.assertIsNone(init["resume_checkpoint"])
        self.assertIsNone(init["actual_connector_state_sha256"])
        self.assertFalse((self.m.OUTPUT / "CPU_PREPARATION_FAILED.json").exists())

    def test_main_failures_preserve_evidence_and_remain_gated(self):
        for fault in ("offline", "token", "capacity"):
            with self.subTest(fault=fault):
                self.fault = fault
                self.m.OUTPUT = self.p1 / ("attempt-" + fault)
                self.process.reset_mock()
                if fault == "capacity":
                    with patch.object(r.declared, "staging_bytes", return_value={str(self.base): 60_000_000_000, str(self.p1): 0}):
                        self.assertEqual(self.run_main(), 2)
                else:
                    self.assertEqual(self.run_main(), 2)
                self.assertEqual(self.process.call_count, 0 if fault == "offline" else 1)
                receipt = json.loads((self.m.OUTPUT / "CPU_PREPARATION_FAILED.json").read_text())
                self.assertEqual(receipt["status"], "CPU_PREPARATION_FAILED_GATED")
                for key in ("ready_for_gpu_submission", "gpu_job_submitted", "scheduler_gate_checked", "oss_checked", "oss_mutated"):
                    self.assertIs(receipt[key], False)
                self.assertFalse((self.m.OUTPUT / "CPU_PREPARED_PENDING_SCHEDULER_GATE.json").exists())
                self.assertTrue((self.m.OUTPUT / "CPU_PREPARATION_INTENT.json").is_file())

    def test_main_rejects_untrusted_release_before_any_attempt_or_process(self):
        with self.assertRaisesRegex(ValueError, "manifest SHA changed"):
            self.m.main(["--release-manifest-sha256", "0" * 64])
        self.assertFalse(self.m.OUTPUT.exists())
        self.process.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
