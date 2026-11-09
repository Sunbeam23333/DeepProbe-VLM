"""Declared formal modes fail closed before any model, scheduler or OSS work."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deepprobe_vlm import declared_training as d

spec = importlib.util.spec_from_file_location("declared_runner_v9", ROOT / "scripts/operations/run_p1_stage_v9.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
BASE = ROOT / "tests/fixtures/training_configs"


def config(method="two_stage", seed=17, stage="sft", language="ByteDance/Ouro-1.4B", loops=4):
    filename = "ouro_r4_sft_s17_save50_limit1.json" if stage == "sft" else "ouro_r4_alignment_a2_resume_s17.json"
    raw = json.loads((BASE / filename).read_text())
    raw["model"].update(language_model=language, language_revision=d.LANGUAGE_REVISIONS[language],
                        language_backend="ouro" if language.startswith("ByteDance/") else "causal_lm", num_loops=loops)
    raw["training"]["seed"] = seed
    raw["training"]["save_total_limit"] = 2 if language.startswith("ByteDance/") and loops in (1, 2) and stage == "alignment" else 1
    raw["experiment_protocol"] = d.declaration(method, seed, language)
    return raw


def initial(raw):
    identity = {"language_id": raw["experiment_protocol"]["language_id"],
                "language_revision": raw["model"]["language_revision"],
                "vision_id": d.VISION_ID, "vision_revision": d.VISION_REVISION}
    for key, identifier, revision in (("language", identity["language_id"], identity["language_revision"]),
                                      ("vision", d.VISION_ID, d.VISION_REVISION)):
        files = {"config.json": {"bytes": 2, "sha256": "a" * 64},
                 "model.safetensors": {"bytes": 8, "sha256": "b" * 64}}
        identity[key] = {"path": "/home/jinhaopeng/hf/models--" + identifier.replace("/", "--") + "/snapshots/" + revision,
                         "files": files, "files_sha256": d.canonical(files)}
    return d.make_initialization(raw, "pretrained_backbones_random_connector", backbones=identity,
                                 projector_sha256="c" * 64)


def contract(raw):
    return {"stage": raw["training"]["stage"], "declared_protocol": raw["experiment_protocol"],
            "training": {"seed": str(raw["training"]["seed"])}, "initialization": initial(raw)}


class DeclaredProtocolTests(unittest.TestCase):
    def test_core_seeds_and_depths_keep_all_recipe_axes(self):
        for seed in d.SEEDS:
            for language, depths in (("ByteDance/Ouro-1.4B", (1, 2, 4)),
                                      ("HuggingFaceTB/SmolLM2-1.7B-Instruct", (1,))):
                for loops in depths:
                    for stage in ("alignment", "sft"):
                        raw = config(seed=seed, language=language, loops=loops, stage=stage)
                        self.assertEqual(r.validate_recipe(raw, stage)["global_batch_size"], 128)
                        mode = d.initialization_mode(raw, init_checkpoint="final" if stage == "sft" else None)
                        self.assertEqual(mode, "matching_alignment_final" if stage == "sft" else "pretrained_backbones_random_connector")

    def test_invalid_seed_or_undeclared_overrides_block(self):
        for seed in (True, "29", 0, 18):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                d.validate_declaration(config(seed=seed))
        for key, value in (("warmup_steps", 100), ("data_seed", 43), ("ignore_data_skip", True),
                           ("dataloader_drop_last", True), ("max_grad_norm", 0), ("deepspeed", "other")):
            raw = config("no_alignment"); raw["training"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                r.validate_recipe(raw, "sft")

    def test_recipe_drift_and_false_equal_compute_claim_block(self):
        for key, value in (("learning_rate", .1), ("num_train_epochs", 2), ("max_steps", 1436),
                           ("bf16", 1), ("gradient_accumulation_steps", 8), ("save_total_limit", 0),
                           ("save_steps", 200), ("save_total_limit", 2)):
            raw = config("no_alignment"); raw["training"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.validate_declaration(raw)
        for key, value in (("equal_compute_claim", True), ("heldout_selection", True), ("target_optimizer_steps", 1436)):
            raw = config("no_alignment"); raw["experiment_protocol"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.validate_declaration(raw)

    def test_explicit_no_alignment_is_not_a_general_fresh_sft_bypass(self):
        raw = config("no_alignment")
        self.assertEqual(d.initialization_mode(raw, no_alignment=True), "pretrained_backbones_random_connector")
        self.assertEqual(d.initialization_mode(raw, no_alignment=True, resume="checkpoint-50"), "exact_resume")
        for kwargs in ({}, {"no_alignment": True, "init_checkpoint": "pretend-final"}):
            with self.assertRaises(ValueError): d.initialization_mode(raw, **kwargs)
        with self.assertRaises(ValueError): d.initialization_mode(config())
        for changed in (config("no_alignment", seed=29), config("no_alignment", loops=2),
                        config("no_alignment", stage="alignment")):
            with self.assertRaises(ValueError): d.initialization_mode(changed, no_alignment=True)

    def test_matching_alignment_seed_and_exact_resume_method_are_required(self):
        raw = config(seed=29); parent = contract(config(seed=29, stage="alignment"))
        d.verify_previous_contract(parent, raw, resume=False)
        aligned = d.make_initialization(raw, "matching_alignment_final", previous=parent,
                                       projector_sha256="d" * 64, checkpoint_provenance_sha256="e" * 64)
        self.assertEqual(aligned["alignment_initialization"], parent["initialization"])
        for bad in (contract(config(seed=17, stage="alignment")), contract(config(seed=29)), {}):
            with self.assertRaises(ValueError): d.verify_previous_contract(bad, raw, resume=False)
        noalign = config("no_alignment"); previous = contract(noalign)
        resumed = d.make_initialization(noalign, "exact_resume", previous=previous)
        self.assertEqual(resumed, previous["initialization"])
        with self.assertRaises(ValueError): d.verify_previous_contract(previous, config(), resume=True)
        previous["initialization"].pop("pretrained_backbones")
        with self.assertRaises(ValueError): d.make_initialization(noalign, "exact_resume", previous=previous)

    def test_completion_requires_target_and_preserves_no_alignment_identity(self):
        raw = config("no_alignment"); prior = contract(raw)
        identity = d.completion_identity(prior, {"global_step": 718, "max_steps": 718})
        self.assertEqual(identity["method"], "no_alignment")
        self.assertEqual(identity["seed"], 17)
        self.assertFalse(identity["two_stage_training_claim"])
        for state in ({"global_step": 50, "max_steps": 718}, {"global_step": 718, "max_steps": 1436}):
            with self.assertRaises(ValueError): d.completion_identity(prior, state)
        prior["initialization"]["initial_connector_state_sha256"] = "missing"
        with self.assertRaises(ValueError): d.completion_identity(prior, {"global_step": 718, "max_steps": 718})

    def test_backbone_bytes_are_all_hashed(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); (root / "config.json").write_text("{}")
            (root / "model.safetensors").write_bytes(b"fixed pretrained bytes")
            first = d.tree_identity(root)
            (root / "model.safetensors").write_bytes(b"changed bytes")
            self.assertNotEqual(first["files_sha256"], d.tree_identity(root)["files_sha256"])
            (root / "broken").symlink_to(root / "absent")
            with self.assertRaises(ValueError): d.tree_identity(root)

    def test_two_roots_and_external_files_charged_without_internal_duplicates(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); a, b = root / "a", root / "b"; a.mkdir(); b.mkdir()
            (a / "p").write_bytes(b"abc"); (b / "internal").symlink_to(a / "p")
            outside = root / "outside"; outside.write_bytes(b"12345"); (b / "outside").symlink_to(outside)
            self.assertEqual(sum(d.staging_bytes([a, b]).values()), 8)
            (b / "bad").symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError): d.staging_bytes([a, b])

    def test_minimal_failure_archive_import_closure_after_partial_snapshot(self):
        with tempfile.TemporaryDirectory() as name:
            source = Path(name) / "source"
            target = source / "scripts/operations/run_p1_stage_v9.py"
            target.parent.mkdir(parents=True)
            target.write_bytes((ROOT / "scripts/operations/run_p1_stage_v9.py").read_bytes())
            r.ensure_failure_archive_dependencies(ROOT, source)
            code = f"""import importlib.util, sys
from pathlib import Path
root = Path({str(source)!r})
sys.argv = ['runner', '--archive-helper']
for label, name in [('runner', 'run_p1_stage_v9.py'), ('assets', 'stage_assets.py')]:
    spec = importlib.util.spec_from_file_location(label, root / 'scripts/operations' / name)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
print('minimal archive imports passed; no upload invoked')
"""
            result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("no upload invoked", result.stdout)

    def test_shell_help_and_unknown_options_have_no_runtime_side_effects(self):
        shell = ROOT / "scripts/operations/run_p1_stage_v9.sh"
        result = subprocess.run(["bash", str(shell), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        result = subprocess.run(["bash", str(shell), "id", "sft", "/data", "/media", "/config", "/formal", "/source",
                                 "--base", "/somewhere"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unsupported", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
