"""Static and JSON gates for the finite formal P1 stage runner."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

try:
    import torch
    from safetensors.torch import save_file as save_safetensors
except ImportError:
    torch = None
    save_safetensors = None


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_p1_stage", ROOT / "scripts/operations/run_p1_stage.py"
)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_safetensors(path, tensors):
    if save_safetensors is None or torch is None:
        raise RuntimeError("torch and safetensors are required for this fixture")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_safetensors({name: torch.as_tensor(value) for name, value in tensors.items()}, str(path))


def write_projector(path, language_dim=3, vision_dim=2, *, dtype=None, gate_shape=None):
    dtype = torch.float32 if dtype is None else dtype
    write_safetensors(path, {
        "gate.weight": torch.zeros(gate_shape or (1, vision_dim), dtype=dtype),
        "gate.bias": torch.zeros((1,), dtype=dtype),
        "projection.0.weight": torch.zeros((language_dim, vision_dim), dtype=dtype),
        "projection.0.bias": torch.zeros((language_dim,), dtype=dtype),
        "projection.2.weight": torch.zeros((language_dim, language_dim), dtype=dtype),
        "projection.2.bias": torch.zeros((language_dim,), dtype=dtype),
    })


class P1RunnerGateTests(unittest.TestCase):
    def test_recipe_is_exact_eight_rank_gb128_formal_seed17(self):
        raw = {
            "model": {"num_loops": 4, "num_frames": 8, "pool_grid": 4},
            "training": {
                "stage": "alignment", "formal_training": True,
                "dataloader_num_workers": 0, "seed": 17,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 16,
                "num_train_epochs": 1, "max_steps": -1, "bf16": True,
                "save_steps": 200,
            },
        }
        identity = runner.validate_recipe(raw, "alignment")
        self.assertEqual(identity["global_batch_size"], 128)
        for field, value in (("formal_training", False), ("dataloader_num_workers", 2),
                             ("seed", 18), ("gradient_accumulation_steps", 8),
                             ("save_only_model", True), ("load_best_model_at_end", True),
                             ("require_final_resume_checkpoint", False),
                             ("save_strategy", "no")):
            changed = json.loads(json.dumps(raw))
            changed["training"][field] = value
            with self.assertRaises(ValueError):
                runner.validate_recipe(changed, "alignment")

    def test_stage_storage_plan_bounds_rotation_and_final_copy(self):
        self.assertEqual(runner.RUN_WORKING_RESERVE_BYTES, 3_000_000_000)
        alignment = runner.stage_storage_plan("alignment", 2, 1_000, 100)
        self.assertEqual(alignment["checkpoint_bytes"], 1_200)
        self.assertEqual(alignment["checkpoint_rotation_peak_bytes"], 3_600)
        self.assertEqual(alignment["checkpoint_plus_final_peak_bytes"], 3_400)
        self.assertEqual(alignment["required_additional_bytes"],
                         3_600 + runner.RUN_WORKING_RESERVE_BYTES)
        sft = runner.stage_storage_plan("sft", 2, 1_000, 100)
        self.assertEqual(sft["checkpoint_bytes"], 3_000)
        self.assertEqual(sft["checkpoint_rotation_peak_bytes"], 9_000)
        self.assertEqual(sft["checkpoint_plus_final_peak_bytes"], 7_000)
        larger = runner.stage_storage_plan("sft", 3, 1_000, 100)
        self.assertGreater(larger["required_additional_bytes"],
                           sft["required_additional_bytes"])

    def test_storage_admission_rejects_limit_equality_and_insufficient_free_space(self):
        reserve = 4_000
        plan = {"required_additional_bytes": reserve}
        with self.assertRaisesRegex(RuntimeError, "operational limit"):
            runner.require_storage_admission(
                runner.STORAGE_OPERATIONAL_LIMIT_BYTES - reserve,
                reserve + runner.FILESYSTEM_FREE_GUARD_BYTES, plan)
        with self.assertRaisesRegex(RuntimeError, "filesystem free bytes"):
            runner.require_storage_admission(
                0, reserve + runner.FILESYSTEM_FREE_GUARD_BYTES - 1, plan)
        receipt = runner.require_storage_admission(
            0, reserve + runner.FILESYSTEM_FREE_GUARD_BYTES, plan)
        self.assertEqual(receipt["status"], "admitted")

    def test_parser_does_not_expose_a_storage_reserve_override(self):
        option_strings = {option for action in runner.parser()._actions
                          for option in action.option_strings}
        self.assertFalse(any("reserve" in option for option in option_strings))

    def _formal_fixture(self, root):
        data, formal, media = root / "data", root / "formal", root / "media"
        data.mkdir(); formal.mkdir(); media.mkdir()
        for name, content in (("train.jsonl", "train\n"),
                              ("validation.jsonl", "validation\n"),
                              ("coverage.jsonl", "coverage\n")):
            (data / name).write_text(content)
        (media / "clip.mp4").write_bytes(b"attested-media")
        media_entries = [{
            "relative_path": "clip.mp4", "source_video_id": "youtube:abcdefghijk",
            "size_bytes": (media / "clip.mp4").stat().st_size,
            "sha256": sha(media / "clip.mp4"),
        }]
        media_inventory = {
            "schema_version": 1, "kind": "formal_training_media_sha256_inventory",
            "media_root": str(media.resolve()), "file_count": 1,
            "total_bytes": media_entries[0]["size_bytes"],
            "entries_sha256": runner.canonical_sha256(media_entries),
            "entries": media_entries,
        }
        (data / "training_media_inventory.json").write_text(json.dumps(media_inventory))
        outputs = {name: sha(data / name) for name in
                   ("train.jsonl", "validation.jsonl", "coverage.jsonl",
                    "training_media_inventory.json")}
        preparation = {
            "schema_version": 1, "status": "formal_training_data_ready", "formal_data_ready": True,
            "media_root": str(media.resolve()), "output_sha256": outputs,
            "training_media_inventory": {
                "relative_path": "training_media_inventory.json",
                "sha256": outputs["training_media_inventory.json"],
                "file_count": 1, "total_bytes": media_entries[0]["size_bytes"],
                "entries_sha256": media_inventory["entries_sha256"],
            },
        }
        (data / "preparation_report.json").write_text(json.dumps(preparation))
        data_ready = {
            "schema_version": 1, "purpose": "FORMAL_TRAINING_DATA", "formal_data_ready": True,
            "report_sha256": sha(data / "preparation_report.json"),
            "manifest_sha256": {"train.jsonl": outputs["train.jsonl"],
                                "validation.jsonl": outputs["validation.jsonl"]},
            "coverage_sha256": outputs["coverage.jsonl"],
            "training_media_inventory_sha256": outputs["training_media_inventory.json"],
            "training_media_entries_sha256": media_inventory["entries_sha256"],
            "training_media_file_count": 1,
            "training_media_total_bytes": media_entries[0]["size_bytes"],
        }
        (data / "READY.json").write_text(json.dumps(data_ready))
        gates = {name: {"status": "passed", "details": []} for name in
                 runner.FORMAL_TRAINING_GATES}
        report = {
            "schema_version": 1, "status": "ready_for_training", "formal_training_ready": True,
            "formal_evaluation_ready": False,
            "benchmark_source_identity_gate": "passed_on_declared_subset",
            "training_readiness_gates": gates,
            "training_manifests": {"valid": True, "sha256": {"train": outputs["train.jsonl"],
                                                "validation": outputs["validation.jsonl"]}},
            "training_candidate_coverage": {"sha256": outputs["coverage.jsonl"], "valid": True},
            "training_data_attestation": {
                "provided": True, "valid": True,
                "report_sha256": sha(data / "preparation_report.json"),
                "ready_sha256": sha(data / "READY.json"),
                "media_inventory": {
                    "sha256": outputs["training_media_inventory.json"],
                    "entries_sha256": media_inventory["entries_sha256"],
                    "file_count": 1, "total_bytes": media_entries[0]["size_bytes"],
                },
            },
            "training_media_attestation": {
                "checked": True, "passed": True,
                "entries_sha256": media_inventory["entries_sha256"],
                "file_count": 1, "total_bytes": media_entries[0]["size_bytes"],
            },
        }
        (formal / "source_gate_report.json").write_text(json.dumps(report))
        (formal / "source_policy.json").write_text(json.dumps({
            "schema_version": 1, "policy_id": "deepprobe-formal-benchmark-source-v1"}))
        (formal / "training_media_inventory.json").write_text(json.dumps({
            "checked": True, "complete": True, "root": str(media.resolve()),
            "expected_files": 1, "present_files": 1, "missing": [], "errors": [],
            "entries": media_entries,
        }))
        artifact_hashes = {name: sha(formal / name) for name in
                           ("source_gate_report.json", "source_policy.json",
                            "training_media_inventory.json")}
        ready = {
            "schema_version": 1, "status": "ready_for_training", "formal_training_ready": True,
            "formal_evaluation_ready": False,
            "benchmark_source_identity_gate": "passed_on_declared_subset",
            "policy_id": "deepprobe-formal-benchmark-source-v1",
            "training_readiness_gates": gates,
            "immutable_input_sha256": {
                "training_train": outputs["train.jsonl"],
                "training_validation": outputs["validation.jsonl"],
                "training_coverage": outputs["coverage.jsonl"],
                "training_media_inventory": outputs["training_media_inventory.json"],
            },
            "artifact_sha256": artifact_hashes,
        }
        (formal / "FORMAL_READY.json").write_text(json.dumps(ready))
        return data, formal / "FORMAL_READY.json", media

    def test_formal_gate_binds_data_reports_and_every_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            data, ready, media = self._formal_fixture(Path(directory))
            identity = runner.verify_formal_inputs(data, ready, media)
            self.assertEqual(identity["data_sha256"]["coverage.jsonl"], sha(data / "coverage.jsonl"))
            (ready.parent / "source_policy.json").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "Formal artifact changed"):
                runner.verify_formal_inputs(data, ready, media)

    def test_formal_gate_fails_if_coverage_bytes_change(self):
        with tempfile.TemporaryDirectory() as directory:
            data, ready, media = self._formal_fixture(Path(directory))
            (data / "coverage.jsonl").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "train/validation/coverage hashes"):
                runner.verify_formal_inputs(data, ready, media)

    def test_formal_gate_rehashes_every_attested_media_file(self):
        with tempfile.TemporaryDirectory() as directory:
            data, ready, media = self._formal_fixture(Path(directory))
            (media / "clip.mp4").write_bytes(b"changed-media")
            with self.assertRaisesRegex(ValueError, "size/SHA-256 differs"):
                runner.verify_formal_inputs(data, ready, media)

    def test_frozen_source_manifest_is_exact_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source, manifest = base / "experiment", base / "SOURCE_MANIFEST.json"
            for name in runner.SOURCE_DIRS:
                (source / name).mkdir(parents=True)
            for relative in (
                "scripts/operations/run_p1_stage.sh",
                "scripts/operations/run_p1_stage.py",
                "scripts/operations/stage_assets.py",
                "scripts/preflight.py", "src/deepprobe_vlm/train.py",
                "src/deepprobe_vlm/formal_integrity.py",
            ):
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative + "\n")
            for name in runner.SOURCE_FILES:
                (source / name).write_text(name + "\n")
            manifest.write_text(json.dumps({"files": runner.source_inventory(source)}))
            identity = runner.verify_source_manifest(source, manifest)
            self.assertEqual(identity["file_count"], len(identity["files"]))
            (source / "src/deepprobe_vlm/train.py").write_text("mutated\n")
            with self.assertRaisesRegex(ValueError, "changed="):
                runner.verify_source_manifest(source, manifest)

    def test_training_outputs_require_train_cli_live_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal = {"formal_ready_sha256": "a" * 64,
                      "data_sha256": {"train.jsonl": "b" * 64,
                                      "validation.jsonl": "c" * 64},
                      "training_media": {"inventory_sha256": "d" * 64,
                                         "entries_sha256": "e" * 64,
                                         "file_count": 1, "total_bytes": 8}}
            with self.assertRaisesRegex(ValueError, "completion receipt"):
                runner.verify_training_outputs(root, "alignment", formal, "f" * 64, "0" * 64)
            contract = {"world_size": 8, "global_batch_size": 128,
                        "formal_training": True, "require_final_resume_checkpoint": True,
                        "config_sha256": "0" * 64,
                        "frozen_source_manifest_sha256": "f" * 64,
                        "formal_ready_sha256": "a" * 64,
                        "train_manifest_sha256": "b" * 64,
                        "eval_manifest_sha256": "c" * 64}
            (root / "run_manifest.json").write_text(json.dumps({
                "world_size": 8, "stage": "alignment", "training_contract": contract}))
            (root / "FORMAL_TRAIN_COMPLETE.json").write_text(json.dumps({
                "schema_version": 1, "status": "formal_training_complete",
                "stage": "alignment", "world_size": 8,
                "frozen_source_manifest_sha256": "f" * 64,
                "formal_ready_sha256": "a" * 64,
                "train_manifest_sha256": "b" * 64,
                "eval_manifest_sha256": "c" * 64,
                "run_manifest_sha256": sha(root / "run_manifest.json"),
                "media_before": {"inventory_sha256": "d" * 64,
                                 "entries_sha256": "e" * 64,
                                 "file_count": 1, "total_bytes": 8,
                                 "metadata_sha256": "1" * 64},
                "media_after": {"inventory_sha256": "d" * 64,
                                "entries_sha256": "e" * 64,
                                "file_count": 1, "total_bytes": 8,
                                "metadata_sha256": "2" * 64},
            }))
            with self.assertRaisesRegex(ValueError, "live source/media contract"):
                runner.verify_training_outputs(root, "alignment", formal, "f" * 64, "0" * 64)

    @unittest.skipUnless(save_safetensors is not None, "safetensors test dependency unavailable")
    def test_sft_initializer_requires_matching_rfg_and_train_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "alignment"
            for component in ("language_model", "vision_model"):
                (root / component).mkdir(parents=True)
                (root / component / "config.json").write_text(json.dumps({
                    "hidden_size": 3 if component == "language_model" else 2}))
                write_safetensors(root / component / "model.safetensors", {f"{component}.weight": [1.0]})
            (root / "tokenizer").mkdir()
            (root / "tokenizer/tokenizer_config.json").write_text("{}")
            (root / "image_processor").mkdir()
            (root / "image_processor/preprocessor_config.json").write_text("{}")
            write_projector(root / "projector.safetensors")
            (root / "training_args.bin").write_bytes(b"args")
            model = {
                "language_model": "/home/jinhaopeng/language", "language_revision": "a" * 40,
                "language_backend": "causal_lm", "vision_model": "/home/jinhaopeng/vision",
                "vision_revision": "b" * 40, "num_loops": 4, "num_frames": 8,
                "pool_grid": 4, "max_text_tokens": 512, "attn_implementation": "sdpa",
            }
            (root / "deepprobe_config.json").write_text(json.dumps(model))
            formal = {"formal_ready_sha256": "c" * 64,
                      "data_sha256": {"train.jsonl": "d" * 64,
                                      "validation.jsonl": "e" * 64}}
            contract = {"stage": "alignment", "formal_training": True,
                        "require_final_resume_checkpoint": True,
                        "frozen_source_manifest_sha256": "f" * 64,
                        "formal_ready_sha256": "c" * 64,
                        "train_manifest_sha256": "d" * 64,
                        "eval_manifest_sha256": "e" * 64,
                        "world_size": 8, "global_batch_size": 128}
            (root / "training_provenance.json").write_text(json.dumps({"training_contract": contract}))
            identity = runner.verify_alignment_final(root, {"model": model}, formal, "f" * 64)
            self.assertEqual(identity["rfg"], {"num_loops": 4, "num_frames": 8, "pool_grid": 4})
            changed = json.loads(json.dumps(model)); changed["num_loops"] = 2
            with self.assertRaisesRegex(ValueError, "num_loops"):
                runner.verify_alignment_final(root, {"model": changed}, formal, "f" * 64)

    @unittest.skipUnless(save_safetensors is not None, "safetensors test dependency unavailable")
    def test_resume_checkpoint_is_complete_and_exactly_contract_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint-7"
            for component in ("language_model", "vision_model"):
                (root / component).mkdir(parents=True)
                (root / component / "config.json").write_text(json.dumps({
                    "hidden_size": 3 if component == "language_model" else 2}))
                write_safetensors(root / component / "model.safetensors", {f"{component}.weight": [1.0]})
            (root / "tokenizer").mkdir()
            (root / "tokenizer/tokenizer_config.json").write_text("{}")
            (root / "image_processor").mkdir()
            (root / "image_processor/preprocessor_config.json").write_text("{}")
            write_projector(root / "projector.safetensors")
            (root / "training_args.bin").write_bytes(b"args")
            (root / "optimizer.pt").write_bytes(b"optimizer")
            (root / "scheduler.pt").write_bytes(b"scheduler")
            for rank in range(8):
                (root / f"rng_state_{rank}.pth").write_bytes(f"rng-{rank}".encode())
            (root / "trainer_state.json").write_text(json.dumps({"global_step": 7,
                                                                   "log_history": [{"loss": 1.0}]}))
            model = {
                "language_model": "/home/jinhaopeng/language", "language_revision": "a" * 40,
                "language_backend": "causal_lm", "vision_model": "/home/jinhaopeng/vision",
                "vision_revision": "b" * 40, "num_loops": 1, "num_frames": 8,
                "pool_grid": 4, "freeze_vision": False, "max_text_tokens": 512,
                "attn_implementation": "sdpa",
            }
            (root / "deepprobe_config.json").write_text(json.dumps(model))
            formal = {"formal_ready_sha256": "c" * 64,
                      "data_sha256": {"train.jsonl": "d" * 64,
                                      "validation.jsonl": "e" * 64}}
            contract = {
                "stage": "sft", "config_sha256": "1" * 64,
                "frozen_source_manifest_sha256": "f" * 64,
                "train_manifest_sha256": "d" * 64, "eval_manifest_sha256": "e" * 64,
                "formal_training": True, "formal_ready_sha256": "c" * 64,
                "require_final_resume_checkpoint": True,
                "global_batch_size": 128, "world_size": 8, "training": {"seed": "17"},
            }
            (root / "training_provenance.json").write_text(json.dumps({"training_contract": contract}))
            identity = runner.verify_resume_checkpoint(
                root, {"model": model}, "1" * 64, formal, "f" * 64, "sft")
            self.assertEqual(identity["optimizer_step"], 7)
            language = root / "language_model"
            (language / "model.safetensors").unlink()
            write_safetensors(language / "model-00001-of-00002.safetensors", {"layer.0": [1.0]})
            write_safetensors(language / "model-00002-of-00002.safetensors", {"layer.1": [2.0]})
            (language / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {"total_size": 8},
                "weight_map": {"layer.0": "model-00001-of-00002.safetensors",
                               "layer.1": "model-00002-of-00002.safetensors"},
            }))
            runner.verify_resume_checkpoint(
                root, {"model": model}, "1" * 64, formal, "f" * 64, "sft")
            (language / "model-00002-of-00002.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "Incomplete sharded checkpoint"):
                runner.verify_resume_checkpoint(
                    root, {"model": model}, "1" * 64, formal, "f" * 64, "sft")
            write_safetensors(language / "model-00002-of-00002.safetensors", {"layer.1": [2.0]})
            (root / "rng_state_7.pth").unlink()
            with self.assertRaisesRegex(ValueError, "rank0..7 RNG"):
                runner.verify_resume_checkpoint(
                    root, {"model": model}, "1" * 64, formal, "f" * 64, "sft")

    @unittest.skipUnless(save_safetensors is not None, "safetensors test dependency unavailable")
    def test_sharded_model_tree_checks_index_paths_and_all_shard_sizes(self):
        def fixture(base):
            root = base / "checkpoint"
            for relative in (
                "deepprobe_config.json", "projector.safetensors", "training_provenance.json",
                "training_args.bin", "language_model/config.json", "vision_model/config.json",
                "tokenizer/tokenizer_config.json", "image_processor/preprocessor_config.json",
                "vision_model/model.safetensors",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if relative == "projector.safetensors":
                    write_projector(path)
                elif path.suffix == ".safetensors":
                    write_safetensors(path, {"vision.weight": [1.0]})
                else:
                    path.write_bytes(b"fixture")
            (root / "language_model/config.json").write_text(json.dumps({"hidden_size": 3}))
            (root / "vision_model/config.json").write_text(json.dumps({"hidden_size": 2}))
            language = root / "language_model"
            shards = ("model-00001-of-00002.safetensors",
                      "model-00002-of-00002.safetensors")
            write_safetensors(language / shards[0], {"layer.0": [1.0]})
            write_safetensors(language / shards[1], {"layer.1": [2.0]})
            index = language / "model.safetensors.index.json"
            document = {"metadata": {"total_size": 8},
                        "weight_map": {"layer.0": shards[0], "layer.1": shards[1]}}
            index.write_text(json.dumps(document))
            return root, language, index, document, shards

        with tempfile.TemporaryDirectory() as directory:
            root, _, _, _, _ = fixture(Path(directory))
            runner._require_model_tree(root)

        def missing_shard(language, index, document, shards):
            (language / shards[1]).unlink()

        def empty_shard(language, index, document, shards):
            (language / shards[1]).write_bytes(b"")

        def unsafe_index_path(language, index, document, shards):
            document["weight_map"]["layer.1"] = "../outside.safetensors"
            index.write_text(json.dumps(document))

        def impossible_total_size(language, index, document, shards):
            document["metadata"]["total_size"] = 10_000
            index.write_text(json.dumps(document))

        def underreported_total_size(language, index, document, shards):
            document["metadata"]["total_size"] = 4
            index.write_text(json.dumps(document))

        def symlinked_shard(language, index, document, shards):
            outside = language.parent / "outside.safetensors"
            outside.write_bytes(b"outside")
            (language / shards[1]).unlink()
            (language / shards[1]).symlink_to(outside)

        def extra_shard(language, index, document, shards):
            write_safetensors(language / "unindexed.safetensors", {"extra": [3.0]})

        def corrupt_projector(language, index, document, shards):
            (language.parent / "projector.safetensors").write_bytes(b"corrupt")

        def corrupt_unsharded_model(language, index, document, shards):
            (language.parent / "vision_model/model.safetensors").write_bytes(b"corrupt")

        def corrupt_shard(language, index, document, shards):
            (language / shards[1]).write_bytes(b"corrupt" * 100)

        def tensor_mapped_to_wrong_shard(language, index, document, shards):
            document["weight_map"] = {"layer.0": shards[1], "layer.1": shards[0]}
            index.write_text(json.dumps(document))

        def undeclared_tensor_in_shard(language, index, document, shards):
            write_safetensors(language / shards[0], {"layer.0": [1.0], "rogue": [3.0]})
            document["metadata"]["total_size"] = 12
            index.write_text(json.dumps(document))

        def declared_tensor_missing(language, index, document, shards):
            document["weight_map"]["layer.missing"] = shards[0]
            index.write_text(json.dumps(document))

        def wrong_projector_contract(language, index, document, shards):
            write_safetensors(language.parent / "projector.safetensors", {"wrong": [1.0]})

        def wrong_projector_shape(language, index, document, shards):
            write_projector(language.parent / "projector.safetensors", gate_shape=(1, 3))

        def wrong_projector_dtype(language, index, document, shards):
            write_projector(language.parent / "projector.safetensors", dtype=torch.bfloat16)

        for label, mutate, error in (
            ("missing", missing_shard, "Incomplete sharded checkpoint weights"),
            ("empty", empty_shard, "Empty or unsafe model checkpoint weight shard"),
            ("unsafe path", unsafe_index_path, "Invalid sharded weight index"),
            ("oversized index", impossible_total_size, "total_size exceeds shard bytes"),
            ("underreported index", underreported_total_size, "total_size mismatch"),
            ("symlink", symlinked_shard, "Empty or unsafe model checkpoint weight shard"),
            ("extra", extra_shard, "Incomplete sharded checkpoint weights"),
            ("corrupt projector", corrupt_projector, "Invalid safetensors file projector"),
            ("corrupt unsharded", corrupt_unsharded_model, "Invalid safetensors file vision_model"),
            ("corrupt shard", corrupt_shard, "Invalid safetensors file language_model"),
            ("wrong tensor owner", tensor_mapped_to_wrong_shard, "Sharded tensor map mismatch"),
            ("undeclared tensor", undeclared_tensor_in_shard, "Sharded tensor map mismatch"),
            ("missing tensor", declared_tensor_missing, "Sharded tensor map mismatch"),
            ("wrong projector", wrong_projector_contract, "FP32 architecture contract"),
            ("wrong projector shape", wrong_projector_shape, "FP32 architecture contract"),
            ("wrong projector dtype", wrong_projector_dtype, "FP32 architecture contract"),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root, language, index, document, shards = fixture(Path(directory))
                mutate(language, index, document, shards)
                with self.assertRaisesRegex(ValueError, error):
                    runner._require_model_tree(root)

    def test_shell_help_is_side_effect_free(self):
        completed = subprocess.run(["bash", str(ROOT / "scripts/operations/run_p1_stage.sh"), "--help"],
                                   text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0)
        self.assertIn("alignment|sft", completed.stdout)
        rejected = subprocess.run([
            "bash", str(ROOT / "scripts/operations/run_p1_stage.sh"),
            "run", "sft", "data", "media", "config", "ready", "source",
        ], text=True, capture_output=True)
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("requires --alignment-final", rejected.stderr)

    def test_cleanup_filter_does_not_remove_training_metadata_or_rng(self):
        self.assertTrue(runner.is_cleanable_weight(Path("training/final/model.safetensors")))
        self.assertTrue(runner.is_cleanable_weight(Path("training/checkpoint-200/optimizer.pt")))
        self.assertFalse(runner.is_cleanable_weight(Path("training/final/training_args.bin")))
        self.assertFalse(runner.is_cleanable_weight(Path("training/checkpoint-200/rng_state.pth")))
        self.assertFalse(runner.should_cleanup_training_payload(
            Path("training/final/model.safetensors"), "alignment"))
        self.assertTrue(runner.should_cleanup_training_payload(
            Path("training/checkpoint-200/optimizer.pt"), "alignment"))
        self.assertTrue(runner.should_cleanup_training_payload(
            Path("training/final/model.safetensors"), "sft"))


if __name__ == "__main__":
    unittest.main()
