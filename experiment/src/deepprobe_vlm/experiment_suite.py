"""Complete declared experiment DAG, without scheduling or artifact deletion.

A plan is an intent, never evidence that a training stage or evaluation succeeded.
Bindings and independently checked receipts are required by the serial executor.
This module uses only the standard library and never imports a model runtime.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from . import declared_training

SCHEMA_VERSION = 1
SUITE_ID = "deepprobe-complete-declared-v1"
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = EXPERIMENT_ROOT / "configs/experiment_suite.json"
RECIPE_INVENTORY = "configs/train/suite/RECIPES.json"
BENCHMARKS = ("videomme899", "tempcompass_mc")
KINDS = {"gate", "train", "evaluate", "latency", "compare_latency", "archive", "cleanup", "aggregate"}
ALLOCATION = {"cluster": "ali-gpu", "queue": "stream", "priority": "low", "nodes": 1,
              "gpus": 8, "restart_policy": "Never", "max_active_project_jobs": 1,
              "max_concurrent_gpu_tasks": 1, "forced_node": None}
TRAINING_DATA = {"train_examples": 91826, "validation_examples": 4472,
                 "total_qa": 96298, "source_clips": 3979,
                 "train_manifest_sha256": "958afecf0f358ae84b5e0c313657db5a5410d53d149c2ba4ea1636f8fa9e1ba9",
                 "validation_manifest_sha256": "533e427260266f6a426b5c4486f16e5ec41881000e1a9b6e01e1e225499d2ab1",
                 "split_scope": "fixed source-disjoint LLaVA-Video subset",
                 "global_batch_size": 128, "target_optimizer_steps_per_stage": 718}
LATENCY_VARIANTS = (
    ("reference", True, False, "reference"),
    ("no_cache", False, False, "reference"),
    ("last_token", True, True, "reference"),
    ("triton_pool", True, False, "triton"),
    ("both", True, True, "triton"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def plan_sha256(plan):
    return hashlib.sha256(canonical(plan)).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_id(kind, model_id=None, qualifier=None):
    return "__".join(str(x) for x in (kind, model_id, qualifier) if x is not None)


def model_specs():
    models = []
    for seed in (17, 29, 43):
        for family, loops in (("ouro_r1", 1), ("ouro_r2", 2), ("ouro_r4", 4), ("smollm2", 1)):
            models.append({"id": f"{family}_s{seed}", "group": "core", "family": family,
                           "seed": seed, "loops": loops, "frames": 8, "pool_grid": 4,
                           "freeze_vision_sft": False, "method": "two_stage"})
    for axis, frames, grid in (("f4", 4, 4), ("f16", 16, 4), ("g2", 8, 2), ("g8", 8, 8)):
        models.append({"id": f"ouro_r4_{axis}_s17", "group": "ablation", "family": "ouro_r4",
                       "seed": 17, "loops": 4, "frames": frames, "pool_grid": grid,
                       "freeze_vision_sft": False, "method": "two_stage"})
    models.extend([
        {"id": "ouro_r4_frozen_vision_s17", "group": "ablation", "family": "ouro_r4",
         "seed": 17, "loops": 4, "frames": 8, "pool_grid": 4, "freeze_vision_sft": True,
         "method": "shared_alignment"},
        {"id": "ouro_r4_no_alignment_s17", "group": "ablation", "family": "ouro_r4",
         "seed": 17, "loops": 4, "frames": 8, "pool_grid": 4, "freeze_vision_sft": False,
         "method": "no_alignment"},
    ])
    priority = {name: index for index, name in enumerate((
        "ouro_r4_s17", "ouro_r4_frozen_vision_s17", "smollm2_s29"))}
    return sorted(models, key=lambda model: priority.get(model["id"], len(priority)))


def _recipe_rows(root):
    inventory = json.loads((Path(root) / RECIPE_INVENTORY).read_text())
    require(inventory.get("schema_version") == 1 and isinstance(inventory.get("files"), list), "Invalid recipe inventory")
    rows = {(r["model_id"], r["stage"]): r for r in inventory["files"]}
    require(len(rows) == len(inventory["files"]) == 34, "Expected 34 unique stage recipes")
    return rows


def _training_hint(model_id, stage):
    accepted = {(family, phase) for family in ("ouro_r1_s17", "ouro_r2_s17") for phase in ("alignment", "sft")}
    accepted.add(("ouro_r4_s17", "alignment"))
    if (model_id, stage) in accepted:
        return {"status": "reuse_candidate", "automatic_skip": False,
                "required_binding": "independent_training_and_archive_acceptance; restore absent bytes before consumption"}
    if (model_id, stage) in {("ouro_r4_s17", "sft"), ("smollm2_s29", "alignment")}:
        return {"status": "resume_candidate", "checkpoint_step": 600 if model_id == "ouro_r4_s17" else 300,
                "resume_accepted": False, "automatic_restart_from_zero": False,
                "required_binding": "full checkpoint/optimizer/scheduler/rank RNG audit, matching source/provenance and fresh capacity"}
    return {"status": "planned", "automatic_skip": False}


def _evaluation_protocol(model, benchmark):
    changed_shape = model["frames"] != 8 or model["pool_grid"] != 4
    if benchmark == "videomme899":
        return "videomme899-reference-shape-ablation-v1" if changed_shape else "videomme-899-no-subtitles-no-audio-v1"
    return "tempcompass-mc-shape-ablation-v1" if changed_shape else "tempcompass-mc-1580-uniform8-noaudio-v1"


def build_plan(config_root=EXPERIMENT_ROOT):
    """Construct the complete fixed matrix from public, hashed stage recipes."""
    recipes = _recipe_rows(config_root)
    models = model_specs()
    tasks = [{"id": "gpu_gate", "kind": "gate", "model_id": None, "dependencies": [],
              "checks": ["eight_gpu_identity", "bf16_attention_forward_backward", "eight_rank_nccl",
                         "cuda_pool_numerical_and_microbenchmark"], "model_quality_claim": False}]
    for model in models:
        mid = model["id"]
        train_ids = []
        for stage in (("alignment", "sft") if model["method"] == "two_stage" else ("sft",)):
            rid = recipes[(mid, stage)]
            tid = task_id("train", mid, stage)
            deps = ["gpu_gate"]
            init = {"kind": "pretrained_backbones_random_connector"}
            if stage == "sft" and model["method"] != "no_alignment":
                source_model = "ouro_r4_s17" if model["method"] == "shared_alignment" else mid
                parent = task_id("train", source_model, "alignment")
                deps.extend([parent, task_id("archive", source_model, "alignment")])
                init = {"kind": "matching_alignment_final", "checkpoint_task_id": parent}
            tasks.append({"id": tid, "kind": "train", "model_id": mid, "dependencies": deps,
                          "stage": stage, "config": {k: rid[k] for k in ("path", "sha256", "bytes")},
                          "entry": rid["entry"], "initializer": init,
                          "expected_optimizer_steps": 718, "historical_hint": _training_hint(mid, stage)})
            tasks.append({"id": task_id("archive", mid, stage), "kind": "archive", "model_id": mid,
                          "dependencies": [tid], "archive_scope": "checkpoint", "checkpoint_task_id": tid,
                          "full_get_sha256_required": True})
            train_ids.append(tid)
        sft = task_id("train", mid, "sft")
        results = []
        for benchmark in BENCHMARKS:
            tid = task_id("evaluate", mid, benchmark)
            tasks.append({"id": tid, "kind": "evaluate", "model_id": mid, "dependencies": [sft],
                          "benchmark": benchmark, "protocol": _evaluation_protocol(model, benchmark),
                          "checkpoint_task_id": sft, "frames": model["frames"], "pool_grid": model["pool_grid"],
                          "max_new_tokens": 16, "audio": False, "subtitles": False, "include_answer": False,
                          "full_declared_coverage_required": True, "checkpoint_selection": "final_step",
                          "frame_sampling": "uniform_indices_from_original_video",
                          "resample_existing_frame_cache": False,
                          "historical_hint": {"status": "reuse_candidate" if mid in ("ouro_r1_s17", "ouro_r2_s17") else "planned",
                                              "automatic_skip": False}})
            results.append(tid)
        if mid in ("ouro_r1_s17", "ouro_r2_s17", "ouro_r4_s17"):
            for variant, cache, last, backend in LATENCY_VARIANTS:
                tid = task_id("latency", mid, variant)
                tasks.append({"id": tid, "kind": "latency", "model_id": mid, "dependencies": [sft],
                              "checkpoint_task_id": sft, "protocol": "deepprobe-development-latency-v2",
                              "cache": cache, "implementation": {"last_token_logits": last, "pool_backend": backend},
                              "sample_count": 32, "warmup_per_sample": 3, "repeats_per_sample": 10,
                              "max_new_tokens": 16, "batch_size": 1, "dtype": "bfloat16", "measured_devices": 1,
                              "selection_uses_heldout": False, "separate_profile": variant == "reference"})
                results.append(tid)
            for variant, _, _, _ in LATENCY_VARIANTS[1:]:
                tid = task_id("compare_latency", mid, variant)
                tasks.append({"id": tid, "kind": "compare_latency", "model_id": mid,
                              "dependencies": [task_id("latency", mid, "reference"), task_id("latency", mid, variant)],
                              "reference_task_id": task_id("latency", mid, "reference"),
                              "candidate_task_id": task_id("latency", mid, variant), "exact_token_parity_required": True})
                results.append(tid)
        result_archive = task_id("archive", mid, "results")
        tasks.append({"id": result_archive, "kind": "archive", "model_id": mid,
                      "dependencies": results, "archive_scope": "results", "result_task_ids": results,
                      "full_get_sha256_required": True})
        for tid in train_ids:
            stage = tid.rsplit("__", 1)[1]
            tasks.append({"id": task_id("cleanup", mid, stage), "kind": "cleanup", "model_id": mid,
                          "dependencies": [task_id("archive", mid, stage), result_archive],
                          "checkpoint_task_id": tid, "checkpoint_only": True,
                          "keep_small_evidence": True, "keep_failed_or_unconsumed_checkpoint": True,
                          "fresh_local_and_oss_sha256_required": True})
    # Add every true checkpoint consumer, including the shared R4 initializer.
    for task in tasks:
        if task["kind"] == "cleanup":
            consumers = _checkpoint_consumers(tasks, task["checkpoint_task_id"])
            task["required_consumers"] = consumers
            task["dependencies"] = list(dict.fromkeys(task["dependencies"] + consumers))
    tasks.append({"id": "aggregate_results", "kind": "aggregate", "model_id": None,
                  "dependencies": [task_id("archive", model["id"], "results") for model in models],
                  "benchmarks": list(BENCHMARKS), "training_seeds": [17, 29, 43],
                  "source_cluster_bootstrap": True, "seed_variance_separate_from_source_bootstrap": True,
                  "paired_inputs_and_coverage_required": True, "no_cross_benchmark_combined_accuracy": True})
    return {"schema_version": SCHEMA_VERSION, "suite_id": SUITE_ID, "allocation": deepcopy(ALLOCATION),
            "training_data": deepcopy(TRAINING_DATA), "models": models, "tasks": tasks,
            "selection": {"checkpoint": "final_step", "heldout_tuning": False, "automatic_skip_from_plan": False},
            "failure_policy": "stop_and_preserve_evidence_resume_only_after_validated_recovery",
            "scope": {"scientific_model_configs": 18, "core_model_seed_configs": 12,
                      "fixed_seed17_ablation_configs": 6, "training_stages": 34,
                      "model_benchmark_evaluations": 36, "latency_runs": 15, "latency_comparisons": 12,
                      "historical_training_acceptance_candidates": 5, "historical_evaluation_acceptance_candidates": 4,
                      "conditional_work_not_claimed_complete": ["mvbench_licensed_media_and_scope", "h20_matched_hardware",
                          "alignment_equal_total_compute", "recurrence_equal_compute", "selective_refresh_budget_kernel_codesign"]}}


def _checkpoint_consumers(task_list, checkpoint_task_id):
    return [task["id"] for task in task_list if task["kind"] in {"train", "evaluate", "latency"} and
            (task.get("checkpoint_task_id") == checkpoint_task_id or
             task.get("initializer", {}).get("checkpoint_task_id") == checkpoint_task_id)]


def checkpoint_consumers(plan, checkpoint_task_id):
    return _checkpoint_consumers(plan["tasks"], checkpoint_task_id)


def tasks(plan):
    return deepcopy(validate_plan(plan)["tasks"])


def _relative_config(value):
    require(isinstance(value, str) and value and "\\" not in value, "Invalid config path")
    p = PurePosixPath(value)
    require(not p.is_absolute() and ".." not in p.parts and p.as_posix() == value
            and value.startswith("configs/train/suite/") and p.suffix == ".json", "Unsafe suite config path")
    return value


def _validate_recipe(raw, model, stage, entry):
    require(isinstance(raw, dict) and set(raw) <= {"model", "training", "experiment_protocol"}, "Unexpected training config sections")
    m, t = raw.get("model", {}), raw.get("training", {})
    language = "HuggingFaceTB/SmolLM2-1.7B-Instruct" if model["family"] == "smollm2" else "ByteDance/Ouro-1.4B"
    expected_model = {"language_model": language, "language_revision": declared_training.LANGUAGE_REVISIONS[language],
                      "language_backend": "causal_lm" if language.startswith("HuggingFaceTB/") else "ouro",
                      "vision_model": declared_training.VISION_ID, "vision_revision": declared_training.VISION_REVISION,
                      "num_frames": model["frames"], "num_loops": model["loops"], "pool_grid": model["pool_grid"],
                      "max_text_tokens": 512, "attn_implementation": "sdpa",
                      "freeze_vision": stage == "alignment" or model["freeze_vision_sft"]}
    require(canonical(m) == canonical(expected_model), "Recipe model/seed/shape/initialization differs")
    expected_training = {"stage": stage, "formal_training": True, "seed": model["seed"],
                         "learning_rate": .001 if stage == "alignment" else .00001,
                         "per_device_train_batch_size": 1, "per_device_eval_batch_size": 1,
                         "gradient_accumulation_steps": 16, "num_train_epochs": 1, "max_steps": -1,
                         "bf16": True, "gradient_checkpointing": True, "warmup_ratio": .03,
                         "weight_decay": 0.0 if stage == "alignment" else .01, "lr_scheduler_type": "cosine",
                         "save_steps": 50, "logging_steps": 10,
                         "save_total_limit": 2 if stage == "alignment" and model["family"] in ("ouro_r1", "ouro_r2") else 1,
                         "eval_strategy": "epoch", "dataloader_num_workers": 0, "optim": "adamw_torch", "report_to": "none"}
    require(canonical(t) == canonical(expected_training), "Recipe training schedule/precision/batch changed")
    declared = "experiment_protocol" in raw
    require(entry == ("deepprobe_vlm.train_declared" if declared else "deepprobe_vlm.train"), "Wrong training entry for recipe")
    if declared:
        declared_training.validate_declaration(raw)
    else:
        require(model["seed"] == 17 and model["family"].startswith("ouro_"), "Nondeclared recipe outside frozen seed17 Ouro scope")


def validate_plan(plan, config_root=None):
    """Validate scientific coverage, DAG, initialization and checkpoint lifetime.

    ``config_root`` additionally checks every exact recipe byte and its semantics.
    Without it, this is schema/plan validation only, not GPU or data admission.
    """
    require(isinstance(plan, dict) and set(plan) == {"schema_version", "suite_id", "allocation", "training_data", "models", "tasks", "selection", "failure_policy", "scope"}, "Unexpected suite plan fields")
    require(type(plan["schema_version"]) is int and plan["schema_version"] == SCHEMA_VERSION and plan["suite_id"] == SUITE_ID, "Unknown suite schema")
    require(canonical(plan["allocation"]) == canonical(ALLOCATION), "Suite requires stream/low/single-node/eight-GPU/Never serialization")
    require(canonical(plan["training_data"]) == canonical(TRAINING_DATA), "Fixed training data or budget changed")
    require(canonical(plan["models"]) == canonical(model_specs()), "Complete unique model/seed/ablation matrix required")
    require(plan["selection"] == {"checkpoint": "final_step", "heldout_tuning": False, "automatic_skip_from_plan": False}, "Invalid selection or implicit skip policy")
    require(plan["failure_policy"] == "stop_and_preserve_evidence_resume_only_after_validated_recovery", "Unsafe failure policy")
    task_list = plan["tasks"]
    require(isinstance(task_list, list) and task_list, "Missing task DAG")
    by_id = {}
    for task in task_list:
        require(isinstance(task, dict) and isinstance(task.get("id"), str) and re.fullmatch(r"[a-z][a-z0-9_]*", task["id"]) and task["id"] not in by_id, "Invalid/duplicate task ID")
        require(task.get("kind") in KINDS and isinstance(task.get("dependencies"), list)
                and all(isinstance(x, str) for x in task["dependencies"])
                and len(set(task["dependencies"])) == len(task["dependencies"]), "Invalid task kind or dependencies")
        by_id[task["id"]] = task
    require(all(dep in by_id and dep != task["id"] for task in task_list for dep in task["dependencies"]), "Missing/self dependency")
    resolved = set()
    while len(resolved) < len(task_list):
        ready = {task["id"] for task in task_list if task["id"] not in resolved and set(task["dependencies"]) <= resolved}
        require(bool(ready), "Task DAG contains a cycle")
        resolved.update(ready)
    # Compare the complete operational intent with a reconstructed declaration;
    # config byte pins are validated separately, allowing a portable package root.
    expected = build_plan(config_root or EXPERIMENT_ROOT)
    require(plan["scope"] == expected["scope"], "Suite completion denominator or conditional scope changed")
    require(len(by_id) == len(expected["tasks"]), "Missing or additional suite task")
    models = {m["id"]: m for m in plan["models"]}
    for expected_task in expected["tasks"]:
        tid = expected_task["id"]
        require(tid in by_id, "Missing declared suite task: " + tid)
        actual = by_id[tid]
        require(canonical(actual) == canonical(expected_task), "Task recipe/dependency/protocol/lifetime differs: " + tid)
        if actual["kind"] == "train":
            pin = actual["config"]
            _relative_config(pin["path"])
            require(type(pin["bytes"]) is int and pin["bytes"] > 0 and isinstance(pin["sha256"], str)
                    and re.fullmatch("[0-9a-f]{64}", pin["sha256"]), "Invalid config pin")
            if config_root is not None:
                root = Path(config_root).resolve(strict=True)
                path = root / pin["path"]
                require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), "Missing or symlink recipe")
                require(path.stat().st_size == pin["bytes"] and file_sha256(path) == pin["sha256"], "Recipe bytes differ")
                _validate_recipe(json.loads(path.read_text()), models[actual["model_id"]], actual["stage"], actual["entry"])
    return plan


def read_plan(path=DEFAULT_PLAN, *, config_root=None):
    return validate_plan(json.loads(Path(path).read_text()), config_root=config_root)


def default_plan():
    return read_plan(DEFAULT_PLAN)


def summarize_plan(plan):
    validate_plan(plan)
    counts = dict(Counter(t["kind"] for t in plan["tasks"]))
    return {"suite_id": plan["suite_id"], "plan_sha256": plan_sha256(plan), "task_count": len(plan["tasks"]),
            "task_counts": counts, "scope": deepcopy(plan["scope"]), "gpu_submitted": False,
            "training_or_evaluation_completed": False, "config_bytes_verified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--config-root", type=Path, default=EXPERIMENT_ROOT)
    args = parser.parse_args(argv)
    plan = read_plan(args.plan, config_root=args.config_root)
    summary = summarize_plan(plan)
    summary["config_bytes_verified"] = True
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
