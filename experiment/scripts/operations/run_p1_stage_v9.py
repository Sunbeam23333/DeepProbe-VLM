#!/usr/bin/env python3
"""Declared v9: independent core seeds and explicit no-alignment SFT.

Forked from the unchanged P1 runner; never deploy over a frozen v8 release.
New declarations require a new exact source manifest and new run directory.

The public entry point is ``run_p1_stage_v9.sh``.  This module is separate so its
JSON/source/config gates can be unit tested without allocating a GPU.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
# Early failure archival has only its minimal import closure copied.
# Its helper must not depend on the as-yet-unverified training package.
if sys.argv[1:2] != ["--archive-helper"]:
    from deepprobe_vlm import declared_training as declared


SHA256 = re.compile(r"[0-9a-f]{64}")
FORMAL_POLICY_ID = "deepprobe-formal-benchmark-source-v1"
FORMAL_TRAINING_GATES = {
    "pinned_annotations", "videomme_fixed_subset", "benchmark_source_identity",
    "training_manifests", "training_candidate_coverage", "training_data_attestation",
    "training_media_integrity", "declared_source_leakage", "artifact_integrity",
}
SOURCE_DIRS = ("src", "scripts", "configs", "requirements")
SOURCE_FILES = ("pyproject.toml", "README.md")
STORAGE_OPERATIONAL_LIMIT_BYTES = 99_000_000_000
FILESYSTEM_FREE_GUARD_BYTES = 1_000_000_000
RUN_WORKING_RESERVE_BYTES = 3_000_000_000
SAFETENSOR_HEADER_ALLOWANCE_BYTES = 1_000_000
SAFETENSOR_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4,
    "U64": 8, "I64": 8, "F64": 8,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object, *, exclusive: bool = True) -> None:
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _safetensor_stats(path: Path, label: str) -> dict:
    """Parse one safetensors header and return its exact key/shape inventory."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required for formal checkpoint verification") from exc
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Empty or unsafe safetensors file: {label}")
    try:
        tensors = {}
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if not isinstance(key, str) or not key:
                    raise ValueError("empty tensor name")
                tensor = handle.get_slice(key)
                shape = list(tensor.get_shape())
                if any(type(item) is not int or item < 0 for item in shape):
                    raise ValueError(f"invalid shape for {key}")
                dtype = str(tensor.get_dtype())
                if dtype not in SAFETENSOR_DTYPE_BYTES:
                    raise ValueError(f"unsupported dtype {dtype} for {key}")
                tensors[key] = {"shape": shape, "numel": math.prod(shape), "dtype": dtype}
    except Exception as exc:
        raise ValueError(f"Invalid safetensors file {label}: {type(exc).__name__}: {exc}") from exc
    if not tensors:
        raise ValueError(f"Safetensors file contains no tensors: {label}")
    payload_bytes = sum(
        item["numel"] * SAFETENSOR_DTYPE_BYTES[item["dtype"]] for item in tensors.values())
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "payload_bytes": payload_bytes,
        "tensor_count": len(tensors),
        "numel": sum(item["numel"] for item in tensors.values()),
        "tensors": tensors,
    }


def _safetensor_collection_stats(paths: list[Path], label: str,
                                 tensor_prefix: str | None = None) -> dict:
    if not paths:
        raise ValueError(f"No safetensors files available for {label}")
    files, owners = [], {}
    for path in sorted(paths):
        item = _safetensor_stats(path, f"{label}/{path.name}")
        if tensor_prefix is not None:
            selected = {key: value for key, value in item["tensors"].items()
                        if key.startswith(tensor_prefix)}
            if not selected:
                continue
            item = dict(item)
            item["tensors"] = selected
            item["tensor_count"] = len(selected)
            item["numel"] = sum(value["numel"] for value in selected.values())
            item["payload_bytes"] = sum(
                value["numel"] * SAFETENSOR_DTYPE_BYTES[value["dtype"]]
                for value in selected.values())
        for key in item["tensors"]:
            if key in owners:
                raise ValueError(
                    f"Duplicate tensor key across {label} files: {key} in "
                    f"{owners[key]} and {path.name}")
            owners[key] = path.name
        files.append(item)
    if not files:
        raise ValueError(f"No matching tensors available for {label}")
    serialized_bytes = (sum(item["bytes"] for item in files)
                        if tensor_prefix is None else
                        sum(item["payload_bytes"] for item in files)
                        + len(files) * SAFETENSOR_HEADER_ALLOWANCE_BYTES)
    total_numel = sum(item["numel"] for item in files)
    # train.py loads backbones in FP32 before saving.  A BF16 upstream snapshot
    # can therefore double on its first native checkpoint; account for that
    # from tensor shapes rather than trusting only current serialized bytes.
    fp32_save_bytes = sum(
        tensor["numel"] * max(4, SAFETENSOR_DTYPE_BYTES[tensor["dtype"]])
        for item in files for tensor in item["tensors"].values()
    ) + len(files) * SAFETENSOR_HEADER_ALLOWANCE_BYTES
    return {
        "files": files,
        "file_count": len(files),
        "tensor_count": len(owners),
        "numel": total_numel,
        "serialized_bytes": serialized_bytes,
        "fp32_save_bytes": fp32_save_bytes,
        "estimated_saved_bytes": max(serialized_bytes, fp32_save_bytes),
        "tensor_prefix": tensor_prefix,
    }


def _hidden_size(config_path: Path, component: str) -> int:
    document = read_json(config_path)
    nested_key = "vision_config" if component == "vision" else "text_config"
    nested = document.get(nested_key)
    candidates = []
    if isinstance(nested, dict):
        candidates.extend(nested.get(key) for key in ("hidden_size", "d_model", "n_embd"))
    candidates.extend(document.get(key) for key in ("hidden_size", "d_model", "n_embd"))
    for value in candidates:
        if type(value) is int and value > 0:
            return value
    raise ValueError(f"Cannot determine {component} hidden size from {config_path}")


def _snapshot_weight_stats(root: Path, label: str) -> dict:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Unsafe or missing offline {label} snapshot: {root}")
    paths = []
    for candidate in sorted(root.rglob("*.safetensors")):
        resolved = candidate.resolve()
        if not resolved.is_relative_to(Path("/home/jinhaopeng")):
            raise ValueError(f"Offline {label} weight escapes personal home: {candidate}")
        paths.append(resolved)
    # The pinned SigLIP repository is a composite text+vision checkpoint, but
    # train.py loads and later saves SiglipVisionModel only.  Counting the
    # unused text tower would reject valid runs without describing a real peak.
    tensor_prefix = "vision_model." if label == "vision" else None
    return _safetensor_collection_stats(paths, f"offline-{label}", tensor_prefix)


def stage_storage_plan(stage: str, save_total_limit: int, model_saved_bytes: int,
                       projector_saved_bytes: int) -> dict:
    """Bound the transient disk peak before Trainer checkpoint rotation."""
    if stage not in {"alignment", "sft"}:
        raise ValueError(f"Unsupported P1 stage for storage planning: {stage}")
    if type(save_total_limit) is not int or save_total_limit < 1:
        raise ValueError("Storage planning requires save_total_limit>=1")
    if type(model_saved_bytes) is not int or model_saved_bytes <= 0:
        raise ValueError("Storage planning requires positive model_saved_bytes")
    if (type(projector_saved_bytes) is not int or projector_saved_bytes <= 0
            or projector_saved_bytes > model_saved_bytes):
        raise ValueError("Storage planning requires a bounded positive projector size")
    checkpoint_bytes = (model_saved_bytes + 2 * projector_saved_bytes
                        if stage == "alignment" else 3 * model_saved_bytes)
    rotation_peak = (save_total_limit + 1) * checkpoint_bytes
    final_model_peak = save_total_limit * checkpoint_bytes + model_saved_bytes
    reserve = max(rotation_peak, final_model_peak) + RUN_WORKING_RESERVE_BYTES
    return {
        "stage": stage,
        "save_total_limit": save_total_limit,
        "model_saved_bytes": model_saved_bytes,
        "projector_saved_bytes": projector_saved_bytes,
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_rotation_peak_bytes": rotation_peak,
        "checkpoint_plus_final_peak_bytes": final_model_peak,
        "run_working_reserve_bytes": RUN_WORKING_RESERVE_BYTES,
        "required_additional_bytes": reserve,
    }


def estimate_stage_storage(runtime_config: dict, stage: str) -> dict:
    model = runtime_config.get("model")
    training = runtime_config.get("training")
    if not isinstance(model, dict) or not isinstance(training, dict):
        raise ValueError("Storage planning requires model and training config objects")
    language_root = Path(model.get("language_model", ""))
    vision_root = Path(model.get("vision_model", ""))
    language = _snapshot_weight_stats(language_root, "language")
    vision = _snapshot_weight_stats(vision_root, "vision")
    language_dim = _hidden_size(language_root / "config.json", "language")
    vision_dim = _hidden_size(vision_root / "config.json", "vision")
    projector_numel = (vision_dim + 1 + language_dim * vision_dim + language_dim
                       + language_dim * language_dim + language_dim)
    projector_saved_bytes = projector_numel * 4 + SAFETENSOR_HEADER_ALLOWANCE_BYTES
    model_saved_bytes = (language["estimated_saved_bytes"]
                         + vision["estimated_saved_bytes"] + projector_saved_bytes)
    plan = stage_storage_plan(
        stage, training.get("save_total_limit", 2), model_saved_bytes, projector_saved_bytes)
    plan["components"] = {
        "language": language,
        "vision": vision,
        "projector": {
            "language_hidden_size": language_dim,
            "vision_hidden_size": vision_dim,
            "numel": projector_numel,
            "estimated_saved_bytes": projector_saved_bytes,
        },
    }
    return plan


def require_storage_admission(current_project_bytes: int, filesystem_free_bytes: int,
                              plan: dict) -> dict:
    reserve = plan.get("required_additional_bytes")
    if (type(current_project_bytes) is not int or current_project_bytes < 0
            or type(filesystem_free_bytes) is not int or filesystem_free_bytes < 0
            or type(reserve) is not int or reserve <= 0):
        raise ValueError("Storage admission requires nonnegative byte counts and a positive reserve")
    projected = current_project_bytes + reserve
    required_free = reserve + FILESYSTEM_FREE_GUARD_BYTES
    if projected >= STORAGE_OPERATIONAL_LIMIT_BYTES:
        raise RuntimeError(
            "P1 storage admission rejected: projected project bytes "
            f"{projected} reach the {STORAGE_OPERATIONAL_LIMIT_BYTES} operational limit")
    if filesystem_free_bytes < required_free:
        raise RuntimeError(
            "P1 storage admission rejected: filesystem free bytes "
            f"{filesystem_free_bytes} are below required {required_free}")
    return {
        "status": "admitted",
        "current_project_bytes": current_project_bytes,
        "filesystem_free_bytes": filesystem_free_bytes,
        "operational_limit_bytes": STORAGE_OPERATIONAL_LIMIT_BYTES,
        "filesystem_free_guard_bytes": FILESYSTEM_FREE_GUARD_BYTES,
        "projected_project_bytes": projected,
        "required_filesystem_free_bytes": required_free,
        "plan": plan,
    }


def source_inventory(root: Path) -> dict[str, str]:
    """Return the exact executable experiment inventory; reject symlinks."""
    files: list[Path] = []
    for name in SOURCE_DIRS:
        directory = root / name
        if not directory.is_dir():
            raise FileNotFoundError(f"Frozen source directory missing: {directory}")
        for path in sorted(directory.rglob("*")):
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path.is_symlink():
                raise ValueError(f"Frozen source rejects symlink: {path}")
            if path.is_file():
                files.append(path)
    for name in SOURCE_FILES:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Frozen source file missing: {path}")
        files.append(path)
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(set(files))}


def verify_source_manifest(root: Path, manifest_path: Path) -> dict:
    if manifest_path.resolve().is_relative_to(root.resolve()):
        raise ValueError("SOURCE_MANIFEST_JSON must be outside the frozen experiment tree")
    document = read_json(manifest_path)
    expected = document.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Frozen source manifest requires a nonempty files object")
    for relative, digest in expected.items():
        if (not isinstance(relative, str) or Path(relative).is_absolute()
                or ".." in Path(relative).parts or not isinstance(digest, str)
                or not SHA256.fullmatch(digest)):
            raise ValueError("Frozen source manifest contains an invalid entry")
    observed = source_inventory(root)
    if expected != observed:
        missing = sorted(set(observed) - set(expected))
        extra = sorted(set(expected) - set(observed))
        changed = sorted(name for name in set(expected) & set(observed)
                         if expected[name] != observed[name])
        raise ValueError(
            f"Frozen source manifest mismatch: missing={missing[:5]}, extra={extra[:5]}, "
            f"changed={changed[:5]}")
    required = {
        "scripts/operations/run_p1_stage_v9.sh",
        "scripts/operations/run_p1_stage_v9.py",
        "scripts/preflight.py",
        "scripts/operations/stage_assets.py",
        "src/deepprobe_vlm/train.py",
        "src/deepprobe_vlm/train_declared.py",
        "src/deepprobe_vlm/declared_training.py",
        "src/deepprobe_vlm/formal_integrity.py",
    }
    absent = sorted(required - set(observed))
    if absent:
        raise ValueError(f"Frozen source manifest omits required P1 files: {absent}")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "file_count": len(observed),
        "files": observed,
    }


def snapshot_source(root: Path, target: Path, identity: dict) -> None:
    target.mkdir(parents=True, exist_ok=False)
    for relative, expected in identity["files"].items():
        source = root / relative
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha256_file(destination) != expected:
            raise RuntimeError(f"Source changed while being snapshotted: {relative}")


def _safe_artifact(parent: Path, name: str) -> Path:
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("FORMAL_READY artifact names must be plain filenames")
    path = parent / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Formal gate artifact missing or unsafe: {name}")
    return path


def verify_attested_training_media(inventory_path: Path, media_root: Path) -> dict:
    """Rehash every selected media file against the preparation inventory."""
    inventory = read_json(inventory_path)
    entries = inventory.get("entries")
    if (inventory.get("schema_version") != 1
            or inventory.get("kind") != "formal_training_media_sha256_inventory"
            or not isinstance(entries, list) or not entries):
        raise ValueError("Formal training media inventory schema/entries are invalid")
    root = media_root.resolve()
    if inventory.get("media_root") != str(root):
        raise ValueError("Training media inventory was prepared against a different MEDIA_ROOT")
    actual = []
    seen = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Training media inventory entry {index} is not an object")
        relative = entry.get("relative_path")
        if (not isinstance(relative, str) or not relative.strip() or "\\" in relative
                or "\x00" in relative or "://" in relative or relative.startswith("~")):
            raise ValueError(f"Unsafe training media path: {relative!r}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts or str(relative_path) == ".":
            raise ValueError(f"Unsafe training media path: {relative!r}")
        if relative in seen:
            raise ValueError(f"Duplicate training media path: {relative}")
        seen.add(relative)
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Training media missing or escapes MEDIA_ROOT: {relative}")
        size = path.stat().st_size
        digest = sha256_file(path)
        observed = {"relative_path": relative,
                    "source_video_id": entry.get("source_video_id"),
                    "size_bytes": size, "sha256": digest}
        if observed != entry:
            raise ValueError(f"Training media size/SHA-256 differs from attestation: {relative}")
        actual.append(observed)
    total = sum(item["size_bytes"] for item in actual)
    entries_sha = canonical_sha256(actual)
    if (inventory.get("file_count") != len(actual)
            or inventory.get("total_bytes") != total
            or inventory.get("entries_sha256") != entries_sha):
        raise ValueError("Training media inventory counts/content digest mismatch")
    return {"inventory_sha256": sha256_file(inventory_path),
            "entries_sha256": entries_sha, "file_count": len(actual),
            "total_bytes": total}


def verify_formal_inputs(data_root: Path, ready_path: Path, media_root: Path) -> dict:
    """Bind all formal receipts to exact train/validation/coverage/report bytes."""
    names = {name: data_root / name for name in (
        "train.jsonl", "validation.jsonl", "coverage.jsonl",
        "READY.json", "preparation_report.json", "training_media_inventory.json")}
    for name, path in names.items():
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"Formal training-data artifact missing or unsafe: {name}")
    hashes = {name: sha256_file(path) for name, path in names.items()}
    media_identity = verify_attested_training_media(
        names["training_media_inventory.json"], media_root)

    ready = read_json(ready_path)
    if (ready.get("schema_version") != 1 or ready.get("policy_id") != FORMAL_POLICY_ID
            or ready.get("formal_training_ready") is not True or ready.get("status") not in {
            "ready_for_training", "ready_for_evaluation"}):
        raise ValueError("FORMAL_READY does not authorize formal training")
    if ready.get("benchmark_source_identity_gate") != "passed_on_declared_subset":
        raise ValueError("Benchmark source identity gate did not pass on the declared subset")
    gates = ready.get("training_readiness_gates")
    if not isinstance(gates, dict) or set(gates) != FORMAL_TRAINING_GATES or any(
            not isinstance(item, dict) or item.get("status") != "passed"
            for item in gates.values()):
        raise ValueError("Every FORMAL_READY training readiness gate must pass")
    if gates.get("training_media_integrity", {}).get("status") != "passed":
        raise ValueError("FORMAL_READY training_media_integrity gate did not pass")

    immutable = ready.get("immutable_input_sha256")
    required_inputs = {
        "training_train": hashes["train.jsonl"],
        "training_validation": hashes["validation.jsonl"],
        "training_coverage": hashes["coverage.jsonl"],
        "training_media_inventory": hashes["training_media_inventory.json"],
    }
    if not isinstance(immutable, dict) or any(immutable.get(k) != v for k, v in required_inputs.items()):
        raise ValueError("FORMAL_READY train/validation/coverage hashes do not match DATA_ROOT")

    artifacts = ready.get("artifact_sha256")
    if not isinstance(artifacts, dict) or "source_gate_report.json" not in artifacts:
        raise ValueError("FORMAL_READY lacks its source-gate report/artifact hash set")
    artifact_identity = {}
    for name, expected in sorted(artifacts.items()):
        if not isinstance(expected, str) or not SHA256.fullmatch(expected):
            raise ValueError(f"Invalid formal artifact hash: {name}")
        path = _safe_artifact(ready_path.parent, name)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Formal artifact changed: {name}")
        artifact_identity[name] = actual

    report = read_json(ready_path.parent / "source_gate_report.json")
    policy = read_json(ready_path.parent / "source_policy.json")
    if policy.get("schema_version") != 1 or policy.get("policy_id") != FORMAL_POLICY_ID:
        raise ValueError("Formal source policy schema/identity differs")
    if (report.get("schema_version") != 1 or report.get("formal_training_ready") is not True
            or report.get("status") not in {"ready_for_training", "ready_for_evaluation"}
            or report.get("benchmark_source_identity_gate") != "passed_on_declared_subset"):
        raise ValueError("source_gate_report does not authorize formal training")
    report_gates = report.get("training_readiness_gates")
    if report_gates != gates:
        raise ValueError("FORMAL_READY/report training gate records differ")
    if report_gates.get("training_media_integrity", {}).get("status") != "passed":
        raise ValueError("source_gate_report training_media_integrity did not pass")
    manifest_hashes = report.get("training_manifests", {}).get("sha256", {})
    if manifest_hashes != {"train": hashes["train.jsonl"],
                           "validation": hashes["validation.jsonl"]}:
        raise ValueError("source_gate_report training manifest hashes differ")
    coverage = report.get("training_candidate_coverage", {})
    if coverage.get("sha256") != hashes["coverage.jsonl"] or coverage.get("valid") is not True:
        raise ValueError("source_gate_report coverage proof differs or is invalid")
    attestation = report.get("training_data_attestation", {})
    if (attestation.get("provided") is not True or attestation.get("valid") is not True
            or attestation.get("report_sha256") != hashes["preparation_report.json"]
            or attestation.get("ready_sha256") != hashes["READY.json"]):
        raise ValueError("Formal training-data attestation differs from DATA_ROOT")
    attested_media = attestation.get("media_inventory", {})
    if (attested_media.get("sha256") != hashes["training_media_inventory.json"]
            or attested_media.get("entries_sha256") != media_identity["entries_sha256"]
            or attested_media.get("file_count") != media_identity["file_count"]
            or attested_media.get("total_bytes") != media_identity["total_bytes"]):
        raise ValueError("source_gate_report media inventory attestation differs")
    media_gate = report.get("training_media_attestation", {})
    if (media_gate.get("passed") is not True
            or media_gate.get("entries_sha256") != media_identity["entries_sha256"]
            or media_gate.get("file_count") != media_identity["file_count"]
            or media_gate.get("total_bytes") != media_identity["total_bytes"]):
        raise ValueError("source_gate_report current training media verification differs")
    formal_media = read_json(ready_path.parent / "training_media_inventory.json")
    data_media = read_json(names["training_media_inventory.json"])
    if (formal_media.get("root") != str(media_root.resolve())
            or formal_media.get("complete") is not True
            or formal_media.get("entries") != data_media.get("entries")):
        raise ValueError("Formal source-gate media inventory differs from attested/current media")

    data_ready = read_json(names["READY.json"])
    data_report = read_json(names["preparation_report.json"])
    if (data_ready.get("purpose") != "FORMAL_TRAINING_DATA"
            or data_ready.get("formal_data_ready") is not True
            or data_report.get("status") != "formal_training_data_ready"
            or data_report.get("formal_data_ready") is not True):
        raise ValueError("Training-data READY/report are not formal-ready")
    if data_ready.get("report_sha256") != hashes["preparation_report.json"]:
        raise ValueError("Training-data READY report hash differs")
    expected_outputs = {
        "train.jsonl": hashes["train.jsonl"],
        "validation.jsonl": hashes["validation.jsonl"],
        "coverage.jsonl": hashes["coverage.jsonl"],
        "training_media_inventory.json": hashes["training_media_inventory.json"],
    }
    if any(data_report.get("output_sha256", {}).get(k) != v for k, v in expected_outputs.items()):
        raise ValueError("Training preparation report output hashes differ")
    preparation_media = data_report.get("training_media_inventory", {})
    if (preparation_media.get("relative_path") != "training_media_inventory.json"
            or preparation_media.get("sha256") != hashes["training_media_inventory.json"]
            or preparation_media.get("entries_sha256") != media_identity["entries_sha256"]
            or preparation_media.get("file_count") != media_identity["file_count"]
            or preparation_media.get("total_bytes") != media_identity["total_bytes"]):
        raise ValueError("Training preparation report media inventory binding differs")
    if (data_ready.get("manifest_sha256", {}).get("train.jsonl") != hashes["train.jsonl"]
            or data_ready.get("manifest_sha256", {}).get("validation.jsonl") != hashes["validation.jsonl"]
            or data_ready.get("coverage_sha256") != hashes["coverage.jsonl"]
            or data_ready.get("training_media_inventory_sha256") != hashes["training_media_inventory.json"]
            or data_ready.get("training_media_entries_sha256") != media_identity["entries_sha256"]
            or data_ready.get("training_media_file_count") != media_identity["file_count"]
            or data_ready.get("training_media_total_bytes") != media_identity["total_bytes"]):
        raise ValueError("Training-data READY manifest/coverage hashes differ")
    reported_media = data_report.get("media_root")
    if not isinstance(reported_media, str) or Path(reported_media).resolve() != media_root.resolve():
        raise ValueError("MEDIA_ROOT differs from formal training-data preparation")

    return {
        "formal_ready_sha256": sha256_file(ready_path),
        "formal_status": ready["status"],
        "policy_id": ready.get("policy_id"),
        "data_sha256": hashes,
        "artifact_sha256": artifact_identity,
        "training_media": media_identity,
    }


def validate_recipe(raw: dict, stage: str) -> dict:
    protocol = declared.validate_declaration(raw)
    model = raw.get("model")
    training = raw.get("training")
    if not isinstance(model, dict) or not isinstance(training, dict):
        raise ValueError("Config requires model and training objects")
    if training.get("stage") != stage:
        raise ValueError("Requested stage differs from config training.stage")
    if training.get("formal_training") is not True:
        raise ValueError("P1 config must declare training.formal_training=true")
    if training.get("dataloader_num_workers") != 0:
        raise ValueError("P1 requires dataloader_num_workers=0")
    if type(training.get("seed")) is not int or training["seed"] not in declared.SEEDS:
        raise ValueError("Declared seed must be 17, 29 or 43")
    for name in ("per_device_train_batch_size", "gradient_accumulation_steps"):
        if not isinstance(training.get(name), int) or isinstance(training.get(name), bool) \
                or training[name] <= 0:
            raise ValueError(f"P1 requires positive integer {name}")
    global_batch = training["per_device_train_batch_size"] * training["gradient_accumulation_steps"] * 8
    if global_batch != 128:
        raise ValueError(f"P1 requires effective global batch 128, got {global_batch}")
    epochs = training.get("num_train_epochs")
    max_steps = training.get("max_steps", -1)
    if not ((isinstance(max_steps, int) and max_steps > 0)
            or (isinstance(epochs, (int, float)) and not isinstance(epochs, bool)
                and math.isfinite(epochs) and epochs > 0)):
        raise ValueError("P1 stage must have a finite positive step or epoch bound")
    if training.get("bf16") is not True:
        raise ValueError("P1 recipe requires bf16=true")
    if training.get("fp16", False) is not False:
        raise ValueError("P1 BF16 recipe forbids fp16")
    if training.get("save_only_model", False) is not False:
        raise ValueError("P1 checkpoints must retain optimizer/scheduler/RNG state")
    if training.get("require_final_resume_checkpoint", True) is not True:
        raise ValueError("P1 requires an exact final-step resumable checkpoint")
    if training.get("save_strategy", "steps") != "steps":
        raise ValueError("P1 low-priority recovery requires periodic step checkpoints")
    save_steps = training.get("save_steps")
    if (not isinstance(save_steps, (int, float)) or isinstance(save_steps, bool)
            or not math.isfinite(save_steps) or save_steps <= 0):
        raise ValueError("P1 low-priority recovery requires positive save_steps")
    if training.get("load_best_model_at_end", False) is not False:
        raise ValueError("P1 final-step resume checkpoint forbids load_best_model_at_end")
    save_limit = training.get("save_total_limit", 2)
    if not isinstance(save_limit, int) or isinstance(save_limit, bool) or save_limit < 1:
        raise ValueError("P1 requires save_total_limit>=1 for its final resume checkpoint")
    for key in ("deepspeed", "fsdp"):
        if training.get(key) not in (None, "", []):
            raise ValueError("P1 runner supports ordinary replicated DDP only")
    for key in ("num_loops", "num_frames", "pool_grid"):
        if not isinstance(model.get(key), int) or isinstance(model.get(key), bool) or model[key] <= 0:
            raise ValueError(f"Invalid model.{key}")
    return {"global_batch_size": global_batch, "declared_protocol": protocol,
            "rfg": {key: model[key] for key in ("num_loops", "num_frames", "pool_grid")}}


def _offline_snapshot(hub_cache: Path, identifier: str, revision: str) -> Path:
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"Model revision must be a pinned 40-character commit: {revision!r}")
    candidate = Path(identifier)
    if candidate.is_absolute():
        path = candidate
    else:
        if identifier.count("/") != 1 or any(part in {"", ".", ".."} for part in identifier.split("/")):
            raise ValueError(f"Expected a Hub repo id or absolute offline path: {identifier!r}")
        path = hub_cache / ("models--" + identifier.replace("/", "--")) / "snapshots" / revision
    if not path.is_dir():
        raise FileNotFoundError(f"Pinned offline model snapshot missing: {path}")
    resolved = path.resolve()
    if not resolved.is_relative_to(Path("/home/jinhaopeng")):
        raise ValueError(f"Offline model must stay in the personal home: {resolved}")
    return resolved


def prepare_offline_config(config_path: Path, output_path: Path, hub_cache: Path, stage: str) -> tuple[dict, dict]:
    raw = read_json(config_path)
    recipe = validate_recipe(raw, stage)
    runtime = json.loads(json.dumps(raw))
    inventory = {}
    specs = (("language", "language_model", "language_revision"),
             ("vision", "vision_model", "vision_revision"))
    for label, model_key, revision_key in specs:
        source_id = raw["model"].get(model_key)
        revision = raw["model"].get(revision_key)
        if not isinstance(source_id, str):
            raise ValueError(f"Config model.{model_key} must be a string")
        root = _offline_snapshot(hub_cache, source_id, revision)
        files = {}
        for path in sorted(root.rglob("*")):
            if path.is_symlink() and not path.resolve().is_relative_to(Path("/home/jinhaopeng")):
                raise ValueError(f"Offline snapshot symlink escapes personal home: {path}")
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if "config.json" not in files or not any(name.endswith(".safetensors") for name in files):
            raise ValueError(f"Offline {label} snapshot lacks config/weights")
        if label == "language" and not any(name in files for name in
                                             ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")):
            raise ValueError("Offline language snapshot lacks tokenizer files")
        if label == "vision" and "preprocessor_config.json" not in files:
            raise ValueError("Offline vision snapshot lacks preprocessor_config.json")
        runtime["model"][model_key] = str(root)
        inventory[label] = {"configured_id": source_id, "revision": revision,
                            "resolved_path": str(root), "files": files,
                            "total_bytes": sum(item["bytes"] for item in files.values())}
    write_json(output_path, runtime)
    return recipe, inventory


def _checkpoint_file_inventory(root: Path) -> dict[str, dict]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Checkpoint rejects symlink: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = {
                "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def is_cleanable_weight(relative: Path) -> bool:
    """Limit cleanup to model tensors and the large optimizer tensor state."""
    return relative.suffix == ".safetensors" or relative.name in {
        "pytorch_model.bin", "optimizer.pt"
    }


def should_cleanup_training_payload(relative: Path, stage: str) -> bool:
    if not is_cleanable_weight(relative):
        return False
    # SFT needs the complete alignment final as its weight initializer.  Keep
    # it locally after archive verification; checkpoint optimizer/tensor state
    # may still be removed because it is never used for stage initialization.
    return not (stage == "alignment" and len(relative.parts) >= 3
                and relative.parts[:2] == ("training", "final"))


def _require_model_tree(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Incomplete model checkpoint: unsafe or missing directory {root}")
    required = ("deepprobe_config.json", "projector.safetensors", "training_provenance.json",
                "training_args.bin", "language_model/config.json", "vision_model/config.json",
                "tokenizer/tokenizer_config.json", "image_processor/preprocessor_config.json")
    for relative in required:
        path = root / relative
        if path.is_symlink() or not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Incomplete model checkpoint: missing {relative}")
    for component in ("language_model", "vision_model"):
        directory = root / component
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Incomplete model checkpoint: unsafe or missing {component}")
    projector = _safetensor_stats(root / "projector.safetensors", "projector.safetensors")
    language_dim = _hidden_size(root / "language_model/config.json", "language")
    vision_dim = _hidden_size(root / "vision_model/config.json", "vision")
    projector_contract = {
        "gate.weight": [1, vision_dim],
        "gate.bias": [1],
        "projection.0.weight": [language_dim, vision_dim],
        "projection.0.bias": [language_dim],
        "projection.2.weight": [language_dim, language_dim],
        "projection.2.bias": [language_dim],
    }
    observed_projector = {
        name: {"shape": item["shape"], "dtype": item["dtype"]}
        for name, item in projector["tensors"].items()
    }
    expected_projector = {
        name: {"shape": shape, "dtype": "F32"}
        for name, shape in projector_contract.items()
    }
    if observed_projector != expected_projector:
        raise ValueError("Projector safetensors do not match the FP32 architecture contract")
    for component in ("language_model", "vision_model"):
        directory = root / component
        weights = {path.name: path for path in directory.glob("*.safetensors")}
        index_path = directory / "model.safetensors.index.json"
        if index_path.is_symlink():
            raise ValueError(f"Unsafe sharded weight index: {component}")
        if index_path.exists():
            if not index_path.is_file() or not index_path.stat().st_size:
                raise ValueError(f"Invalid sharded weight index: {component}")
            index = read_json(index_path)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map or any(
                    not isinstance(name, str) or not name
                    or not isinstance(shard, str) or not shard
                    or any(char in shard for char in ("/", "\\", ":", "\x00"))
                    or Path(shard).name != shard
                    or not shard.endswith(".safetensors")
                    for name, shard in weight_map.items()):
                raise ValueError(f"Invalid sharded weight index: {component}")
            metadata = index.get("metadata")
            total_size = metadata.get("total_size") if isinstance(metadata, dict) else None
            if type(total_size) is not int or total_size <= 0:
                raise ValueError(f"Invalid sharded weight index total_size: {component}")
            expected = set(weight_map.values())
            if set(weights) != expected:
                raise ValueError(
                    f"Incomplete sharded checkpoint weights: {component}; "
                    f"expected={sorted(expected)}, observed={sorted(weights)}")
        elif set(weights) != {"model.safetensors"}:
            raise ValueError(
                f"Unsharded checkpoint requires exactly model.safetensors: {component}")
        for name, path in weights.items():
            if path.is_symlink() or not path.is_file() or not path.stat().st_size:
                raise ValueError(f"Empty or unsafe model checkpoint weight shard: {component}/{name}")
        if index_path.exists() and total_size > sum(path.stat().st_size for path in weights.values()):
            raise ValueError(f"Sharded weight index total_size exceeds shard bytes: {component}")
        parsed = {name: _safetensor_stats(path, f"{component}/{name}")
                  for name, path in sorted(weights.items())}
        if index_path.exists():
            observed_total_size = sum(item["payload_bytes"] for item in parsed.values())
            if total_size != observed_total_size:
                raise ValueError(
                    f"Sharded weight index total_size mismatch: {component}; "
                    f"declared={total_size}, observed={observed_total_size}")
            expected_keys = {name: set() for name in weights}
            for tensor, shard in weight_map.items():
                expected_keys[shard].add(tensor)
            for name, item in parsed.items():
                actual = set(item["tensors"])
                if actual != expected_keys[name]:
                    raise ValueError(
                        f"Sharded tensor map mismatch: {component}/{name}; "
                        f"missing={sorted(expected_keys[name] - actual)[:5]}, "
                        f"extra={sorted(actual - expected_keys[name])[:5]}")


def verify_alignment_final(path: Path, runtime_config: dict, formal: dict,
                           source_manifest_sha256: str) -> dict:
    _require_model_tree(path)
    saved = read_json(path / "deepprobe_config.json")
    current = runtime_config["model"]
    for name in ("language_model", "language_revision", "language_backend", "vision_model",
                 "vision_revision", "num_loops", "num_frames", "pool_grid", "max_text_tokens",
                 "attn_implementation"):
        if saved.get(name) != current.get(name):
            raise ValueError(f"SFT alignment initializer mismatch: {name}")
    provenance = read_json(path / "training_provenance.json")
    contract = provenance.get("training_contract")
    if not isinstance(contract, dict) or contract.get("stage") != "alignment":
        raise ValueError("SFT initializer is not an alignment final under train.py contract")
    declared.verify_previous_contract(contract, runtime_config, resume=False)
    declared.verify_alignment_completion(path, contract)
    expected = formal["data_sha256"]
    if (contract.get("formal_training") is not True
            or contract.get("require_final_resume_checkpoint") is not True
            or contract.get("formal_ready_sha256") != formal["formal_ready_sha256"]
            or contract.get("frozen_source_manifest_sha256") != source_manifest_sha256
            or contract.get("train_manifest_sha256") != expected["train.jsonl"]
            or contract.get("eval_manifest_sha256") != expected["validation.jsonl"]
            or contract.get("world_size") != 8 or contract.get("global_batch_size") != 128):
        raise ValueError("Alignment final has a different formal data/DDP/training contract")
    inventory = _checkpoint_file_inventory(path)
    return {"path": str(path.resolve()), "files": inventory,
            "rfg": {key: saved[key] for key in ("num_loops", "num_frames", "pool_grid")}}


def _verify_native_resume_state(path: Path, world_size: int = 8) -> dict:
    match = re.fullmatch(r"checkpoint-([1-9][0-9]*)", path.name)
    if not match:
        raise ValueError("Resume path must be a checkpoint-<positive-step> directory")
    step = int(match.group(1))
    _require_model_tree(path)
    state = read_json(path / "trainer_state.json")
    if (type(state.get("global_step")) is not int or state["global_step"] != step
            or step > declared.TARGET_STEPS or type(state.get("max_steps")) is not int
            or state["max_steps"] != declared.TARGET_STEPS):
        raise ValueError("Resume checkpoint directory/state optimizer step mismatch")
    for name in ("optimizer.pt", "scheduler.pt"):
        item = path / name
        if not item.is_file() or not item.stat().st_size:
            raise ValueError(f"Resume checkpoint is incomplete: {name}")
    expected_rng = {f"rng_state_{rank}.pth" for rank in range(world_size)}
    observed_rng = {item.name for item in path.glob("rng_state*.pth") if item.is_file()}
    if observed_rng != expected_rng or any(not (path / name).stat().st_size for name in expected_rng):
        raise ValueError(
            f"Resume checkpoint requires exact rank0..{world_size - 1} RNG state; "
            f"observed={sorted(observed_rng)}")
    losses: list[dict] = []
    _finite_losses(state, "resume.trainer_state", losses)
    return {"optimizer_step": step, "finite_losses": losses}


def verify_resume_checkpoint(path: Path, runtime_config: dict, runtime_config_sha256: str,
                             formal: dict, source_manifest_sha256: str, stage: str) -> dict:
    """Authenticate an external prior-attempt checkpoint before launching GPUs."""
    state = _verify_native_resume_state(path)
    saved = read_json(path / "deepprobe_config.json")
    if saved != runtime_config.get("model"):
        raise ValueError("Exact resume requires an unchanged model configuration")
    provenance = read_json(path / "training_provenance.json")
    contract = provenance.get("training_contract")
    expected_data = formal["data_sha256"]
    if not isinstance(contract, dict):
        raise ValueError("Resume checkpoint lacks train.py training_contract")
    required = {
        "stage": stage,
        "config_sha256": runtime_config_sha256,
        "frozen_source_manifest_sha256": source_manifest_sha256,
        "train_manifest_sha256": expected_data["train.jsonl"],
        "eval_manifest_sha256": expected_data["validation.jsonl"],
        "formal_training": True,
        "require_final_resume_checkpoint": True,
        "formal_ready_sha256": formal["formal_ready_sha256"],
        "global_batch_size": 128,
        "world_size": 8,
    }
    changed = [name for name, value in required.items() if contract.get(name) != value]
    if changed:
        raise ValueError(f"Resume checkpoint training contract mismatch: {changed}")
    declared.verify_previous_contract(contract, runtime_config, resume=True)
    return {"path": str(path.resolve()), **state,
            "training_contract": contract, "files": _checkpoint_file_inventory(path)}


def verify_preflight(path: Path) -> dict:
    report = read_json(path)
    rows = report.get("rank_reports")
    if (report.get("status") != "gpu_gate_passed" or report.get("world_size") != 8
            or not isinstance(rows, list) or len(rows) != 8
            or {row.get("rank") for row in rows} != set(range(8))
            or any(row.get("attention_forward_backward") != "pass"
                   or row.get("distributed_all_reduce") != "pass" or row.get("errors") for row in rows)):
        raise ValueError("Eight-rank BF16/NCCL preflight is incomplete")
    return report


def _finite_losses(value: object, location: str, seen: list[dict]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{location}.{key}"
            if "loss" in key.lower() and isinstance(item, (int, float)) and not isinstance(item, bool):
                if not math.isfinite(item):
                    raise ValueError(f"Nonfinite loss at {child}")
                seen.append({"field": child, "value": item})
            _finite_losses(item, child, seen)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_losses(item, f"{location}[{index}]", seen)


def verify_training_outputs(training: Path, stage: str, formal: dict,
                            source_manifest_sha256: str, runtime_config_sha256: str,
                            runtime_config: dict) -> dict:
    complete_path = training / "FORMAL_TRAIN_COMPLETE.json"
    if complete_path.is_symlink() or not complete_path.is_file():
        raise ValueError("train.py did not produce its formal media/source completion receipt")
    complete = read_json(complete_path)
    manifest = read_json(training / "run_manifest.json")
    contract = manifest.get("training_contract", {})
    if (manifest.get("world_size") != 8 or manifest.get("stage") != stage
            or contract.get("world_size") != 8 or contract.get("global_batch_size") != 128
            or contract.get("formal_training") is not True
            or contract.get("require_final_resume_checkpoint") is not True
            or contract.get("config_sha256") != runtime_config_sha256
            or contract.get("frozen_source_manifest_sha256") != source_manifest_sha256
            or contract.get("formal_ready_sha256") != formal["formal_ready_sha256"]
            or contract.get("train_manifest_sha256") != formal["data_sha256"]["train.jsonl"]
            or contract.get("eval_manifest_sha256") != formal["data_sha256"]["validation.jsonl"]):
        raise ValueError("train.py run manifest violates the formal eight-rank contract")
    media_before, media_after = complete.get("media_before"), complete.get("media_after")
    if (complete.get("schema_version") != 1
            or complete.get("status") != "formal_training_complete"
            or complete.get("stage") != stage or complete.get("world_size") != 8
            or complete.get("frozen_source_manifest_sha256") != source_manifest_sha256
            or complete.get("formal_ready_sha256") != formal["formal_ready_sha256"]
            or complete.get("train_manifest_sha256") != formal["data_sha256"]["train.jsonl"]
            or complete.get("eval_manifest_sha256") != formal["data_sha256"]["validation.jsonl"]
            or complete.get("run_manifest_sha256") != sha256_file(training / "run_manifest.json")
            or not isinstance(media_before, dict) or media_before != media_after
            or any(media_before.get(key) != formal["training_media"].get(key) for key in (
                "inventory_sha256", "entries_sha256", "file_count", "total_bytes"))
            or not isinstance(media_before.get("metadata_sha256"), str)
            or not SHA256.fullmatch(media_before["metadata_sha256"])):
        raise ValueError("train.py formal completion receipt violates live source/media contract")
    state = read_json(training / "trainer_state.json")
    if (complete.get("trainer_state_sha256") != sha256_file(training / "trainer_state.json")
            or complete.get("optimizer_step") != state.get("global_step")):
        raise ValueError("train.py formal completion receipt differs from trainer state")
    declaration = declared.completion_identity(contract, state)
    if declaration["declared_protocol"] != declared.validate_declaration(runtime_config):
        raise ValueError("Completed method/seed/target differs from requested protocol")
    if any(complete.get(k) != v for k, v in declaration.items()):
        raise ValueError("Formal completion does not bind the declared initialization/target")
    losses: list[dict] = []
    _finite_losses(state, "trainer_state", losses)
    metrics = {}
    for name in ("train_results.json", "eval_results.json"):
        path = training / name
        if not path.is_file():
            raise ValueError(f"Missing metrics: {name}")
        metrics[name] = read_json(path)
        _finite_losses(metrics[name], name, losses)
    if not losses or not any(item["field"].endswith("eval_loss") for item in losses):
        raise ValueError("Finite train/evaluation loss evidence is incomplete")

    final = training / "final"
    _require_model_tree(final)
    if read_json(final / "training_provenance.json").get("training_contract") != contract:
        raise ValueError("Final model lost its declared training contract")
    checkpoints = []
    for checkpoint in sorted(training.glob("checkpoint-*"),
                             key=lambda p: int(p.name.removeprefix("checkpoint-"))):
        step = int(checkpoint.name.removeprefix("checkpoint-"))
        _require_model_tree(checkpoint)
        checkpoint_state = read_json(checkpoint / "trainer_state.json")
        if checkpoint_state.get("global_step") != step:
            raise ValueError(f"Checkpoint step mismatch: {checkpoint.name}")
        _verify_native_resume_state(checkpoint)
        if read_json(checkpoint / "training_provenance.json").get("training_contract") != contract:
            raise ValueError("Resume checkpoint lost its declared training contract")
        checkpoints.append({"name": checkpoint.name, "step": step})
    if not checkpoints:
        raise ValueError("Formal P1 stage produced no complete resumable checkpoint")
    if checkpoints[-1]["step"] != state["global_step"]:
        raise ValueError("Formal P1 stage lacks a complete checkpoint at its exact final optimizer step")
    return {"optimizer_steps": state["global_step"], "finite_losses": losses,
            "metrics": metrics, "checkpoints": checkpoints,
            "formal_completion_sha256": sha256_file(complete_path),
            "final_files": _checkpoint_file_inventory(final)}


def archive_helper(source: Path, out: Path, uri: str, run_id: str, stage_name: str) -> int:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "p1_assets", source / "scripts/operations/stage_assets.py")
    assets = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(assets)
    listing = subprocess.run(["ossutil", "ls", uri + "/"], check=True,
                             capture_output=True, text=True, timeout=300).stdout
    if any(row.split() and row.split()[-1].startswith(uri + "/") for row in listing.splitlines()):
        raise FileExistsError("Run bucket prefix already contains objects; no overwrite/retry")
    files = []
    for path in sorted(out.rglob("*")):
        relative = path.relative_to(out)
        if (relative.parts and relative.parts[0] in {"tmp", "cache"}) or relative.as_posix() in {
                "stdout.log", "ARCHIVE_MANIFEST.json", "ARCHIVED.json"}:
            continue
        if path.is_symlink():
            raise ValueError(f"Archive rejects symlink: {relative}")
        if path.is_file():
            files.append({"path": relative.as_posix(), "bytes": path.stat().st_size,
                          "sha256": sha256_file(path)})
    manifest = {"run_id": run_id, "stage": stage_name, "uri": uri, "files": files,
                "file_count": len(files), "total_bytes": sum(item["bytes"] for item in files),
                "verification": "OSS cp CRC + exact-key length; local SHA-256 inventory",
                "excluded": ["active stdout.log (stdout.snapshot.log archived)", "tmp/", "cache/"]}
    manifest_path = out / "ARCHIVE_MANIFEST.json"
    write_json(manifest_path, manifest)
    for item in files:
        assets.upload_file(out / item["path"], uri + "/" + item["path"])
        if ((out / item["path"]).stat().st_size != item["bytes"]
                or sha256_file(out / item["path"]) != item["sha256"]):
            raise RuntimeError(f"Payload changed during archive: {item['path']}")
    assets.upload_file(manifest_path, uri + "/ARCHIVE_MANIFEST.json")
    for name in ("ARCHIVE_MANIFEST.json", "P1_STAGE_COMPLETE.json"):
        if not (out / name).is_file():
            continue
        readback = out / "tmp" / ("readback-" + name)
        subprocess.run(["ossutil", "cp", uri + "/" + name, str(readback), "-f", "--no-progress"],
                       check=True, timeout=1800)
        if sha256_file(readback) != sha256_file(out / name):
            raise RuntimeError(f"Archive readback differs: {name}")
    receipt = {"status": "archive_verified", "run_id": run_id, "stage": stage_name,
               "uri": uri, "archive_manifest_sha256": sha256_file(manifest_path),
               "file_count": len(files), "total_bytes": manifest["total_bytes"],
               "remote_manifest_and_completion_sha256_readback": True}
    pending = out / "ARCHIVED.json.pending"
    write_json(pending, receipt)
    assets.upload_file(pending, uri + "/ARCHIVED.json")
    receipt_readback = out / "tmp/readback-ARCHIVED.json"
    subprocess.run(["ossutil", "cp", uri + "/ARCHIVED.json", str(receipt_readback),
                    "-f", "--no-progress"], check=True, timeout=300)
    if sha256_file(receipt_readback) != sha256_file(pending):
        raise RuntimeError("Archive completion receipt readback differs")
    pending.replace(out / "ARCHIVED.json")
    return 0


def ensure_failure_archive_dependencies(experiment: Path, source: Path) -> None:
    """Complete the minimal import closure even if snapshotting stopped midway."""
    required = ("scripts/operations/run_p1_stage_v9.py", "scripts/operations/stage_assets.py",
                "scripts/data/download_hf.py", "src/deepprobe_vlm/__init__.py",
                "src/deepprobe_vlm/manifest.py")
    for relative in required:
        target = source / relative
        if target.is_symlink():
            raise ValueError("Unsafe failure-archive dependency")
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(experiment / relative, target)


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        declared.initialization_mode(read_json(args.config), init_checkpoint=args.alignment_final,
                                     resume=args.resume_checkpoint, no_alignment=args.no_alignment)
        self.experiment = args.experiment.resolve()
        self.base = args.base.resolve()
        self.data = args.data_root.resolve()
        self.media = args.media_root.resolve()
        self.config = args.config.resolve()
        self.formal_ready = args.formal_ready.resolve()
        self.source_manifest = args.source_manifest.resolve()
        self.alignment_final = args.alignment_final.resolve() if args.alignment_final else None
        self.resume_checkpoint = args.resume_checkpoint.resolve() if args.resume_checkpoint else None
        self.out = self.base / "runs" / args.run_id
        self.uri = "oss://ss-devbox-stream/jinhaopeng/Paper/DeepProbe-VLM/runs/" + args.run_id
        if self.out.exists() or self.out.is_symlink():
            raise FileExistsError("Run directory already exists; never overwrite/retry automatically")
        for path in (self.experiment, self.base, self.data, self.media, self.config,
                     self.formal_ready, self.source_manifest,
                     *([self.alignment_final] if self.alignment_final else []),
                     *([self.resume_checkpoint] if self.resume_checkpoint else [])):
            if not path.is_relative_to(Path("/home/jinhaopeng")):
                raise ValueError(f"All P1 paths must stay in the personal home: {path}")
        self.out.mkdir(parents=True, exist_ok=False)
        for name in ("tmp", "cache", "inputs"):
            (self.out / name).mkdir()
        self.source = self.out / "source"
        self.training = self.out / "training"
        self.log = (self.out / "stdout.log").open("x", encoding="utf-8")
        self.log_lock = threading.Lock()
        self.active: subprocess.Popen | None = None
        self.stages: list[dict] = []
        os.environ.update({
            "TMPDIR": str(self.out / "tmp"), "TMP": str(self.out / "tmp"),
            "TEMP": str(self.out / "tmp"), "XDG_CACHE_HOME": str(self.out / "cache"),
            "CUDA_CACHE_PATH": str(self.out / "cache/cuda"),
            "TRITON_CACHE_DIR": str(self.out / "cache/triton"),
            "TORCH_HOME": str(self.out / "cache/torch"),
            "TORCH_EXTENSIONS_DIR": str(self.out / "cache/extensions"),
            "TORCHINDUCTOR_CACHE_DIR": str(self.out / "cache/inductor"),
        })
        for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK"):
            os.environ.pop(key, None)

    def emit(self, value: object) -> None:
        text = str(value).rstrip("\n")
        with self.log_lock:
            self.log.write(text + "\n")
            self.log.flush()
            print(text, flush=True)

    def cache_bytes(self) -> int:
        return sum(declared.staging_bytes([self.base, Path(str(self.base) + "-p1")]).values())

    def stop(self) -> None:
        process = self.active
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=15)

    def run_stage(self, name: str, command: list[object], timeout: int,
                  *, monitor_storage: bool = True) -> None:
        env = dict(os.environ, PYTHONPATH=str(self.source / "src"))
        record = {"name": name, "command": [str(item) for item in command],
                  "timeout_seconds": timeout, "started_utc": datetime.now(timezone.utc).isoformat()}
        self.stages.append(record)
        self.emit(f"START {name} (timeout {timeout}s)")
        started = time.monotonic()
        self.active = subprocess.Popen(record["command"], cwd=self.source, env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, errors="replace", start_new_session=True)
        process = self.active

        def pump() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                self.emit(line)

        thread = threading.Thread(target=pump, daemon=True)
        thread.start()
        next_storage = started + 60
        try:
            while process.poll() is None:
                now = time.monotonic()
                if now - started > timeout:
                    raise TimeoutError(f"{name} exceeded its finite time budget")
                if monitor_storage and now >= next_storage:
                    if self.cache_bytes() >= STORAGE_OPERATIONAL_LIMIT_BYTES:
                        raise RuntimeError("Project cache reached the 99 GB guard")
                    if shutil.disk_usage(self.base).free < FILESYSTEM_FREE_GUARD_BYTES:
                        raise RuntimeError("Filesystem free space fell below the 1 GB guard")
                    next_storage = now + 60
                time.sleep(1)
            if process.returncode:
                raise RuntimeError(f"{name} exited with {process.returncode}")
        finally:
            self.stop()
            thread.join(timeout=30)
            record["returncode"] = process.returncode
            record["elapsed_seconds"] = time.monotonic() - started
            self.active = None
            self.emit(f"END {name}: code={process.returncode}, elapsed={record['elapsed_seconds']:.1f}s")

    def launch_prefix(self, offset: int) -> list[object]:
        port = 29000 + int(hashlib.sha256(self.args.run_id.encode()).hexdigest()[:4], 16) % 5000 + offset
        return [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--node-rank=0",
                "--nproc-per-node=8", "--master-addr=127.0.0.1", f"--master-port={port}"]

    def cleanup_weights(self) -> None:
        archived = read_json(self.out / "ARCHIVED.json")
        manifest_path = self.out / "ARCHIVE_MANIFEST.json"
        if (archived.get("status") != "archive_verified"
                or archived.get("remote_manifest_and_completion_sha256_readback") is not True
                or sha256_file(manifest_path) != archived.get("archive_manifest_sha256")):
            raise ValueError("Archive/readback receipt is not valid; retaining all weights")
        removed, retained = [], []
        for item in read_json(manifest_path)["files"]:
            relative = Path(item["path"])
            if not relative.parts or relative.parts[0] != "training" or not is_cleanable_weight(relative):
                continue
            if not should_cleanup_training_payload(relative, self.args.stage):
                retained.append(item)
                continue
            path = self.out / relative
            if (path.is_symlink() or not path.resolve().is_relative_to(self.training.resolve())
                    or path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]):
                raise ValueError(f"Archived weight changed; refusing cleanup: {relative}")
            path.unlink()
            removed.append(item)
        write_json(self.out / "CLEANED_WEIGHTS.json",
                   {"status": "only_verified_archived_weights_removed", "files": removed,
                    "retained_archived_alignment_final": retained,
                    "retention_reason": ("complete alignment final retained locally for matching SFT initialization"
                                         if self.args.stage == "alignment" else None),
                    "archive_manifest_sha256": archived["archive_manifest_sha256"]})

    def execute(self) -> int:
        identity = verify_source_manifest(self.experiment, self.source_manifest)
        snapshot_source(self.experiment, self.source, identity)
        write_json(self.out / "source_manifest.json", identity)
        inputs = self.out / "inputs"
        data_inputs, formal_inputs = inputs / "formal_training_data", inputs / "formal_gate"
        data_inputs.mkdir()
        formal_inputs.mkdir()
        formal = verify_formal_inputs(self.data, self.formal_ready, self.media)
        for name in ("train.jsonl", "validation.jsonl", "coverage.jsonl", "READY.json",
                     "preparation_report.json", "training_media_inventory.json"):
            shutil.copyfile(self.data / name, data_inputs / name)
        for name in formal["artifact_sha256"]:
            shutil.copyfile(self.formal_ready.parent / name, formal_inputs / name)
        shutil.copyfile(self.formal_ready, formal_inputs / "FORMAL_READY.json")
        shutil.copyfile(self.config, inputs / "config.requested.json")
        shutil.copyfile(self.source_manifest, inputs / "frozen_source_manifest.json")
        recipe, offline = prepare_offline_config(
            inputs / "config.requested.json", inputs / "config.runtime.json",
            Path(os.environ["HF_HUB_CACHE"]), self.args.stage)
        write_json(inputs / "offline_models.json", offline)
        runtime = read_json(inputs / "config.runtime.json")
        alignment = None
        resume = None
        runtime_sha = sha256_file(inputs / "config.runtime.json")
        if self.alignment_final is not None:
            alignment = verify_alignment_final(
                self.alignment_final, runtime, formal, identity["manifest_sha256"])
            write_json(inputs / "alignment_initializer.json", alignment)
        if self.resume_checkpoint is not None:
            resume = verify_resume_checkpoint(
                self.resume_checkpoint, runtime, runtime_sha, formal,
                identity["manifest_sha256"], self.args.stage)
            write_json(inputs / "resume_initializer.json", resume)

        storage_plan = estimate_stage_storage(runtime, self.args.stage)
        storage = require_storage_admission(
            self.cache_bytes(), shutil.disk_usage(self.base).free, storage_plan)
        storage["checked_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(inputs / "storage_admission.json", storage)
        self.run_stage("full-node-inventory", [sys.executable, "-c",
                       "import torch; assert torch.cuda.device_count()==8, 'Expected exactly eight visible GPUs'"], 120)
        before_launch = self.launch_prefix(0)
        train_launch = self.launch_prefix(1)
        after_launch = self.launch_prefix(2)
        self.run_stage("preflight-before", before_launch + [self.source / "scripts/preflight.py",
                       "--expected-gpu", "B300", "--output", self.out / "preflight-before.json"], 600)
        verify_preflight(self.out / "preflight-before.json")
        command = train_launch + ["-m", "deepprobe_vlm.train_declared",
                            "--config", inputs / "config.runtime.json",
                            "--train-manifest", data_inputs / "train.jsonl",
                            "--eval-manifest", data_inputs / "validation.jsonl",
                            "--media-root", self.media,
                            "--formal-ready", formal_inputs / "FORMAL_READY.json",
                            "--formal-training-data-dir", data_inputs,
                            "--frozen-source-manifest", inputs / "frozen_source_manifest.json",
                            "--frozen-source-manifest-sha256", identity["manifest_sha256"],
                            "--output-dir", self.training]
        if self.args.no_alignment:
            command += ["--no-alignment"]
        if self.alignment_final is not None:
            command += ["--init-checkpoint", self.alignment_final]
        if self.resume_checkpoint is not None:
            command += ["--resume", self.resume_checkpoint]
        # A formal epoch is much larger than P0.  The stage remains bounded,
        # while allowing two wall-clock days before it is stopped and archived.
        self.run_stage("formal-train-" + self.args.stage, command, 172_800)
        outputs = verify_training_outputs(
            self.training, self.args.stage, formal, identity["manifest_sha256"], runtime_sha, runtime)
        log_text = (self.out / "stdout.log").read_text(encoding="utf-8", errors="replace")
        if re.search(r"cuda\s+out\s+of\s+memory|outofmemoryerror|\bnan\b", log_text, re.IGNORECASE):
            raise ValueError("Training log contains an OOM or NaN marker")
        self.run_stage("preflight-after", after_launch + [self.source / "scripts/preflight.py",
                       "--expected-gpu", "B300", "--output", self.out / "preflight-after.json"], 600)
        verify_preflight(self.out / "preflight-after.json")
        if verify_formal_inputs(self.data, self.formal_ready, self.media) != formal:
            raise ValueError("Formal inputs changed during the training attempt")
        if source_inventory(self.source) != identity["files"]:
            raise ValueError("Frozen source snapshot changed during the training attempt")
        if alignment is not None and _checkpoint_file_inventory(self.alignment_final) != alignment["files"]:
            raise ValueError("Alignment initializer changed during SFT")
        if resume is not None and _checkpoint_file_inventory(self.resume_checkpoint) != resume["files"]:
            raise ValueError("Resume checkpoint changed during the new attempt")
        write_json(self.out / "P1_STAGE_COMPLETE.json", {
            "status": "p1_formal_stage_complete", "formal_training_result": True,
            "held_out_benchmark_result": False,
            "run_id": self.args.run_id, "stage": self.args.stage,
            "world_size": 8, "effective_global_batch": recipe["global_batch_size"],
            "rfg": recipe["rfg"], "seed": runtime["training"]["seed"], "dataloader_num_workers": 0,
            "runner_version": "declared-v9",
            **declared.completion_identity(
                read_json(self.training / "run_manifest.json")["training_contract"],
                read_json(self.training / "trainer_state.json")),
            "source_identity": {k: identity[k] for k in ("manifest_sha256", "file_count")},
            "formal_identity": formal, "offline_models": offline,
            "requested_config_sha256": sha256_file(inputs / "config.requested.json"),
            "runtime_config_sha256": runtime_sha,
            "storage_admission_sha256": sha256_file(inputs / "storage_admission.json"),
            "storage_admission": storage,
            "alignment_initializer": alignment, "resume_initializer": resume,
            "local_weight_retention": ("complete alignment final retained after archive for matching SFT"
                                       if self.args.stage == "alignment"
                                       else "successful SFT weights may be removed after verified OSS readback"),
            "training_result": outputs,
            "preflight_before_sha256": sha256_file(self.out / "preflight-before.json"),
            "preflight_after_sha256": sha256_file(self.out / "preflight-after.json"),
        })
        return 0

    def finalize(self, exit_code: int) -> int:
        self.stop()
        write_json(self.out / "EXIT_STATUS.json", {
            "exit_code": exit_code, "run_id": self.args.run_id, "stage": self.args.stage,
            "stages": self.stages, "automatic_retry": False,
            "signal_limit": "SIGKILL or node disappearance can prevent bounded archival; inspect before manual recovery.",
        })
        try:
            ensure_failure_archive_dependencies(self.experiment, self.source)
            self.log.flush()
            shutil.copyfile(self.out / "stdout.log", self.out / "stdout.snapshot.log")
            self.run_stage("archive", [sys.executable,
                           self.source / "scripts/operations/run_p1_stage_v9.py", "--archive-helper",
                           str(self.source), str(self.out), self.uri, self.args.run_id, self.args.stage],
                           14_400, monitor_storage=False)
            archived = read_json(self.out / "ARCHIVED.json")
            if archived.get("status") != "archive_verified":
                raise ValueError("Archive helper did not produce a verified receipt")
            if exit_code == 0:
                self.cleanup_weights()
        except BaseException:
            self.emit("ARCHIVE/CLEANUP FAILURE: local evidence retained; no automatic retry.\n" + traceback.format_exc())
            exit_code = exit_code or 74
        self.log.close()
        return exit_code


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--stage", choices=("alignment", "sft"), required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--media-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--formal-ready", type=Path, required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--no-alignment", action="store_true", help="Declared R4 seed17 random-connector SFT only")
    p.add_argument("--alignment-final", type=Path)
    p.add_argument("--resume-checkpoint", type=Path)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--write-source-manifest":
        if len(argv) != 3:
            raise ValueError("source manifest writer expects EXPERIMENT OUTPUT")
        root, output = Path(argv[1]).resolve(), Path(argv[2]).resolve()
        if output.is_relative_to(root):
            raise ValueError("Source manifest output must be outside the experiment tree")
        write_json(output, {
            "schema_version": 1,
            "purpose": "DEEPPROBE_FROZEN_P1_SOURCE",
            "files": source_inventory(root),
        })
        print(json.dumps({"output": str(output), "sha256": sha256_file(output)}))
        return 0
    if argv and argv[0] == "--archive-helper":
        if len(argv) != 6:
            raise ValueError("archive helper expects SOURCE OUT URI RUN_ID STAGE")
        return archive_helper(Path(argv[1]), Path(argv[2]), argv[3], argv[4], argv[5])
    args = parser().parse_args(argv)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_id):
        raise ValueError("Invalid run id")
    if args.alignment_final and args.resume_checkpoint:
        raise ValueError("Choose --alignment-final OR --resume-checkpoint")
    if args.stage == "alignment" and args.alignment_final:
        raise ValueError("alignment forbids --alignment-final")
    declared.initialization_mode(read_json(args.config), init_checkpoint=args.alignment_final,
                                 resume=args.resume_checkpoint, no_alignment=args.no_alignment)
    if read_json(args.config)["training"]["stage"] != args.stage:
        raise ValueError("Requested stage differs from declared config")
    runner = Runner(args)
    exit_code = 1

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        exit_code = runner.execute()
    except BaseException as exc:
        runner.emit(traceback.format_exc())
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
    return runner.finalize(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
