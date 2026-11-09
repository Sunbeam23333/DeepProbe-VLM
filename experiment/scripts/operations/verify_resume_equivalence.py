#!/usr/bin/env python3
"""Prove or reject exact checkpoint-resume equivalence.

The comparison is deliberately stricter than comparing serialized file hashes.
It directly compares every model, optimizer, scheduler and per-rank RNG tensor,
including tensor dtype and shape, after a control run and a run resumed from the
control run's intermediate checkpoint.  File SHA-256 values are recorded only
as serialization evidence; they are not an acceptance criterion.

Only compare checkpoints produced by this experiment runner.  PyTorch state
files are deserialized with ``weights_only=False`` because Hugging Face RNG
state contains Python and NumPy values in addition to tensors.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control-checkpoint", type=Path, required=True)
    p.add_argument("--resumed-checkpoint", type=Path, required=True)
    p.add_argument("--fork-checkpoint", type=Path, required=True)
    p.add_argument("--source-manifest", type=Path, required=True,
                   help="Exact executable-source manifest recorded in both training contracts")
    p.add_argument("--control-final", type=Path,
                   help="Optional final/ model copy; must equal the control final checkpoint")
    p.add_argument("--resumed-final", type=Path,
                   help="Optional final/ model copy; must equal the resumed final checkpoint")
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--eval-manifest", type=Path, required=True)
    p.add_argument("--expected-world-size", type=int, required=True)
    p.add_argument("--expected-split-step", type=int, required=True)
    p.add_argument("--expected-final-step", type=int, required=True)
    p.add_argument("--output", type=Path, required=True, help="New JSON report; never overwritten")
    return p


_FILE_DIGEST_CACHE: dict[tuple[str, int, int], str] = {}


def sha256_file(path: Path) -> str:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key in _FILE_DIGEST_CACHE:
        return _FILE_DIGEST_CACHE[key]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _FILE_DIGEST_CACHE[key] = value
    return value


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def _tensor_bytes(tensor) -> memoryview:
    # view(uint8) works for BF16, whose NumPy conversion is otherwise unsupported.
    array = tensor.detach().cpu().contiguous().view(-1).view(__import__("torch").uint8).numpy()
    return memoryview(array)


def _tensor_digest(tensor) -> str:
    digest = hashlib.sha256()
    digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


class DirectStateComparator:
    """Recursively compare loaded states without treating serialization as state."""

    def __init__(self, *, mismatch_limit: int = 50):
        self.tensor_count = 0
        self.tensor_bytes = 0
        self.scalar_count = 0
        self.mismatch_limit = mismatch_limit
        self.mismatches: list[dict[str, Any]] = []
        self._content = hashlib.sha256()

    def mismatch(self, path: str, reason: str, left: Any = None, right: Any = None) -> None:
        if len(self.mismatches) >= self.mismatch_limit:
            return
        row: dict[str, Any] = {"path": path, "reason": reason}
        if left is not None:
            row["control"] = left
        if right is not None:
            row["resumed"] = right
        self.mismatches.append(row)

    def _record_tensor(self, path: str, left) -> None:
        self.tensor_count += 1
        self.tensor_bytes += left.numel() * left.element_size()
        descriptor = {"path": path, "dtype": str(left.dtype), "shape": list(left.shape),
                      "sha256": _tensor_digest(left)}
        self._content.update(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode())

    def compare(self, left: Any, right: Any, path: str = "$") -> None:
        import numpy as np
        import torch

        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
                self.mismatch(path, "tensor/non-tensor type mismatch", type(left).__name__,
                              type(right).__name__)
                return
            self._record_tensor(path, left)
            if left.dtype != right.dtype:
                self.mismatch(path, "tensor dtype mismatch", str(left.dtype), str(right.dtype))
            elif tuple(left.shape) != tuple(right.shape):
                self.mismatch(path, "tensor shape mismatch", list(left.shape), list(right.shape))
            elif not torch.equal(left.detach().cpu(), right.detach().cpu()):
                self.mismatch(path, "tensor values differ", _tensor_digest(left),
                              _tensor_digest(right))
            return
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
                self.mismatch(path, "ndarray/non-ndarray type mismatch", type(left).__name__,
                              type(right).__name__)
                return
            self.scalar_count += 1
            if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
                self.mismatch(path, "NumPy RNG state differs",
                              {"dtype": str(left.dtype), "shape": list(left.shape)},
                              {"dtype": str(right.dtype), "shape": list(right.shape)})
            else:
                self._content.update(path.encode())
                self._content.update(memoryview(np.ascontiguousarray(left)).cast("B"))
            return
        if isinstance(left, dict) or isinstance(right, dict):
            if not isinstance(left, dict) or not isinstance(right, dict):
                self.mismatch(path, "mapping/non-mapping type mismatch", type(left).__name__,
                              type(right).__name__)
                return
            left_keys, right_keys = set(left), set(right)
            if left_keys != right_keys:
                self.mismatch(path, "mapping keys differ", sorted(map(repr, left_keys - right_keys)),
                              sorted(map(repr, right_keys - left_keys)))
            for key in sorted(left_keys & right_keys, key=lambda item: (type(item).__name__, repr(item))):
                self.compare(left[key], right[key], f"{path}/{type(key).__name__}:{key!r}")
            return
        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            if type(left) is not type(right):
                self.mismatch(path, "sequence type mismatch", type(left).__name__, type(right).__name__)
                return
            if len(left) != len(right):
                self.mismatch(path, "sequence length mismatch", len(left), len(right))
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                self.compare(left_item, right_item, f"{path}/{index}")
            return
        if isinstance(left, np.generic) or isinstance(right, np.generic):
            if type(left) is not type(right) or left.item() != right.item():
                self.mismatch(path, "NumPy scalar differs", repr(left), repr(right))
            self.scalar_count += 1
            return
        if left is None or isinstance(left, (bool, int, float, str, bytes)):
            self.scalar_count += 1
            if type(left) is not type(right) or left != right:
                self.mismatch(path, "scalar differs", repr(left), repr(right))
            else:
                self._content.update(f"{path}:{type(left).__name__}:{left!r}".encode())
            return
        raise TypeError(f"Unsupported state value at {path}: {type(left).__module__}.{type(left).__name__}")

    def result(self) -> dict[str, Any]:
        return {"status": "passed" if not self.mismatches else "failed",
                "direct_tensor_value_comparison": True,
                "tensor_count": self.tensor_count, "tensor_bytes": self.tensor_bytes,
                "scalar_or_array_count": self.scalar_count,
                "control_state_content_sha256": self._content.hexdigest(),
                "mismatch_count_shown": len(self.mismatches), "mismatches": self.mismatches}


def compare_model_state(left_root: Path, right_root: Path) -> dict[str, Any]:
    """Directly compare safetensors one tensor at a time to bound host RAM."""
    import torch
    from safetensors import safe_open

    left_files = {p.relative_to(left_root).as_posix(): p for p in left_root.rglob("*.safetensors")}
    right_files = {p.relative_to(right_root).as_posix(): p for p in right_root.rglob("*.safetensors")}
    result: dict[str, Any] = {
        "status": "running", "direct_tensor_value_comparison": True,
        "control_files": [file_record(left_files[key], left_root) for key in sorted(left_files)],
        "resumed_files": [file_record(right_files[key], right_root) for key in sorted(right_files)],
        "tensor_count": 0, "tensor_bytes": 0, "mismatches": [],
    }
    if not left_files:
        result["mismatches"].append({"path": "$", "reason": "no model safetensors found"})
    if set(left_files) != set(right_files):
        result["mismatches"].append({"path": "$", "reason": "model file names differ",
                                     "control_only": sorted(set(left_files) - set(right_files)),
                                     "resumed_only": sorted(set(right_files) - set(left_files))})
    aggregate = hashlib.sha256()
    for relative in sorted(set(left_files) & set(right_files)):
        with safe_open(left_files[relative], framework="pt", device="cpu") as left, \
             safe_open(right_files[relative], framework="pt", device="cpu") as right:
            left_keys, right_keys = set(left.keys()), set(right.keys())
            if left_keys != right_keys:
                result["mismatches"].append({"path": relative, "reason": "tensor names differ",
                                             "control_only": sorted(left_keys - right_keys),
                                             "resumed_only": sorted(right_keys - left_keys)})
            for key in sorted(left_keys & right_keys):
                left_tensor, right_tensor = left.get_tensor(key), right.get_tensor(key)
                result["tensor_count"] += 1
                result["tensor_bytes"] += left_tensor.numel() * left_tensor.element_size()
                descriptor = {"path": relative + "::" + key, "dtype": str(left_tensor.dtype),
                              "shape": list(left_tensor.shape), "sha256": _tensor_digest(left_tensor)}
                aggregate.update(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode())
                if left_tensor.dtype != right_tensor.dtype:
                    result["mismatches"].append({"path": descriptor["path"],
                                                 "reason": "tensor dtype mismatch",
                                                 "control": str(left_tensor.dtype),
                                                 "resumed": str(right_tensor.dtype)})
                elif tuple(left_tensor.shape) != tuple(right_tensor.shape):
                    result["mismatches"].append({"path": descriptor["path"],
                                                 "reason": "tensor shape mismatch",
                                                 "control": list(left_tensor.shape),
                                                 "resumed": list(right_tensor.shape)})
                elif not torch.equal(left_tensor, right_tensor):
                    result["mismatches"].append({"path": descriptor["path"],
                                                 "reason": "tensor values differ",
                                                 "control_sha256": descriptor["sha256"],
                                                 "resumed_sha256": _tensor_digest(right_tensor)})
                del left_tensor, right_tensor
                if len(result["mismatches"]) >= 50:
                    break
        if len(result["mismatches"]) >= 50:
            break
    result["control_tensor_content_sha256"] = aggregate.hexdigest()
    result["serialized_files_all_identical"] = (
        set(left_files) == set(right_files) and all(
            sha256_file(left_files[name]) == sha256_file(right_files[name]) for name in left_files))
    result["serialization_note"] = (
        "Informational only. Acceptance uses direct dtype/shape/torch.equal tensor comparison, "
        "not safetensors file hashes.")
    result["status"] = "passed" if not result["mismatches"] else "failed"
    return result


def torch_load(path: Path):
    import torch
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:  # PyTorch versions before mmap/weights_only keyword support.
        return torch.load(path, map_location="cpu")


def compare_torch_state(left: Path, right: Path, label: str) -> dict[str, Any]:
    comparator = DirectStateComparator()
    left_value = torch_load(left)
    right_value = torch_load(right)
    comparator.compare(left_value, right_value)
    result = comparator.result()
    result.update(label=label,
                  control_file=file_record(left, left.parent),
                  resumed_file=file_record(right, right.parent),
                  serialized_files_identical=sha256_file(left) == sha256_file(right),
                  serialization_note="Informational only; direct loaded-state comparison decides acceptance.")
    del left_value, right_value
    gc.collect()
    return result


def rng_files(checkpoint: Path, world_size: int) -> dict[int, Path]:
    plain = checkpoint / "rng_state.pth"
    ranked: dict[int, Path] = {}
    for path in checkpoint.glob("rng_state_*.pth"):
        suffix = path.stem.removeprefix("rng_state_")
        if suffix.isdigit():
            ranked[int(suffix)] = path
    if world_size == 1 and plain.is_file() and not ranked:
        return {0: plain}
    expected = set(range(world_size))
    if set(ranked) != expected or plain.exists():
        raise ValueError(f"{checkpoint}: expected one RNG state for ranks 0..{world_size - 1}; "
                         f"found ranked={sorted(ranked)}, plain={plain.exists()}")
    return ranked


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def checkpoint_metadata(checkpoint: Path, world_size: int, expected_step: int) -> dict[str, Any]:
    required = ("training_provenance.json", "trainer_state.json", "optimizer.pt", "scheduler.pt")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise ValueError(f"{checkpoint}: incomplete checkpoint, missing {missing}")
    state = load_json(checkpoint / "trainer_state.json")
    if state.get("global_step") != expected_step:
        raise ValueError(f"{checkpoint}: expected global_step={expected_step}, got {state.get('global_step')}")
    provenance = load_json(checkpoint / "training_provenance.json")
    contract = provenance.get("training_contract")
    if not isinstance(contract, dict) or contract.get("world_size") != world_size:
        raise ValueError(f"{checkpoint}: training contract world_size is not {world_size}")
    rng = rng_files(checkpoint, world_size)
    payload_paths = sorted({*(checkpoint.rglob("*.safetensors")),
                            *(checkpoint / name for name in required), *rng.values()})
    return {"path": str(checkpoint.resolve()), "global_step": state["global_step"],
            "world_size": contract["world_size"], "training_contract": contract,
            "training_contract_sha256": canonical_sha256(contract),
            "rng_ranks": sorted(rng), "rng_files": {str(k): file_record(v, checkpoint) for k, v in rng.items()},
            "payload_inventory": [file_record(path, checkpoint) for path in payload_paths]}


def trainer_progress(checkpoint: Path) -> dict[str, Any]:
    state = load_json(checkpoint / "trainer_state.json")
    keys = ("epoch", "global_step", "max_steps", "num_train_epochs", "train_batch_size", "total_flos")
    return {key: state.get(key) for key in keys}


def manifest_identity(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def run_identity(checkpoint: Path) -> dict[str, Any]:
    run_dir = checkpoint.parent
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing run manifest beside checkpoint: {manifest_path}")
    manifest = load_json(manifest_path)
    return {"run_dir": str(run_dir.resolve()), "manifest": manifest,
            "manifest_file": file_record(manifest_path, run_dir)}


def run(args: argparse.Namespace, report: dict[str, Any]) -> None:
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite report: {args.output}")
    if args.expected_world_size <= 0:
        raise ValueError("expected-world-size must be positive")
    if not 0 < args.expected_split_step < args.expected_final_step:
        raise ValueError("Require 0 < expected-split-step < expected-final-step")
    for path in (args.control_checkpoint, args.resumed_checkpoint, args.fork_checkpoint):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if bool(args.control_final) != bool(args.resumed_final):
        raise ValueError("Specify both --control-final and --resumed-final, or neither")

    manifests = {"train": manifest_identity(args.train_manifest),
                 "eval": manifest_identity(args.eval_manifest)}
    control_meta = checkpoint_metadata(args.control_checkpoint, args.expected_world_size,
                                       args.expected_final_step)
    resumed_meta = checkpoint_metadata(args.resumed_checkpoint, args.expected_world_size,
                                       args.expected_final_step)
    fork_meta = checkpoint_metadata(args.fork_checkpoint, args.expected_world_size,
                                    args.expected_split_step)
    control_run, resumed_run = run_identity(args.control_checkpoint), run_identity(args.resumed_checkpoint)
    if args.fork_checkpoint.parent.resolve() != args.control_checkpoint.parent.resolve():
        raise ValueError("Fork checkpoint must belong to the control run")
    if control_run["manifest"].get("resume") is not None:
        raise ValueError("Control run must be uninterrupted, not itself a resume")
    recorded_resume = resumed_run["manifest"].get("resume")
    if not recorded_resume or Path(recorded_resume).resolve() != args.fork_checkpoint.resolve():
        raise ValueError("Resumed run manifest does not identify the supplied fork checkpoint")
    for field in ("model", "training", "stage", "train_manifest_sha256", "eval_manifest_sha256",
                  "training_contract", "packages", "git_revision"):
        if control_run["manifest"].get(field) != resumed_run["manifest"].get(field):
            raise ValueError(f"Control and resumed run manifests differ in {field}")
    report.update(manifests=manifests, control=control_meta, resumed=resumed_meta, fork=fork_meta)
    report["run_identity"] = {"control": control_run, "resumed": resumed_run,
                              "resumed_from_exact_fork_path": True}
    source_manifest = manifest_identity(args.source_manifest)
    contracts_equal = (control_meta["training_contract"] == resumed_meta["training_contract"] ==
                       fork_meta["training_contract"])
    report["contract_checks"] = {
        "status": "passed" if contracts_equal else "failed",
        "contracts_identical": contracts_equal,
        "world_size": args.expected_world_size,
        "train_manifest_sha256_matches": all(
            item["training_contract"].get("train_manifest_sha256") == manifests["train"]["sha256"]
            for item in (control_meta, resumed_meta, fork_meta)),
        "eval_manifest_sha256_matches": all(
            item["training_contract"].get("eval_manifest_sha256") == manifests["eval"]["sha256"]
            for item in (control_meta, resumed_meta, fork_meta)),
        "frozen_source_manifest_sha256_matches": all(
            item["training_contract"].get("frozen_source_manifest_sha256") == source_manifest["sha256"]
            for item in (control_meta, resumed_meta, fork_meta)),
    }
    report["source_manifest"] = source_manifest
    if not all(value for key, value in report["contract_checks"].items()
               if key.endswith("_matches")):
        report["contract_checks"]["status"] = "failed"

    report["comparisons"] = {
        "model": compare_model_state(args.control_checkpoint, args.resumed_checkpoint),
        "optimizer": compare_torch_state(args.control_checkpoint / "optimizer.pt",
                                         args.resumed_checkpoint / "optimizer.pt", "optimizer"),
        "scheduler": compare_torch_state(args.control_checkpoint / "scheduler.pt",
                                         args.resumed_checkpoint / "scheduler.pt", "scheduler"),
    }
    control_rng = rng_files(args.control_checkpoint, args.expected_world_size)
    resumed_rng = rng_files(args.resumed_checkpoint, args.expected_world_size)
    rng_results = {str(rank): compare_torch_state(control_rng[rank], resumed_rng[rank], f"rng_rank_{rank}")
                   for rank in range(args.expected_world_size)}
    report["comparisons"]["rng"] = {
        "status": "passed" if all(row["status"] == "passed" for row in rng_results.values()) else "failed",
        "rank_count": args.expected_world_size, "ranks": rng_results,
    }
    control_progress, resumed_progress = trainer_progress(args.control_checkpoint), trainer_progress(args.resumed_checkpoint)
    report["comparisons"]["trainer_progress"] = {
        "status": "passed" if control_progress == resumed_progress else "failed",
        "control": control_progress, "resumed": resumed_progress,
        "scope": "Core progression fields only; wall-clock throughput and log serialization are excluded."}
    if args.control_final:
        report["comparisons"]["control_checkpoint_vs_final"] = compare_model_state(
            args.control_checkpoint, args.control_final)
        report["comparisons"]["resumed_checkpoint_vs_final"] = compare_model_state(
            args.resumed_checkpoint, args.resumed_final)

    statuses = [report["contract_checks"]["status"]]
    statuses.extend(item["status"] for item in report["comparisons"].values())
    proven = all(status == "passed" for status in statuses)
    report["strict_tensorwise_resume_equivalence_proven"] = proven
    report["acceptance_definition"] = (
        "Same frozen executable-source manifest, package versions, git revision, training "
        "contract/data/world size and final optimizer step; exact dtype, shape and "
        "torch.equal values for every model/optimizer/scheduler tensor; exact scalar/container state; "
        "and exact Python/NumPy/CPU/CUDA RNG state for every rank.")
    report["serialization_hashes_used_for_acceptance"] = False
    report["status"] = "strict_resume_equivalence_passed" if proven else "strict_resume_equivalence_failed"


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report: dict[str, Any] = {
        "status": "running", "purpose": "P0_05_STRICT_RESUME_ENGINEERING_EVIDENCE_NOT_A_QUALITY_RESULT",
        "utc": datetime.now(timezone.utc).isoformat(), "errors": [],
    }
    try:
        run(args, report)
    except Exception as exc:
        report["status"] = "error"
        report["strict_tensorwise_resume_equivalence_proven"] = False
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.output.is_symlink():
        print(json.dumps({"status": "error", "error": f"output exists: {args.output}"}))
        return 2
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "report": str(args.output),
                      "errors": report["errors"]}))
    return 0 if report.get("strict_tensorwise_resume_equivalence_proven") else 1


if __name__ == "__main__":
    raise SystemExit(main())
