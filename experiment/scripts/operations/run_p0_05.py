#!/usr/bin/env python3
"""Run the finite P0-05 strict-resume and short-overfit engineering gates.

This runner never submits a scheduler job and never uses held-out test data.
Submit it through the established low-priority, full-node GPU wrapper.  It
expects one allocated node and defaults to eight DDP ranks for the resume gate;
the short-overfit gate intentionally uses one visible GPU on that same node.

The control final model copy is checked and archived before local removal.
The resumed branch keeps its complete checkpoint-20 for direct strict model,
optimizer, scheduler and per-rank RNG comparison, without writing a duplicate
``final/`` model.  The control final checkpoint and the overfit final model
remain for archival.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
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
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = "oss://ss-devbox-stream/jinhaopeng/Paper/DeepProbe-VLM/runs"
PRECISION_RECEIPT_PREFIX = "P0_BF16_FORWARD_RANK"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--eval-manifest", type=Path, required=True)
    p.add_argument("--media-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--master-port", type=int)
    p.add_argument("--overfit-samples", type=int, default=4)
    p.add_argument("--overfit-steps", type=int, default=60)
    p.add_argument("--overfit-learning-rate", type=float, default=1e-5)
    p.add_argument("--overfit-warmup-steps", type=int, default=5)
    p.add_argument("--overfit-max-loss-ratio", type=float, default=0.5)
    p.add_argument("--overfit-min-token-accuracy", type=float, default=0.95)
    p.add_argument("--project-storage-root", type=Path,
                   help="Project cache counted against the 100 GB staging guard")
    p.add_argument("--project-storage-limit-bytes", type=int, default=100_000_000_000)
    p.add_argument("--reserved-additional-bytes", type=int, default=75_000_000_000)
    p.add_argument("--training-timeout-seconds", type=int, default=7200)
    p.add_argument("--verification-timeout-seconds", type=int, default=3600)
    p.add_argument("--keep-duplicates", action="store_true",
                   help="For forensic debugging only; may exceed the staging budget")
    p.add_argument("--diagnostic-step", type=int,
                   help="Trace only fork-step + 1 in both strict branches; keep max_steps unchanged")
    p.add_argument("--dry-run", action="store_true", help="Validate inputs and print the plan only")
    return p


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def p0_source_inventory(raw_config: dict[str, Any], *, require_model_code: bool) -> dict[str, Any]:
    """Inventory every local executable source plus pinned Hub model code."""
    files: dict[str, dict[str, Any]] = {}
    roots = [ROOT / name for name in ("src", "scripts", "configs", "requirements")]
    for base in roots:
        if not base.is_dir():
            raise FileNotFoundError(base)
        for path in sorted(base.rglob("*")):
            if (not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc"):
                continue
            relative = path.relative_to(ROOT).as_posix()
            files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    for name in ("pyproject.toml", "README.md"):
        path = ROOT / name
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}

    model = raw_config.get("model", {})
    hub = Path(os.environ.get("HF_HUB_CACHE", ""))
    model_code = {}
    for label, key, revision_key in (("language", "language_model", "language_revision"),
                                     ("vision", "vision_model", "vision_revision")):
        identifier, revision = model.get(key), model.get(revision_key)
        if not isinstance(identifier, str) or not isinstance(revision, str):
            raise ValueError(f"Config lacks pinned model.{key}/{revision_key}")
        if Path(identifier).is_absolute():
            snapshot = Path(identifier)
        elif identifier.count("/") == 1 and hub:
            snapshot = hub / ("models--" + identifier.replace("/", "--")) / "snapshots" / revision
        else:
            snapshot = Path("/__missing_hub_snapshot__")
        code = {}
        if snapshot.is_dir():
            for path in sorted(snapshot.rglob("*.py")):
                if path.is_file():
                    code[path.relative_to(snapshot).as_posix()] = {
                        "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if require_model_code and label == "language" and model.get("language_backend") == "ouro" and not code:
            raise FileNotFoundError(f"Pinned Ouro remote-code snapshot is missing: {snapshot}")
        model_code[label] = {"identifier": identifier, "revision": revision,
                             "snapshot": str(snapshot.resolve()) if snapshot.is_dir() else None,
                             "files": code}
    return {"schema_version": 1, "purpose": "P0_05_FROZEN_EXECUTABLE_SOURCE",
            "files": files, "pinned_model_code": model_code}


def assert_source_inventory_unchanged(expected: dict[str, Any], raw_config: dict[str, Any]) -> None:
    observed = p0_source_inventory(raw_config, require_model_code=True)
    if observed != expected:
        raise RuntimeError("Executable source/model code changed during P0-05")


def write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_atomic_json(path: Path, value: Any) -> None:
    """Replace a mutable local bookkeeping file without exposing partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def verify_bf16_forward_receipts(run_dir: Path, expected_world_size: int) -> dict[str, Any]:
    """Require per-rank proof of BF16 autocast on real training forwards."""
    if expected_world_size <= 0:
        raise ValueError("Expected a positive precision-receipt world size")
    expected = {f"{PRECISION_RECEIPT_PREFIX}{rank}.json" for rank in range(expected_world_size)}
    observed = {path.name for path in run_dir.glob(f"{PRECISION_RECEIPT_PREFIX}*.json")}
    if observed != expected:
        raise RuntimeError(
            f"BF16 forward receipts differ in {run_dir}: "
            f"missing={sorted(expected - observed)}, unexpected={sorted(observed - expected)}"
        )
    receipts = {}
    for rank in range(expected_world_size):
        path = run_dir / f"{PRECISION_RECEIPT_PREFIX}{rank}.json"
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"BF16 forward receipt is missing or unsafe: {path}")
        value = load_json(path)
        required = {
            "schema_version": 1,
            "status": "cuda_bf16_autocast_observed_on_training_forward",
            "rank": rank,
            "world_size": expected_world_size,
            "input_device": "cuda",
            "cuda_autocast_enabled": True,
            "cuda_autocast_dtype": "torch.bfloat16",
            "output_dir": str(run_dir.resolve()),
        }
        if (value != required or type(value.get("schema_version")) is not int
                or type(value.get("rank")) is not int
                or type(value.get("world_size")) is not int
                or type(value.get("cuda_autocast_enabled")) is not bool):
            raise RuntimeError(f"BF16 forward receipt content differs: {path}")
        receipts[str(rank)] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    return {"status": "passed", "world_size": expected_world_size, "ranks": receipts}


def tree_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for directory, folders, names in os.walk(root, followlinks=False):
        folders[:] = [name for name in folders if not (Path(directory) / name).is_symlink()]
        for name in names:
            path = Path(directory) / name
            try:
                if not path.is_symlink():
                    total += path.stat().st_size
            except FileNotFoundError:
                pass
    return total


def _remote_object_length(uri: str) -> int:
    report = subprocess.run(["ossutil", "ls", uri], check=True, text=True,
                            capture_output=True, timeout=300).stdout
    rows = [line.split() for line in report.splitlines()
            if line.split() and line.split()[-1] == uri]
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one archived object: {uri}")
    return int(rows[0][4])


def _upload_file(source: Path, uri: str) -> None:
    subprocess.run(["ossutil", "cp", str(source), uri, "-f", "--no-progress"],
                   check=True, timeout=7200)
    if _remote_object_length(uri) != source.stat().st_size:
        raise RuntimeError(f"Archived object length differs: {uri}")


class VerifiedArchive:
    """Incremental, hash-inventoried archive used before any local pruning.

    Large exact duplicates have to be removed during P0-05 to stay below the
    personal 100 GB staging cap.  Each such tree is uploaded first and gets a
    small SHA-verified fragment receipt at its final object key.  The final
    manifest is the union of those durable fragments and all files still
    present when the run exits.
    """

    _EXCLUDED_TOP_LEVEL = {
        "runtime", "ARCHIVE_LEDGER.json", "ARCHIVE_MANIFEST.json",
        "ARCHIVED.json", "CLEANED_WEIGHTS.json",
    }

    def __init__(self, root: Path, run_id: str):
        if shutil.which("ossutil") is None:
            raise FileNotFoundError("ossutil is required before starting the GPU gate")
        self.root = root.resolve()
        self.run_id = run_id
        self.uri = f"{ARCHIVE_ROOT}/{run_id}"
        self.entries: dict[str, dict[str, Any]] = {}
        self.fragments: list[dict[str, Any]] = []
        listing = subprocess.run(["ossutil", "ls", self.uri + "/"], check=True,
                                 text=True, capture_output=True, timeout=300).stdout
        objects = [parts[-1] for line in listing.splitlines()
                   if (parts := line.split()) and parts[-1].startswith(self.uri + "/")]
        if objects:
            raise FileExistsError(f"Archive prefix is not empty: {self.uri}")
        self._write_ledger()

    def _relative(self, path: Path) -> str:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root) or path.is_symlink():
            raise ValueError(f"Archive path is outside the run or is a symlink: {path}")
        return resolved.relative_to(self.root).as_posix()

    def _record(self, path: Path) -> dict[str, Any]:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Archive only accepts regular files: {path}")
        relative = self._relative(path)
        record = {"path": relative, "bytes": path.stat().st_size,
                  "sha256": sha256_file(path)}
        previous = self.entries.get(relative)
        if previous is not None and previous != record:
            raise RuntimeError(f"Previously archived file changed: {relative}")
        if previous is None:
            _upload_file(path, self.uri + "/" + relative)
            self.entries[relative] = record
            self._write_ledger()
        elif _remote_object_length(self.uri + "/" + relative) != record["bytes"]:
            raise RuntimeError(f"Previously archived object length changed: {relative}")
        return record

    def _write_ledger(self) -> None:
        write_atomic_json(self.root / "ARCHIVE_LEDGER.json", {
            "schema_version": 1, "run_id": self.run_id, "uri": self.uri,
            "files": [self.entries[name] for name in sorted(self.entries)],
            "fragments": self.fragments,
        })

    def _readback_sha(self, local: Path, uri: str, label: str) -> None:
        target = self.root / "runtime/archive-readback" / label
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        subprocess.run(["ossutil", "cp", uri, str(target), "-f", "--no-progress"],
                       check=True, timeout=1800)
        try:
            if sha256_file(target) != sha256_file(local):
                raise RuntimeError(f"Archived receipt readback differs: {uri}")
        finally:
            target.unlink(missing_ok=True)

    def archive_tree_before_prune(self, tag: str, path: Path) -> dict[str, Any]:
        if not re.fullmatch(r"[a-z0-9_-]+", tag):
            raise ValueError(f"Invalid archive fragment tag: {tag}")
        if not path.is_dir() or path.is_symlink():
            raise ValueError(f"Expected regular directory for archive fragment: {path}")
        records = [self._record(item) for item in sorted(path.rglob("*")) if item.is_file()]
        receipt = self.root / "evidence/archive-fragments" / f"{tag}.json"
        write_new_json(receipt, {
            "schema_version": 1, "status": "archive_fragment_verified",
            "run_id": self.run_id, "tag": tag, "uri": self.uri,
            "tree": self._relative(path), "file_count": len(records),
            "total_bytes": sum(item["bytes"] for item in records), "files": records,
            "verification": "ossutil CRC upload, exact object length, receipt SHA-256 readback",
        })
        receipt_record = self._record(receipt)
        self._readback_sha(receipt, self.uri + "/" + receipt_record["path"],
                           f"fragment-{tag}.json")
        fragment = {"tag": tag, "receipt": receipt_record["path"],
                    "receipt_sha256": receipt_record["sha256"],
                    "file_count": len(records), "total_bytes": sum(item["bytes"] for item in records)}
        self.fragments.append(fragment)
        self._write_ledger()
        return fragment

    def finalize(self, *, exit_code: int) -> dict[str, Any]:
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(self.root)
            if relative.parts[0] in self._EXCLUDED_TOP_LEVEL or path.name.endswith(".pending"):
                continue
            self._record(path)
        files = [self.entries[name] for name in sorted(self.entries)]
        manifest_path = self.root / "ARCHIVE_MANIFEST.json"
        write_new_json(manifest_path, {
            "schema_version": 1, "status": "complete_payload_inventory",
            "run_id": self.run_id, "exit_code": exit_code, "uri": self.uri,
            "file_count": len(files), "total_bytes": sum(item["bytes"] for item in files),
            "files": files, "fragment_receipts": self.fragments,
        })
        _upload_file(manifest_path, self.uri + "/ARCHIVE_MANIFEST.json")
        self._readback_sha(manifest_path, self.uri + "/ARCHIVE_MANIFEST.json",
                           "ARCHIVE_MANIFEST.json")
        receipt = {
            "schema_version": 1, "status": "archive_verified", "run_id": self.run_id,
            "exit_code": exit_code, "uri": self.uri,
            "archive_manifest_sha256": sha256_file(manifest_path),
            "file_count": len(files), "total_bytes": sum(item["bytes"] for item in files),
            "remote_manifest_and_completion_sha256_readback": True,
            "verification": "ossutil CRC upload, exact payload lengths, manifest/receipt SHA-256 readback",
        }
        pending = self.root / "ARCHIVED.json.pending"
        write_atomic_json(pending, receipt)
        _upload_file(pending, self.uri + "/ARCHIVED.json")
        self._readback_sha(pending, self.uri + "/ARCHIVED.json", "ARCHIVED.json")
        pending.replace(self.root / "ARCHIVED.json")
        return receipt


def cleanup_verified_archived_weights(root: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    manifest_path = root / "ARCHIVE_MANIFEST.json"
    if (receipt.get("status") != "archive_verified"
            or receipt.get("remote_manifest_and_completion_sha256_readback") is not True
            or receipt.get("archive_manifest_sha256") != sha256_file(manifest_path)):
        raise ValueError("Refusing cleanup without a verified archive/readback receipt")
    removed = []
    for item in load_json(manifest_path)["files"]:
        relative = Path(item["path"])
        if relative.suffix != ".safetensors" and relative.name not in {
                "pytorch_model.bin", "optimizer.pt"}:
            continue
        path = root / relative
        if not path.exists():
            # Exact duplicate trees may already have their own verified fragment
            # receipt and have been pruned to stay below the staging cap.
            continue
        if (path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
                or path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]):
            raise ValueError(f"Archived weight changed before cleanup: {relative}")
        path.unlink()
        removed.append(item)
    result = {"status": "only_verified_archived_weights_removed",
              "archive_manifest_sha256": receipt["archive_manifest_sha256"],
              "files": removed, "bytes_removed": sum(item["bytes"] for item in removed)}
    write_new_json(root / "CLEANED_WEIGHTS.json", result)
    return result


def select_fixed_sources(path: Path, count: int) -> list[dict[str, Any]]:
    if not 1 <= count <= 8:
        raise ValueError("overfit-samples must be in 1..8")
    selected, seen = [], set()
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError(f"{path}:{lineno}: expected train split")
            source = row.get("source_video_id")
            if not isinstance(source, str) or not source:
                raise ValueError(f"{path}:{lineno}: missing source_video_id")
            if source not in seen:
                selected.append(row)
                seen.add(source)
            if len(selected) == count:
                break
    if len(selected) != count:
        raise ValueError(f"Need {count} distinct train sources; found {len(selected)}")
    return selected


def build_configs(raw: dict[str, Any], *, overfit_samples: int, overfit_steps: int,
                  overfit_learning_rate: float, overfit_warmup_steps: int) -> tuple[dict, dict]:
    if not isinstance(raw.get("model"), dict) or not isinstance(raw.get("training"), dict):
        raise ValueError("Config needs model and training objects")
    if not 1 <= overfit_samples <= 8:
        raise ValueError("overfit samples must be in 1..8")
    if overfit_steps <= 0 or overfit_learning_rate <= 0:
        raise ValueError("overfit steps and learning rate must be positive")
    if not 0 <= overfit_warmup_steps < overfit_steps:
        raise ValueError("overfit warmup steps must be in 0..steps-1")
    strict = json.loads(json.dumps(raw))
    training = strict["training"]
    final_step = training.get("max_steps")
    split_step = training.get("save_steps")
    if isinstance(final_step, bool) or not isinstance(final_step, int) or final_step <= 1:
        raise ValueError("Strict resume config needs max_steps > 1")
    if isinstance(split_step, bool) or not isinstance(split_step, int) or not 0 < split_step < final_step:
        raise ValueError("Strict resume config needs 0 < save_steps < max_steps")
    if final_step % split_step:
        raise ValueError("Final step must be a save step so optimizer/scheduler/RNG can be compared")
    if int(training.get("save_total_limit", 0)) < 2:
        raise ValueError("Strict resume config must retain both split and final checkpoints")
    if training.get("bf16") is not True or training.get("fp16", False) is not False:
        raise ValueError("P0-05 strict equivalence requires bf16=true and fp16=false")
    training["dataloader_num_workers"] = 0

    overfit = json.loads(json.dumps(strict))
    overfit_training = overfit["training"]
    overfit_training.update({
        "stage": "sft", "learning_rate": overfit_learning_rate,
        "per_device_train_batch_size": 1, "per_device_eval_batch_size": 1,
        # One optimizer update sees the fixed subset. v7 used single-example
        # full-model updates during a failed gate; causality is unproven.
        "gradient_accumulation_steps": overfit_samples, "num_train_epochs": 1,
        "max_steps": overfit_steps, "warmup_steps": overfit_warmup_steps, "warmup_ratio": 0.0,
        "weight_decay": 0.0, "lr_scheduler_type": "constant_with_warmup",
        "save_strategy": "no", "eval_strategy": "no", "logging_steps": 1,
        "dataloader_num_workers": 0, "report_to": "none", "disable_tqdm": True,
    })
    # These step intervals are meaningless when their strategies are disabled.
    overfit_training.pop("eval_steps", None)
    overfit_training.pop("save_steps", None)
    overfit_training.pop("save_total_limit", None)
    return strict, overfit


def training_command(args: argparse.Namespace, config: Path, output: Path, *, distributed: bool,
                     train_manifest: Path | None = None, resume: Path | None = None,
                     init_checkpoint: Path | None = None,
                     diagnostic_step: int | None = None,
                     skip_resumed_final_export: bool = False) -> list[str]:
    entry = ROOT / "scripts/operations/deterministic_train.py"
    command = [str(args.python)]
    if distributed:
        command += ["-m", "torch.distributed.run", "--nnodes=1", "--node-rank=0",
                    f"--nproc-per-node={args.world_size}", "--master-addr=127.0.0.1",
                    f"--master-port={args.master_port}"]
    command += [str(entry), "--config", str(config), "--train-manifest",
                str((train_manifest or args.train_manifest).resolve()),
                "--eval-manifest", str(args.eval_manifest.resolve()), "--media-root",
                str(args.media_root.resolve()), "--output-dir", str(output),
                "--frozen-source-manifest-sha256", args.frozen_source_manifest_sha256]
    if resume:
        command += ["--resume", str(resume)]
    if init_checkpoint:
        command += ["--init-checkpoint", str(init_checkpoint)]
    if skip_resumed_final_export:
        if not resume:
            raise ValueError("Only an exact-resume branch may skip its final model export")
        command.append("--p0-skip-resumed-final-export")
    if diagnostic_step is not None:
        command += ["--diagnostic-step", str(diagnostic_step)]
    return command


def build_plan(args: argparse.Namespace, paths: dict[str, Path], *, split_step: int,
               final_step: int) -> dict[str, Any]:
    control_checkpoint = paths["control"] / f"checkpoint-{final_step}"
    fork_checkpoint = paths["control"] / f"checkpoint-{split_step}"
    resumed_checkpoint = paths["resumed"] / f"checkpoint-{final_step}"
    plan = {
        "gpu_gate": [str(args.python), "-c",
                     ("import torch; n=torch.cuda.device_count(); "
                      f"assert n=={args.world_size}, f'expected {args.world_size} visible GPUs, got {{n}}'; "
                      "assert torch.cuda.is_bf16_supported(); print(torch.__version__, torch.version.cuda, n)")],
        "control": training_command(args, paths["strict_config"], paths["control"], distributed=True,
                                    diagnostic_step=args.diagnostic_step),
        "control_copy_verify": [
            str(args.python), str(ROOT / "scripts/operations/verify_model_copy.py"),
            "--checkpoint", str(control_checkpoint), "--final", str(paths["control"] / "final"),
            "--output", str(paths["control_copy_report"])],
        "resume": training_command(args, paths["strict_config"], paths["resumed"], distributed=True,
                                   resume=fork_checkpoint, diagnostic_step=args.diagnostic_step,
                                   skip_resumed_final_export=not args.keep_duplicates),
        "strict_verify": [str(args.python), str(ROOT / "scripts/operations/verify_resume_equivalence.py"),
                          "--control-checkpoint", str(control_checkpoint),
                          "--resumed-checkpoint", str(resumed_checkpoint),
                          "--fork-checkpoint", str(fork_checkpoint),
                          "--source-manifest", str(paths["source_manifest"]),
                          "--train-manifest", str(args.train_manifest.resolve()),
                          "--eval-manifest", str(args.eval_manifest.resolve()),
                          "--expected-world-size", str(args.world_size),
                          "--expected-split-step", str(split_step),
                          "--expected-final-step", str(final_step),
                          "--output", str(paths["strict_report"])],
        "overfit": training_command(args, paths["overfit_config"], paths["overfit"], distributed=False,
                                    train_manifest=paths["overfit_manifest"],
                                    init_checkpoint=control_checkpoint),
        "overfit_verify": [str(args.python), str(ROOT / "scripts/operations/verify_short_overfit.py"),
                           "--config", str(paths["overfit_config"]),
                           "--train-manifest", str(paths["overfit_manifest"]),
                           "--media-root", str(args.media_root.resolve()),
                           "--initial-checkpoint", str(control_checkpoint),
                           "--final-checkpoint", str(paths["overfit"] / "final"),
                           "--training-output", str(paths["overfit"]),
                           "--expected-samples", str(args.overfit_samples),
                           "--expected-world-size", "1", "--expected-final-step", str(args.overfit_steps),
                           "--max-loss-ratio", str(args.overfit_max_loss_ratio),
                           "--min-final-token-accuracy", str(args.overfit_min_token_accuracy),
                           "--output", str(paths["overfit_report"])],
    }
    if args.keep_duplicates:
        plan["resumed_copy_verify"] = [
            str(args.python), str(ROOT / "scripts/operations/verify_model_copy.py"),
            "--checkpoint", str(resumed_checkpoint), "--final", str(paths["resumed"] / "final"),
            "--output", str(paths["resumed_copy_report"])]
    if args.diagnostic_step is not None:
        plan["diagnostic_compare"] = [
            str(args.python), str(ROOT / "scripts/operations/compare_step_diagnostic.py"),
            "--control", str(paths["control"]), "--resumed", str(paths["resumed"]),
            "--step", str(args.diagnostic_step), "--world-size", str(args.world_size),
            "--output", str(paths["diagnostic_report"])]
    return plan


class StageRunner:
    def __init__(self, *, root: Path, cwd: Path, env: dict[str, str], storage_root: Path,
                 storage_limit: int):
        self.root, self.cwd, self.env = root, cwd, env
        self.storage_root, self.storage_limit = storage_root, storage_limit
        self.stages: list[dict[str, Any]] = []
        self.active: subprocess.Popen[str] | None = None

    def stop(self) -> None:
        process = self.active
        if process is None or process.poll() is not None:
            return
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

    def run(self, name: str, command: list[str], timeout: int, *, one_gpu: bool = False) -> None:
        log_path = self.root / "logs" / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stage_env = dict(self.env)
        if one_gpu:
            visible = stage_env.get("CUDA_VISIBLE_DEVICES")
            stage_env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[0] if visible else "0"
        record = {"name": name, "command": command, "started_utc": datetime.now(timezone.utc).isoformat(),
                  "timeout_seconds": timeout, "one_gpu": one_gpu, "log": str(log_path)}
        self.stages.append(record)
        started = time.monotonic()
        with log_path.open("x", encoding="utf-8") as log:
            self.active = subprocess.Popen(command, cwd=self.cwd, env=stage_env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, errors="replace",
                                           start_new_session=True)
            process = self.active

            def pump() -> None:
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line); log.flush()
                    print(line, end="", flush=True)

            thread = threading.Thread(target=pump, daemon=True)
            thread.start()
            next_storage = started + 30
            try:
                while process.poll() is None:
                    now = time.monotonic()
                    if now - started > timeout:
                        raise TimeoutError(f"{name} exceeded {timeout}s")
                    if now >= next_storage:
                        used = tree_bytes(self.storage_root)
                        record["last_project_storage_bytes"] = used
                        if used >= self.storage_limit - 1_000_000_000:
                            raise RuntimeError("Project storage reached the configured limit minus 1 GB guard")
                        next_storage = now + 30
                    time.sleep(1)
                if process.returncode:
                    raise RuntimeError(f"{name} exited with {process.returncode}")
            finally:
                self.stop()
                thread.join(timeout=30)
                record["returncode"] = process.returncode
                record["elapsed_seconds"] = time.monotonic() - started
                record["finished_utc"] = datetime.now(timezone.utc).isoformat()
                self.active = None


def remove_tree(path: Path, run_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_relative_to(run_root.resolve()):
        raise ValueError(f"Refusing to prune outside this run: {path}")
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"Expected owned directory for pruning: {path}")
    files = sum(1 for item in path.rglob("*") if item.is_file() and not item.is_symlink())
    size = tree_bytes(path)
    shutil.rmtree(path)
    return {"path": str(path.relative_to(run_root)), "files_removed": files, "bytes_removed": size}


def verify_resumed_final_export_omission(run_dir: Path, final_step: int,
                                         world_size: int) -> dict[str, Any]:
    """Require the complete resumable checkpoint used instead of ``final/``.

    This is a structural receipt only.  The later strict verifier directly
    compares model, optimizer, scheduler and each rank's RNG values; the final
    archive hashes and uploads every checkpoint file.
    """
    final_dir = run_dir / "final"
    if final_dir.exists() or final_dir.is_symlink():
        raise RuntimeError(f"Resumed final export was not omitted: {final_dir}")
    checkpoint = run_dir / f"checkpoint-{final_step}"
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise RuntimeError(f"Resumed final checkpoint is missing or unsafe: {checkpoint}")
    if world_size <= 0:
        raise ValueError("Expected a positive resumed checkpoint world size")
    essentials = ["deepprobe_config.json", "training_provenance.json",
                  "trainer_state.json", "optimizer.pt", "scheduler.pt",
                  "projector.safetensors"]
    expected_rng = {f"rng_state_{rank}.pth" for rank in range(world_size)}
    observed_rng = {item.name for item in checkpoint.glob("rng_state*.pth")}
    if observed_rng != expected_rng:
        raise RuntimeError(f"Resumed checkpoint RNG ranks differ: {sorted(observed_rng)}")
    essential_files = [checkpoint / name for name in essentials + sorted(expected_rng)]
    for path in essential_files:
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Resumed final checkpoint file is missing or unsafe: {path}")
    model_files = []
    for component in ("language_model", "vision_model"):
        directory = checkpoint / component
        if not directory.is_dir() or directory.is_symlink():
            raise RuntimeError(f"Resumed checkpoint model component is missing: {directory}")
        files = sorted(directory.rglob("*.safetensors"))
        if not files:
            raise RuntimeError(f"Resumed checkpoint has no {component} tensors")
        for path in files:
            if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"Resumed checkpoint model file is missing or unsafe: {path}")
        model_files.extend(files)
    state = load_json(checkpoint / "trainer_state.json")
    provenance = load_json(checkpoint / "training_provenance.json")
    if type(state.get("global_step")) is not int or state["global_step"] != final_step:
        raise RuntimeError("Resumed checkpoint has the wrong final optimizer step")
    contract = provenance.get("training_contract")
    if not isinstance(contract, dict) or contract.get("world_size") != world_size:
        raise RuntimeError("Resumed checkpoint has the wrong training world size")
    return {
        "schema_version": 1,
        "status": "p0_resumed_final_export_intentionally_omitted",
        "reason": ("The complete final-step checkpoint already contains model, optimizer, "
                   "scheduler and every rank's RNG; a second final/ model would duplicate local bytes."),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_step": final_step,
        "world_size": world_size,
        "final_directory_absent": True,
        "structural_check_only": True,
        "direct_value_comparison_required": True,
        "files": [{"path": path.relative_to(checkpoint).as_posix(), "bytes": path.stat().st_size}
                  for path in sorted(essential_files + model_files)],
    }


def run(args: argparse.Namespace) -> int:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("run-id must contain only letters, digits, underscores and hyphens")
    for path in (args.config, args.train_manifest, args.eval_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.media_root.is_dir() or not args.python.is_file():
        raise FileNotFoundError("media-root and runtime python must exist")
    if args.world_size <= 0 or not 1024 <= (args.master_port or 0) <= 65535:
        raise ValueError("world-size must be positive and master-port must be 1024..65535")
    raw = load_json(args.config)
    strict, overfit = build_configs(raw, overfit_samples=args.overfit_samples,
                                    overfit_steps=args.overfit_steps,
                                    overfit_learning_rate=args.overfit_learning_rate,
                                    overfit_warmup_steps=args.overfit_warmup_steps)
    source_inventory = p0_source_inventory(raw, require_model_code=not args.dry_run)
    args.frozen_source_manifest_sha256 = hashlib.sha256(
        canonical_json_bytes(source_inventory)).hexdigest()
    split_step, final_step = strict["training"]["save_steps"], strict["training"]["max_steps"]
    if args.diagnostic_step is not None and args.diagnostic_step != split_step + 1:
        raise ValueError("P0-05 diagnostic step must be exactly the fork step plus one")
    selected = select_fixed_sources(args.train_manifest, args.overfit_samples)
    output = args.output_dir.resolve()
    paths = {"root": output, "strict_config": output / "inputs/strict_config.json",
             "overfit_config": output / "inputs/overfit_config.json",
             "overfit_manifest": output / "inputs/overfit_train.jsonl",
             "source_manifest": output / "inputs/p0_source_manifest.json",
             "control": output / "strict/control", "resumed": output / "strict/resumed",
             "overfit": output / "overfit/training",
             "control_copy_report": output / "evidence/control_checkpoint_vs_final.json",
             "resumed_copy_report": output / "evidence/resumed_checkpoint_vs_final.json",
             "resumed_omission_report": output / "evidence/resumed_final_export_omitted.json",
             "strict_report": output / "evidence/strict_resume.json",
             "diagnostic_report": output / "evidence/step_diagnostic_comparison.json",
             "overfit_report": output / "evidence/short_overfit.json"}
    plan = build_plan(args, paths, split_step=split_step, final_step=final_step)
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "run_id": args.run_id, "split_step": split_step,
                          "final_step": final_step, "diagnostic_step": args.diagnostic_step,
                          "fixed_sample_ids": [r["sample_id"] for r in selected],
                          "plan": plan}, indent=2))
        return 0
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output must not already exist: {output}")
    storage_root = (args.project_storage_root.resolve() if args.project_storage_root else output.parent)
    if not storage_root.is_dir():
        raise FileNotFoundError(storage_root)
    existing = tree_bytes(storage_root)
    free = shutil.disk_usage(storage_root).free
    operational_limit = args.project_storage_limit_bytes - 1_000_000_000
    if existing + args.reserved_additional_bytes >= operational_limit:
        raise RuntimeError(
            "Project storage admission failed: existing bytes plus requested reserve reaches "
            "the limit minus the 1 GB in-run guard"
        )
    if free < args.reserved_additional_bytes + 1_000_000_000:
        raise RuntimeError("Filesystem free-space admission failed")

    output.mkdir(parents=True)
    runtime_tmp = output / "runtime/tmp"
    runtime_cache = output / "runtime/cache"
    runtime_tmp.mkdir(parents=True)
    runtime_cache.mkdir(parents=True)
    archive = VerifiedArchive(output, args.run_id)
    write_new_json(paths["strict_config"], strict)
    write_new_json(paths["overfit_config"], overfit)
    write_new_json(paths["source_manifest"], source_inventory)
    if sha256_file(paths["source_manifest"]) != args.frozen_source_manifest_sha256:
        raise AssertionError("Frozen source manifest serialization differs from planned hash")
    paths["overfit_manifest"].parent.mkdir(parents=True, exist_ok=True)
    with paths["overfit_manifest"].open("x", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    inputs = {name: {"path": str(path.resolve()), "bytes": path.stat().st_size,
                     "sha256": sha256_file(path)}
              for name, path in (("requested_config", args.config), ("strict_config", paths["strict_config"]),
                                 ("overfit_config", paths["overfit_config"]),
                                 ("train_manifest", args.train_manifest), ("eval_manifest", args.eval_manifest),
                                 ("overfit_manifest", paths["overfit_manifest"]),
                                 ("source_manifest", paths["source_manifest"]))}
    env = dict(os.environ)
    env.update({"PYTHONPATH": str(ROOT / "src"), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1", "PYTHONHASHSEED": str(strict["training"].get("seed", 17)),
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "NCCL_ALGO": "Ring", "NCCL_PROTO": "Simple",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                "TMPDIR": str(runtime_tmp), "TMP": str(runtime_tmp), "TEMP": str(runtime_tmp),
                "XDG_CACHE_HOME": str(runtime_cache), "CUDA_CACHE_PATH": str(runtime_cache / "cuda"),
                "TRITON_CACHE_DIR": str(runtime_cache / "triton"),
                "TORCH_EXTENSIONS_DIR": str(runtime_cache / "extensions"),
                "TORCHINDUCTOR_CACHE_DIR": str(runtime_cache / "inductor")})
    runner = StageRunner(root=output, cwd=ROOT, env=env, storage_root=storage_root,
                         storage_limit=args.project_storage_limit_bytes)
    exit_code = 1
    failure: BaseException | None = None
    precision_receipts: dict[str, Any] = {}
    try:
        run_manifest = {
            "status": "running", "run_id": args.run_id, "purpose": "P0_05_ENGINEERING_NOT_QUALITY",
            "started_utc": datetime.now(timezone.utc).isoformat(), "inputs": inputs,
            "source_inventory": source_inventory,
            "frozen_source_manifest_sha256": args.frozen_source_manifest_sha256,
            "world_size": args.world_size, "split_step": split_step, "final_step": final_step,
            "overfit_steps": args.overfit_steps, "overfit_samples": args.overfit_samples,
            "determinism": {key: env[key] for key in ("PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG",
                                                       "NCCL_ALGO", "NCCL_PROTO")},
            "storage_admission": {"root": str(storage_root), "existing_bytes": existing,
                                  "reserved_additional_bytes": args.reserved_additional_bytes,
                                  "limit_bytes": args.project_storage_limit_bytes,
                                  "operational_limit_bytes": operational_limit,
                                  "filesystem_free_bytes": free}, "plan": plan}
        if args.diagnostic_step is not None:
            run_manifest["diagnostic_step"] = args.diagnostic_step
        write_new_json(output / "RUN_MANIFEST.json", run_manifest)
        runner.run("gpu_gate", plan["gpu_gate"], 300)
        assert_source_inventory_unchanged(source_inventory, raw)
        runner.run("strict_control", plan["control"], args.training_timeout_seconds)
        precision_receipts["control"] = verify_bf16_forward_receipts(
            paths["control"], args.world_size)
        runner.run("control_copy_verify", plan["control_copy_verify"],
                   args.verification_timeout_seconds, one_gpu=True)
        if not load_json(paths["control_copy_report"]).get("exact_model_copy_proven"):
            raise RuntimeError("Control final model is not equal to its final checkpoint")
        early_pruned = []
        if not args.keep_duplicates:
            archived = archive.archive_tree_before_prune(
                "control-final-copy", paths["control"] / "final")
            early_pruned.append({**remove_tree(paths["control"] / "final", output),
                                 "archive_fragment": archived})
        assert_source_inventory_unchanged(source_inventory, raw)
        runner.run("strict_resume", plan["resume"], args.training_timeout_seconds)
        precision_receipts["resumed"] = verify_bf16_forward_receipts(
            paths["resumed"], args.world_size)
        if not args.keep_duplicates:
            write_new_json(paths["resumed_omission_report"],
                           verify_resumed_final_export_omission(
                               paths["resumed"], final_step, args.world_size))
        if args.diagnostic_step is not None:
            runner.run("diagnostic_compare", plan["diagnostic_compare"],
                       args.verification_timeout_seconds)
        if args.keep_duplicates:
            runner.run("resumed_copy_verify", plan["resumed_copy_verify"],
                       args.verification_timeout_seconds, one_gpu=True)
            if not load_json(paths["resumed_copy_report"]).get("exact_model_copy_proven"):
                raise RuntimeError("Resumed final model is not equal to its final checkpoint")
        else:
            write_new_json(output / "evidence/MODEL_COPIES_PRUNED.json", {
                "status": "control_final_copy_verified_archived_then_pruned",
                "scope": ("The control final model was compared with checkpoint-20, fully archived "
                          "and then removed locally. The resumed final export was intentionally "
                          "omitted; its complete checkpoint-20 remains for direct strict comparison."),
                "control_copy_report_sha256": sha256_file(paths["control_copy_report"]),
                "resumed_final_omission_report_sha256": sha256_file(
                    paths["resumed_omission_report"]),
                "removed": early_pruned})
        runner.run("strict_verify", plan["strict_verify"], args.verification_timeout_seconds, one_gpu=True)
        strict_report = load_json(paths["strict_report"])
        if not strict_report.get("strict_tensorwise_resume_equivalence_proven"):
            raise RuntimeError("Strict resume report did not pass")

        if not args.keep_duplicates:
            # The fork checkpoint is unique evidence: it is the exact optimizer
            # and eight-rank RNG state from which the resumed branch started.
            # Keep it through the final archive.  The resumed final checkpoint
            # is directly equal to the control final checkpoint and may be
            # pruned only after its own durable fragment archive is read back.
            resumed_checkpoint = paths["resumed"] / f"checkpoint-{final_step}"
            archived = archive.archive_tree_before_prune(
                "resumed-final-checkpoint", resumed_checkpoint)
            removed = {**remove_tree(resumed_checkpoint, output),
                       "archive_fragment": archived}
            write_new_json(output / "evidence/STRICT_COMPARED_COPY_PRUNED.json", {
                "status": "strictly_compared_copy_removed_after_verified_fragment_archive",
                "strict_report_sha256": sha256_file(paths["strict_report"]), "removed": removed,
                "retained_fork_checkpoint": str(paths["control"] / f"checkpoint-{split_step}"),
                "retained_control_final_checkpoint": str(paths["control"] / f"checkpoint-{final_step}")})

        assert_source_inventory_unchanged(source_inventory, raw)
        runner.run("short_overfit", plan["overfit"], args.training_timeout_seconds, one_gpu=True)
        precision_receipts["overfit"] = verify_bf16_forward_receipts(paths["overfit"], 1)
        runner.run("short_overfit_verify", plan["overfit_verify"], args.verification_timeout_seconds,
                   one_gpu=True)
        overfit_report = load_json(paths["overfit_report"])
        if not overfit_report.get("short_overfit_engineering_gate_passed"):
            raise RuntimeError("Short-overfit report did not pass")
        assert_source_inventory_unchanged(source_inventory, raw)
        write_new_json(output / "P0_05_ACCEPTED.json", {
            "status": "p0_05_engineering_gates_passed", "run_id": args.run_id,
            "strict_tensorwise_resume_equivalence_proven": True,
            "strict_world_size": args.world_size, "strict_fork_step": split_step,
            "strict_final_step": final_step, "short_overfit_engineering_gate_passed": True,
            "overfit_samples": args.overfit_samples, "overfit_steps": args.overfit_steps,
            "strict_report_sha256": sha256_file(paths["strict_report"]),
            "overfit_report_sha256": sha256_file(paths["overfit_report"]),
            "control_copy_report_sha256": sha256_file(paths["control_copy_report"]),
            "resumed_final_export": (
                {"status": "verified_duplicate_copy_present_before_archive",
                 "copy_report_sha256": sha256_file(paths["resumed_copy_report"])}
                if args.keep_duplicates else
                {"status": "intentionally_omitted_complete_checkpoint_compared",
                 "omission_report_sha256": sha256_file(paths["resumed_omission_report"])}),
            "bf16_forward_receipts": precision_receipts,
            "quality_result": False,
            "archive_status": "pending_verified_archive_in_runner_finalization",
            "interpretation": "Engineering checkpoint/recovery and memorization sanity checks only."})
        exit_code = 0
    except BaseException as exc:
        failure = exc
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        details = traceback.format_exc()
        print(details, file=sys.stderr, flush=True)
        write_new_json(output / "FAILURE.json", {
            "status": "p0_05_failed", "run_id": args.run_id,
            "exception_type": type(exc).__name__, "exception": str(exc),
            "traceback": details, "automatic_retry": False,
        })
    finally:
        runner.stop()
        write_new_json(output / "EXIT_STATUS.json", {
            "exit_code": exit_code, "run_id": args.run_id, "stages": runner.stages,
            "finished_utc": datetime.now(timezone.utc).isoformat(), "automatic_retry": False})
        try:
            receipt = archive.finalize(exit_code=exit_code)
            if exit_code == 0:
                cleanup_verified_archived_weights(output, receipt)
        except BaseException as archive_error:
            print("ARCHIVE/CLEANUP FAILURE:\n" + traceback.format_exc(), file=sys.stderr, flush=True)
            if not (output / "ARCHIVE_FAILURE.json").exists():
                write_new_json(output / "ARCHIVE_FAILURE.json", {
                    "status": "archive_or_cleanup_failed", "run_id": args.run_id,
                    "experiment_exit_code": exit_code,
                    "exception_type": type(archive_error).__name__,
                    "exception": str(archive_error),
                })
            exit_code = exit_code or 74
    if failure is not None and exit_code == 0:
        raise AssertionError("A failed experiment cannot return success")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.master_port is None:
        args.master_port = 29000 + int(hashlib.sha256(args.run_id.encode()).hexdigest()[:4], 16) % 5000
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        return run(args)
    except BaseException:
        traceback.print_exc()
        return 130 if isinstance(sys.exc_info()[1], KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
