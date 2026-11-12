"""Structural checks for ordinary Trainer/DDP recovery checkpoints.

Transformers permits missing optimizer, scheduler, scaler and RNG files when
resuming. Our exact-resume entry points must reject that fallback. These cheap
checks do not load tensors, authenticate their contents, or prove numerical
resume equivalence; the native loaders and separate integrity/equivalence
audits still provide those checks.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import stat


def _require_state_file(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"Exact resume checkpoint is incomplete: missing {path.name}") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
        raise ValueError(
            f"Exact resume requires a nonempty regular state file: {path.name}")


def verify_resume_checkpoint(path: str | Path, *, world_size: int,
                             require_scaler: bool = False,
                             expected_max_steps: int | None = None) -> dict:
    """Reject incomplete native recovery state before any model is loaded.

    The accepted RNG layout is exactly the one written by the same-size
    ordinary Trainer/DDP job. Weight-only stage initialization does not call
    this function. ``expected_max_steps`` is optional for epoch-based training,
    where Trainer has not yet computed the effective update count.
    """
    if type(world_size) is not int or world_size < 1:
        raise ValueError("Exact resume requires a positive integer world_size")
    if type(require_scaler) is not bool:
        raise ValueError("Exact resume require_scaler must be boolean")
    if expected_max_steps is not None and (
            type(expected_max_steps) is not int or expected_max_steps < 1):
        raise ValueError("Exact resume expected_max_steps must be positive or None")
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Exact resume requires a regular checkpoint directory")
    match = re.fullmatch(r"checkpoint-([1-9][0-9]*)", root.name)
    if match is None:
        raise ValueError("Exact resume path must be a checkpoint-<positive-step> directory")

    state_path = root / "trainer_state.json"
    _require_state_file(state_path)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Exact resume requires valid trainer_state.json") from exc
    step = int(match.group(1))
    if (not isinstance(state, dict) or type(state.get("global_step")) is not int
            or state["global_step"] != step or type(state.get("max_steps")) is not int
            or state["max_steps"] < step):
        raise ValueError("Exact resume checkpoint directory/trainer_state step mismatch")
    if expected_max_steps is not None and state["max_steps"] != expected_max_steps:
        raise ValueError("Exact resume trainer_state max_steps differs from the training schedule")

    required = {"trainer_state.json", "optimizer.pt", "scheduler.pt"}
    if require_scaler:
        required.add("scaler.pt")
    rng_files = ({"rng_state.pth"} if world_size == 1 else
                 {f"rng_state_{rank}.pth" for rank in range(world_size)})
    observed_rng = {item.name for item in root.glob("rng_state*.pth")}
    if observed_rng != rng_files:
        raise ValueError(
            f"Exact resume requires the complete RNG state for world_size={world_size}; "
            f"expected={sorted(rng_files)}, observed={sorted(observed_rng)}")
    required.update(rng_files)
    for name in sorted(required):
        _require_state_file(root / name)
    return {
        "checkpoint": str(root), "optimizer_step": step,
        "max_steps": state["max_steps"], "world_size": world_size,
        "state_files": sorted(required),
        "validation_scope": "file_presence_and_trainer_state_structure_only",
    }
