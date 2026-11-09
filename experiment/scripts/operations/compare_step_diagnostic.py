#!/usr/bin/env python3
"""Compare the first resumed optimizer step's per-rank diagnostic receipts.

This read-only report locates the earliest observed mismatch.  Tensor
fingerprints are intentionally bounded, so an equal report is not a proof of
full tensor equality; the P0-05 strict checkpoint comparator remains the gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


PURPOSE = "P0_05_FIRST_DIVERGENT_STEP_DIAGNOSTIC_NOT_ACCEPTANCE"
EMBEDDING_BOUNDARY_PURPOSE = "P0_05_EMBEDDING_GRADIENT_BOUNDARY_NOT_ACCEPTANCE"
EMBEDDING_GRADIENTS = ("multimodal_output", "lookup_output", "local_embedding_weight_rows")


def _at(value: dict, *path: str):
    for component in path:
        value = value[component]
    return value


def _microbatches(value: dict, key: str):
    return [item[key] for item in value["microbatches"]]


def _embedding_gradients(value: dict, key: str):
    return [item["embedding_gradient_boundary"]["gradients"][key]
            for item in value["microbatches"]]


def _embedding_boundary(value: dict, key: str):
    return [item["embedding_gradient_boundary"][key]
            for item in value["microbatches"]]


def _validate_tensor_hash(value: Any, *, where: str) -> None:
    if not isinstance(value, dict) or not isinstance(value.get("dtype"), str) or not value["dtype"]:
        raise ValueError(f"Missing or invalid {where} tensor dtype")
    shape = value.get("shape")
    if not isinstance(shape, list) or not shape or any(type(size) is not int or size < 0 for size in shape):
        raise ValueError(f"Missing or invalid {where} tensor shape")
    if type(value.get("bytes")) is not int or value["bytes"] <= 0:
        raise ValueError(f"Missing or invalid {where} tensor byte length")
    if not isinstance(value.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"]):
        raise ValueError(f"Missing or invalid {where} tensor SHA-256")


def _validate_embedding_boundaries(receipt: dict, *, branch: str, rank: int) -> None:
    batches = receipt.get("microbatches")
    if not isinstance(batches, list) or not batches:
        raise ValueError(f"{branch} rank {rank} has no diagnostic microbatches")
    for index, batch in enumerate(batches):
        where = f"{branch} rank {rank} microbatch {index}"
        boundary = batch.get("embedding_gradient_boundary") if isinstance(batch, dict) else None
        if (not isinstance(boundary, dict) or boundary.get("schema_version") != 1
                or boundary.get("purpose") != EMBEDDING_BOUNDARY_PURPOSE):
            raise ValueError(f"{where} is missing a valid embedding gradient boundary receipt")
        count = boundary.get("selected_token_count")
        if type(count) is not int or count <= 0:
            raise ValueError(f"{where} has no selected embedding token rows")
        _validate_tensor_hash(boundary.get("selected_token_ids"), where=f"{where} selected_token_ids")
        selected = boundary["selected_token_ids"]
        if selected["dtype"] != "torch.int64" or selected["shape"] != [count] or selected["bytes"] != 8 * count:
            raise ValueError(f"{where} selected token hash does not match its row count")
        gradients = boundary.get("gradients")
        if not isinstance(gradients, dict):
            raise ValueError(f"{where} is missing embedding gradients")
        for name in EMBEDDING_GRADIENTS:
            _validate_tensor_hash(gradients.get(name), where=f"{where} {name}")
        _validate_tensor_hash(boundary.get("post_backward_embedding_weight_rows"),
                              where=f"{where} post_backward_embedding_weight_rows")
        if gradients["lookup_output"]["shape"] != gradients["multimodal_output"]["shape"]:
            raise ValueError(f"{where} embedding activation gradient shapes differ")
        local = gradients["local_embedding_weight_rows"]
        reduced = boundary["post_backward_embedding_weight_rows"]
        if local["shape"] != reduced["shape"] or local["dtype"] != reduced["dtype"] or local["shape"][0] != count:
            raise ValueError(f"{where} embedding weight gradient rows are inconsistent")
        synchronized = boundary.get("post_backward_ddp_synchronized")
        if type(synchronized) is not bool:
            raise ValueError(f"{where} has no Boolean DDP synchronization flag")
        if index == len(batches) - 1 and not synchronized:
            raise ValueError(f"{where} final embedding weight gradient was not DDP synchronized")


PHASES = [
    ("before_forward_model", lambda r: _at(r, "events", "before_forward", "model")),
    ("before_forward_optimizer", lambda r: _at(r, "events", "before_forward", "optimizer")),
    ("before_forward_scheduler", lambda r: _at(r, "events", "before_forward", "scheduler")),
    ("before_forward_rng", lambda r: _at(r, "events", "before_forward", "rng_sha256_before_forward")),
    ("actual_sample_ids", lambda r: _microbatches(r, "sample_ids")),
    ("actual_input_tensor_hashes", lambda r: _microbatches(r, "input_tensors")),
    ("batch_rng_before_forward", lambda r: _microbatches(r, "rng_sha256_before_forward")),
    ("per_rank_loss", lambda r: _microbatches(r, "loss")),
    ("embedding_token_rows", lambda r: _embedding_boundary(r, "selected_token_ids")),
    ("embedding_token_count", lambda r: _embedding_boundary(r, "selected_token_count")),
    ("embedding_ddp_sync", lambda r: _embedding_boundary(r, "post_backward_ddp_synchronized")),
    ("multimodal_embedding_gradient", lambda r: _embedding_gradients(r, "multimodal_output")),
    ("lookup_embedding_gradient", lambda r: _embedding_gradients(r, "lookup_output")),
    ("local_embedding_weight_rows", lambda r: _embedding_gradients(r, "local_embedding_weight_rows")),
    ("post_backward_embedding_weight_rows", lambda r: _embedding_boundary(
        r, "post_backward_embedding_weight_rows")),
    ("gradient_before_clip", lambda r: _at(r, "events", "after_backward_before_clip")),
    ("gradient_after_clip", lambda r: _at(r, "events", "after_clip_before_optimizer")),
    ("after_optimizer_model", lambda r: _at(r, "events", "after_optimizer_before_scheduler", "model")),
    ("after_optimizer_adam", lambda r: _at(r, "events", "after_optimizer_before_scheduler", "optimizer")),
    ("after_optimizer_rng", lambda r: _at(r, "events", "after_optimizer_before_scheduler", "rng_sha256")),
    ("after_scheduler", lambda r: _at(r, "events", "after_scheduler", "scheduler")),
]


def first_difference(left: Any, right: Any, path: str = "") -> dict[str, Any] | None:
    if type(left) is not type(right):
        return {"path": path, "control": repr(left)[:200], "resumed": repr(right)[:200]}
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else key
            if key not in left or key not in right:
                return {"path": child, "control": "present" if key in left else "missing",
                        "resumed": "present" if key in right else "missing"}
            difference = first_difference(left[key], right[key], child)
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return {"path": path + ".length", "control": len(left), "resumed": len(right)}
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            difference = first_difference(a, b, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if left != right:
        return {"path": path, "control": repr(left)[:200], "resumed": repr(right)[:200]}
    return None


def compare(control: Path, resumed: Path, *, step: int, world_size: int) -> dict[str, Any]:
    if step <= 0 or world_size <= 0:
        raise ValueError("Positive optimizer step and world size required")
    findings = []
    for rank in range(world_size):
        relative = Path("diagnostics") / f"optimizer-step-{step}" / f"rank-{rank}.json"
        a = json.loads((control / relative).read_text(encoding="utf-8"))
        b = json.loads((resumed / relative).read_text(encoding="utf-8"))
        for branch, receipt in (("control", a), ("resumed", b)):
            if (receipt.get("schema_version") != 1 or receipt.get("purpose") != PURPOSE
                    or receipt.get("status") != "complete"
                    or receipt.get("target_optimizer_step") != step
                    or receipt.get("rank") != rank or receipt.get("world_size") != world_size):
                raise ValueError(f"Invalid or incomplete {branch} step diagnostic for rank {rank}")
            _validate_embedding_boundaries(receipt, branch=branch, rank=rank)
        first = None
        for index, (phase, select) in enumerate(PHASES):
            difference = first_difference(select(a), select(b))
            if difference:
                first = {"phase_index": index, "phase": phase, **difference}
                break
        findings.append({"rank": rank, "first_difference": first})
    diverged = [item for item in findings if item["first_difference"] is not None]
    earliest = min(diverged, key=lambda item: item["first_difference"]["phase_index"]) if diverged else None
    return {"schema_version": 1, "purpose": "P0_05_STEP_DIAGNOSTIC_COMPARISON_NOT_ACCEPTANCE",
            "status": "observed_difference" if diverged else "no_observed_difference",
            "target_optimizer_step": step, "world_size": world_size,
            "earliest_observed_difference": earliest,
            "rank_findings": findings,
            "limitation": "Large model/Adam/gradient tensors have sampled exact bytes and reductions, not complete hashes."}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--resumed", type=Path, required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--world-size", type=int, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    result = compare(args.control, args.resumed, step=args.step, world_size=args.world_size)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
