"""Opt-in, one-optimizer-step evidence for locating a resume divergence.

The normal training path never constructs these objects.  This trace reads
already-loaded tensors without changing them or drawing random numbers.  It
records exact input bytes and loss values, while large parameter/optimizer
tensors get bounded exact byte samples plus whole-tensor reductions.  Those
summaries locate a difference; they are not a tensorwise equality proof.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
from typing import Any

import numpy as np
import torch
from transformers import TrainerCallback


SAMPLE_ID_KEY = "__deepprobe_diagnostic_sample_ids__"
_WINDOW = 8
_FULL_SAMPLE_MAX_ELEMENTS = 4096
_EMBEDDING_GRADIENT_TAP_ATTRIBUTE = "_deepprobe_step_embedding_gradient_tap"
_MAX_EMBEDDING_GRADIENT_CAPTURE_BYTES = 32 * 1024 * 1024
_DDP_LOGGING_FIELDS = (
    "backend_name", "rank", "world_size", "bucket_cap_bytes", "bucket_sizes",
    "rebuilt_bucket_sizes", "has_rebuilt_buckets", "find_unused_parameters",
    "gradient_as_bucket_view", "static_graph",
)


def ddp_bucket_snapshot(model) -> dict[str, Any]:
    """Best-effort bucket layout from the DDP wrapper used by training_step.

    The logging API is private and this receipt is diagnostic only. An API
    change must never turn a normal optimizer step into a failed run.
    """
    if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return {"status": "not_ddp"}
    try:
        raw = model._get_ddp_logging_data()
        if not isinstance(raw, dict):
            raise TypeError("DDP logging data is not a dictionary")
        result: dict[str, Any] = {"status": "available"}
        for key in _DDP_LOGGING_FIELDS:
            value = raw.get(key)
            if isinstance(value, (bool, int, str)):
                result[key] = value
        python_rebuilt = getattr(model, "_has_rebuilt_buckets", None)
        if isinstance(python_rebuilt, bool):
            result["python_has_rebuilt_buckets"] = python_rebuilt
        logged_rebuilt = result.get("has_rebuilt_buckets")
        if type(logged_rebuilt) is int and logged_rebuilt in (0, 1):
            rebuilt = bool(logged_rebuilt)
        elif type(logged_rebuilt) is bool:
            rebuilt = logged_rebuilt
        else:
            rebuilt = python_rebuilt if isinstance(python_rebuilt, bool) else None
        active_key = ("rebuilt_bucket_sizes" if rebuilt else "bucket_sizes") if rebuilt is not None else None
        result["active_bucket_source"] = active_key if active_key in result else None
        result["active_bucket_sizes"] = result.get(active_key) if active_key else None
        return result
    except Exception as exc:
        return {"status": "unavailable", "error_type": type(exc).__name__}


def _json_number(value: float | int) -> float | int | str:
    if isinstance(value, int):
        return value
    number = float(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "+inf" if number > 0 else "-inf"
    return number


def _rng_sha256() -> str:
    """Hash the four RNG families used by this Trainer rank, without drawing."""
    digest = hashlib.sha256()
    digest.update(pickle.dumps(random.getstate(), protocol=4))
    digest.update(pickle.dumps(np.random.get_state(), protocol=4))
    digest.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        digest.update(torch.cuda.get_rng_state(torch.cuda.current_device()).cpu().numpy().tobytes())
    return digest.hexdigest()


def _without_rng_change(action):
    before = _rng_sha256()
    result = action()
    if _rng_sha256() != before:
        raise RuntimeError("Step diagnostic changed a training RNG state")
    return result, before


def _full_tensor_sha256(tensor: torch.Tensor) -> dict[str, Any]:
    """Hash exact bytes of an input tensor, including its dtype and shape."""
    value = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
            "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def _sample_spans(numel: int) -> list[tuple[int, int]]:
    if numel <= _FULL_SAMPLE_MAX_ELEMENTS:
        return [(0, numel)] if numel else []
    midpoint = numel // 2
    return [(0, _WINDOW), (midpoint - _WINDOW // 2, midpoint + _WINDOW // 2),
            (numel - _WINDOW, numel)]


def summarize_tensors(named_tensors: list[tuple[str, torch.Tensor]]) -> dict[str, Any]:
    """Small exact windows and whole-tensor sum/L2 for every named tensor.

    Samples and scalar reductions are copied to the host in batches per
    device, so a large model does not incur a host transfer per parameter.
    Noncontiguous tensors are reported without forcing a full-size copy.
    """
    entries: dict[str, dict[str, Any]] = {}
    samples_by_device: dict[str, list[tuple[str, torch.Tensor]]] = {}
    scalars_by_device: dict[str, list[tuple[str, str, torch.Tensor]]] = {}
    total_elements = 0
    for name, tensor in named_tensors:
        if name in entries:
            raise ValueError(f"Duplicate diagnostic tensor name: {name}")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Diagnostic tensor expected for {name}")
        value = tensor.detach()
        numel = value.numel()
        total_elements += numel
        entry: dict[str, Any] = {
            "dtype": str(value.dtype), "shape": list(value.shape), "numel": numel,
            "sample_policy": "complete" if numel <= _FULL_SAMPLE_MAX_ELEMENTS
                             else "first_middle_last_8",
        }
        entries[name] = entry
        if numel and value.layout == torch.strided and (
                torch.is_floating_point(value) or torch.is_complex(value)):
            # These reductions are read-only and do not allocate a full tensor.
            # CUDA summaries are diagnostic fingerprints, not exact equality.
            device = str(value.device)
            scalars_by_device.setdefault(device, []).extend((
                (name, "sum", value.sum(dtype=torch.float64) if not torch.is_complex(value)
                 else value.real.sum(dtype=torch.float64)),
                (name, "l2_norm", torch.linalg.vector_norm(value)),
            ))
        if value.layout != torch.strided or not value.is_contiguous():
            entry["sample_policy"] = "unsupported_noncontiguous_or_sparse"
            entry["sample_sha256"] = None
            continue
        flat = value.reshape(-1)
        spans = _sample_spans(numel)
        entry["sampled_elements"] = sum(end - start for start, end in spans)
        if not spans:
            entry["sample_sha256"] = hashlib.sha256(b"").hexdigest()
        else:
            for start, end in spans:
                part = flat[start:end].view(torch.uint8)
                samples_by_device.setdefault(str(part.device), []).append((name, part))
    hashers = {name: hashlib.sha256() for name, entry in entries.items()
               if entry.get("sampled_elements", 0) > 0}
    for parts in samples_by_device.values():
        packed = torch.cat([part for _, part in parts]).cpu().numpy().tobytes()
        offset = 0
        for name, part in parts:
            size = part.numel()
            hashers[name].update(packed[offset:offset + size])
            offset += size
        if offset != len(packed):
            raise AssertionError("Diagnostic sample packing length differs")
    for name, hasher in hashers.items():
        entries[name]["sample_sha256"] = hasher.hexdigest()
    for items in scalars_by_device.values():
        packed = torch.stack([scalar.to(dtype=torch.float64) for _, _, scalar in items])
        for (name, key, _), number in zip(items, packed.cpu().tolist(), strict=True):
            entries[name][key] = _json_number(number)
    return {"tensor_count": len(entries), "total_elements": total_elements, "tensors": entries}


def _named_model_tensors(model: torch.nn.Module) -> list[tuple[str, torch.Tensor]]:
    values = [("parameter." + name, value) for name, value in model.named_parameters()]
    values.extend(("buffer." + name, value) for name, value in model.named_buffers())
    return values


def _optimizer_snapshot(model: torch.nn.Module, optimizer) -> dict[str, Any]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = []
    states: list[tuple[str, torch.Tensor]] = []
    scalar_states: dict[str, Any] = {}
    for group_index, group in enumerate(optimizer.param_groups):
        parameter_names = []
        for slot, parameter in enumerate(group["params"]):
            name = names.get(id(parameter))
            if name is None:
                raise RuntimeError(f"Optimizer parameter {group_index}:{slot} lacks a model name")
            parameter_names.append(name)
            state = optimizer.state.get(parameter, {})
            for key, value in sorted(state.items()):
                state_name = f"group{group_index}.slot{slot}.{name}.{key}"
                if isinstance(value, torch.Tensor):
                    states.append((state_name, value))
                elif isinstance(value, (bool, int, float, str)) or value is None:
                    scalar_states[state_name] = _json_number(value) if isinstance(value, float) else value
                else:
                    raise TypeError(f"Unsupported optimizer state type for {state_name}")
        groups.append({"index": group_index, "parameter_names": parameter_names,
                       "lr": _json_number(group["lr"]),
                       "weight_decay": _json_number(group.get("weight_decay", 0.0)),
                       "betas": [_json_number(value) for value in group.get("betas", ())],
                       "eps": _json_number(group.get("eps", 0.0))})
    return {"param_groups": groups, "scalar_states": scalar_states,
            "tensor_states": summarize_tensors(states)}


def _gradient_snapshot(model: torch.nn.Module) -> dict[str, Any]:
    parameters = list(model.named_parameters())
    present = [(name, parameter.grad) for name, parameter in parameters
               if parameter.grad is not None]
    return {"missing_names": [name for name, parameter in parameters
                              if parameter.requires_grad and parameter.grad is None],
            "gradients": summarize_tensors(present)}


class EmbeddingGradientTap:
    """Capture a traced microbatch's small embedding-boundary gradients.

    The two activation hooks locate a difference before embedding lookup
    backward.  The leaf-parameter hook observes the local embedding gradient
    before DDP's reduction; the ordinary post-backward gradient is captured
    separately.  Hooks return ``None`` and never replace a training gradient.
    """

    _ACTIVATIONS = ("multimodal_output", "lookup_output")
    _LOCAL_ROWS = "local_embedding_weight_rows"

    @staticmethod
    def supports(model: torch.nn.Module) -> bool:
        language = getattr(model, "language_model", None)
        return (callable(getattr(language, "get_input_embeddings", None))
                and callable(getattr(model, "prepare_embeddings", None)))

    def __init__(self, model: torch.nn.Module, input_ids: torch.Tensor):
        if not self.supports(model):
            raise TypeError("Embedding gradient tap requires a DeepProbe-style model")
        if not isinstance(input_ids, torch.Tensor) or input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("Embedding gradient tap requires integer input_ids")
        self.model = model
        self.embedding = model.language_model.get_input_embeddings()
        weight = getattr(self.embedding, "weight", None)
        if not isinstance(weight, torch.nn.Parameter) or not weight.requires_grad:
            raise ValueError("Embedding gradient tap requires trainable lookup weights")
        self.token_ids = torch.unique(input_ids.detach().reshape(-1).to(device="cpu", dtype=torch.int64))
        if self.token_ids.numel() == 0:
            raise ValueError("Embedding gradient tap requires at least one token ID")
        self._saved: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []
        self._installed = False

    def install(self) -> None:
        if self._installed or hasattr(self.model, _EMBEDDING_GRADIENT_TAP_ATTRIBUTE):
            raise RuntimeError("An embedding gradient tap is already installed")
        setattr(self.model, _EMBEDDING_GRADIENT_TAP_ATTRIBUTE, self)
        try:
            self._handles.append(self.embedding.weight.register_hook(self._capture_local_rows))
        except BaseException:
            delattr(self.model, _EMBEDDING_GRADIENT_TAP_ATTRIBUTE)
            raise
        self._installed = True

    def capture_activation(self, label: str, value: torch.Tensor) -> None:
        if not self._installed or label not in self._ACTIVATIONS:
            raise RuntimeError(f"Unexpected embedding gradient tap boundary: {label}")
        if not isinstance(value, torch.Tensor) or not value.requires_grad:
            raise RuntimeError(f"Embedding boundary has no autograd path: {label}")
        self._handles.append(value.register_hook(
            lambda gradient, name=label: self._capture(name, gradient)))

    def _capture(self, label: str, gradient: torch.Tensor) -> None:
        if label in self._saved:
            raise RuntimeError(f"Embedding gradient boundary fired twice: {label}")
        byte_count = gradient.numel() * gradient.element_size()
        if byte_count > _MAX_EMBEDDING_GRADIENT_CAPTURE_BYTES:
            raise RuntimeError(f"Embedding gradient boundary is too large: {label} ({byte_count} bytes)")
        self._saved[label] = gradient.detach().clone()
        return None

    def _capture_local_rows(self, gradient: torch.Tensor) -> None:
        selected = gradient.index_select(0, self.token_ids.to(device=gradient.device))
        self._capture(self._LOCAL_ROWS, selected)
        return None

    def summary(self, *, sync_gradients: bool) -> dict[str, Any]:
        expected = {*self._ACTIVATIONS, self._LOCAL_ROWS}
        if set(self._saved) != expected:
            raise RuntimeError(f"Incomplete embedding gradient tap: missing={sorted(expected - set(self._saved))}")
        post_backward = self.embedding.weight.grad
        if post_backward is None:
            raise RuntimeError("Embedding lookup has no post-backward gradient")
        rows = post_backward.detach().index_select(0, self.token_ids.to(device=post_backward.device))
        return {
            "schema_version": 1,
            "purpose": "P0_05_EMBEDDING_GRADIENT_BOUNDARY_NOT_ACCEPTANCE",
            "selected_token_ids": _full_tensor_sha256(self.token_ids),
            "selected_token_count": self.token_ids.numel(),
            "maximum_capture_bytes_per_tensor": _MAX_EMBEDDING_GRADIENT_CAPTURE_BYTES,
            "gradients": {name: _full_tensor_sha256(self._saved[name]) for name in sorted(expected)},
            "post_backward_embedding_weight_rows": _full_tensor_sha256(rows),
            "post_backward_ddp_synchronized": bool(sync_gradients),
        }

    def close(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        if getattr(self.model, _EMBEDDING_GRADIENT_TAP_ATTRIBUTE, None) is self:
            delattr(self.model, _EMBEDDING_GRADIENT_TAP_ATTRIBUTE)
        self._saved.clear()
        self._installed = False


def _scheduler_snapshot(scheduler) -> dict[str, Any] | None:
    if scheduler is None:
        return None
    state = scheduler.state_dict()
    # The pinned cosine scheduler has only scalar/list metadata.  Omit a
    # large/unrecognized object instead of accidentally serializing tensors.
    result = {}
    for key, value in state.items():
        if isinstance(value, (bool, int, float, str)) or value is None:
            result[key] = _json_number(value) if isinstance(value, float) else value
        elif isinstance(value, (tuple, list)) and len(value) <= 32 and all(
                isinstance(item, (bool, int, float, str)) or item is None for item in value):
            result[key] = [_json_number(item) if isinstance(item, float) else item for item in value]
        else:
            result[key] = {"type": type(value).__name__, "omitted": True}
    return result


class DiagnosticCollator:
    """Attach sample IDs to a batch; the diagnostic Trainer removes them."""

    def __init__(self, collator):
        self.collator = collator

    def __call__(self, examples):
        batch = self.collator(examples)
        if SAMPLE_ID_KEY in batch:
            raise ValueError("Reserved diagnostic batch key already exists")
        batch[SAMPLE_ID_KEY] = [example["record"]["sample_id"] for example in examples]
        return batch


class FirstStepTrace:
    """One rank's step trace, written incrementally for failure forensics."""

    def __init__(self, output_dir: str | Path, *, target_step: int, rank: int,
                 world_size: int, resume_checkpoint: str | None):
        if target_step <= 0 or rank < 0 or rank >= world_size:
            raise ValueError("Invalid target step/rank/world size for diagnostic")
        self.target_step = target_step
        self.path = (Path(output_dir) / "diagnostics" / f"optimizer-step-{target_step}"
                     / f"rank-{rank}.json")
        self.record: dict[str, Any] = {
            "schema_version": 1,
            "purpose": "P0_05_FIRST_DIVERGENT_STEP_DIAGNOSTIC_NOT_ACCEPTANCE",
            "status": "not_started", "target_optimizer_step": target_step,
            "rank": rank, "world_size": world_size,
            "resume_checkpoint": resume_checkpoint,
            "events": {}, "microbatches": [],
            "summary_limit": "Large tensors use exact first/middle/last windows and full sum/L2; not exact equality",
        }
        self._started = False
        self._raw_losses: list[torch.Tensor] = []

    def should_trace(self, global_step: int) -> bool:
        return global_step == self.target_step - 1

    def _persist(self, status: str) -> None:
        self.record["status"] = status
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self._started and (self.path.exists() or self.path.is_symlink()):
            raise FileExistsError(f"Diagnostic output already exists: {self.path}")
        pending = self.path.with_suffix(".json.pending")
        with pending.open("w", encoding="utf-8") as stream:
            json.dump(self.record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        pending.replace(self.path)
        self._started = True

    def before_forward(self, *, state, model, optimizer, scheduler) -> None:
        if not self.should_trace(state.global_step):
            return
        if self._started:
            raise RuntimeError("Diagnostic target optimizer step started twice")

        def capture():
            return {"trainer_global_step_before": state.global_step,
                    "model": summarize_tensors(_named_model_tensors(model)),
                    "optimizer": _optimizer_snapshot(model, optimizer),
                    "scheduler": _scheduler_snapshot(scheduler)}

        event, rng = _without_rng_change(capture)
        event["rng_sha256_before_forward"] = rng
        self.record["events"]["before_forward"] = event
        self._persist("before_forward_recorded")

    def batch_before_forward(self, sample_ids: list[str], inputs: dict[str, Any],
                             ddp_bucket_before_forward: dict[str, Any] | None = None) -> None:
        if not self._started:
            raise RuntimeError("Diagnostic batch arrived before step-begin state capture")
        if not sample_ids or not all(isinstance(item, str) and item for item in sample_ids):
            raise ValueError("Diagnostic batch requires actual nonempty sample IDs")

        def capture():
            if any(not isinstance(value, torch.Tensor) for value in inputs.values()):
                raise TypeError("Diagnostic training inputs must be tensors after metadata removal")
            fields = {name: _full_tensor_sha256(value) for name, value in sorted(inputs.items())}
            digest = hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode())
            return {"sample_ids": list(sample_ids), "input_tensors": fields,
                    "input_batch_sha256": digest.hexdigest()}

        item, rng = _without_rng_change(capture)
        item["rng_sha256_before_forward"] = rng
        if ddp_bucket_before_forward is not None:
            item["ddp_bucket_before_forward"] = ddp_bucket_before_forward
        self.record["microbatches"].append(item)
        self._persist("input_recorded")

    def retain_raw_loss(self, loss: torch.Tensor) -> None:
        self._raw_losses.append(loss.detach())

    def after_backward(self, *, model, scaled_loss: torch.Tensor, sync_gradients: bool,
                       embedding_gradients: dict[str, Any] | None = None,
                       ddp_bucket_after_backward: dict[str, Any] | None = None) -> None:
        if not self.record["microbatches"]:
            raise RuntimeError("Diagnostic loss arrived before its input batch")
        batch = self.record["microbatches"][-1]
        if "loss" in batch:
            raise RuntimeError("Diagnostic microbatch loss was recorded twice")
        if len(self._raw_losses) != 1:
            raise RuntimeError("Expected one raw forward loss for diagnostic microbatch")

        def capture():
            return {"raw_forward_loss": _json_number(self._raw_losses.pop().item()),
                    "trainer_scaled_loss": _json_number(scaled_loss.detach().item())}

        batch["loss"], _ = _without_rng_change(capture)
        if embedding_gradients is not None:
            batch["embedding_gradient_boundary"] = embedding_gradients
        if ddp_bucket_after_backward is not None:
            batch["ddp_bucket_after_backward"] = ddp_bucket_after_backward
        if sync_gradients:
            gradients, rng = _without_rng_change(lambda: _gradient_snapshot(model))
            self.record["events"]["after_backward_before_clip"] = {
                "rng_sha256": rng, **gradients}
        self._persist("backward_recorded")

    def before_optimizer(self, *, model) -> None:
        if not self._started:
            raise RuntimeError("Optimizer hook arrived before diagnostic target step")
        gradients, rng = _without_rng_change(lambda: _gradient_snapshot(model))
        self.record["events"]["after_clip_before_optimizer"] = {"rng_sha256": rng, **gradients}
        self._persist("before_optimizer_recorded")

    def after_optimizer(self, *, model, optimizer) -> None:
        if "after_clip_before_optimizer" not in self.record["events"]:
            raise RuntimeError("Post-optimizer hook arrived without pre-optimizer evidence")

        def capture():
            return {"model": summarize_tensors(_named_model_tensors(model)),
                    "optimizer": _optimizer_snapshot(model, optimizer)}

        event, rng = _without_rng_change(capture)
        event["rng_sha256"] = rng
        self.record["events"]["after_optimizer_before_scheduler"] = event
        self._persist("after_optimizer_recorded")

    def after_step(self, *, state, scheduler) -> None:
        if state.global_step != self.target_step:
            raise RuntimeError("Diagnostic step-end did not reach the target optimizer step")
        self.record["events"]["after_scheduler"] = {
            "trainer_global_step_after": state.global_step,
            "scheduler": _scheduler_snapshot(scheduler)}
        self._persist("complete")


class FirstStepTraceCallback(TrainerCallback):
    def __init__(self, trace: FirstStepTrace):
        self.trace = trace

    def on_step_begin(self, args, state, control, **kwargs):
        if self.trace.should_trace(state.global_step):
            self.trace.before_forward(state=state, model=kwargs["model"],
                                      optimizer=kwargs["optimizer"],
                                      scheduler=kwargs["lr_scheduler"])
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self.trace.should_trace(state.global_step):
            self.trace.before_optimizer(model=kwargs["model"])
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        if self.trace.should_trace(state.global_step):
            self.trace.after_optimizer(model=kwargs["model"], optimizer=kwargs["optimizer"])
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.trace.target_step:
            self.trace.after_step(state=state, scheduler=kwargs["lr_scheduler"])
        return control
