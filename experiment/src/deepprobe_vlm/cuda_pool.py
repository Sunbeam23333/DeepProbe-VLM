"""Explicit inference-only CUDA weighted spatial pooling.

Importing this module never imports Triton or initializes CUDA. The accelerated
entry point has no fallback: unsupported input, missing Triton and launch errors
fail visibly. Gate, sigmoid, clamping and projection remain with the caller.
"""
from __future__ import annotations

import importlib
import math
import threading

import torch


_LAUNCH = threading.local()


def _validate_inputs(states, weights, grid, *, require_cuda):
    if not isinstance(states, torch.Tensor) or not isinstance(weights, torch.Tensor):
        raise TypeError("states and weights must be torch tensors")
    if type(grid) is not int or grid < 1:
        raise ValueError("grid must be a positive integer")
    if states.ndim != 3 or any(d <= 0 for d in states.shape):
        raise ValueError("states must be a nonempty [N,S,C] tensor")
    if tuple(weights.shape) != (*states.shape[:2], 1):
        raise ValueError("weights must have shape [N,S,1]")
    side = math.isqrt(states.shape[1])
    if side * side != states.shape[1] or grid > side:
        raise ValueError("S must be square and grid must not exceed its side")
    if states.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("states must be BF16 or FP32")
    if weights.dtype != torch.float32:
        raise TypeError("weights must be FP32 after the original gate/sigmoid/clamp")
    if states.device != weights.device:
        raise ValueError("states and weights must be on the same device")
    if not states.is_contiguous() or not weights.is_contiguous():
        raise ValueError("states and weights must be contiguous; no implicit copy")
    if require_cuda and states.device.type != "cuda":
        raise ValueError("the Triton pooling backend requires CUDA tensors")
    if require_cuda:
        longest_bin = max(((i + 1) * side + grid - 1) // grid - i * side // grid
                          for i in range(grid))
        if longest_bin * longest_bin > 1024:
            raise ValueError("Triton pooling supports at most 1024 pixels per bin; no fallback")
    return side


def reference_weighted_spatial_pool(states, weights, grid):
    """The existing separable FP32 matmul path, isolated for comparison.

Weights must already be finite and strictly positive, as produced by the
caller's unchanged sigmoid/FP32/clamp_min(1e-6) path. This reference preserves
autograd and is also usable in CPU numerical contract tests.
    """
    side = _validate_inputs(states, weights, grid, require_cuda=False)
    with torch.autocast(device_type=states.device.type, enabled=False):
        values = (states.float() * weights).transpose(1, 2).reshape(states.shape[0], -1, side, side)
        spatial_weights = weights.transpose(1, 2).reshape(states.shape[0], 1, side, side)
        bins = values.new_zeros((grid, side))
        for index in range(grid):
            start = index * side // grid
            end = ((index + 1) * side + grid - 1) // grid
            bins[index, start:end] = 1.0 / (end - start)
        numerator = torch.matmul(torch.matmul(bins, values), bins.T)
        denominator = torch.matmul(torch.matmul(bins, spatial_weights), bins.T)
        return (numerator / denominator).flatten(2).transpose(1, 2)


def weighted_spatial_pool(states, weights, grid):
    """Return CUDA FP32 [N,G*G,C] pooling without changing spatial bin semantics.

This entry point is inference-only and requires an explicit no_grad or
inference_mode context. It neither detaches training tensors nor supplies a
backward implementation. Finite, strictly positive weights are an input
precondition; checking their values would add a host/device synchronization to
every call. The benchmark validates this precondition outside its timings.
Each spatial bin is bounded to 1024 pixels to keep the reduction tile bounded.
    """
    _LAUNCH.last = None
    if torch.is_grad_enabled():
        raise RuntimeError("Triton pooling is inference-only; use an explicit no_grad/inference_mode context")
    side = _validate_inputs(states, weights, grid, require_cuda=True)
    if states.requires_grad or weights.requires_grad:
        raise RuntimeError("Triton pooling does not accept tensors requiring gradients")
    try:
        backend = importlib.import_module("._triton_pool", __package__)
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError("explicit Triton pooling requested but Triton is unavailable; no fallback") from error
    result, launch = backend.launch(states, weights, grid, side)
    _LAUNCH.last = launch
    return result


def last_launch_metadata():
    """Metadata for the last launch enqueued by this thread, not a speed claim.

Callers must synchronize CUDA before interpreting the launch as completed.
Kernel execution is independently observable in the optional profiler trace.
    """
    value = getattr(_LAUNCH, "last", None)
    return None if value is None else dict(value)
