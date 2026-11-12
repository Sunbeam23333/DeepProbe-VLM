"""Lazy-loaded Triton implementation; do not import from CPU-only entry points."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _weighted_spatial_pool_kernel(
    states, weights, output,
    SIDE: tl.constexpr, CHANNELS: tl.constexpr, GRID: tl.constexpr,
    BLOCK_PIXELS: tl.constexpr, BLOCK_CHANNELS: tl.constexpr,
):
    frame = tl.program_id(0)
    spatial_bin = tl.program_id(1)
    channel_block = tl.program_id(2)
    bin_y, bin_x = spatial_bin // GRID, spatial_bin % GRID
    y0, x0 = bin_y * SIDE // GRID, bin_x * SIDE // GRID
    y1 = ((bin_y + 1) * SIDE + GRID - 1) // GRID
    x1 = ((bin_x + 1) * SIDE + GRID - 1) // GRID
    width = x1 - x0
    pixel = tl.arange(0, BLOCK_PIXELS)
    channel = channel_block * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    inside = pixel < (y1 - y0) * width
    spatial = (y0 + pixel // width) * SIDE + x0 + pixel % width
    index = frame * SIDE * SIDE + spatial
    w = tl.load(weights + index, mask=inside, other=0.0).to(tl.float32)
    x = tl.load(states + index[:, None] * CHANNELS + channel[None, :],
                mask=inside[:, None] & (channel[None, :] < CHANNELS), other=0.0).to(tl.float32)
    numerator = tl.sum(x * w[:, None], axis=0)
    denominator = tl.sum(w, axis=0)
    pooled = numerator / denominator
    destination = (frame * GRID * GRID + spatial_bin) * CHANNELS + channel
    tl.store(output + destination, pooled, mask=channel < CHANNELS)


def launch(states, weights, grid, side):
    frames, _, channels = states.shape
    longest_bin = max(((i + 1) * side + grid - 1) // grid - i * side // grid
                      for i in range(grid))
    block_pixels = triton.next_power_of_2(longest_bin * longest_bin)
    # Bound each program's reduction tile as large bins approach a whole image.
    block_channels = 32 if block_pixels <= 256 else 16
    warps = 4 if block_pixels * block_channels <= 8192 else 8
    launch_grid = (frames, grid * grid, triton.cdiv(channels, block_channels))
    output = torch.empty((frames, grid * grid, channels), device=states.device, dtype=torch.float32)
    # Explicitly honor the input device instead of assuming the caller's current
    # device. torch/triton share that device's current stream.
    with torch.cuda.device(states.device):
        compiled = _weighted_spatial_pool_kernel[launch_grid](
            states, weights, output, side, channels, grid, block_pixels, block_channels,
            num_warps=warps, enable_fp_fusion=False,
        )
    metadata = {
        "backend": "triton", "status": "kernel_launch_enqueued",
        "kernel_function": "_weighted_spatial_pool_kernel",
        "compiled_kernel_name": getattr(compiled, "name", None),
        "triton_version": triton.__version__, "device": str(states.device),
        "launch_grid": list(launch_grid), "block_pixels": block_pixels,
        "block_channels": block_channels, "num_warps": warps,
        "accumulation_dtype": "float32", "output_dtype": "float32",
        "enable_fp_fusion": False, "fallback_used": False,
    }
    return output, metadata
