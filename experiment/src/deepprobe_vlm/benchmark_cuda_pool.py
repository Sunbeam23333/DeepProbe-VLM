"""Opt-in CUDA microbenchmark for the weighted-pooling kernel, not model latency.

The default invocation prints the protocol without loading torch or Triton.
Only --execute --output NEW_DIRECTORY performs CUDA work. Inputs are synthetic
with the specified model-shaped dimensions; no quality or end-to-end speedup
is inferred. Profiling, when requested, is a separate pass after all timings.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import time
from datetime import datetime, timezone


PROTOCOL = "deepprobe-cuda-weighted-pool-micro-v1"


def _write(path, value):
    # Reuse the established exclusive, fsynced publication implementation.
    from .benchmark_latency import write_json
    write_json(path, value)


def _summary(values):
    values = sorted(values)
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("invalid measured times")
    point = (len(values) - 1) * .95
    low, high = math.floor(point), math.ceil(point)
    return {"count": len(values), "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "p95": values[low] + (values[high] - values[low]) * (point - low),
            "min": values[0], "max": values[-1]}


def _validate_settings(args):
    if args.dtype not in ("bfloat16", "float32") or type(args.profile) is not bool:
        raise ValueError("unsupported dtype or non-Boolean profile setting")
    for key in ("frames", "side", "channels", "grid", "warmup", "repeats"):
        if type(getattr(args, key)) is not int or getattr(args, key) < 1:
            raise ValueError(f"{key} must be a positive integer")
    if args.grid > args.side:
        raise ValueError("grid must not exceed the patch-grid side")
    if args.frames * args.side * args.side * args.channels > 64 * 1024 * 1024:
        raise ValueError("microbenchmark input exceeds 64M elements")
    if args.side > 128 or args.grid > 16 or args.warmup > 1000 or args.repeats > 10000:
        raise ValueError("microbenchmark setting exceeds its explicit safety bound")
    longest_bin = max(((i + 1) * args.side + args.grid - 1) // args.grid - i * args.side // args.grid
                      for i in range(args.grid))
    if longest_bin * longest_bin > 1024:
        raise ValueError("Triton pooling supports at most 1024 pixels per bin")
    return {
        "protocol": PROTOCOL, "shape": [args.frames, args.side * args.side, args.channels],
        "patch_grid_side": args.side, "pool_grid": args.grid, "states_dtype": args.dtype,
        "weights_dtype": "float32", "seed": 17,
        "input_scope": "synthetic model-shaped states; no checkpoint loaded",
        "weights": "sigmoid of seeded synthetic gate logits in states dtype, then FP32 clamp_min(1e-6)",
        "warmup_calls_per_backend": args.warmup, "measured_calls_per_backend": args.repeats,
        "pair_order": "reference,triton on even repeats; triton,reference on odd repeats",
        "numerical_gate": {"rtol": 3e-5, "atol": 3e-5},
        "timed_scope": "weighted pooling only, including output allocation; gate and projector excluded",
        "cuda_events": "same device/current stream, synchronized after every call",
        "wall_time": "call plus event recording and completion synchronization",
        "warmup_and_first_triton_call_excluded": True, "profiler_in_timing": False,
        "separate_profile_requested": args.profile, "num_inference_devices": 1,
        "end_to_end_model_latency_measured": False, "quality_measured": False,
        "hardware_admission_owned_by_caller": True,
    }


def _source_pins():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("benchmark_cuda_pool.py", "cuda_pool.py", "_triton_pool.py", "benchmark_latency.py")}


def execute(args, settings):
    output = args.output
    if output is None:
        raise ValueError("--execute requires a fresh --output directory")
    if any(p.is_symlink() for p in (output.absolute(), *output.absolute().parents)):
        raise ValueError("output must not contain symlink components")
    output.mkdir(parents=True, exist_ok=False)
    source_pins = _source_pins()
    _write(output / "INTENT.json", {"protocol": settings, "source_sha256": source_pins,
           "utc": datetime.now(timezone.utc).isoformat(), "automatic_retry": False})
    rows = []
    try:
        import torch
        from .cuda_pool import weighted_spatial_pool, reference_weighted_spatial_pool, last_launch_metadata
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("microbenchmark measures one process/device, not DDP throughput")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no CPU fallback")
        if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 CUDA support is required")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        props = torch.cuda.get_device_properties(device)
        runtime = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                   "triton": importlib.metadata.version("triton"), "device": str(device),
                   "gpu_name": props.name, "gpu_uuid": str(getattr(props, "uuid", "unavailable")),
                   "compute_capability": [props.major, props.minor], "total_memory_bytes": props.total_memory,
                   "visible_devices": torch.cuda.device_count(), "world_size": 1,
                   "tf32": False, "profiler_enabled_during_timing": False,
                   "allocation_and_hardware_isolation_verified_by_runner": False}
        _write(output / "RUNTIME.json", runtime)
        dtype = getattr(torch, args.dtype)
        with torch.inference_mode():
            states = torch.randn(settings["shape"], dtype=dtype, device=device)
            logits = torch.randn((*settings["shape"][:2], 1), dtype=dtype, device=device)
            weights = logits.sigmoid().float().clamp_min(1e-6)
            del logits
            if not bool(torch.isfinite(states).all()) or not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
                raise ValueError("synthetic inputs violate the finite positive-weight contract")
            def input_pins():
                result = {}
                for name, tensor in (("states", states), ("weights", weights)):
                    raw = tensor.detach().view(torch.uint8).cpu().numpy().tobytes()
                    result[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                                    "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                return result
            before_inputs = input_pins()
            _write(output / "INPUTS.json", {"scope": settings["input_scope"], "seed": 17,
                   "payloads_saved": False, "same_resident_inputs_for_both_backends": True,
                   "hashes_outside_timings": True, "tensors": before_inputs})
            funcs = {"reference": lambda: reference_weighted_spatial_pool(states, weights, args.grid),
                     "triton": lambda: weighted_spatial_pool(states, weights, args.grid)}
            expected = funcs["reference"]()
            torch.cuda.synchronize(device)
            first_started = time.perf_counter()
            actual = funcs["triton"]()
            torch.cuda.synchronize(device)
            first_seconds = time.perf_counter() - first_started
            launch = last_launch_metadata()
            if launch is None or launch["backend"] != "triton" or launch["fallback_used"]:
                raise RuntimeError("no explicit Triton launch evidence")
            error = (actual - expected).abs()
            numerical = {"output_shape": list(actual.shape), "output_dtype": str(actual.dtype),
                         "max_abs_error": error.max().item(),
                         "max_relative_error_with_1e_minus8_floor": (error / expected.abs().clamp_min(1e-8)).max().item(),
                         "rmse": torch.mean((actual - expected).square()).sqrt().item(),
                         "finite": bool(torch.isfinite(actual).all()),
                         **settings["numerical_gate"],
                         "passed": bool(torch.allclose(actual, expected, **settings["numerical_gate"]))}
            _write(output / "NUMERICAL.json", numerical)
            torch.testing.assert_close(actual, expected, **settings["numerical_gate"])
            _write(output / "KERNEL_LAUNCH.json", {**launch, "cuda_synchronized_after_call": True,
                   "first_call_seconds_including_possible_jit": first_seconds,
                   "first_call_is_steady_state_measurement": False})
            del expected, actual, error
            for _ in range(args.warmup):
                for backend in ("reference", "triton"):
                    value = funcs[backend]()
                    del value
            torch.cuda.synchronize(device)
            with (output / "measurements.jsonl").open("x", encoding="utf-8") as handle:
                for repeat in range(args.repeats):
                    order = ("reference", "triton") if repeat % 2 == 0 else ("triton", "reference")
                    for order_index, backend in enumerate(order):
                        torch.cuda.synchronize(device)
                        baseline_allocated = torch.cuda.memory_allocated(device)
                        baseline_reserved = torch.cuda.memory_reserved(device)
                        torch.cuda.reset_peak_memory_stats(device)
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        began = time.perf_counter()
                        start.record()
                        value = funcs[backend]()
                        end.record()
                        end.synchronize()
                        wall_seconds = time.perf_counter() - began
                        milliseconds = start.elapsed_time(end)
                        row = {"backend": backend, "repeat": repeat, "order_index": order_index,
                               "cuda_event_ms": milliseconds, "synchronized_call_seconds": wall_seconds,
                               "baseline_allocated_bytes": baseline_allocated,
                               "baseline_reserved_bytes": baseline_reserved,
                               "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                               "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
                        if not all(math.isfinite(x) and x > 0 for x in (milliseconds, wall_seconds)):
                            raise RuntimeError("invalid nonpositive/nonfinite measured duration")
                        del value
                        rows.append(row)
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
            profile = {"requested": args.profile, "performed": False}
            if args.profile:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
                    for backend in ("reference", "triton"):
                        with torch.profiler.record_function("deepprobe.pool_micro." + backend):
                            value = funcs[backend]()
                            torch.cuda.synchronize(device)
                            del value
                trace = output / "pool.trace.json"
                prof.export_chrome_trace(str(trace))
                data = json.loads(trace.read_text())
                names = [e.get("name", "") for e in data.get("traceEvents", []) if e.get("cat") == "kernel"]
                matched = sorted({name for name in names if "weighted_spatial_pool_kernel" in name})
                profile = {"requested": True, "performed": True, "included_in_measurements": False,
                           "cuda_kernel_events": len(names), "triton_kernel_names": matched,
                           "actual_triton_kernel_event_present": bool(matched),
                           "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest()}
                operators = []
                for item in prof.key_averages(group_by_input_shape=True):
                    operators.append({"name": item.key, "count": item.count,
                        "self_cpu_time_total_us": item.self_cpu_time_total,
                        "self_device_time_total_us": getattr(item, "self_device_time_total", 0),
                        "device_time_total_us": getattr(item, "device_time_total", 0),
                        "input_shapes": item.input_shapes})
                _write(output / "PROFILE.json", {**profile, "operators": operators})
                if not matched:
                    raise RuntimeError("requested profile contains no actual Triton pooling kernel event")
            if input_pins() != before_inputs:
                raise RuntimeError("pooling benchmark changed its synthetic input bytes")
            _write(output / "INPUTS_UNCHANGED.json", {"verified": True, "tensors": before_inputs,
                   "checked_after_timings_and_optional_profile": True})
        summaries = {}
        for backend in ("reference", "triton"):
            subset = [r for r in rows if r["backend"] == backend]
            if len(subset) != args.repeats:
                raise RuntimeError("incomplete measured backend")
            summaries[backend] = {"cuda_event_ms": _summary([r["cuda_event_ms"] for r in subset]),
                "synchronized_call_seconds": _summary([r["synchronized_call_seconds"] for r in subset]),
                "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in subset),
                "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in subset)}
        if _source_pins() != source_pins:
            raise RuntimeError("benchmark executable source changed during execution")
        summary = {"status": "CUDA_POOL_MICROBENCHMARK_COMPLETE", "protocol": settings,
                   "numerical": numerical, "backends": summaries, "profile": profile,
                   "source_sha256": source_pins, "quantile_method": "R7 linear interpolation",
                   "timings_are_operator_microbenchmarks": True,
                   "model_speedup_or_quality_claim": False,
                   "completed_utc": datetime.now(timezone.utc).isoformat()}
        _write(output / "SUMMARY.json", summary)
        return summary
    except Exception as error:
        _write(output / "FAILURE.json", {"error_type": type(error).__name__, "error": str(error),
               "durable_measurements": len(rows), "completed_summary": False, "automatic_retry": False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--side", type=int, default=27)
    parser.add_argument("--channels", type=int, default=1152)
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    settings = _validate_settings(args)
    if not args.execute:
        print(json.dumps({"status": "MICROBENCHMARK_PLAN_ONLY", "settings": settings,
                          "cuda_used": False, "model_loaded": False}, indent=2))
        return 0
    execute(args, settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
