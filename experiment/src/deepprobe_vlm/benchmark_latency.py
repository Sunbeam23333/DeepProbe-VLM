"""Opt-in, per-device development latency measurements; imports no GPU library by default.

Create a plan with ``freeze_plan`` before inspecting any measured output, then
use ``python -m deepprobe_vlm.benchmark_latency --plan ... --plan-sha256 ...``
to validate its schema. Only ``--execute --output NEW_DIRECTORY`` loads a model.
Scheduling, hardware admission and cross-run comparisons belong to the caller.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time

PROTOCOL = "deepprobe-development-latency-v1"
OPTIMIZATION_PROTOCOL = "deepprobe-development-latency-v2"
SETTINGS = {"sample_count": 32, "seed": 17, "warmup_requests_per_sample": 3,
            "measured_requests_per_sample": 10, "batch_size": 1,
            "max_new_tokens": 16, "dtype": "bfloat16", "decoding": "greedy_argmax",
            "include_answer": False, "audio": False, "subtitles": False,
            "sample_order": "sha256_seed_sample_id", "request_order": "sample_major",
            "input_reuse": "reprepare_every_request", "num_inference_devices": 1}
DEPENDENCIES = ("__init__.py", "benchmark_latency.py", "data.py", "evaluate.py", "manifest.py", "modeling.py")
OPTIMIZATION_DEPENDENCIES = DEPENDENCIES + ("cuda_pool.py", "_triton_pool.py")
PACKAGE = Path(__file__).resolve().parent
TIMINGS = ("prepare_seconds", "model_seconds", "postprocess_seconds", "request_seconds")
SCOPE = {
    "prepare_seconds": "media decode, image processing, tokenization, host-to-device transfer and synchronization",
    "model_seconds": "greedy_generate: vision encoding, prefill and autoregressive decode, with CUDA synchronization",
    "postprocess_seconds": "generated-token device-to-host transfer and tokenizer text decoding, with synchronization",
    "request_seconds": "sum of preparation, model execution and text postprocessing",
    "excluded": ["model/tokenizer/processor loading", "input byte hashing", "quality scoring", "artifact writing"],
    "ttft_measured": False, "isolated_vision_prefill_decode_measured": False,
    "serving_or_eight_gpu_throughput_measured": False,
    "memory": "absolute per-request CUDA allocated/reserved peaks including resident model and inputs; warmed allocator retained",
    "warmup": "fixed warmup per sample; no convergence/adaptive stopping claim",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def safe(path):
    path = Path(path).absolute()
    require(".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)), "unsafe or symlink path")
    return path


def inventory(root):
    root = safe(root)
    require(not any(path.is_symlink() for path in root.rglob("*")), "symlink in inventory")
    return [{"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha(safe(path))}
            for path in sorted(root.rglob("*")) if path.is_file()]


def pin(path):
    path = safe(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def verify_pin(item, root=None):
    path = Path(item["path"])
    if root is not None:
        require(not path.is_absolute() and ".." not in path.parts, "unsafe inventory path")
        path = Path(root) / path
    path = safe(path)
    require(path.is_file() and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "pinned bytes changed")
    return path


def select_records(records):
    """Selection depends on IDs and seed only, never labels, scores or speed."""
    require(len(records) >= SETTINGS["sample_count"], "development manifest needs at least 32 samples")
    require(len({r["sample_id"] for r in records}) == len(records), "duplicate sample IDs")
    require(all(r.get("split") in {"train", "validation"} for r in records), "only train/development validation inputs allowed")
    forbidden = {"videomme", "video-mme", "tempcompass", "mvbench"}
    require(all(str(r.get("dataset", "")).lower() not in forbidden for r in records), "benchmark inputs are outside development scope")
    require(not any(r.get(k) for r in records for k in ("audio", "audio_path", "subtitle", "subtitles", "subtitle_path")), "audio/subtitle inputs forbidden")
    return sorted(records, key=lambda r: (hashlib.sha256(f"17\0{r['sample_id']}".encode()).hexdigest(), r["sample_id"]))[:32]


def sample_pins(records):
    return [{"sample_id": r["sample_id"], "record_sha256": digest(r)} for r in select_records(records)]


def media_names(records):
    return sorted({name for row in records for name in ([row["video_path"]] if "video_path" in row else row["image_paths"])})


def comparison_identity(plan):
    """Only declared implementation/cache axes may differ within a comparison."""
    excluded = {"cache", "comparison_identity"}
    if plan.get("schema_version") == 2:
        excluded.add("implementation")
    return digest({k: v for k, v in plan.items() if k not in excluded})


def implementation_options(plan):
    """Legacy v1 never implicitly opts into an experimental execution path."""
    if plan["schema_version"] == 1:
        return {}
    value = plan.get("implementation")
    require(isinstance(value, dict) and set(value) == {"last_token_logits", "pool_backend"}
            and type(value["last_token_logits"]) is bool
            and isinstance(value["pool_backend"], str)
            and value["pool_backend"] in {"reference", "triton"}, "invalid inference implementation")
    return dict(value)


def validate_plan(plan):
    version = plan.get("schema_version")
    require(type(version) is int and version in {1, 2}, "unknown latency protocol")
    extra = {"implementation"} if version == 2 else set()
    require(set(plan) == {"schema_version", "protocol", "settings", "cache", "data_scope", "quality_score_based_selection",
            "manifest", "source_metadata", "media_root", "media_files", "samples", "checkpoint", "model_config",
            "source_files", "comparison_identity"} | extra, "unknown or missing plan field")
    require(plan["protocol"] == (PROTOCOL if version == 1 else OPTIMIZATION_PROTOCOL), "unknown latency protocol")
    implementation_options(plan)
    require(canonical(plan["settings"]) == canonical(SETTINGS), "fixed latency settings changed")
    require(type(plan["cache"]) is bool and plan["data_scope"] == "development_only" and
            plan["quality_score_based_selection"] is False, "invalid cache/scope/selection policy")
    require(len(plan["samples"]) == 32 and len({r["sample_id"] for r in plan["samples"]}) == 32, "fixed input set incomplete")
    for name in ("manifest", "source_metadata"):
        require(Path(plan[name]["path"]).is_absolute(), "input paths must be absolute")
    require(Path(plan["media_root"]).is_absolute() and Path(plan["checkpoint"]["path"]).is_absolute(), "roots must be absolute")
    dependencies = DEPENDENCIES if version == 1 else OPTIMIZATION_DEPENDENCIES
    require({r["path"] for r in plan["source_files"]} == set(dependencies) and
            len(plan["source_files"]) == len(dependencies), "complete executable dependency inventory required")
    for rows in (plan["source_files"], plan["media_files"], plan["checkpoint"]["files"]):
        require(rows and len({r["path"] for r in rows}) == len(rows), "empty or duplicate inventory")
        for row in rows:
            require(not Path(row["path"]).is_absolute() and ".." not in Path(row["path"]).parts, "unsafe inventory path")
            require(type(row["bytes"]) is int and row["bytes"] >= 0 and len(row["sha256"]) == 64 and
                    all(c in "0123456789abcdef" for c in row["sha256"]), "invalid byte/hash pin")
    required = {"deepprobe_config.json", "training_provenance.json", "projector.safetensors",
                "language_model/config.json", "vision_model/config.json"}
    require(required <= {r["path"] for r in plan["checkpoint"]["files"]}, "checkpoint inventory incomplete")
    require(plan["comparison_identity"] == comparison_identity(plan), "comparison identity changed")
    return plan


def freeze_plan(manifest, checkpoint, media_root, source_metadata, *, cache=True, implementation=None):
    """Hash local files and return an immutable-plan value; never load weights.

    Source metadata must declare scope=development, manifest_sha256,
    quality_score_based_selection=false and a nonempty source_description.
    This helper performs no writes; callers save its output before execution.
    """
    from .manifest import load_manifest
    manifest, checkpoint, media_root = safe(manifest), safe(checkpoint), safe(media_root)
    records = load_manifest(manifest, check_media=False)
    selected = select_records(records)
    plan = {"schema_version": 1, "protocol": PROTOCOL, "settings": dict(SETTINGS), "cache": cache,
            "data_scope": "development_only", "quality_score_based_selection": False,
            "manifest": pin(manifest), "source_metadata": pin(source_metadata), "media_root": str(media_root),
            "media_files": [{**pin(media_root / name), "path": name} for name in media_names(selected)],
            "samples": sample_pins(records), "checkpoint": {"path": str(checkpoint), "files": inventory(checkpoint)},
            "model_config": json.loads((checkpoint / "deepprobe_config.json").read_text()),
            "source_files": [{**pin(PACKAGE / name), "path": name} for name in DEPENDENCIES]}
    if implementation is not None:
        plan.update(schema_version=2, protocol=OPTIMIZATION_PROTOCOL, implementation=dict(implementation))
        plan["source_files"] = [{**pin(PACKAGE / name), "path": name} for name in OPTIMIZATION_DEPENDENCIES]
    plan["comparison_identity"] = comparison_identity(plan)
    validate_plan(plan)
    verify_metadata(plan)
    return plan


def verify_metadata(plan):
    meta = json.loads(verify_pin(plan["source_metadata"]).read_text())
    require(meta.get("scope") == "development" and meta.get("manifest_sha256") == plan["manifest"]["sha256"] and
            meta.get("quality_score_based_selection") is False and isinstance(meta.get("source_description"), str) and
            meta["source_description"].strip(), "development source metadata does not bind this manifest")
    return meta


def verify_inputs(plan):
    from .manifest import load_manifest
    validate_plan(plan)
    for row in plan["source_files"]:
        verify_pin(row, PACKAGE)
    verify_metadata(plan)
    records = load_manifest(verify_pin(plan["manifest"]), check_media=False)
    require(sample_pins(records) == plan["samples"], "fixed development sample set changed")
    selected = select_records(records)
    require(media_names(selected) == sorted(r["path"] for r in plan["media_files"]), "media inventory coverage changed")
    for row in plan["media_files"]:
        verify_pin(row, plan["media_root"])
    checkpoint = safe(plan["checkpoint"]["path"])
    require(inventory(checkpoint) == plan["checkpoint"]["files"], "checkpoint inventory or bytes changed")
    require(json.loads((checkpoint / "deepprobe_config.json").read_text()) == plan["model_config"], "model config changed")
    return selected


def measure_request(*, prepare, generate, decode, synchronize, reset_peak, memory, clock=time.perf_counter):
    """Measure one synchronized request; adapters return only Python output data."""
    synchronize()
    before = memory()
    reset_peak()
    started = clock()
    batch = prepare()
    synchronize()
    prepared = clock()
    tokens = generate(batch)
    synchronize()
    generated = clock()
    output = decode(tokens)
    synchronize()
    finished = clock()
    peak = memory()
    ids = output["generated_token_ids"]
    require(isinstance(ids, list) and 1 <= len(ids) <= SETTINGS["max_new_tokens"] and
            all(type(i) is int and i >= 0 for i in ids), "invalid actual generated token IDs")
    require(isinstance(output["prediction"], str), "prediction must be text")
    times = {"prepare_seconds": prepared-started, "model_seconds": generated-prepared,
             "postprocess_seconds": finished-generated, "request_seconds": finished-started}
    require(all(math.isfinite(v) and v >= 0 for v in times.values()) and times["model_seconds"] > 0, "invalid monotonic timing")
    require(all(type(v) is int and v >= 0 for v in [*before.values(), *peak.values()]), "invalid memory counters")
    return {**times, **output, "generated_tokens": len(ids), "memory": {
        "baseline_allocated_bytes": before["allocated"], "baseline_reserved_bytes": before["reserved"],
        "peak_allocated_bytes": peak["peak_allocated"], "peak_reserved_bytes": peak["peak_reserved"]}}


def stats(values):
    ordered = sorted(values)
    require(bool(ordered) and all(math.isfinite(v) for v in ordered), "empty or nonfinite measurements")
    index = (len(ordered)-1) * .95
    lo, hi = math.floor(index), math.ceil(index)
    return {"count": len(ordered), "mean": statistics.fmean(ordered), "median": statistics.median(ordered),
            "p95": ordered[lo] + (ordered[hi]-ordered[lo]) * (index-lo), "min": ordered[0], "max": ordered[-1]}


def summarize(rows, sample_ids, warmups=3, repeats=10):
    require(len(sample_ids) == len(set(sample_ids)), "duplicate summary sample IDs")
    groups = defaultdict(list)
    for row in rows:
        groups[row["sample_id"]].append(row)
    require(set(groups) == set(sample_ids), "missing or extra samples")
    measured, per_sample, quality = [], [], []
    for sid in sample_ids:
        group = groups[sid]
        expected = {(phase, index) for phase, count in (("warmup", warmups), ("measured", repeats)) for index in range(count)}
        require(len(group) == len(expected) and {(r["phase"], r["repeat_index"]) for r in group} == expected, "incomplete or duplicate trials")
        trials = sorted((r for r in group if r["phase"] == "measured"), key=lambda r: r["repeat_index"])
        measured.extend(trials)
        quality.append({k: trials[0][k] for k in ("sample_id", "prediction", "answer", "quality")})
        per_sample.append({"sample_id": sid, "timings": {k: stats([r[k] for r in trials]) for k in TIMINGS},
                           "generated_tokens": stats([r["generated_tokens"] for r in trials]),
                           "distinct_measured_token_sequences": len({tuple(r["generated_token_ids"]) for r in trials})})
    return {"measured_requests": len(measured), "warmup_requests_excluded": len(rows)-len(measured),
            "quantile_method": "linear interpolation at (n-1)*p, R7", "aggregation": "equal repetitions per sample; request observations are correlated",
            "timings": {k: stats([r[k] for r in measured]) for k in TIMINGS},
            "generated_tokens": stats([r["generated_tokens"] for r in measured]),
            "total_measured_generated_tokens": sum(r["generated_tokens"] for r in measured),
            "peak_allocated_bytes": max(r["memory"]["peak_allocated_bytes"] for r in measured),
            "peak_reserved_bytes": max(r["memory"]["peak_reserved_bytes"] for r in measured),
            "per_sample": per_sample, "first_measured_quality": quality,
            "quality_scope": "reference scorer diagnostics on first measured repeat of each development sample; not official benchmark accuracy",
            "quality_score_based_selection": False, "failed_or_missing_requests": 0}


def write_json(path, value):
    """Publish a complete, fsynced receipt exclusively; failed staging stays pending."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.pending-", delete=False) as out:
        pending = Path(out.name)
        out.write(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n")
        out.flush(); os.fsync(out.fileno())
    os.link(pending, path)  # Atomic publication fails if the destination exists.
    try:
        pending.unlink()
    except Exception:
        # Publication is already committed. A leftover pending hard link must
        # not turn a successful receipt into a caught execution failure.
        pass


def execute(plan, plan_sha, output, *, plan_bytes=None, profile=False):
    require(type(profile) is bool, "profile must be Boolean")
    plan_bytes = canonical(plan) if plan_bytes is None else plan_bytes
    require(hashlib.sha256(plan_bytes).hexdigest() == plan_sha and json.loads(plan_bytes) == plan, "execution plan bytes changed")
    output = safe(output)
    output.mkdir(exist_ok=False)
    with (output / "PLAN.json").open("xb") as out:
        out.write(plan_bytes); out.flush(); os.fsync(out.fileno())
    write_json(output / "INTENT.json", {"utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": plan_sha,
               "measurement_scope": SCOPE, "automatic_retry": False, "data_scope": "development_only"})
    completed_rows = []
    try:
        records = verify_inputs(plan)
        implementation = implementation_options(plan)
        checkpoint = Path(plan["checkpoint"]["path"])
        provenance = json.loads((checkpoint / "training_provenance.json").read_text())
        for key in ("training_sources", "tuning_sources"):
            require(isinstance(provenance.get(key), list) and all(isinstance(s, str) and s for s in provenance[key]), "invalid checkpoint training provenance")
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        import torch
        from transformers import AutoImageProcessor, AutoTokenizer
        from .data import MultimodalCollator, VideoManifestDataset
        from .evaluate import score_answer
        from .modeling import DeepProbeVLM
        require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA device required")
        require(int(os.environ.get("WORLD_SIZE", "1")) == 1, "runner measures one process/device, not distributed throughput")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.manual_seed(17); torch.cuda.manual_seed_all(17)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        model = DeepProbeVLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", local_files_only=True, trust_remote_code=False)
        processor = AutoImageProcessor.from_pretrained(checkpoint / "image_processor", local_files_only=True, use_fast=False)
        dataset = VideoManifestDataset(records, processor, model.deep_config.num_frames, plan["media_root"])
        collator = MultimodalCollator(tokenizer, num_visual_tokens=model.deep_config.num_visual_tokens,
                                     max_text_tokens=model.deep_config.max_text_tokens, include_answer=False)
        props = torch.cuda.get_device_properties(device)
        driver = Path("/proc/driver/nvidia/version")
        packages = {}
        for name in ("torch", "transformers", "tokenizers", "numpy", "pillow", "av", "safetensors", "huggingface-hub", "triton"):
            try: packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: packages[name] = None
        runtime = {"packages": packages, "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                   "gpu_name": props.name, "gpu_total_memory": props.total_memory, "compute_capability": [props.major, props.minor],
                   "gpu_uuid": str(getattr(props, "uuid", "unavailable")), "driver": driver.read_text() if driver.is_file() else "unavailable",
                   "visible_device_count": torch.cuda.device_count(), "measured_device": str(device),
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "pid": os.getpid(),
                   "python": os.sys.version, "torch_num_threads": torch.get_num_threads(),
                   "platform": platform.platform(), "machine": platform.machine(), "logical_cpu_count": os.cpu_count(),
                   "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                   "actual_parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
                   "tf32": False, "cudnn_benchmark": False, "profiler_enabled": False,
                   "checkpoint_training_provenance_sha256": sha(checkpoint / "training_provenance.json"),
                   "development_overlap_permitted_no_heldout_claim": True, "hardware_isolation_verified_by_runner": False}
        runtime.update(implementation=implementation or {"last_token_logits": False, "pool_backend": "reference"},
                       profile_diagnostic_requested=profile,
                       first_warmup_may_include_jit_compilation=implementation.get("pool_backend") == "triton")
        write_json(output / "RUNTIME.json", runtime)

        def memory():
            return {"allocated": torch.cuda.memory_allocated(device), "reserved": torch.cuda.memory_reserved(device),
                    "peak_allocated": torch.cuda.max_memory_allocated(device), "peak_reserved": torch.cuda.max_memory_reserved(device)}

        def decode(tokens):
            require(tokens.ndim == 2 and tokens.shape[0] == 1, "expected one generated token sequence")
            ids = tokens[0].detach().cpu().tolist()
            return {"generated_token_ids": ids, "prediction": tokenizer.decode(ids, skip_special_tokens=True)}

        with (output / "requests.jsonl").open("xb") as out, torch.inference_mode():
            for index, record in enumerate(records):
                prefix, suffix, answer = collator.tokenize_record(record)
                require(answer == [], "gold answer must not enter prompt")
                input_info = {"prompt_text_tokens": len(prefix)+len(suffix), "visual_tokens": model.deep_config.num_visual_tokens,
                              "prompt_sequence_tokens": len(prefix)+len(suffix)+model.deep_config.num_visual_tokens,
                              "eos_token_id": tokenizer.eos_token_id, "model_config": plan["model_config"]}
                def prepare():
                    example = dataset[index]
                    batch = {key: value.to(device) for key, value in collator([example]).items()}
                    input_info["tensor_shapes"] = {key: list(value.shape) for key, value in batch.items()}
                    input_info["tensor_dtypes"] = {key: str(value.dtype) for key, value in batch.items()}
                    return batch
                def generate(batch):
                    return model.greedy_generate(**batch, max_new_tokens=16, eos_token_id=tokenizer.eos_token_id,
                                                 use_cache=plan["cache"], **implementation)
                for phase, count in (("warmup", 3), ("measured", 10)):
                    for repeat in range(count):
                        row = measure_request(prepare=prepare, generate=generate, decode=decode,
                            synchronize=lambda: torch.cuda.synchronize(device),
                            reset_peak=lambda: torch.cuda.reset_peak_memory_stats(device), memory=memory)
                        row.update(sample_id=record["sample_id"], source_video_id=record["source_video_id"], split=record["split"],
                            phase=phase, repeat_index=repeat, record_sha256=digest(record), plan_sha256=plan_sha,
                            input=dict(input_info), implementation=runtime["implementation"], answer=record["answer"],
                            quality=score_answer(row["prediction"], record["answer"], record.get("choices")))
                        out.write(canonical(row)+b"\n"); out.flush(); os.fsync(out.fileno())
                        completed_rows.append(row)
        summary = summarize(completed_rows, [r["sample_id"] for r in records])
        summary.update(status="DEVELOPMENT_LATENCY_MEASUREMENTS_COMPLETE", plan_sha256=plan_sha,
            protocol=plan["protocol"], comparison_identity=plan["comparison_identity"], cache=plan["cache"],
            implementation=runtime["implementation"], measurement_scope=SCOPE,
            requests_sha256=sha(output / "requests.jsonl"), runtime_sha256=sha(output / "RUNTIME.json"),
            finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "SUMMARY.json", summary)
        if profile:
            # Separate diagnostic pass AFTER all unprofiled measurements. Its
            # failure cannot turn a published timing summary into a failed run.
            try:
                example = dataset[0]
                batch = {key: value.to(device) for key, value in collator([example]).items()}
                torch.cuda.synchronize(device)
                with torch.inference_mode(), torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=True, profile_memory=True) as prof:
                    tokens = model.greedy_generate(**batch, max_new_tokens=16, eos_token_id=tokenizer.eos_token_id,
                                                  use_cache=plan["cache"], profile_ranges=True, **implementation)
                    torch.cuda.synchronize(device)
                prof.export_chrome_trace(str(output / "profile.trace.json"))
                table = prof.key_averages(group_by_input_shape=True).table(sort_by="self_device_time_total", row_limit=80)
                (output / "profile.operators.txt").write_text(table, encoding="utf-8")
                write_json(output / "PROFILE.json", {"status": "DIAGNOSTIC_PROFILE_COMPLETE", "included_in_measurements": False,
                    "sample_id": records[0]["sample_id"], "generated_token_ids": decode(tokens)["generated_token_ids"],
                    "trace_sha256": sha(output / "profile.trace.json"), "operators_sha256": sha(output / "profile.operators.txt"),
                    "plan_sha256": plan_sha, "runtime_sha256": summary["runtime_sha256"],
                    "timing_summary_sha256": sha(output / "SUMMARY.json"),
                    "implementation": runtime["implementation"], "cache": plan["cache"],
                    "profile_ranges_enabled": True, "isolated_kernel_speedup_claimed": False})
            except Exception as error:
                write_json(output / "PROFILE_FAILURE.json", {"error_type": type(error).__name__, "error": str(error),
                    "timing_summary_preserved": True, "plan_sha256": plan_sha})
        return summary
    except Exception as error:
        write_json(output / "FAILURE.json", {"utc": datetime.now(timezone.utc).isoformat(), "error_type": type(error).__name__,
            "error": str(error), "completed_requests": len(completed_rows), "automatic_retry": False,
            "complete_summary_written": (output / "SUMMARY.json").is_file(), "plan_sha256": plan_sha})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", action="store_true", help="Separate first-sample CUDA diagnostic after unprofiled measurements")
    args = parser.parse_args(argv)
    require(sha(args.plan) == args.plan_sha256, "plan SHA changed")
    plan = validate_plan(json.loads(args.plan.read_text()))
    if not args.execute:
        require(not args.profile, "--profile requires --execute")
        print(json.dumps({"status": "PLAN_SCHEMA_VALID_NO_EXECUTION", "plan_sha256": args.plan_sha256,
                          "input_bytes_verified": False, "model_loaded": False, "gpu_used": False}))
        return 0
    require(args.output is not None, "--execute requires a fresh --output directory")
    execute(plan, args.plan_sha256, args.output, plan_bytes=args.plan.read_bytes(), profile=args.profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
