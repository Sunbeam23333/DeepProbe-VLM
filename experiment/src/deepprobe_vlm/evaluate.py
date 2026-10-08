"""Auditable reference QA evaluation; not a substitute for official scorers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import warnings

import torch
from transformers import AutoTokenizer, AutoImageProcessor

from .data import MultimodalCollator, VideoManifestDataset
from .manifest import load_manifest
from .modeling import DeepProbeVLM


def normalise(text):
    return " ".join(text.strip().lower().split())


def score_answer(prediction, answer, choices=None):
    if choices:
        # Accept only a single explicit choice (option prefixes optional), not
        # the first arbitrary A-Z letter in free prose.
        match = re.fullmatch(r"\s*(?:answer\s*:\s*)?\(?([A-Z])\)?[.\s]*", prediction, re.I)
        selected = match.group(1).upper() if match else None
        if selected is None:
            matches = [chr(65+i) for i, choice in enumerate(choices) if normalise(choice) == normalise(prediction)]
            selected = matches[0] if len(matches) == 1 else None
        return {"metric": "strict_mcq_accuracy", "parsed_prediction": selected,
                "correct": int(selected == answer.strip().upper())}
    return {"metric": "normalised_exact_match", "parsed_prediction": normalise(prediction),
            "correct": int(normalise(prediction) == normalise(answer))}


def synchronise():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def validate_output_paths(output: Path) -> None:
    for path in (output, output.with_suffix(".summary.json"), output.with_suffix(".trace.json")):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite evaluation artifact: {path}")


def check_provenance(checkpoint: Path, records, *, allow_unverified=False):
    path = checkpoint / "training_provenance.json"
    if not path.is_file():
        if not allow_unverified:
            raise ValueError("Checkpoint lacks training_provenance.json; use --allow-unverified-provenance only for an explicitly unverified imported checkpoint")
        warnings.warn("UNVERIFIED CHECKPOINT: no training/tuning source provenance; no clean-held-out evaluation claim is supported", RuntimeWarning)
        return "UNVERIFIED_MISSING_TRAINING_PROVENANCE"
    provenance = json.loads(path.read_text())
    for name in ("training_sources", "tuning_sources"):
        values = provenance.get(name) if isinstance(provenance, dict) else None
        if not isinstance(values, list) or not all(isinstance(x, str) and x for x in values):
            raise ValueError(f"Invalid checkpoint provenance: {name} must be a list of source IDs")
    overlap = set(provenance["training_sources"]) & {x["source_video_id"] for x in records}
    if overlap:
        raise ValueError(f"Evaluation sources occurred in training: {sorted(overlap)[:8]}")
    tuned_test_overlap = set(provenance["tuning_sources"]) & {
        x["source_video_id"] for x in records if x["split"] == "test"}
    if tuned_test_overlap:
        raise ValueError(f"Test sources occurred in training-time validation/tuning: {sorted(tuned_test_overlap)[:8]}")
    return "passed_recorded_training_source_check"


def record_metadata(record):
    return {
        "duration": record.get("duration", record.get("duration_category", record.get("duration_group"))),
        "task": record.get("task", record.get("task_type", record.get("question_type"))),
        "question_type": record.get("question_type"), "task_type": record.get("task_type"),
        "duration_group": record.get("duration_group"), "duration_category": record.get("duration_category"),
    }


def summarise_metrics(rows):
    result = {}
    for metric in sorted({row["metric"] for row in rows}):
        group = [row for row in rows if row["metric"] == metric]
        result[metric] = {"num_examples": len(group), "num_correct": sum(x["correct"] for x in group),
                          "score": sum(x["correct"] for x in group) / len(group)}
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True, help="New JSONL predictions path")
    p.add_argument("--media-root")
    p.add_argument("--max-samples", type=int)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--profile", action="store_true", help="Save CPU/CUDA trace for first example")
    p.add_argument("--no-cache", action="store_true", help="Explicit uncached reference comparison")
    p.add_argument("--allow-unverified-provenance", action="store_true",
                   help="Explicitly allow imported checkpoints missing training provenance; results marked unverified")
    cli = p.parse_args(argv)
    output = Path(cli.output)
    validate_output_paths(output)
    media_root = Path(cli.media_root) if cli.media_root else Path(cli.manifest).resolve().parent
    records = load_manifest(cli.manifest, media_root=media_root)
    if cli.max_samples is not None:
        if cli.max_samples <= 0:
            raise ValueError("max-samples must be positive")
        records = records[:cli.max_samples]
    checkpoint = Path(cli.checkpoint)
    contamination_check = check_provenance(checkpoint, records,
                                            allow_unverified=cli.allow_unverified_provenance)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = DeepProbeVLM.from_pretrained(checkpoint, torch_dtype=dtype).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", trust_remote_code=False)
    processor = AutoImageProcessor.from_pretrained(checkpoint / "image_processor", use_fast=False)
    dataset = VideoManifestDataset(records, processor, model.deep_config.num_frames, media_root)
    collator = MultimodalCollator(tokenizer, num_visual_tokens=model.deep_config.num_visual_tokens,
                                 max_text_tokens=model.deep_config.max_text_tokens, include_answer=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with output.open("x", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            synchronise()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            batch = {k: v.to(device) for k, v in collator([dataset[index]]).items()}
            synchronise()
            prepared = time.perf_counter()
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            if cli.profile and index == 0:
                with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
                    tokens = model.greedy_generate(**batch, max_new_tokens=cli.max_new_tokens,
                                                   eos_token_id=tokenizer.eos_token_id, use_cache=not cli.no_cache)
                prof.export_chrome_trace(str(output.with_suffix(".trace.json")))
            else:
                tokens = model.greedy_generate(**batch, max_new_tokens=cli.max_new_tokens,
                                               eos_token_id=tokenizer.eos_token_id, use_cache=not cli.no_cache)
            synchronise()
            finished = time.perf_counter()
            prediction = tokenizer.decode(tokens[0], skip_special_tokens=True)
            row = {"sample_id": record["sample_id"], "source_video_id": record["source_video_id"],
                   "split": record["split"], "prediction": prediction, "answer": record["answer"],
                   **score_answer(prediction, record["answer"], record.get("choices")),
                   "prepare_seconds": prepared-started, "model_seconds": finished-prepared,
                   "end_to_end_seconds": finished-started, "generated_tokens": tokens.shape[1],
                   "latency_s": finished-started, "model_name": model.deep_config.language_model,
                   **record_metadata(record),
                   "cache": not cli.no_cache, "num_loops": model.deep_config.num_loops,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                   "peak_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                   "timing_status": "profiled_not_benchmark" if cli.profile and index == 0 else "raw_cold_or_mixed_no_warmup",
                   "contamination_check": contamination_check}
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
    summary = {"status": "REFERENCE_SCORER_NOT_OFFICIAL_BENCHMARK", "num_examples": len(rows),
               "by_metric": summarise_metrics(rows),
               "manifest_sha256": hashlib.sha256(Path(cli.manifest).read_bytes()).hexdigest(),
               "checkpoint": str(checkpoint), "device": str(device), "torch": torch.__version__,
               "cuda": torch.version.cuda, "contamination_check": contamination_check,
               "timing_note": "Includes cold/mixed runs; do not present as warmed p50/p95 speed claim."}
    with output.with_suffix(".summary.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
