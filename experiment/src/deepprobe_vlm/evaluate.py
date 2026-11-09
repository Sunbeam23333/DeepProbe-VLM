"""Auditable reference QA evaluation; not a substitute for official scorers."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import time
import tempfile
import warnings

import torch
from transformers import AutoTokenizer, AutoImageProcessor

from .data import MultimodalCollator, VideoManifestDataset
from .manifest import load_manifest
from .modeling import DeepProbeVLM


VIDEOMME_PROTOCOL = "videomme-899-no-subtitles-no-audio-v1"


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_videomme_subset(records):
    """Validate the predeclared 899-group subset, never select using answers."""
    groups = defaultdict(list)
    for row in records:
        match = re.fullmatch(r"videomme:(\d{3})-([123])", row["sample_id"])
        if not match or row.get("dataset") != "videomme" or row["split"] != "test":
            raise ValueError("Video-MME requires its original question IDs and test split")
        if len(row.get("choices", [])) != 4 or row["answer"] not in "ABCD":
            raise ValueError("Video-MME requires four choices and a canonical gold letter")
        if any(row.get(key) for key in ("subtitle", "subtitles", "subtitle_path", "audio", "audio_path")):
            raise ValueError("Subtitle/audio inputs are forbidden by this Video-MME protocol")
        groups[int(match.group(1))].append(row)
    expected_groups = set(range(1, 901)) - {490}
    if set(groups) != expected_groups or len(records) != 2697:
        raise ValueError("Video-MME declared coverage must be 899 groups / 2697 questions, excluding group 490")
    sources = set()
    for rows in groups.values():
        if len(rows) != 3 or {r["sample_id"][-1] for r in rows} != {"1", "2", "3"}:
            raise ValueError("Each Video-MME group must retain all three questions")
        ids = {row["source_video_id"] for row in rows}
        if len(ids) != 1 or sources & ids:
            raise ValueError("Video-MME group/source mapping must be one-to-one")
        sources |= ids
    durations = Counter(record_metadata(row)["duration"] for row in records)
    if durations != {"short": 900, "medium": 897, "long": 900}:
        raise ValueError(f"Unexpected fixed-subset duration coverage: {dict(durations)}")
    return {"num_questions": len(records), "num_video_groups": len(groups),
            "by_duration": dict(sorted(durations.items())), "excluded_video_groups": [490]}


def shard_records(records, shard_index, num_shards):
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid shard index/count")
    # Shard whole sources, retaining canonical manifest order within a shard.
    owners = {source: index % num_shards for index, source in
              enumerate(dict.fromkeys(row["source_video_id"] for row in records))}
    return [row for row in records if owners[row["source_video_id"]] == shard_index]


def shard_output(output, shard_index, num_shards):
    if num_shards == 1:
        return output
    return output.with_name(f"{output.stem}.rank-{shard_index:05d}-of-{num_shards:05d}{output.suffix}")


def normalise(text):
    return " ".join(text.strip().lower().split())


def score_answer(prediction, answer, choices=None):
    if choices:
        # Accept only a single explicit choice (option prefixes optional), not
        # the first arbitrary A-Z letter in free prose.
        match = re.fullmatch(r"\s*(?:answer\s*:\s*)?(?:([A-Z])|\(([A-Z])\))\.?\s*", prediction, re.I)
        selected = (match.group(1) or match.group(2)).upper() if match else None
        if selected is None:
            matches = [chr(65+i) for i, choice in enumerate(choices) if normalise(choice) == normalise(prediction)]
            selected = matches[0] if len(matches) == 1 else None
        if selected not in [chr(65+i) for i in range(len(choices))]:
            selected = None
        return {"metric": "strict_mcq_accuracy", "parsed_prediction": selected,
                "invalid_prediction": selected is None,
                "correct": int(selected == answer.strip().upper())}
    return {"metric": "normalised_exact_match", "parsed_prediction": normalise(prediction),
            "correct": int(normalise(prediction) == normalise(answer))}


def synchronise():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def validate_output_paths(output: Path) -> None:
    for path in (output, output.with_suffix(".summary.json"), output.with_suffix(".trace.json"),
                 output.with_suffix(".protocol.json")):
        if path.exists() or path.is_symlink():
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
                          "num_invalid": sum(bool(x.get("invalid_prediction")) for x in group),
                          "score": sum(x["correct"] for x in group) / len(group)}
    return result


def summarise_groups(rows, key):
    return {str(value): summarise_metrics([row for row in rows if row.get(key) == value])
            for value in sorted({row.get(key) for row in rows}, key=str)}


def write_json_once(path, value):
    """A completed receipt is immutable and may only be reused byte-for-byte."""
    encoded = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if path.is_symlink():
        raise ValueError(f"Symlink output is not allowed: {path}")
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError(f"Existing receipt differs: {path}")
        return
    # Link a fully flushed temporary file into place: a killed writer cannot
    # leave a truncated protocol/completion receipt that blocks safe resume.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-receipt-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_prediction_rows(output, records, protocol_sha256, *, recover_tail=False):
    """Validate a durable prefix; torn last writes are saved before recovery."""
    if output.is_symlink():
        raise ValueError(f"Symlink predictions are forbidden: {output}")
    if not output.exists():
        return []
    payload = output.read_bytes()
    if payload and not payload.endswith(b"\n"):
        if not recover_tail:
            raise ValueError("Predictions have an incomplete trailing line")
        boundary = payload.rfind(b"\n") + 1
        tail = payload[boundary:]
        recovery = output.with_name(output.name + ".incomplete-tail-" + hashlib.sha256(tail).hexdigest() + ".bin")
        if recovery.exists():
            if recovery.read_bytes() != tail:
                raise ValueError("Prediction recovery receipt mismatch")
        else:
            with recovery.open("xb") as handle:
                handle.write(tail)
                handle.flush()
                os.fsync(handle.fileno())
        with output.open("r+b") as handle:
            handle.truncate(boundary)
            handle.flush()
            os.fsync(handle.fileno())
        payload = payload[:boundary]
    rows = [json.loads(line) for line in payload.splitlines()]
    if len(rows) > len(records):
        raise ValueError("Prediction count exceeds assigned shard")
    for row, record in zip(rows, records):
        expected = {"sample_id": record["sample_id"], "source_video_id": record["source_video_id"],
                    "answer": record["answer"], "split": record["split"],
                    "record_sha256": canonical_sha256(record), "protocol_sha256": protocol_sha256,
                    **record_metadata(record),
                    **score_answer(row["prediction"], record["answer"], record.get("choices"))}
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Predictions mismatch frozen input/protocol at {record['sample_id']}")
    return rows


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
    p.add_argument("--protocol", choices=["reference", VIDEOMME_PROTOCOL], default="reference")
    p.add_argument("--run-provenance", help="Frozen runner receipt with full checkpoint/source/plan identities")
    p.add_argument("--resume", action="store_true", help="Resume only an identical protocol and verified prediction prefix")
    p.add_argument("--distributed-env", action="store_true", help="Use torchrun ranks; one independent model per GPU")
    p.add_argument("--allow-unverified-provenance", action="store_true",
                   help="Explicitly allow imported checkpoints missing training provenance; results marked unverified")
    cli = p.parse_args(argv)
    if cli.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    rank, world, local_rank = 0, 1, 0
    if cli.distributed_env:
        rank, world, local_rank = (int(os.environ[name]) for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
        if not torch.cuda.is_available() or not 0 <= local_rank < torch.cuda.device_count():
            raise ValueError("Distributed evaluation requires an allocated local CUDA device")
        torch.cuda.set_device(local_rank)
    output = Path(cli.output)
    output = shard_output(output, rank, world)
    if not cli.resume:
        validate_output_paths(output)
    elif cli.profile:
        raise ValueError("Profiling cannot resume an evaluation")
    media_root = Path(cli.media_root) if cli.media_root else Path(cli.manifest).resolve().parent
    records = load_manifest(cli.manifest, media_root=media_root)
    if cli.max_samples is not None:
        if cli.max_samples <= 0:
            raise ValueError("max-samples must be positive")
        records = records[:cli.max_samples]
    full_coverage = None
    if cli.protocol == VIDEOMME_PROTOCOL:
        if cli.max_samples is not None or cli.max_new_tokens != 16 or cli.no_cache or cli.profile:
            raise ValueError("Frozen Video-MME protocol requires all samples, greedy 16 tokens, cache on, no profiling")
        if not cli.run_provenance or cli.allow_unverified_provenance:
            raise ValueError("Frozen Video-MME protocol requires verified runner/checkpoint provenance")
        full_coverage = validate_videomme_subset(records)
    checkpoint = Path(cli.checkpoint).resolve()
    contamination_check = check_provenance(checkpoint, records,
                                            allow_unverified=cli.allow_unverified_provenance)
    config = json.loads((checkpoint / "deepprobe_config.json").read_text())
    if cli.protocol == VIDEOMME_PROTOCOL and (config["num_frames"] != 8 or config["pool_grid"] != 4):
        raise ValueError("Video-MME protocol requires frozen 8-frame / pool-grid-4 checkpoint config")
    provenance = json.loads(Path(cli.run_provenance).read_text()) if cli.run_provenance else None
    if provenance is not None:
        if provenance.get("checkpoint") != str(checkpoint) or not provenance.get("all_checkpoint_files_sha256_verified"):
            raise ValueError("Runner receipt does not attest this checkpoint")
        if provenance.get("checkpoint_config_sha256") != sha256_file(checkpoint / "deepprobe_config.json"):
            raise ValueError("Checkpoint config differs from runner receipt")
    records = shard_records(records, rank, world)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    protocol = {"protocol": cli.protocol, "manifest_sha256": sha256_file(cli.manifest),
                "checkpoint": str(checkpoint), "checkpoint_config": config,
                "run_provenance_sha256": sha256_file(cli.run_provenance) if cli.run_provenance else None,
                "checkpoint_provenance_sha256": sha256_file(checkpoint / "training_provenance.json")
                if (checkpoint / "training_provenance.json").is_file() else None,
                "ordered_sample_ids_sha256": canonical_sha256([row["sample_id"] for row in records]),
                "num_shards": world, "shard_index": rank, "num_examples": len(records),
                "max_new_tokens": cli.max_new_tokens, "cache": not cli.no_cache, "decoding": "greedy_argmax",
                "dtype": str(dtype), "seed": 17, "subtitles": False, "audio": False,
                "prompt_contract": "MultimodalCollator: chronological frames, question, choices, option-letter only; no gold",
                "full_coverage": full_coverage, "torch": torch.__version__, "cuda": torch.version.cuda,
                "contamination_check": contamination_check}
    protocol_sha = canonical_sha256(protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol_path = output.with_suffix(".protocol.json")
    if cli.resume and output.exists() and not protocol_path.is_file():
        raise ValueError("Cannot resume predictions without their frozen protocol")
    write_json_once(protocol_path, protocol)
    rows = read_prediction_rows(output, records, protocol_sha, recover_tail=cli.resume)
    # Greedy decoding and eval() disable sampling/dropout; these additionally
    # fix RNG-dependent initialization and disallow TF32 mode drift.
    torch.manual_seed(17)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(17)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    model = DeepProbeVLM.from_pretrained(checkpoint, torch_dtype=dtype).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", trust_remote_code=False)
    processor = AutoImageProcessor.from_pretrained(checkpoint / "image_processor", use_fast=False)
    dataset = VideoManifestDataset(records, processor, model.deep_config.num_frames, media_root)
    collator = MultimodalCollator(tokenizer, num_visual_tokens=model.deep_config.num_visual_tokens,
                                 max_text_tokens=model.deep_config.max_text_tokens, include_answer=False)
    last_media, last_pixels = None, None
    with output.open("a" if cli.resume else "x", encoding="utf-8") as handle:
        for index in range(len(rows), len(records)):
            record = records[index]
            synchronise()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            media_key = canonical_sha256({key: record.get(key) for key in
                                         ("video_path", "image_paths", "start_seconds", "end_seconds")})
            if media_key != last_media:
                last_pixels = dataset[index]["pixel_values"]
                last_media = media_key
            batch = {k: v.to(device) for k, v in collator([{"record": record, "pixel_values": last_pixels}]).items()}
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
                   "record_sha256": canonical_sha256(record), "protocol_sha256": protocol_sha,
                   "shard_index": rank, "num_shards": world,
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
            os.fsync(handle.fileno())
            if (index + 1) % 10 == 0:
                print(json.dumps({"event": "evaluation_progress", "rank": rank,
                                  "completed": index + 1, "total": len(records)}), flush=True)
    summary = {"status": "REFERENCE_SCORER_NOT_OFFICIAL_BENCHMARK", "num_examples": len(rows),
               "by_metric": summarise_metrics(rows),
               "by_duration": summarise_groups(rows, "duration"), "by_task": summarise_groups(rows, "task"),
               "num_invalid": sum(bool(row.get("invalid_prediction")) for row in rows),
               "protocol": cli.protocol, "protocol_sha256": protocol_sha,
               "run_provenance": provenance, "predictions_sha256": sha256_file(output),
               "shard_index": rank, "num_shards": world, "full_coverage": full_coverage,
               "manifest_sha256": hashlib.sha256(Path(cli.manifest).read_bytes()).hexdigest(),
               "checkpoint": str(checkpoint), "device": str(device), "torch": torch.__version__,
               "cuda": torch.version.cuda, "contamination_check": contamination_check,
               "timing_note": "Includes cold/mixed runs; do not present as warmed p50/p95 speed claim."}
    write_json_once(output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
