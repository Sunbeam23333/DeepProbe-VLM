#!/usr/bin/env python3
"""Convert official local annotations to strict DeepProbe JSONL, no downloading."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from deepprobe_vlm.manifest import ManifestError, resolve_media_path, safe_relative_path, validate_records

# Source: OpenGVLab/Ask-Anything video_chat2/mvbench.ipynb data_list.
# Bounded tasks intentionally target RAW source videos, not pretrimmed Hub clips:
# applying original timestamps to already-trimmed videos would be incorrect.
MV_TASKS = {
    "action_sequence": ("star/Charades_v1_480", "charades", True),
    "action_prediction": ("star/Charades_v1_480", "charades", True),
    "object_interaction": ("star/Charades_v1_480", "charades", True),
    "action_localization": ("sta/sta_video", "charades", True),
    "moving_count": ("clevrer/video_validation", "clevrer", False),
    "moving_direction": ("clevrer/video_validation", "clevrer", False),
    "moving_attribute": ("clevrer/video_validation", "clevrer", False),
    "object_existence": ("clevrer/video_validation", "clevrer", False),
    "counterfactual_inference": ("clevrer/video_validation", "clevrer", False),
    "action_count": ("perception/videos", "perceptiontest", False),
    "character_order": ("perception/videos", "perceptiontest", False),
    "object_shuffle": ("perception/videos", "perceptiontest", False),
    "state_change": ("perception/videos", "perceptiontest", False),
    "fine_grained_pose": ("nturgbd_convert", "nturgbd", False),
    "egocentric_navigation": ("vlnqa", "vlnqa", False),
    "fine_grained_action": ("Moments_in_Time_Raw/videos", "momentsintime", False),
    "scene_transition": ("scene_qa/video", "sceneqa", False),
    "action_antonym": ("ssv2_video_mp4", "ssv2", False),
    "unexpected_action": ("FunQA_test/test", "funqa", False),
    "episodic_reasoning": ("tvqa/frames_fps3_hq", "tvqa", True),
}


def rows_from(path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            yield from batch.to_pylist()
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("annotation JSON must be a list; refuse to guess a nested schema")
        yield from data


def clean_choices(values):
    return [re.sub(r"^\s*(?:\([A-Z]\)|[A-Z][.)])\s*", "", str(v)).strip() for v in values]


def source_id(video, family=None, source_map=None):
    """Conservative original-source grouping. Explicit source map overrides all.

    Unknown families fail closed instead of pretending clip names prove provenance.
    """
    video = safe_relative_path(video)
    if source_map and video in source_map:
        value = source_map[video]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"invalid source-map value for {video}")
        return value
    stem = PurePosixPath(video).stem
    lower = video.lower()
    if family == "youtube" or "youtube" in lower or stem.startswith("ytb_"):
        match = re.fullmatch(r"(?:ytb_|v_)?([A-Za-z0-9_-]{11})(?:[_-]\d+(?:\.\d+)?[_-]\d+(?:\.\d+)?)?", stem)
        if not match:
            raise ValueError(f"cannot determine original YouTube ID: {video}; provide --source-map")
        return "youtube:" + match.group(1)
    if family == "charades" or "charades" in lower:
        return "charades:" + stem.split("_")[0]
    if family is None:
        families = {"perception": "perceptiontest", "nextqa": "nextqa", "activitynet": "activitynet", "clevrer": "clevrer"}
        family = next((name for part, name in families.items() if part in lower), None)
    if family is None:
        raise ValueError(f"unverified source-video naming for {video}; supply --source-map")
    # Remove only known explicit time-segment suffixes, never question IDs.
    stem = re.sub(r"(?:_segment_\d+)?_\d+(?:\.\d+)?s?_\d+(?:\.\d+)?s?$", "", stem)
    return f"{family.lower()}:{stem}"


def convert_llava(row, args, source_map):
    video = safe_relative_path(row["video"])
    source = source_id(video, source_map=source_map)
    conversations = row["conversations"]
    if len(conversations) % 2:
        raise ValueError(f"unpaired conversation: {row.get('id')}")
    for index in range(0, len(conversations), 2):
        human, assistant = conversations[index:index + 2]
        if human["from"] not in {"human", "user"} or assistant["from"] not in {"gpt", "assistant"}:
            raise ValueError("expected alternating human/gpt pairs; no silent reordering")
        question = re.sub(r"<(?:image|video)>", "", human["value"]).strip()
        identity = f"{args.input.name}|{row.get('id')}|{video}|{index//2}"
        yield {"sample_id": "llava:" + hashlib.sha256(identity.encode()).hexdigest()[:24],
               "source_video_id": source, "video_path": video, "question": question,
               "answer": assistant["value"].strip(), "split": args.split,
               "dataset": "llava_video_178k", "annotation_file": args.input.name}


def convert_videomme(row, args, source_map):
    vid = row["videoID"]  # NOT video_id (a benchmark index such as 001).
    video = safe_relative_path(f"{args.video_prefix}/{vid}.mp4")
    yield {"sample_id": "videomme:" + row["question_id"], "source_video_id": "youtube:" + vid,
           "video_path": video, "question": row["question"], "choices": clean_choices(row["options"]),
           "answer": row["answer"].strip().upper(), "split": "test", "dataset": "videomme",
           "duration_category": row["duration"], "task_type": row["task_type"], "domain": row["domain"]}


def convert_herbench(row, args, source_map):
    video = safe_relative_path(row["video_path"])
    family = row.get("source_dataset") or row.get("metadata", {}).get("source_dataset")
    source = source_id(video, family=family, source_map=source_map)
    yield {"sample_id": "herbench:" + row["question_id"], "source_video_id": source,
           "video_path": video, "question": row["question"], "choices": clean_choices(row["choices"]),
           "answer": row["answer"].strip().upper(), "split": "test", "dataset": "herbench",
           "task_type": row["task_type"]}


def convert_mvbench(row, args, source_map):
    folder, family, bounded = MV_TASKS[args.task]
    video = safe_relative_path(f"{args.video_prefix}/{folder}/{row['video']}")
    choices = clean_choices(row["candidates"])
    if choices.count(row["answer"]) != 1:
        raise ValueError("MVBench answer must uniquely match one candidate")
    result = {"sample_id": "mvbench:" + args.task + ":" + hashlib.sha256(
                json.dumps(row, sort_keys=True).encode()).hexdigest()[:24],
              "source_video_id": source_id(row["video"], family=family, source_map=source_map),
              "video_path": video, "question": row["question"], "choices": choices,
              "answer": chr(65 + choices.index(row["answer"])), "split": "test",
              "dataset": "mvbench", "task_type": args.task}
    if bounded:
        if "start" not in row or "end" not in row:
            raise ValueError(f"{args.task} requires official start/end annotation")
        result.update(start_seconds=float(row["start"]), end_seconds=float(row["end"]))
    if args.task == "episodic_reasoning":
        directory = resolve_media_path(video, args.media_root)
        if not directory.is_dir():
            raise FileNotFoundError(f"original TVQA 3fps frame directory missing: {directory}")
        # Original MVBench midpoint-segment sampling, first frame index = 1.
        frames = list(directory.glob("*.jpg"))
        max_frame = len(frames)
        start = max(1, round(float(row["start"]) * 3))
        end = min(round(float(row["end"]) * 3), max_frame)
        if end < start:
            raise ValueError("TVQA temporal bounds exceed the available frame sequence")
        segment = (end - start) / args.num_frames
        indices = [int(start + segment / 2 + round(segment * i)) for i in range(args.num_frames)]
        result.pop("video_path")
        result.pop("start_seconds")
        result.pop("end_seconds")
        result["image_paths"] = [f"{video}/{i:05d}.jpg" for i in indices]
        result["frame_sampling"] = {"fps": 3, "num_frames": args.num_frames,
                                     "start_seconds": row["start"], "end_seconds": row["end"]}
    yield result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", choices=["llava", "videomme", "mvbench", "herbench"])
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--media-root", required=True, type=Path)
    p.add_argument("--split", default="train", choices=["train", "validation"])
    p.add_argument("--video-prefix", default="videos", help="Relative extracted video directory; MVBench commonly 'video'")
    p.add_argument("--source-map", type=Path, help="JSON mapping annotation media paths to original source_video_id")
    p.add_argument("--task", choices=sorted(MV_TASKS))
    p.add_argument("--num-frames", type=int, default=16, help="TVQA frame-folder sampling count")
    p.add_argument("--available-only", action="store_true", help="Explicit shard pilot: omit missing media and write a coverage report; NOT full-benchmark evaluation")
    args = p.parse_args()
    if args.dataset == "mvbench" and not args.task:
        p.error("--task required for MVBench")
    if args.num_frames <= 0:
        p.error("--num-frames must be positive")
    if args.available_only and args.dataset != "llava":
        p.error("--available-only is restricted to training pilots; benchmarks must be complete")
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".coverage.json").exists():
        raise FileExistsError("output/coverage report already exists; select a new filename")
    source_map = json.loads(args.source_map.read_text()) if args.source_map else {}
    convert = {"llava": convert_llava, "videomme": convert_videomme,
               "mvbench": convert_mvbench, "herbench": convert_herbench}[args.dataset]
    all_rows = [new for row in rows_from(args.input) for new in convert(row, args, source_map)]
    validate_records(all_rows, args.media_root, check_media=False)
    selected, omitted = [], []
    for row in all_rows:
        paths = [row["video_path"]] if "video_path" in row else row["image_paths"]
        present = all(resolve_media_path(path, args.media_root).is_file() for path in paths)
        if not present and args.available_only:
            omitted.append(row["sample_id"])
        else:
            selected.append(row)
    validate_records(selected, args.media_root, check_media=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    coverage = {"input_rows_after_qa_expansion": len(all_rows), "written": len(selected),
                "omitted_missing_media": len(omitted), "omitted_sample_ids": omitted,
                "partial_training_pilot": args.available_only,
                "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest()}
    with args.output.with_suffix(args.output.suffix + ".coverage.json").open("x") as stream:
        json.dump(coverage, stream, indent=2)
    print(json.dumps({k: v for k, v in coverage.items() if k != "omitted_sample_ids"}, indent=2))


if __name__ == "__main__":
    main()
