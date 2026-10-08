"""Validated source-disjoint manifests and deterministic bounded visual inputs."""
from __future__ import annotations

from pathlib import Path
from .manifest import load_manifest, validate_records, validate_disjoint, resolve_media_path


def uniform_indices(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0:
        raise ValueError("Frame counts must be positive")
    if count == 1:
        return [total // 2]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def decode_video(path: str | Path, num_frames: int, start_seconds=None, end_seconds=None):
    """Two deterministic sequential passes; bounded RAM, no imprecise seek.

    Long videos pay a full decode scan. This reference decoder is correct but
    not a throughput claim. Cache selected frames offline for large training.
    """
    import av
    def eligible_frames(container):
        for frame in container.decode(video=0):
            if start_seconds is not None or end_seconds is not None:
                if frame.time is None:
                    raise ValueError("Bounded clips require decoded frame timestamps")
                if start_seconds is not None and frame.time < start_seconds:
                    continue
                if end_seconds is not None and frame.time >= end_seconds:
                    continue
            yield frame
    with av.open(str(path)) as container:
        count = sum(1 for _ in eligible_frames(container))
    indices = uniform_indices(count, num_frames)
    wanted = set(indices)
    selected = {}
    with av.open(str(path)) as container:
        for index, frame in enumerate(eligible_frames(container)):
            if index in wanted:
                selected[index] = frame.to_image().convert("RGB")
            if index >= indices[-1]:
                break
    if len(selected) != len(wanted):
        raise ValueError(f"Video changed or could not be decoded consistently: {path}")
    return [selected[i].copy() for i in indices]


def load_frames(record, media_root, num_frames):
    from PIL import Image
    if record.get("video_path"):
        return decode_video(resolve_media_path(record["video_path"], media_root), num_frames,
                            record.get("start_seconds"), record.get("end_seconds"))
    paths = record["image_paths"]
    images = []
    for index in uniform_indices(len(paths), num_frames):
        with Image.open(resolve_media_path(paths[index], media_root)) as source:
            images.append(source.convert("RGB"))
    return images


class VideoManifestDataset:
    def __init__(self, records, image_processor, num_frames, media_root):
        self.records, self.image_processor = records, image_processor
        self.num_frames, self.media_root = num_frames, media_root

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        frames = load_frames(record, self.media_root, self.num_frames)
        pixels = self.image_processor(images=frames, return_tensors="pt")["pixel_values"]
        return {"record": record, "pixel_values": pixels}


class MultimodalCollator:
    def __init__(self, tokenizer, *, num_visual_tokens, max_text_tokens,
                 include_answer=True):
        self.tokenizer = tokenizer
        self.num_visual_tokens = num_visual_tokens
        self.max_text_tokens = max_text_tokens
        self.include_answer = include_answer
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs an EOS token")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize_record(self, item, *, check_budget=True):
        """Shared exact text construction for collation and CPU-only prechecks.

        Does not load/decode media or import torch. Disabling the check only
        exposes full lengths to the reporting tool; it never truncates tokens.
        """
        prefix = self.tokenizer.encode("Video frames in chronological order:\n", add_special_tokens=False)
        if self.tokenizer.bos_token_id is not None:
            prefix = [self.tokenizer.bos_token_id] + prefix
        question = "\nQuestion: " + item["question"]
        if item.get("choices"):
            question += "\n" + "\n".join(f"{chr(65+i)}. {x}" for i, x in enumerate(item["choices"]))
            question += "\nAnswer with the option letter only."
        question += "\nAnswer:"
        suffix = self.tokenizer.encode(question, add_special_tokens=False)
        answer = (self.tokenizer.encode(" " + item["answer"], add_special_tokens=False)
                  + [self.tokenizer.eos_token_id]) if self.include_answer else []
        if check_budget and len(prefix) + len(suffix) + len(answer) > self.max_text_tokens:
            raise ValueError(f"Text exceeds max_text_tokens for {item['sample_id']}; curate explicitly (no silent truncation)")
        return prefix, suffix, answer

    def __call__(self, examples):
        import torch
        rows, masks, labels = [], [], []
        for example in examples:
            item = example["record"]
            prefix, suffix, answer = self.tokenize_record(item)
            visual_ids = [self.tokenizer.pad_token_id] * self.num_visual_tokens
            row = prefix + visual_ids + suffix + answer
            visual = [False] * len(prefix) + [True] * len(visual_ids) + [False] * (len(suffix) + len(answer))
            rows.append(row)
            masks.append(visual)
            labels.append([-100] * (len(row) - len(answer)) + answer)
        width = max(map(len, rows))
        batch = {
            "input_ids": torch.tensor([r + [self.tokenizer.pad_token_id] * (width-len(r)) for r in rows]),
            "attention_mask": torch.tensor([[1] * len(r) + [0] * (width-len(r)) for r in rows]),
            "visual_mask": torch.tensor([r + [False] * (width-len(r)) for r in masks]),
            "pixel_values": torch.stack([x["pixel_values"] for x in examples]),
        }
        if self.include_answer:
            batch["labels"] = torch.tensor([r + [-100] * (width-len(r)) for r in labels])
        return batch
