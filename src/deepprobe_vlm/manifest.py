"""Strict, dependency-free JSONL validation shared by training and evaluation.

Gold answers belong in manifests for scoring, never in the model's input prompt.
All media paths are relative to one explicit media_root (symlinks cannot escape it).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Iterable


class ManifestError(ValueError):
    """A manifest cannot be used safely or unambiguously."""


def safe_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError("media path must be a non-empty string")
    if "\\" in value or "\x00" in value or re.match(r"^[A-Za-z]:", value):
        raise ManifestError(f"unsafe media path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.startswith("~"):
        raise ManifestError(f"media path must remain below media_root: {value!r}")
    if "://" in value or str(path) == ".":
        raise ManifestError(f"remote/empty media path not allowed: {value!r}")
    return path.as_posix()


def resolve_media_path(relative_path: str, media_root: str | Path) -> Path:
    root = Path(media_root).expanduser().resolve()
    path = (root / safe_relative_path(relative_path)).resolve()
    if not path.is_relative_to(root):
        raise ManifestError(f"media path/symlink escapes media_root: {relative_path!r}")
    return path


def validate_disjoint(*record_groups: Iterable[dict]) -> None:
    """Reject duplicate question IDs and any source shared by distinct splits.

    source_video_id identifies an original source, NOT a question or cropped clip.
    This detects declared provenance overlap; it cannot discover renamed videos.
    """
    ids: set[str] = set()
    sources: dict[str, str] = {}
    media_splits: dict[str, str] = {}
    for group in record_groups:
        for row in group:
            sid, source, split = row["sample_id"], row["source_video_id"], row["split"]
            if sid in ids:
                raise ManifestError(f"duplicate sample_id: {sid}")
            ids.add(sid)
            if source in sources and sources[source] != split:
                raise ManifestError(f"source-video leakage: {source!r} in {sources[source]} and {split}")
            sources[source] = split
            media = [row["video_path"]] if "video_path" in row else row["image_paths"]
            for path in media:
                key = safe_relative_path(path)
                if key in media_splits and media_splits[key] != split:
                    raise ManifestError(f"same media in different splits: {key}")
                media_splits[key] = split


def validate_records(records: Iterable[dict], media_root: str | Path | None = None,
                     check_media: bool = True) -> list[dict]:
    rows = list(records)
    if not rows:
        raise ManifestError("manifest is empty")
    if check_media and media_root is None:
        raise ManifestError("media_root is required when check_media=True")
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ManifestError(f"row {index}: expected JSON object")
        for key in ("sample_id", "source_video_id", "question", "answer", "split"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ManifestError(f"row {index}: non-empty string {key!r} required")
        for key in ("sample_id", "source_video_id"):
            if row[key] != row[key].strip():
                raise ManifestError(f"row {index}: {key} cannot contain leading/trailing whitespace")
        if row["split"] not in {"train", "validation", "test"}:
            raise ManifestError(f"row {index}: split must be train, validation, or test")
        if ("video_path" in row) == ("image_paths" in row):
            raise ManifestError(f"row {index}: exactly one of video_path/image_paths required")
        paths = [row["video_path"]] if "video_path" in row else row["image_paths"]
        if not isinstance(paths, list) or not paths:
            raise ManifestError(f"row {index}: image_paths must be a non-empty list")
        for path in paths:
            safe_relative_path(path)
            if media_root is not None:
                resolved = resolve_media_path(path, media_root)
                if check_media and not resolved.is_file():
                    raise ManifestError(f"row {index}: missing media file: {resolved}")
        if "choices" in row:
            choices = row["choices"]
            if not isinstance(choices, list) or not 2 <= len(choices) <= 26 or any(
                    not isinstance(x, str) or not x.strip() for x in choices):
                raise ManifestError(f"row {index}: choices must contain 2–26 non-empty strings")
            if row["answer"] not in [chr(65 + i) for i in range(len(choices))]:
                raise ManifestError(f"row {index}: MC answer must be an uppercase choice letter")
        if ("start_seconds" in row) != ("end_seconds" in row):
            raise ManifestError(f"row {index}: start_seconds/end_seconds must occur together")
        if "start_seconds" in row:
            start, end = row["start_seconds"], row["end_seconds"]
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or
                   not math.isfinite(x) for x in (start, end)) or not 0 <= start < end:
                raise ManifestError(f"row {index}: invalid temporal bounds")
    validate_disjoint(rows)
    return rows


def load_manifest(path: str | Path, media_root: str | Path | None = None,
                  check_media: bool = True) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{path}:{lineno}: invalid JSON: {exc.msg}") from exc
    return validate_records(rows, media_root, check_media)
