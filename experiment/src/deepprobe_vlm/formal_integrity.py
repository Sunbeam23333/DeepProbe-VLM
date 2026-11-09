"""Live byte and source checks required by formal P1 training.

This module intentionally has no torch dependency.  The training entry point
coordinates one verifier across distributed ranks; the verifier itself never
accepts a cached success receipt in place of current media bytes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re


SHA256 = re.compile(r"[0-9a-f]{64}")
SOURCE_DIRS = ("src", "scripts", "configs", "requirements")
SOURCE_FILES = ("pyproject.toml", "README.md")
REQUIRED_P1_SOURCE = {
    "scripts/operations/run_p1_stage.sh",
    "scripts/operations/run_p1_stage.py",
    "scripts/preflight.py",
    "scripts/operations/stage_assets.py",
    "src/deepprobe_vlm/train.py",
    "src/deepprobe_vlm/formal_integrity.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def source_inventory(root: Path) -> dict[str, str]:
    """Hash the exact executable experiment tree, rejecting symlinks."""
    files: list[Path] = []
    for name in SOURCE_DIRS:
        directory = root / name
        if directory.is_symlink() or not directory.is_dir():
            raise FileNotFoundError(f"Frozen source directory missing or unsafe: {directory}")
        for path in sorted(directory.rglob("*")):
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path.is_symlink():
                raise ValueError(f"Frozen source rejects symlink: {path}")
            if path.is_file():
                files.append(path)
    for name in SOURCE_FILES:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Frozen source file missing or unsafe: {path}")
        files.append(path)
    return {path.relative_to(root).as_posix(): sha256_file(path)
            for path in sorted(set(files))}


def verify_frozen_source_manifest(root: Path, manifest_path: Path,
                                  expected_sha256: str) -> dict:
    """Bind a claimed manifest digest to the code actually executing train.py."""
    root = root.resolve()
    manifest_path = Path(manifest_path)
    if (manifest_path.is_symlink() or not manifest_path.is_file()
            or manifest_path.resolve().is_relative_to(root)):
        raise ValueError("Frozen source manifest must be a regular file outside the source tree")
    if not isinstance(expected_sha256, str) or not SHA256.fullmatch(expected_sha256):
        raise ValueError("Frozen source manifest requires a lowercase SHA-256")
    actual_sha256 = sha256_file(manifest_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("Frozen source manifest SHA-256 differs from the training contract")
    document = _read_json_object(manifest_path)
    files = document.get("files")
    if (document.get("schema_version") != 1
            or document.get("purpose") != "DEEPPROBE_FROZEN_P1_SOURCE"
            or not isinstance(files, dict) or not files):
        raise ValueError("Frozen source manifest schema/purpose/files are invalid")
    for relative, digest in files.items():
        if (not isinstance(relative, str) or not relative
                or PurePosixPath(relative).is_absolute()
                or PurePosixPath(relative).as_posix() != relative
                or ".." in PurePosixPath(relative).parts
                or "\\" in relative or "://" in relative
                or not isinstance(digest, str) or not SHA256.fullmatch(digest)):
            raise ValueError("Frozen source manifest contains an unsafe or invalid file entry")
    observed = source_inventory(root)
    if files != observed:
        missing = sorted(set(observed) - set(files))
        extra = sorted(set(files) - set(observed))
        changed = sorted(name for name in set(files) & set(observed)
                         if files[name] != observed[name])
        raise ValueError(f"Frozen source tree differs from manifest: "
                         f"missing={missing[:5]}, extra={extra[:5]}, changed={changed[:5]}")
    absent = sorted(REQUIRED_P1_SOURCE - set(observed))
    if absent:
        raise ValueError(f"Frozen source manifest omits required P1 files: {absent}")
    return {"manifest_sha256": actual_sha256, "file_count": len(observed),
            "files_sha256": canonical_sha256(observed)}


def _safe_media_relative(value: object) -> str:
    if (not isinstance(value, str) or not value or "\\" in value
            or "\x00" in value or "://" in value or value.startswith("~")):
        raise ValueError(f"Unsafe formal training media path: {value!r}")
    path = PurePosixPath(value)
    if (path.is_absolute() or ".." in path.parts or path.as_posix() != value
            or path.as_posix() == "."):
        raise ValueError(f"Unsafe formal training media path: {value!r}")
    return value


def _load_media_inventory(inventory_path: Path, media_root: Path,
                          expected_inventory_sha256: str,
                          expected_media: dict[str, str]) -> tuple[dict, list[dict], str]:
    if inventory_path.is_symlink() or not inventory_path.is_file():
        raise ValueError("Formal training media inventory is missing or unsafe")
    inventory_sha256 = sha256_file(inventory_path)
    if inventory_sha256 != expected_inventory_sha256:
        raise ValueError("Formal training media inventory changed from FORMAL_READY")
    inventory = _read_json_object(inventory_path)
    entries = inventory.get("entries")
    if (inventory.get("schema_version") != 1
            or inventory.get("kind") != "formal_training_media_sha256_inventory"
            or inventory.get("media_root") != str(media_root)
            or not isinstance(entries, list) or not entries):
        raise ValueError("Formal training media inventory schema/root/entries differ")
    observed_media: dict[str, str] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {
                "relative_path", "source_video_id", "size_bytes", "sha256"}:
            raise ValueError(f"Invalid formal training media inventory entry {index}")
        relative = _safe_media_relative(entry["relative_path"])
        source = entry["source_video_id"]
        size = entry["size_bytes"]
        digest = entry["sha256"]
        if (not isinstance(source, str) or not source.strip()
                or type(size) is not int or size <= 0
                or not isinstance(digest, str) or not SHA256.fullmatch(digest)
                or relative in observed_media):
            raise ValueError(f"Invalid formal training media inventory entry {index}")
        observed_media[relative] = source
    if observed_media != expected_media:
        raise ValueError("Formal media inventory does not exactly cover train/validation media")
    if (inventory.get("file_count") != len(entries)
            or inventory.get("total_bytes") != sum(entry["size_bytes"] for entry in entries)
            or inventory.get("entries_sha256") != canonical_sha256(entries)):
        raise ValueError("Formal training media inventory counts/content digest differ")
    return inventory, entries, inventory_sha256


def manifest_media(records: list[dict]) -> dict[str, str]:
    """Return exact media-path/source bindings actually consumed by training."""
    result: dict[str, str] = {}
    for row in records:
        source = row["source_video_id"]
        paths = [row["video_path"]] if "video_path" in row else row["image_paths"]
        for relative in paths:
            relative = _safe_media_relative(relative)
            previous = result.setdefault(relative, source)
            if previous != source:
                raise ValueError(f"Formal media path maps to multiple sources: {relative}")
    return result


def _stat_identity(path: Path, stat_result: os.stat_result) -> list[object]:
    return [str(path), stat_result.st_dev, stat_result.st_ino,
            stat_result.st_size, stat_result.st_mtime_ns, stat_result.st_ctime_ns]


def _verify_media(inventory_path: Path, media_root: Path,
                  expected_inventory_sha256: str, expected_media: dict[str, str],
                  *, hash_bytes: bool) -> dict:
    media_root = Path(media_root).resolve()
    inventory_path = Path(inventory_path)
    inventory, entries, inventory_sha256 = _load_media_inventory(
        inventory_path, media_root, expected_inventory_sha256, expected_media)
    identities = []
    for entry in entries:
        relative = entry["relative_path"]
        raw_path = media_root / relative
        path = raw_path.resolve(strict=True)
        if not path.is_relative_to(media_root) or not path.is_file():
            raise ValueError(f"Formal training media is missing or escapes MEDIA_ROOT: {relative}")
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if before.st_size != entry["size_bytes"]:
                raise ValueError(f"Formal training media size differs: {relative}")
            if hash_bytes:
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
                if digest.hexdigest() != entry["sha256"]:
                    raise ValueError(f"Formal training media SHA-256 differs: {relative}")
            after = os.fstat(stream.fileno())
        if (_stat_identity(path, before) != _stat_identity(path, after)
                or raw_path.resolve(strict=True) != path
                or _stat_identity(path, path.stat()) != _stat_identity(path, after)):
            raise ValueError(f"Formal training media changed during verification: {relative}")
        identities.append([relative, *_stat_identity(path, after)])
    if sha256_file(inventory_path) != inventory_sha256:
        raise ValueError("Formal training media inventory changed during verification")
    return {"inventory_sha256": inventory_sha256,
            "entries_sha256": inventory["entries_sha256"],
            "file_count": len(entries), "total_bytes": inventory["total_bytes"],
            "metadata_sha256": canonical_sha256(identities)}


def verify_attested_training_media(inventory_path: Path, media_root: Path,
                                   expected_inventory_sha256: str,
                                   expected_media: dict[str, str]) -> dict:
    """Rehash every selected file once, detecting edits during the hash pass."""
    return _verify_media(inventory_path, media_root, expected_inventory_sha256,
                         expected_media, hash_bytes=True)


def verify_attested_media_metadata(inventory_path: Path, media_root: Path,
                                   expected_inventory_sha256: str,
                                   expected_media: dict[str, str]) -> dict:
    """Cheap pre-train identity check; valid only after a full byte check."""
    return _verify_media(inventory_path, media_root, expected_inventory_sha256,
                         expected_media, hash_bytes=False)
