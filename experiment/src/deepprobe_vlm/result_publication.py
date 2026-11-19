"""Publish a sealed result bundle and verify every downloaded byte.

Credentials are provided through an environment variable or a private SDK
credential directory. This module never changes repository visibility, deletes
remote files, uploads an unlisted file, or authenticates on import.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any


MANIFEST = "PUBLICATION_MANIFEST.json"
SECRET = re.compile(
    rb"(?:hf_[A-Za-z0-9]{24,}|ms-[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}"
    rb"|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)"
)
TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".md", ".txt", ".tex", ".svg"}
ALLOWED_SUFFIXES = TEXT_SUFFIXES | {".pdf", ".png"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_relative(name: str) -> str:
    p = PurePosixPath(name)
    if (not name or p.is_absolute() or ".." in p.parts or "\\" in name
            or p.as_posix() != name or any(x.startswith(".") for x in p.parts)):
        raise ValueError(f"Unsafe publication path: {name!r}")
    return name


def inventory(folder: Path) -> list[dict[str, Any]]:
    folder = folder.absolute()
    if not folder.is_dir() or any(p.is_symlink() for p in (folder, *folder.parents)):
        raise ValueError("Publication root must be an ordinary directory")
    rows = []
    for p in sorted(folder.rglob("*")):
        if p.is_symlink():
            raise ValueError("Publication cannot contain symlinks")
        if p.is_dir():
            continue
        if not p.is_file():
            raise ValueError("Publication contains a nonregular file")
        name = safe_relative(p.relative_to(folder).as_posix())
        if name == MANIFEST:
            continue
        if p.suffix not in ALLOWED_SUFFIXES:
            raise ValueError(f"Not a result/document asset: {name}")
        raw = p.read_bytes()
        if p.suffix in TEXT_SUFFIXES and SECRET.search(raw):
            raise ValueError(f"Credential pattern in publication: {name}")
        rows.append({"path": name, "bytes": len(raw), "sha256": digest(raw)})
    if not rows:
        raise ValueError("Empty result bundle")
    return rows


def seal(folder: Path, *, release_id: str, provenance: dict | None = None) -> dict:
    """Seal an existing curated directory without overwriting a release."""
    safe_relative(release_id)
    rows = inventory(folder)
    doc = {"schema_version": 1, "release_id": release_id,
           "kind": "CURATED_EXPERIMENT_RESULTS", "created_utc": utc(),
           "files": rows, "file_count": len(rows),
           "payload_bytes": sum(r["bytes"] for r in rows),
           "provenance": provenance or {},
           "scope": "Results and documents only; no media, weights or private cluster snapshots."}
    raw = (json.dumps(doc, ensure_ascii=False, indent=2) + "\n").encode()
    if SECRET.search(raw):
        raise ValueError("Credential pattern in release metadata")
    with (folder / MANIFEST).open("xb") as f:
        f.write(raw)
    verify(folder)
    return doc


def verify(folder: Path) -> dict:
    p = folder / MANIFEST
    if p.is_symlink():
        raise ValueError("Manifest cannot be a symlink")
    raw = p.read_bytes()
    if SECRET.search(raw):
        raise ValueError("Credential pattern in manifest")
    doc = json.loads(raw)
    if doc.get("schema_version") != 1 or doc.get("kind") != "CURATED_EXPERIMENT_RESULTS":
        raise ValueError("Unsupported result manifest")
    rows = inventory(folder)
    if (rows != doc["files"] or len(rows) != doc["file_count"]
            or sum(r["bytes"] for r in rows) != doc["payload_bytes"]):
        raise ValueError("Result bundle changed after sealing")
    return doc


def publish(api: Any, folder: Path, *, repo_id: str, prefix: str,
            revision: str = "master", receipt_dir: Path) -> dict:
    """Upload a new prefix; an existing prefix must already match exactly.

    On a transport exception inspect the remote prefix before reusing the
    intent. No unconditional retry or destructive synchronization is performed.
    """
    safe_relative(prefix)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id):
        raise ValueError("Expected owner/repository")
    doc = verify(folder)
    manifest_raw = (folder / MANIFEST).read_bytes()
    rows = doc["files"] + [{"path": MANIFEST, "bytes": len(manifest_raw),
                             "sha256": digest(manifest_raw)}]
    expected = {prefix + "/" + row["path"]: row for row in rows}
    receipt_dir.mkdir(parents=True, exist_ok=False)
    intent = {"utc": utc(), "repo_id": repo_id, "repo_type": "dataset",
              "prefix": prefix, "revision": revision, "manifest_sha256": digest(manifest_raw),
              "files": expected, "sync_remote_repo": False, "automatic_retry": False}
    (receipt_dir / "INTENT.json").write_text(json.dumps(intent, indent=2) + "\n")
    existing = {f.path: f for f in api.list_repo_files(repo_id, "dataset", revision=revision)
                if f.path.startswith(prefix + "/") and f.type != "tree"}
    if existing and set(existing) != set(expected):
        raise ValueError("Remote prefix is partial or differs; inspect and reconcile before retry")
    if not existing:
        api.upload_folder(repo_id, "dataset", folder,
                          path_in_repo=prefix, revision=revision,
                          commit_message="Publish verified DeepProbe-VLM experiment results",
                          allow_patterns=[r["path"] for r in rows],
                          max_workers=4, use_cache=False, disable_tqdm=True,
                          sync_remote_repo=False)
    # Recheck local bytes and the complete destination set after upload.
    verify(folder)
    remote = {f.path: f for f in api.list_repo_files(repo_id, "dataset", revision=revision)
              if f.path.startswith(prefix + "/") and f.type != "tree"}
    if set(remote) != set(expected):
        raise ValueError("Published prefix has missing or extra files")
    verified = []
    with tempfile.TemporaryDirectory(prefix="deepprobe-results-readback-") as tmp:
        for name, row in expected.items():
            if remote[name].size != row["bytes"]:
                raise ValueError(f"Remote size mismatch: {name}")
            p = api.download_file(repo_id, "dataset", name, revision=revision,
                                  local_dir=Path(tmp), force=True,
                                  expected_sha256=row["sha256"], file_size=row["bytes"])
            raw = Path(p).read_bytes()
            if len(raw) != row["bytes"] or digest(raw) != row["sha256"]:
                raise ValueError(f"Full readback failed: {name}")
            verified.append({"path": name, "bytes": len(raw), "sha256": digest(raw)})
    final = {f.path: f.size for f in api.list_repo_files(repo_id, "dataset", revision=revision)
             if f.path.startswith(prefix + "/") and f.type != "tree"}
    if final != {name: r["bytes"] for name, r in expected.items()}:
        raise ValueError("Remote inventory changed during readback")
    result = {"status": "MODELSCOPE_RESULTS_FULL_READBACK_VERIFIED", "utc": utc(),
              "repo_id": repo_id, "prefix": prefix, "revision": revision,
              "manifest_sha256": digest(manifest_raw), "files": verified,
              "file_count": len(verified), "total_bytes": sum(r["bytes"] for r in verified),
              "remote_files_deleted": False, "visibility_changed": False,
              "already_present": bool(existing)}
    (receipt_dir / "VERIFIED.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["seal", "verify", "upload"])
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--release-id")
    p.add_argument("--repo", default="JasonJin/DeepProbe-VLM")
    p.add_argument("--prefix")
    p.add_argument("--revision", default="master")
    p.add_argument("--endpoint", default="https://modelscope.ai")
    p.add_argument("--token-env", default="MODELSCOPE_API_TOKEN")
    p.add_argument("--credentials-dir", type=Path)
    p.add_argument("--receipt-dir", type=Path)
    a = p.parse_args(argv)
    if a.mode == "seal":
        if not a.release_id:
            p.error("seal requires --release-id")
        doc = seal(a.bundle, release_id=a.release_id)
        print(json.dumps({"file_count": doc["file_count"], "payload_bytes": doc["payload_bytes"]}))
    elif a.mode == "verify":
        doc = verify(a.bundle)
        print(json.dumps({"verified": True, "file_count": doc["file_count"]}))
    else:
        if not a.prefix or not a.receipt_dir:
            p.error("upload requires --prefix and --receipt-dir")
        token = os.environ.get(a.token_env)
        if not token and not a.credentials_dir:
            p.error("Set the token environment variable or use an existing private credential directory")
        from modelscope_hub import HubApi
        from modelscope_hub.config import HubConfig
        with tempfile.TemporaryDirectory(prefix="deepprobe-publisher-sdk-") as tmp:
            config = HubConfig(endpoint=a.endpoint, token=token,
                               config_dir=a.credentials_dir or Path(tmp) / "credentials",
                               cache_dir=Path(tmp) / "cache")
            result = publish(HubApi(config=config), a.bundle, repo_id=a.repo,
                             prefix=a.prefix, revision=a.revision, receipt_dir=a.receipt_dir)
        print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
