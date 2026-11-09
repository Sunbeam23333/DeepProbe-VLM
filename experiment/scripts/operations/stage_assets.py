#!/usr/bin/env python3
"""Stage the pinned P0 inputs on a devbox and commit verified objects to OSS.

Does not execute Hub model code. Download sizes are checked before transfer.
An OSS receipt means CRC-checked upload plus remote length checks, not a GPU result.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("download_hf", ROOT / "scripts/data/download_hf.py")
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def entries():
    cfg = json.loads((ROOT / "configs/train/ouro_pilot.json").read_text())["model"]
    datasets = json.loads((ROOT / "configs/datasets.json").read_text())["datasets"]
    out = []
    for name, model, revision in [
        ("ouro", cfg["language_model"], cfg["language_revision"]),
        ("siglip", cfg["vision_model"], cfg["vision_revision"]),
    ]:
        out.append(dict(name=name, repo_id=model, revision=revision, repo_type="model",
                        patterns=["*.json", "*.safetensors", "README.md"] + (["*.py", "*.txt"] if name == "ouro" else [])))
    patterns = {
        "llava_video_178k": ["README.md", "0_30_s_youtube_v0_1/0_30_s_youtube_oe_v0_1_qa_processed.json",
                            "0_30_s_youtube_v0_1/0_30_s_youtube_v0_1_videos_1.tar.gz"],
        "videomme": ["README.md", "videomme/test-00000-of-00001.parquet"],
        "mvbench": ["README.md", "json/*.json", "video/MVBench_videos_ntu.txt"],
    }
    for name, pats in patterns.items():
        out.append(dict(name=name, repo_id=datasets[name]["repo_id"], revision=datasets[name]["revision"],
                        repo_type="dataset", patterns=pats, rights=datasets[name]["rights"]))
    return out


def upload_file(source, uri):
    subprocess.run(["ossutil", "cp", str(source), uri, "-f", "--no-progress"], check=True)
    # This account can list/read/write objects but cannot call GetObjectAcl;
    # ossutil stat unnecessarily requests that permission, so use exact listing.
    report = subprocess.run(["ossutil", "ls", uri], check=True, text=True, capture_output=True).stdout
    rows = [line.split() for line in report.splitlines() if line.split() and line.split()[-1] == uri]
    if len(rows) != 1 or int(rows[0][4]) != source.stat().st_size:
        raise RuntimeError(f"Remote length mismatch: {uri}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--bucket", default="oss://ss-devbox-stream/jinhaopeng/Paper/DeepProbe-VLM/assets/p0-v1")
    p.add_argument("--max-gb", type=float, default=15)
    p.add_argument("--download", action="store_true")
    p.add_argument("--only", nargs="+", choices=[e["name"] for e in entries()])
    args = p.parse_args()
    if args.max_gb <= 0 or not args.max_gb < float("inf"):
        p.error("positive finite max-gb required")
    from huggingface_hub import HfApi, snapshot_download
    plans = []
    for item in entries():
        if args.only and item["name"] not in args.only:
            continue
        info = HfApi().repo_info(item["repo_id"], repo_type=item["repo_type"], revision=item["revision"], files_metadata=True)
        if info.sha != item["revision"]:
            raise RuntimeError("Unexpected resolved revision")
        files = download.choose_files(info.siblings, item["patterns"])
        item.update(files=files, total_bytes=sum(f["size_bytes"] for f in files))
        plans.append(item)
    total = sum(x["total_bytes"] for x in plans)
    if total > args.max_gb * 1e9:
        raise ValueError(f"{total} selected bytes exceed cap")
    args.root.mkdir(parents=True, exist_ok=True)
    plan_path = args.root / "download_plan.json"
    plan_path.write_text(json.dumps({"total_bytes": total, "entries": plans}, indent=2) + "\n")
    print(json.dumps({"phase": "plan", "total_bytes": total, "entries": [{k:x[k] for k in ("name", "revision", "total_bytes")} for x in plans]}), flush=True)
    if not args.download:
        return
    receipts = []
    for item in plans:
        local = args.root / item["name"] / item["revision"]
        uri = args.bucket.rstrip("/") + "/" + item["name"] + "/" + item["revision"]
        complete = local / "ASSET_COMMITTED.json"
        if complete.exists():
            saved = json.loads(complete.read_text())
            if saved["uri"] != uri or saved["revision"] != item["revision"]:
                raise RuntimeError("Existing receipt does not match requested asset")
            receipts.append(saved)
            continue
        print(json.dumps({"phase": "download", "name": item["name"]}), flush=True)
        snapshot_download(repo_id=item["repo_id"], repo_type=item["repo_type"], revision=item["revision"],
                          allow_patterns=[f["path"] for f in item["files"]], local_dir=local, max_workers=4)
        for f in item["files"]:
            download.verify_download(download.resolve_media_path(f["path"], local), f)
        for f in item["files"]:
            upload_file(local / f["path"], uri + "/" + f["path"])
        receipt = {**item, "downloaded": True, "dataset": item["name"], "uri": uri, "local_path": str(local), "utc": datetime.now(timezone.utc).isoformat(),
                   "verification": "upstream LFS SHA256 / Git blob SHA1; OSS cp CRC; exact object listing length"}
        temp = local / "ASSET_COMMITTED.json.pending"
        temp.write_text(json.dumps(receipt, indent=2) + "\n")
        upload_file(temp, uri + "/ASSET_COMMITTED.json")
        temp.replace(complete)
        receipts.append(receipt)
        print(json.dumps({"phase": "committed", "name": item["name"], "bytes": item["total_bytes"], "uri": uri}), flush=True)
    (args.root / "ASSETS_READY.json").write_text(json.dumps({"assets": receipts}, indent=2) + "\n")


if __name__ == "__main__":
    main()
