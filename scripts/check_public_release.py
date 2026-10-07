#!/usr/bin/env python3
"""Check tracked/staged public files, never print a suspected secret value."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "HF credential": re.compile(rb"\bhf_[A-Za-z0-9]{24,}\b"),
    "GitHub credential": re.compile(rb"\b(?:ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{25,}\b"),
    "private key": re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "local user path": re.compile(rb"/Users/[A-Za-z][^/\s]+/"),
    "student identifier": re.compile(rb"(?i)(?:student[ _-]?id|BUPT[ _-]?ID|QMUL[ _-]?ID|class[ _-]?id)\s*[:=]?\s*[0-9]{8,12}\b"),
}
FORBIDDEN = {".safetensors", ".pth", ".pt", ".ckpt", ".mp4", ".mkv", ".zip", ".parquet", ".bin"}


def main():
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    issues = []
    for name in filter(None, names):
        file = ROOT / name
        if not file.is_file():
            continue
        if file.is_symlink():
            issues.append((name, "symlink not permitted in this initial public release"))
            continue
        if file.suffix.lower() in FORBIDDEN or file.stat().st_size > 20 * 2**20:
            issues.append((name, "large/binary research artifact requires explicit review"))
        if name == "scripts/check_public_release.py":
            continue  # scan patterns are examples, not credentials
        if file.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf"}:
            continue
        data = file.read_bytes()
        for label, pattern in PATTERNS.items():
            if pattern.search(data):
                issues.append((name, label))
    for name, label in issues:
        print(f"FAIL {name}: {label}")
    if not issues:
        print(f"Public-release scan passed for {sum(bool(x) for x in names)} tracked files.")
    return bool(issues)


if __name__ == "__main__":
    sys.exit(main())
