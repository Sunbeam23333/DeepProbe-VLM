#!/usr/bin/env python3
"""Stream a portable, credential-redacted source/configuration/log snapshot.

Does not mutate the source tree. Binary assets belong in the separate data
transfer plan. A snapshot of live logs is a point-in-time copy, not acceptance.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tarfile

TOKENS = re.compile(rb"hf_[A-Za-z0-9]{24,}|ms-[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}|AKIA[A-Z0-9]{16}|LTAI[A-Za-z0-9]{12,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")
ASSIGNMENT = re.compile(rb"(?i)((?:accesskeysecret|access_key_secret|client-key-data|client-certificate-data|authorization|api_key|password)[\"']?\s*[:=]\s*[\"'])([^\"'\r\n]{12,})([\"'])")
SUFFIXES = {'.py', '.sh', '.bash', '.json', '.jsonl', '.yaml', '.yml', '.toml', '.md', '.txt', '.csv', '.tsv', '.log', '.tex', '.bib', '.sty', '.cls', '.jinja', '.ini', '.cfg', '.lock', '.stderr', '.stdout'}
SKIP_DIRS = {'.git', '__pycache__', '.pytest_cache', '.ruff_cache', 'node_modules', 'site-packages', 'runtime-cache', 'suite-runtime', 'env', '.venv'}


def redact(raw):
    raw = TOKENS.sub(b'[REDACTED_CREDENTIAL]', raw)
    return ASSIGNMENT.sub(lambda m: m[1] + b'[REDACTED_CREDENTIAL]' + m[3], raw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('roots', nargs='+', help='LABEL=/absolute/root')
    a = p.parse_args()
    rows, skipped = [], []
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as tar:
        for spec in a.roots:
            label, value = spec.split('=', 1)
            root = Path(value).resolve()
            assert label and '/' not in label and label not in {'.', '..'}
            for directory, dirs, files in os.walk(root):
                dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith('.venv') and not (Path(directory)/d).is_symlink() and not d.startswith('migration-20261121'))
                for name in sorted(files):
                    path = Path(directory)/name
                    rel = label + '/' + path.relative_to(root).as_posix()
                    if path.is_symlink() or not path.is_file():
                        continue
                    if path.suffix not in SUFFIXES and name not in {'stdout', 'stderr', 'Dockerfile', 'Makefile', 'requirements.txt', 'LICENSE', '.gitignore', '.gitattributes'}:
                        continue
                    if name.startswith('.env') or name in {'token', 'credentials', '.ossutilconfig', 'aliconfig'}:
                        skipped.append({'path': rel, 'reason': 'credential file'})
                        continue
                    size = path.stat().st_size
                    if size > 128 * 1024**2:
                        skipped.append({'path': rel, 'bytes': size, 'reason': 'large asset; separate transfer plan'})
                        continue
                    raw = path.read_bytes()
                    if b'\x00' in raw[:4096]:
                        skipped.append({'path': rel, 'reason': 'binary payload'})
                        continue
                    clean = redact(raw)
                    info = tarfile.TarInfo(rel)
                    info.size, info.mode, info.mtime = len(clean), 0o644, 0
                    tar.addfile(info, io.BytesIO(clean))
                    rows.append({'path': rel, 'bytes': len(clean), 'sha256': hashlib.sha256(clean).hexdigest(), 'redacted': clean != raw})
        doc = json.dumps({'files': rows, 'skipped': skipped, 'live_logs_are_not_acceptance': True}, indent=2).encode()
        info = tarfile.TarInfo('SNAPSHOT_MANIFEST.json')
        info.size, info.mode = len(doc), 0o644
        tar.addfile(info, io.BytesIO(doc))


if __name__ == '__main__':
    main()
