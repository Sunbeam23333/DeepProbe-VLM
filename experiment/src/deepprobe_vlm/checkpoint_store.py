"""Stream-bucket adapter using the installed ossutil v2 API.

Reads stream directly through ``ossutil cat -q``. Large writes use explicit
multipart uploads with overwrite protection at both initiation and completion;
``ossutil cp --ignore-existing`` alone is not a race-safe overwrite guard.
Only one 64 MiB part file is staged, never a second whole checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import tempfile
import time
import uuid

from .checkpoint_retention import (check_pin, digest, file_pin, require, safe,
                                   write_once)


def _field(document, name):
    """ossutil API JSON is XML-shaped; response headers may be one-item arrays."""
    found = []
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.replace("-", "").replace("_", "").lower() == name.lower():
                    while isinstance(item, list) and len(item) == 1:
                        item = item[0]
                    require(isinstance(item, str) and item, "invalid API response field")
                    found.append(item)
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(document)
    require(found and len(set(found)) == 1, "missing or ambiguous API response field: " + name)
    return found[0]


class OssutilStore:
    """No credentials are accepted in argv; an existing config file is used.

    ``state_dir`` belongs inside the new run's excluded ``.retention`` directory.
    Its small multipart journals survive interruption. Unknown initiation or
    completion outcomes stop for reconciliation; they do not start a new upload.
    """
    PART_BYTES = 64 * 1024 * 1024

    def __init__(self, executable, config_file, *, allowed_prefix, state_dir,
                 timeout_seconds=3600):
        self.executable = safe(executable)
        self.config_file = safe(config_file)
        require(self.executable.is_file() and os.access(self.executable, os.X_OK)
                and self.config_file.is_file(), "installed ossutil or config unavailable")
        require(isinstance(allowed_prefix, str) and allowed_prefix.startswith("oss://ss-devbox-stream/")
                and "/runs/" in allowed_prefix and allowed_prefix.endswith("/")
                and ".." not in allowed_prefix.split("/") and not any(c.isspace() for c in allowed_prefix),
                "store must be restricted to one stream run prefix")
        require(type(timeout_seconds) is int and 1 <= timeout_seconds <= 86400,
                "invalid transfer timeout")
        self.allowed_prefix = allowed_prefix
        self.state_dir = safe(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout_seconds

    def _uri(self, uri):
        require(isinstance(uri, str) and uri.startswith(self.allowed_prefix)
                and ".." not in uri.split("/") and "\\" not in uri, "object escapes allowed run prefix")
        return "ss-devbox-stream", uri.removeprefix("oss://ss-devbox-stream/")

    def _argv(self, args):
        return [str(self.executable), *args, "-c", str(self.config_file), "--retry-times", "0"]

    def _json(self, args):
        operation = self.state_dir / "requests" / uuid.uuid4().hex
        operation.parent.mkdir(exist_ok=True)
        # API output is small metadata; binary object bodies use iter_object.
        result = subprocess.run(self._argv(["api", *args, "--output-format", "json", "-q"]),
                                capture_output=True, timeout=self.timeout)
        write_once(operation.with_suffix(".json"), {
            "operation": args[0], "returncode": result.returncode,
            "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
            "automatic_retry": False})
        require(result.returncode == 0, "ossutil API failed; retained request hashes; no automatic retry")
        require(len(result.stdout) <= 16 * 1024 * 1024, "unexpectedly large API metadata response")
        document = json.loads(result.stdout)
        require(isinstance(document, dict), "invalid ossutil API JSON")
        write_once(operation.with_suffix(".response.json"), document)
        return document

    def list_prefix(self, prefix):
        bucket, key = self._uri(prefix)
        token, seen_tokens, result = None, set(), []
        while True:
            args = ["list-objects-v2", "--bucket", bucket, "--prefix", key, "--max-keys", "1000"]
            if token is not None:
                args += ["--continuation-token", token]
            value = self._json(args)
            require(value.get("Name") == bucket and value.get("Prefix") == key,
                    "list response bucket or prefix differs")
            if "ContinuationToken" in value:
                require(value["ContinuationToken"] == token, "list continuation binding differs")
            contents = value.get("Contents") or []
            if isinstance(contents, dict):
                contents = [contents]
            require(isinstance(contents, list) and str(value.get("KeyCount")) == str(len(contents)),
                    "list response item count differs")
            for row in contents:
                name = row.get("Key")
                size = row.get("Size")
                require(isinstance(name, str) and name.startswith(key)
                        and ((type(size) is int and size >= 0) or (isinstance(size, str) and size.isdigit()))
                        and isinstance(row.get("ETag"), str) and row["ETag"]
                        and isinstance(row.get("LastModified"), str) and row["LastModified"],
                        "invalid object list entry")
                result.append({"uri": "oss://" + bucket + "/" + name, "bytes": int(size),
                               "etag": row["ETag"], "last_modified": row["LastModified"]})
            truncated = value.get("IsTruncated")
            require(truncated in ("true", "false", True, False), "unknown list truncation state")
            if truncated in ("false", False):
                break
            token = value.get("NextContinuationToken")
            require(isinstance(token, str) and token and token not in seen_tokens,
                    "missing or repeated pagination token")
            seen_tokens.add(token)
        require(len({r["uri"] for r in result}) == len(result), "duplicate object across list pages")
        return sorted(result, key=lambda x: x["uri"])

    def iter_object(self, uri):
        self._uri(uri)
        # The installed v2 cat command was independently checked to emit exactly
        # the object body with -q, without elapsed-time/progress decoration.
        with tempfile.TemporaryFile() as stderr:
            process = subprocess.Popen(self._argv(["cat", uri, "-q"]),
                                       stdout=subprocess.PIPE, stderr=stderr)
            try:
                assert process.stdout is not None
                deadline = time.monotonic() + self.timeout
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    while True:
                        remaining = deadline - time.monotonic()
                        require(remaining > 0, "streaming object GET exceeded its time budget")
                        if not selector.select(min(1.0, remaining)):
                            continue
                        block = os.read(process.stdout.fileno(), 8 * 1024 * 1024)
                        if not block:
                            break
                        yield block
                code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
                require(code == 0, "streaming object GET failed")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                if process.stdout is not None:
                    process.stdout.close()

    def put_if_absent(self, uri, source, expected_sha256):
        bucket, key = self._uri(uri)
        source = safe(source)
        pin = file_pin(source)
        require(pin["sha256"] == expected_sha256, "upload source SHA differs")
        require(not any(r["uri"] == uri for r in self.list_prefix(uri)), "object already exists")
        if pin["bytes"] <= self.PART_BYTES:
            self._json(["put-object", "--bucket", bucket, "--key", key,
                        "--body", "file://" + str(source), "--metadata", "sha256=" + expected_sha256,
                        "--forbid-overwrite"])
            check_pin(pin)
            return
        require(math.ceil(pin["bytes"] / self.PART_BYTES) <= 10000, "too many upload parts")
        state = safe(self.state_dir / "multipart" / digest(uri))
        state.mkdir(parents=True, exist_ok=True)
        binding = {"uri": uri, "source": pin, "part_bytes": self.PART_BYTES}
        init = state / "INITIATED.json"
        if init.exists():
            value = json.loads(init.read_text())
            require(value.get("binding") == binding and isinstance(value.get("upload_id"), str)
                    and value["upload_id"], "multipart state/source binding differs")
            upload_id = value["upload_id"]
        else:
            require(not (state / "INIT_INTENT.json").exists(),
                    "unknown initiation outcome; reconcile its upload ID before resuming")
            write_once(state / "INIT_INTENT.json", binding)
            response = self._json(["initiate-multipart-upload", "--bucket", bucket, "--key", key,
                                   "--metadata", "sha256=" + expected_sha256, "--forbid-overwrite", "true"])
            upload_id = _field(response, "uploadid")
            write_once(init, {"binding": binding, "upload_id": upload_id})
        require(not (state / "COMPLETE_INTENT.json").exists(),
                "unknown completion outcome; read the exact object, never blindly complete again")
        parts = []
        with source.open("rb") as stream:
            for index in range(1, math.ceil(pin["bytes"] / self.PART_BYTES) + 1):
                wanted = min(self.PART_BYTES, pin["bytes"] - (index - 1) * self.PART_BYTES)
                record = state / f"part-{index:05}.json"
                if record.is_file():
                    value = json.loads(record.read_text())
                    require(value.get("part_number") == index and value.get("bytes") == wanted
                            and value.get("source_sha256") == expected_sha256
                            and isinstance(value.get("etag"), str) and value["etag"],
                            "multipart part journal differs")
                    stream.seek(wanted, os.SEEK_CUR)
                else:
                    scratch = safe(state / "part.pending")
                    if scratch.exists():
                        require(scratch.is_file() and scratch.stat().st_nlink == 1,
                                "unsafe owned multipart scratch")
                        scratch.unlink()
                    remaining = wanted
                    with scratch.open("xb") as target:
                        while remaining:
                            block = stream.read(min(8 * 1024 * 1024, remaining))
                            require(block, "upload source shortened")
                            target.write(block)
                            remaining -= len(block)
                        target.flush()
                        os.fsync(target.fileno())
                    response = self._json(["upload-part", "--bucket", bucket, "--key", key,
                                           "--upload-id", upload_id, "--part-number", str(index),
                                           "--body", "file://" + str(scratch)])
                    value = {"part_number": index, "bytes": wanted, "source_sha256": expected_sha256,
                             "etag": _field(response, "etag")}
                    write_once(record, value)
                    scratch.unlink()
                parts.append({"PartNumber": index, "ETag": value["etag"]})
        check_pin(pin)
        completion = {"binding": binding, "upload_id": upload_id, "parts": parts}
        write_once(state / "COMPLETE_INTENT.json", completion)
        response = self._json(["complete-multipart-upload", "--bucket", bucket, "--key", key,
                               "--upload-id", upload_id, "--complete-multipart-upload",
                               json.dumps({"Part": parts}, separators=(",", ":")),
                               "--forbid-overwrite", "true"])
        write_once(state / "COMPLETED.json", {"binding": binding, "response": response})
        check_pin(pin)
