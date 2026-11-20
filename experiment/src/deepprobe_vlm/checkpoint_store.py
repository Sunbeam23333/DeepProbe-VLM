"""Stream-bucket adapter using the installed ossutil v2 API.

Reads stream directly through ``ossutil cat -q``. Large writes use explicit
multipart uploads with overwrite protection at both initiation and completion;
``ossutil cp --ignore-existing`` alone is not a race-safe overwrite guard.
By default one 64 MiB part file is staged. Opt-in parallel uploads stage at
most ``64 MiB * part_workers``; never a second whole checkpoint.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import stat
import subprocess
import tempfile
import threading
import time
import uuid

from .checkpoint_retention import (check_pin, digest, file_pin, fingerprint, require, safe,
                                   write_once)

STREAM_ENDPOINT = "https://oss-cn-wulanchabu-internal.aliyuncs.com"


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
                 timeout_seconds=3600, part_workers=1):
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
        require(type(part_workers) is int and 1 <= part_workers <= 4,
                "part_workers must be an integer from 1 to 4")
        self.allowed_prefix = allowed_prefix
        self.state_dir = safe(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout_seconds
        self.part_workers = part_workers

    @property
    def multipart_buffer_bytes(self):
        """Maximum private-disk part staging per active upload; reserve upfront.

        Copying additionally uses at most 8 MiB of RAM per worker. The caller
        must reserve separately for concurrent store objects and small journals.
        """
        return self.PART_BYTES * self.part_workers

    def _uri(self, uri):
        require(isinstance(uri, str) and uri.startswith(self.allowed_prefix)
                and ".." not in uri.split("/") and "\\" not in uri, "object escapes allowed run prefix")
        return "ss-devbox-stream", uri.removeprefix("oss://ss-devbox-stream/")

    def _argv(self, args):
        return [str(self.executable), *args, "-c", str(self.config_file),
                "--endpoint", STREAM_ENDPOINT, "--retry-times", "0"]

    def _json(self, args):
        operation = self.state_dir / "requests" / uuid.uuid4().hex
        operation.parent.mkdir(exist_ok=True)
        # API output is small metadata; binary object bodies use iter_object.
        try:
            result = subprocess.run(self._argv(["api", *args, "--output-format", "json", "-q"]),
                                    capture_output=True, timeout=self.timeout)
        except Exception as exc:
            # An interrupted request may have reached OSS. Record the unknown
            # result without including exception text, credentials or URLs.
            write_once(operation.with_suffix(".json"), {
                "operation": args[0], "returncode": None, "outcome": "UNKNOWN",
                "error_type": type(exc).__name__, "automatic_retry": False})
            raise
        # Error bodies can contain signed requests or access-key identifiers.
        # Keep bounded diagnostic text only after removing sensitive lines and
        # URLs; service codes remain useful even when a whole line is removed.
        output = (result.stdout + result.stderr).decode('utf-8', errors='replace')
        messages = []
        for line in output.splitlines()[:16]:
            if re.search(r'(?i)authorization|access.?key|secret|credential|signature|'
                         r'security.?token|string.?to.?sign|canonical.?request|'
                         r'\b(?:hf_|LTAI|sk-)[A-Za-z0-9_+-]+', line):
                messages.append('[sensitive request detail redacted]')
                continue
            line = re.sub(r'https?://\S+', '[URL redacted]', line)
            line = re.sub(r'(?i)\b(?:token|password)\s*[:=]\s*\S+', '[credential redacted]', line)
            line = re.sub(r'[A-Za-z0-9_+/=-]{32,}', '[long token redacted]', line)
            messages.append(line[:300])
        diagnostic = {'service_codes': sorted({code for code in (
            'AccessDenied', 'InvalidAccessKeyId', 'SignatureDoesNotMatch', 'RequestTimeTooSkewed',
            'NoSuchKey', 'NoSuchBucket', 'ServiceUnavailable', 'InternalError', 'RequestTimeout',
            'InvalidArgument', 'InvalidRequest', 'AuthorizationFailed', 'OperationNotSupported',
            'NotImplemented', 'MethodNotAllowed') if re.search(r'\b' + code + r'\b', output)}),
            'http_statuses': sorted(set(re.findall(r'\b[45][0-9]{2}\b', output))),
            'message_redacted': '\n'.join(messages)[:1200]}
        write_once(operation.with_suffix(".json"), {
            "operation": args[0], "returncode": result.returncode,
            "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
            "error_diagnostic": diagnostic if result.returncode else None,
            "automatic_retry": False})
        require(result.returncode == 0, "ossutil API failed; " + json.dumps(diagnostic)
                + "; retained request hashes; no automatic retry")
        require(len(result.stdout) <= 16 * 1024 * 1024, "unexpectedly large API metadata response")
        # DeleteObject may legitimately return HTTP 204 with no serialized
        # metadata. Its caller still requires a fresh exact-key absence check.
        document = {} if args[0] == 'delete-object' and not result.stdout.strip() else json.loads(result.stdout)
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

    def delete_exact_verified(self, uri, expected):
        """Delete one immutable owned object, under the caller's shared lock.

        OSS unversioned deletion has no cross-client compare-and-delete promise.
        Safety here is the suite's shared reference lock, unique immutable keys,
        forbid-overwrite writes, and fresh full-byte verification. The caller
        also proves a remaining recovery copy and journals intent before this
        request. No recursive or prefix deletion is used.
        """
        bucket, key = self._uri(uri)
        require(isinstance(expected, dict) and expected.get('uri') == uri
                and isinstance(expected.get('sha256'), str), 'Exact deletion SHA and identity required')
        current = [r for r in self.list_prefix(uri) if r['uri'] == uri]
        require(len(current) == 1 and all(current[0].get(k) == v for k, v in expected.items()
                                        if k != 'sha256'), 'Remote object metadata changed before deletion')
        h, size = hashlib.sha256(), 0
        for block in self.iter_object(uri):
            size += len(block); h.update(block)
        require(size == expected['bytes'] and h.hexdigest() == expected['sha256'],
                'Remote object bytes changed before deletion')
        require([r for r in self.list_prefix(uri) if r['uri'] == uri] == current,
                'Remote object metadata changed during SHA verification')
        args = ['delete-object', '--bucket', bucket, '--key', key]
        if expected.get('version_id') not in (None, '', 'null'):
            args += ['--version-id', expected['version_id']]
        self._json(args)
        require(not any(r['uri'] == uri for r in self.list_prefix(uri)), 'Exact object remains after deletion')

    def put_if_absent(self, uri, source, expected_sha256):
        bucket, key = self._uri(uri)
        source = safe(source)
        source_fingerprint = fingerprint(source.lstat())
        pin = file_pin(source)
        require(fingerprint(source.lstat()) == source_fingerprint, "upload source changed while pinning")
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
        with self._multipart_lock(state):
            self._put_multipart(bucket, key, source, pin, state, binding, expected_sha256, source_fingerprint)

    @contextmanager
    def _multipart_lock(self, state):
        # The caller also owns its run lock. This lock prevents two store
        # instances from dispatching the same object's requests concurrently.
        path = safe(state / ".upload.lock")
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                    and info.st_uid == os.getuid(), "unsafe multipart lock")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("multipart upload already active for this object") from None
            yield
        finally:
            os.close(fd)

    def _put_multipart(self, bucket, key, source, pin, state, binding, expected_sha256, source_fingerprint):
        protocol = safe(state / "PARALLEL_PROTOCOL.json")
        parallel = self.part_workers > 1 or protocol.exists()
        if parallel:
            # Persist the protocol so resuming with one worker cannot bypass
            # remote-part reconciliation or ambiguous-request protections.
            write_once(protocol, {"schema_version": 1, "protocol": "verified-parallel-parts"})
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
        if parallel:
            parts = self._parallel_parts(bucket, key, source, pin, state, upload_id, source_fingerprint)
        else:
            parts = self._serial_parts(bucket, key, source, pin, state, upload_id, expected_sha256)
        check_pin(pin)
        completion = {"binding": binding, "upload_id": upload_id, "parts": parts}
        write_once(state / "COMPLETE_INTENT.json", completion)
        response = self._json(["complete-multipart-upload", "--bucket", bucket, "--key", key,
                               "--upload-id", upload_id, "--complete-multipart-upload",
                               json.dumps({"Part": parts}, separators=(",", ":")),
                               "--forbid-overwrite", "true"])
        write_once(state / "COMPLETED.json", {"binding": binding, "response": response})
        check_pin(pin)

    def _serial_parts(self, bucket, key, source, pin, state, upload_id, expected_sha256):
        # Keep the default request sequence and existing serial journals
        # compatible. Parallel journals always use the verified path above.
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
        return parts

    def _list_parts(self, bucket, key, upload_id):
        """Fresh, paginated inventory for exactly one known multipart upload."""
        marker, result = 0, {}
        while True:
            args = ["list-parts", "--bucket", bucket, "--key", key,
                    "--upload-id", upload_id, "--max-parts", "1000"]
            if marker:
                args += ["--part-number-marker", str(marker)]
            value = self._json(args)
            require(value.get("Bucket") == bucket and value.get("Key") == key
                    and value.get("UploadId") == upload_id, "multipart list identity differs")
            if "PartNumberMarker" in value:
                require(str(value["PartNumberMarker"]) == str(marker), "multipart list marker differs")
            rows = value.get("Part") or []
            if isinstance(rows, dict):
                rows = [rows]
            require(isinstance(rows, list), "invalid multipart list entries")
            for row in rows:
                require(isinstance(row, dict), "invalid multipart list entry")
                number, size, etag = row.get("PartNumber"), row.get("Size"), row.get("ETag")
                require(type(number) in (int, str) and str(number).isdigit()
                        and 1 <= int(number) <= 10000 and int(number) > marker
                        and int(number) not in result
                        and type(size) in (int, str) and str(size).isdigit() and int(size) > 0
                        and isinstance(etag, str) and etag, "invalid or duplicate multipart list part")
                result[int(number)] = {"bytes": int(size), "etag": etag}
            truncated = value.get("IsTruncated")
            require(truncated in (True, False, "true", "false"), "unknown multipart truncation state")
            if truncated in (False, "false"):
                return result
            following = value.get("NextPartNumberMarker")
            require(type(following) in (int, str) and str(following).isdigit()
                    and rows and int(following) == max(result) and int(following) > marker,
                    "invalid or repeated multipart pagination marker")
            marker = int(following)

    def _verified_part_records(self, state, pin, upload_id, remote):
        total = math.ceil(pin["bytes"] / self.PART_BYTES)
        records = {}
        for record in state.glob("part-?????.json"):
            require(re.fullmatch(r"part-[0-9]{5}\.json", record.name), "invalid multipart journal name")
            index = int(record.name[5:10])
            value = json.loads(safe(record).read_text())
            wanted = min(self.PART_BYTES, pin["bytes"] - (index - 1) * self.PART_BYTES)
            require(1 <= index <= total and isinstance(value, dict)
                    and value.get("part_number") == index and value.get("bytes") == wanted
                    and value.get("source_sha256") == pin["sha256"]
                    and value.get("upload_id", upload_id) == upload_id
                    and isinstance(value.get("etag"), str) and value["etag"],
                    "multipart part journal differs")
            if "part_sha256" in value:
                require(isinstance(value["part_sha256"], str)
                        and re.fullmatch(r"[a-f0-9]{64}", value["part_sha256"]), "invalid part SHA journal")
            require(remote.get(index) == {"bytes": wanted, "etag": value["etag"]},
                    "remote multipart part differs from successful journal")
            records[index] = value
        require(set(remote) == set(records),
                "remote part has no successful local receipt; reconcile ambiguous part before resuming")
        return records

    @staticmethod
    def _remove_part_scratch(path):
        path = safe(path)
        if path.exists():
            info = path.lstat()
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_uid == os.getuid(),
                    "unsafe owned multipart scratch")
            path.unlink()

    def _parallel_parts(self, bucket, key, source, pin, state, upload_id, source_fingerprint):
        # A failed request is retried only on an explicit later invocation,
        # after a fresh exact-upload inventory confirms that part is absent.
        require(fingerprint(source.lstat()) == source_fingerprint, "multipart upload source changed")
        remote = self._list_parts(bucket, key, upload_id)
        records = self._verified_part_records(state, pin, upload_id, remote)
        # Clear only our stale buffers under the object lock before reserving
        # new ones; lowering the worker count must not leave extra buffers.
        for path in state.iterdir():
            if path.name == "part.pending" or re.fullmatch(r"part-[0-9]{5}\.pending", path.name):
                self._remove_part_scratch(path)
        missing = iter(index for index in range(1, math.ceil(pin["bytes"] / self.PART_BYTES) + 1)
                       if index not in records)
        failed = threading.Event()
        error = None
        with ThreadPoolExecutor(max_workers=self.part_workers, thread_name_prefix="oss-part") as executor:
            active = set()

            def refill():
                while len(active) < self.part_workers and not failed.is_set():
                    index = next(missing, None)
                    if index is None:
                        break
                    active.add(executor.submit(self._upload_parallel_part, bucket, key, source,
                                               pin, state, upload_id, index, failed, source_fingerprint))

            refill()
            while active:
                done, active = wait(active, return_when=FIRST_COMPLETED)
                # Consume every finished future before replenishing the
                # window; a failure stops dispatch while inflight calls settle.
                for future in done:
                    try:
                        future.result()
                    except BaseException as exc:
                        failed.set()
                        if error is None:
                            error = exc
                if error is None:
                    refill()
        if error is not None:
            raise error
        remote = self._list_parts(bucket, key, upload_id)
        records = self._verified_part_records(state, pin, upload_id, remote)
        require(len(records) == math.ceil(pin["bytes"] / self.PART_BYTES), "multipart upload incomplete")
        require(fingerprint(source.lstat()) == source_fingerprint, "multipart upload source changed")
        return [{"PartNumber": index, "ETag": records[index]["etag"]} for index in sorted(records)]

    def _upload_parallel_part(self, bucket, key, source, pin, state, upload_id, index, failed,
                              source_fingerprint):
        scratch = safe(state / f"part-{index:05}.pending")
        wanted = min(self.PART_BYTES, pin["bytes"] - (index - 1) * self.PART_BYTES)
        try:
            remaining, h = wanted, hashlib.sha256()
            source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(source_fd, "rb", buffering=0) as stream, scratch.open("xb") as target:
                info = os.fstat(stream.fileno())
                require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                        and info.st_size == pin["bytes"] and fingerprint(info) == source_fingerprint,
                        "multipart upload source changed")
                stream.seek((index - 1) * self.PART_BYTES)
                block = memoryview(bytearray(min(8 * 1024 * 1024, wanted)))
                while remaining:
                    count = stream.readinto(block[:min(len(block), remaining)])
                    require(count, "upload source shortened")
                    target.write(block[:count])
                    h.update(block[:count])
                    remaining -= count
                require(fingerprint(os.fstat(stream.fileno())) == source_fingerprint,
                        "multipart upload source changed while copying")
                target.flush()
                os.fsync(target.fileno())
            intent = {"part_number": index, "bytes": wanted, "source_sha256": pin["sha256"],
                      "upload_id": upload_id, "part_sha256": h.hexdigest()}
            write_once(state / f"part-{index:05}.intent.json", intent)
            attempt = state / f"part-{index:05}-requests" / uuid.uuid4().hex
            write_once(attempt.with_suffix(".intent.json"), intent)
            try:
                response = self._json(["upload-part", "--bucket", bucket, "--key", key,
                                       "--upload-id", upload_id, "--part-number", str(index),
                                       "--body", "file://" + str(scratch)])
                value = {**intent, "etag": _field(response, "etag")}
                write_once(state / f"part-{index:05}.json", value)
                write_once(attempt.with_suffix(".result.json"), {"outcome": "SUCCEEDED", **value})
            except BaseException as exc:
                failed.set()
                write_once(attempt.with_suffix(".result.json"), {
                    "outcome": "UNKNOWN", "error_type": type(exc).__name__, "automatic_retry": False})
                raise
        except BaseException:
            failed.set()
            raise
        finally:
            self._remove_part_scratch(scratch)
