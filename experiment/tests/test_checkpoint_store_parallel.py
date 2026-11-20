"""CPU-only multipart contract tests; no network or production speed claims."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time
from unittest.mock import patch

import pytest

from deepprobe_vlm.checkpoint_retention import digest, file_pin
from deepprobe_vlm.checkpoint_store import OssutilStore


def make_store(root, workers=1):
    executable, config = root / "ossutil", root / "config"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    config.write_text("fixture-not-a-credential")
    return OssutilStore(executable, config, allowed_prefix="oss://ss-devbox-stream/test/runs/parallel/",
                        state_dir=root / "state", part_workers=workers)


class FakeMultipart:
    """Blocking fake API: multiple upload calls really execute concurrently."""
    def __init__(self, store, source_bytes=b"abcdefghijklmnopqrstuvwxyz"):
        self.store = store
        store.PART_BYTES = 4
        self.source = store.state_dir.parent / "source"
        self.source.write_bytes(source_bytes)
        self.sha = file_pin(self.source)["sha256"]
        self.uri = store.allowed_prefix + "weight"
        self.key = self.uri.removeprefix("oss://ss-devbox-stream/")
        self.state = store.state_dir / "multipart" / digest(self.uri)
        self.lock = threading.Lock()
        self.calls, self.uploads, self.finished = [], [], []
        self.parts, self.completed = {}, None
        self.active = self.peak = 0
        self.hook = lambda number: None
        self.completed_events = {i: threading.Event() for i in range(1, 101)}
        self.fail_after_save = set()
        self.special_failure = None

    def run(self):
        with patch.object(self.store, "list_prefix", return_value=[]), \
                patch.object(self.store, "_json", side_effect=self.api):
            self.store.put_if_absent(self.uri, self.source, self.sha)

    def api(self, args):
        operation = args[0]
        with self.lock:
            self.calls.append(list(args))
        if operation == self.special_failure:
            raise TimeoutError("fixture request outcome unknown")
        if operation == "initiate-multipart-upload":
            return {"UploadId": "same-upload"}
        if operation == "list-parts":
            assert args[args.index("--upload-id") + 1] == "same-upload"
            with self.lock:
                rows = [{"PartNumber": str(n), "Size": str(len(v["body"])), "ETag": v["etag"]}
                        for n, v in sorted(self.parts.items())]
            return {"Bucket": "ss-devbox-stream", "Key": self.key, "UploadId": "same-upload",
                    "IsTruncated": False, "Part": rows}
        if operation == "upload-part":
            number = int(args[args.index("--part-number") + 1])
            path = Path(args[args.index("--body") + 1].removeprefix("file://"))
            body = path.read_bytes()
            assert 0 < len(body) <= self.store.PART_BYTES
            if self.store.part_workers > 1 or (self.state / "PARALLEL_PROTOCOL.json").exists():
                assert path.name == f"part-{number:05}.pending"
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
                self.uploads.append(number)
            try:
                self.hook(number)
                etag = '"' + hashlib.md5(body).hexdigest() + '"'
                with self.lock:
                    self.parts[number] = {"body": body, "etag": etag}
                    self.finished.append(number)
                self.completed_events[number].set()
                if number in self.fail_after_save:
                    raise TimeoutError("server stored part but acknowledgement lost")
                return {"Header": {"Etag": [etag]}}
            finally:
                with self.lock:
                    self.active -= 1
        if operation == "complete-multipart-upload":
            assert self.active == 0
            self.completed = json.loads(args[args.index("--complete-multipart-upload") + 1])["Part"]
            return {"ETag": "completed"}
        raise AssertionError("unexpected fake API operation: " + operation)


@pytest.mark.parametrize("workers", [2, 3, 4])
def test_real_concurrency_is_bounded_and_completion_is_ordered(tmp_path, workers):
    store = make_store(tmp_path, workers)
    fake = FakeMultipart(store)
    barrier = threading.Barrier(workers)
    staged_peaks = []
    staged_checked = threading.Event()

    def upload(number):
        if number <= workers:
            barrier.wait(timeout=5)
            # Every initial worker has fully staged exactly its own part.
            if number == 1:
                staged_peaks.append(sum(p.stat().st_size for p in fake.state.glob("part-*.pending")))
                staged_checked.set()
                assert fake.completed_events[2].wait(5)
            else:
                assert staged_checked.wait(5)

    fake.hook = upload
    fake.run()
    assert fake.peak == workers
    assert fake.finished.index(2) < fake.finished.index(1)
    assert max(staged_peaks) <= store.multipart_buffer_bytes == 4 * workers
    assert [part["PartNumber"] for part in fake.completed] == list(range(1, 8))
    assert b"".join(fake.parts[n]["body"] for n in sorted(fake.parts)) == fake.source.read_bytes()
    assert not list(fake.state.glob("*.pending"))
    assert len(list(fake.state.glob("part-*.json"))) == 14  # one intent and one success per part
    requests = list(fake.state.glob("part-*-requests/*.result.json"))
    assert len(requests) == 7
    assert all(json.loads(p.read_text())["outcome"] == "SUCCEEDED" for p in requests)
    guarded = [c for c in fake.calls if c[0] in ("initiate-multipart-upload", "complete-multipart-upload")]
    assert all(c[c.index("--forbid-overwrite") + 1] == "true" for c in guarded)


def test_partial_failure_waits_for_inflight_receipt_and_explicit_resume_reuses_it(tmp_path):
    fake = FakeMultipart(make_store(tmp_path, 2))
    both_started = threading.Barrier(2)
    failure_seen, release_success = threading.Event(), threading.Event()

    def upload(number):
        assert number in (1, 2), "no new part may dispatch after the first failure"
        both_started.wait(timeout=5)
        if number == 1:
            failure_seen.set()
            raise RuntimeError("fixture part failed")
        assert release_success.wait(5)

    fake.hook = upload
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fake.run)
        try:
            assert failure_seen.wait(5)
            time.sleep(0.05)  # the uploader must remain blocked on the other inflight request
            assert not future.done()
            assert sorted(fake.uploads) == [1, 2]
        finally:
            release_success.set()
        with pytest.raises(RuntimeError, match="fixture part failed"):
            future.result(timeout=5)
    assert fake.active == 0
    assert json.loads((fake.state / "part-00002.json").read_text())["upload_id"] == "same-upload"
    assert not (fake.state / "part-00001.json").exists()
    assert not (fake.state / "COMPLETE_INTENT.json").exists()
    assert not list(fake.state.glob("*.pending"))
    outcomes = sorted(json.loads(p.read_text())["outcome"]
                      for p in fake.state.glob("part-*-requests/*.result.json"))
    assert outcomes == ["SUCCEEDED", "UNKNOWN"]
    original_receipt = (fake.state / "part-00002.json").read_bytes()
    fake.hook = lambda number: None
    fake.run()
    assert Counter(fake.uploads) == Counter({1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1})
    assert sum(c[0] == "initiate-multipart-upload" for c in fake.calls) == 1
    assert (fake.state / "part-00002.json").read_bytes() == original_receipt
    assert (fake.state / "COMPLETED.json").exists()


def test_ambiguous_part_is_not_adopted_or_overwritten_on_resume_with_one_worker(tmp_path):
    fake = FakeMultipart(make_store(tmp_path, 2))
    fake.fail_after_save.add(1)
    with pytest.raises(TimeoutError):
        fake.run()
    uploads, calls = list(fake.uploads), len(fake.calls)
    # The persisted protocol must survive a later constructor's serial default.
    fake.store.part_workers = 1
    fake.fail_after_save.clear()
    with pytest.raises(ValueError, match="no successful local receipt"):
        fake.run()
    assert fake.uploads == uploads
    assert [c[0] for c in fake.calls[calls:]] == ["list-parts"]
    assert not (fake.state / "COMPLETE_INTENT.json").exists()


@pytest.mark.parametrize("operation, message", [("initiate-multipart-upload", "unknown initiation"),
                                                ("complete-multipart-upload", "unknown completion")])
def test_ambiguous_init_or_completion_never_reissues_request(tmp_path, operation, message):
    fake = FakeMultipart(make_store(tmp_path, 3))
    fake.special_failure = operation
    with pytest.raises(TimeoutError):
        fake.run()
    calls = list(fake.calls)
    fake.special_failure = None
    with pytest.raises(ValueError, match=message):
        fake.run()
    assert fake.calls == calls


def test_serial_default_preserves_prior_request_order_and_buffer_size(tmp_path):
    store = make_store(tmp_path)
    assert store.part_workers == 1
    assert store.multipart_buffer_bytes == 64 * 1024 * 1024
    fake = FakeMultipart(store, b"abcdefghij")
    fake.run()
    assert fake.peak == 1
    assert [c[0] for c in fake.calls] == ["initiate-multipart-upload", "upload-part", "upload-part",
                                        "upload-part", "complete-multipart-upload"]
    assert not (fake.state / "PARALLEL_PROTOCOL.json").exists()


@pytest.mark.parametrize("workers", [0, 5, -1, True, 1.0, "2"])
def test_worker_option_rejects_invalid_values(tmp_path, workers):
    with pytest.raises(ValueError, match="part_workers"):
        make_store(tmp_path, workers)


def test_request_timeout_is_journaled_without_exception_secrets(tmp_path):
    store = make_store(tmp_path, 2)
    error = subprocess.TimeoutExpired("https://signed.example/?secret=hidden", 1)
    with patch("deepprobe_vlm.checkpoint_store.subprocess.run", side_effect=error):
        with pytest.raises(subprocess.TimeoutExpired):
            store._json(["upload-part"])
    logs = list((store.state_dir / "requests").glob("*.json"))
    assert len(logs) == 1
    assert json.loads(logs[0].read_text()) == {"operation": "upload-part", "returncode": None,
        "outcome": "UNKNOWN", "error_type": "TimeoutExpired", "automatic_retry": False}
    assert "hidden" not in logs[0].read_text()


def test_list_parts_pagination_binds_upload_and_rejects_duplicate_parts(tmp_path):
    store = make_store(tmp_path, 2)
    common = {"Bucket": "ss-devbox-stream", "Key": "test/runs/parallel/weight", "UploadId": "u"}
    first = {**common, "IsTruncated": True, "PartNumberMarker": "0", "NextPartNumberMarker": "1",
             "Part": {"PartNumber": "1", "Size": "4", "ETag": "one"}}
    second = {**common, "IsTruncated": False, "PartNumberMarker": "1",
              "Part": {"PartNumber": "2", "Size": "3", "ETag": "two"}}
    with patch.object(store, "_json", side_effect=[first, second]) as api:
        assert store._list_parts(common["Bucket"], common["Key"], "u") == {
            1: {"bytes": 4, "etag": "one"}, 2: {"bytes": 3, "etag": "two"}}
        assert api.call_args_list[1].args[0][-2:] == ["--part-number-marker", "1"]
    second["Part"]["PartNumber"] = "1"
    with patch.object(store, "_json", side_effect=[first, second]):
        with pytest.raises(ValueError, match="duplicate multipart list part"):
            store._list_parts(common["Bucket"], common["Key"], "u")
    first["UploadId"] = "another-upload"
    with patch.object(store, "_json", return_value=first):
        with pytest.raises(ValueError, match="identity differs"):
            store._list_parts(common["Bucket"], common["Key"], "u")


def test_changed_remote_part_blocks_resume_without_new_uploads(tmp_path):
    fake = FakeMultipart(make_store(tmp_path, 2))
    fake.special_failure = "complete-multipart-upload"
    with pytest.raises(TimeoutError):
        fake.run()
    # Exercise the part verifier directly: completion ambiguity must itself
    # remain non-resumable, so never remove or rewrite its intent receipt.
    remote = {n: {"bytes": len(v["body"]), "etag": v["etag"]} for n, v in fake.parts.items()}
    remote[1]["etag"] = "changed"
    with pytest.raises(ValueError, match="remote multipart part differs"):
        fake.store._verified_part_records(fake.state, file_pin(fake.source), "same-upload", remote)


def test_concurrent_store_for_same_object_is_rejected_before_any_new_request(tmp_path):
    store = make_store(tmp_path, 2)
    fake = FakeMultipart(store)
    fake.state.mkdir(parents=True)
    with store._multipart_lock(fake.state):
        with pytest.raises(ValueError, match="already active"):
            fake.run()
    assert fake.calls == []


def test_source_change_cannot_publish_completion(tmp_path):
    fake = FakeMultipart(make_store(tmp_path, 2))

    def change_after_upload(number):
        if number == 7:
            fake.source.write_bytes(b"Z" * 26)

    fake.hook = change_after_upload
    with pytest.raises(ValueError, match="source changed|pinned file changed"):
        fake.run()
    assert fake.completed is None
    assert not (fake.state / "COMPLETE_INTENT.json").exists()


def test_source_mutation_after_hash_but_before_parts_cannot_create_bad_receipts(tmp_path):
    fake = FakeMultipart(make_store(tmp_path, 2))
    original = fake.source.read_bytes()
    api = fake.api

    def mutate_during_init(args):
        result = api(args)
        if args[0] == "initiate-multipart-upload":
            fake.source.write_bytes(b"X" * len(original))
        return result

    fake.api = mutate_during_init
    with pytest.raises(ValueError, match="source changed"):
        fake.run()
    assert fake.uploads == []
    assert not list(fake.state.glob("part-*.json"))
    fake.api = api
    fake.source.write_bytes(original)
    fake.run()
    assert b"".join(fake.parts[n]["body"] for n in sorted(fake.parts)) == original
