"""Exercise long-lived budget clients on a real local POSIX filesystem.

These processes use real flock, reads, fsync and atomic replacement. Passing
does not establish that a remote shared filesystem provides cross-node flock
or cache coherence; that requires a separate observation on that filesystem.
"""
from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
from queue import Empty
import time
import traceback

import pytest

from deepprobe_vlm import suite_storage as storage


WORKERS = 4
ROUNDS = 5


def _empty_remote():
    return {"complete": True, "objects": [], "multipart": []}


def _zero_local(_roots):
    # Isolate reservation arithmetic from the growing test journal itself.
    return {"bytes": 0, "files": 0, "inventory_sha256": "a" * 64}


def _owner(worker, round_number):
    return {
        "fleet_id": "real-process-budget-regression",
        "worker_id": f"worker-{worker}",
        "job_uid": f"job-{worker}",
        "task_id": f"round-{round_number}",
        "claim_id": f"claim-{worker}-{round_number}",
    }


def _quiescent(owner):
    return {"owner": owner, "active_writers": [], "active_uploads": []}


def _disk_bytes(budget):
    # Independently inspect durable bytes rather than an implementation cache.
    with budget.lock():
        return Path(budget.path).read_bytes()


def _assert_phase(budget, round_number, phase, own_token, worker):
    ledger = json.loads(_disk_bytes(budget))
    rows = ledger["reservations"]
    assert len(rows) == (round_number + 1) * WORKERS
    assert rows[own_token]["owner"] == _owner(worker, round_number)
    expected_owners = {
        tuple(sorted(_owner(w, r).items()))
        for r in range(round_number + 1) for w in range(WORKERS)
    }
    assert {tuple(sorted(row["owner"].items())) for row in rows.values()} == expected_owners
    for token, row in rows.items():
        owner = row["owner"]
        r = int(owner["task_id"].removeprefix("round-"))
        w = int(owner["worker_id"].removeprefix("worker-"))
        closed = r < round_number or phase == "released"
        extended = r < round_number or phase in {"extended", "released"}
        assert row["state"] == ("RELEASED" if closed else "ACTIVE")
        assert row["home_bytes"] == (31 if extended else 11) + w
        assert row["oss_bytes"] == (41 if extended else 21) + w
        events = [event for event in ledger["events"] if event["token"] == token]
        assert [event["kind"] for event in events] == (
            ["RESERVED"] + (["EXTENDED"] if extended else []) + (["RELEASED"] if closed else [])
        )
        if closed:
            assert row["quiescence"] == _quiescent(owner)
    assert len(ledger["events"]) == round_number * WORKERS * 3 + WORKERS * {
        "reserved": 1, "extended": 2, "released": 3,
    }[phase]
    snapshot = budget.snapshot()
    active = 0 if phase == "released" else WORKERS
    assert snapshot["active_reservations"] == active
    assert snapshot["reserved_bytes"] == {
        "home": 0 if not active else sum((31 if phase == "extended" else 11) + w for w in range(WORKERS)),
        "oss": 0 if not active else sum((41 if phase == "extended" else 21) + w for w in range(WORKERS)),
    }


def _long_lived_worker(config, barrier, results, worker):
    try:
        # Keep this instance alive across other processes' atomic replacements.
        budget = storage.SharedBudget(config, _empty_remote, local_scanner=_zero_local)
        barrier.wait()
        tokens = []
        for round_number in range(ROUNDS):
            owner = _owner(worker, round_number)
            token = budget.reserve(owner, 11 + worker, 21 + worker)
            tokens.append(token)
            barrier.wait()
            _assert_phase(budget, round_number, "reserved", token, worker)
            barrier.wait()

            before = _disk_bytes(budget)
            try:
                budget.release(token, lambda exact: _quiescent({**exact, "job_uid": "wrong-job"}))
            except storage.StorageError as error:
                assert "not been proven quiescent" in str(error)
            else:
                raise AssertionError("A different owner's quiescence released an active token")
            try:
                budget.reserve({**owner, "claim_id": owner["claim_id"] + "-over-cap"},
                               config["home_limit_bytes"] - 1, 0)
            except storage.StorageError as error:
                assert "home capacity gated" in str(error)
            else:
                raise AssertionError("Concurrent active reservations were not counted for admission")
            assert _disk_bytes(budget) == before
            barrier.wait()

            budget.extend(token, 31 + worker, 41 + worker)
            barrier.wait()
            _assert_phase(budget, round_number, "extended", token, worker)
            barrier.wait()
            budget.release(token, _quiescent)
            barrier.wait()
            _assert_phase(budget, round_number, "released", token, worker)
            barrier.wait()
        results.put({"status": "ok", "worker": worker, "tokens": tokens})
    except BaseException:
        results.put({"status": "error", "worker": worker, "traceback": traceback.format_exc()})
        barrier.abort()
        raise


@pytest.mark.parametrize("unlimited_oss", [False, True])
def test_four_long_lived_processes_preserve_reservations_and_history(tmp_path, unlimited_oss):
    owned = tmp_path / "owned"
    owned.mkdir()
    config = {
        "schema_version": 2, "roots": [str(owned)], "ledger_root": str(owned / "ledger"),
        "home_limit_bytes": 1_000_000, "oss_limit_bytes": None if unlimited_oss else 1_000_000,
        "oss_prefixes": ["oss://ss-devbox-stream/fixture/process-regression/"],
        "home_safety_bytes": 0, "oss_safety_bytes": 0, "free_guard_bytes": 0,
    }
    context = multiprocessing.get_context("spawn")
    barrier, results = context.Barrier(WORKERS, timeout=15), context.Queue()
    processes = [context.Process(target=_long_lived_worker, args=(config, barrier, results, w))
                 for w in range(WORKERS)]
    deadline = time.monotonic() + 90
    messages = []
    try:
        for process in processes:
            process.start()
        for _ in processes:
            try:
                messages.append(results.get(timeout=max(0.01, deadline - time.monotonic())))
            except Empty:
                pytest.fail(f"Budget worker result timeout; exits={[p.exitcode for p in processes]}")
        for process in processes:
            process.join(timeout=max(0.01, deadline - time.monotonic()))
        assert all(not process.is_alive() and process.exitcode == 0 for process in processes), messages
        assert {message["worker"] for message in messages} == set(range(WORKERS))
        assert all(message["status"] == "ok" for message in messages), messages
        tokens = [token for message in messages for token in message["tokens"]]
        assert len(set(tokens)) == WORKERS * ROUNDS
        ledger = json.loads((owned / "ledger" / "LEDGER.json").read_bytes())
        assert set(ledger["reservations"]) == set(tokens)
        assert all(row["state"] == "RELEASED" for row in ledger["reservations"].values())
        assert len(ledger["events"]) == WORKERS * ROUNDS * 3
    finally:
        # A failed assertion or child crash must never leave blocked children.
        # Do not acquire the barrier's internal lock here: a killed child may
        # have left it held. Reap processes directly, with bounded joins.
        for process in processes:
            if process.pid is not None and process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
                assert not process.is_alive(), f"Could not reap budget worker {process.pid}"
                process.close()
        results.close()
        results.join_thread()
