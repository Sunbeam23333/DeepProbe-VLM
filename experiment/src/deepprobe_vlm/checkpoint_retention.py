"""Owned-run archival and consumer-gated checkpoint retention.

No command in this module submits a GPU job. Public functions default to planning
where destructive work is possible. A caller must hold its suite lock and supply
a live quiescence guard; this module additionally locks the individual run.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import uuid


OWNER = ".suite-owner.json"
REMOTE_MANIFEST = "_suite_archive_manifest.json"
KINDS = {"training", "evaluation", "latency", "comparison"}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def safe(path):
    path = Path(path).absolute()
    require(".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
            "unsafe or symlink path")
    return path


def relative(value):
    require(isinstance(value, str) and value and "\\" not in value, "invalid relative file name")
    p = PurePosixPath(value)
    require(not p.is_absolute() and ".." not in p.parts and p.as_posix() == value,
            "unsafe relative file name")
    return p


def fingerprint(s):
    return (s.st_dev, s.st_ino, s.st_mode, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def file_pin(path):
    path = safe(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as handle:
        before = os.fstat(handle.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
                "only single-link regular files can be attested")
        h = hashlib.sha256()
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
        after = os.fstat(handle.fileno())
    require(fingerprint(before) == fingerprint(after) == fingerprint(path.lstat()),
            "file changed while hashing")
    return {"path": str(path), "bytes": before.st_size, "sha256": h.hexdigest()}


def check_pin(pin):
    require(isinstance(pin, dict) and set(pin) == {"path", "bytes", "sha256"}
            and type(pin["bytes"]) is int and pin["bytes"] >= 0 and valid_sha(pin["sha256"]),
            "invalid file pin")
    require(file_pin(pin["path"]) == pin, "pinned file changed")
    return safe(pin["path"])


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_once(path, value):
    """Exclusive durable JSON publication; an identical existing receipt is reusable."""
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value) + b"\n"
    if path.exists():
        require(path.read_bytes() == data, "existing receipt belongs to a different operation")
        return path
    pending = path.with_name(path.name + ".pending-" + uuid.uuid4().hex)
    with pending.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(pending, path)
        fsync_directory(path.parent)
    finally:
        pending.unlink()
    return path


def claim_owned_run(runs_root, run_id, suite_id, source_sha256):
    """Claim a fresh run only; never turn an existing unowned run into cleanup scope."""
    require(all(isinstance(x, str) and re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", x)
                for x in (run_id, suite_id)), "invalid suite or run ID")
    require(valid_sha(source_sha256), "invalid executable source identity")
    runs = safe(runs_root)
    require(runs.is_dir(), "runs root must already exist")
    root = safe(runs / run_id)
    owner = {"schema_version": 1, "suite_id": suite_id, "run_id": run_id,
             "runs_root": str(runs), "run_root": str(root), "source_sha256": source_sha256}
    if root.exists():
        require(root.is_dir(), "run root is not a directory")
        if (root / OWNER).is_file():
            require(json.loads((root / OWNER).read_text()) == owner, "run ownership differs")
            return owner
        require(not any(root.iterdir()), "cannot claim an existing nonempty unowned run")
    else:
        root.mkdir()
    write_once(root / OWNER, owner)
    return owner


def check_owner(owner):
    require(isinstance(owner, dict) and owner.get("schema_version") == 1,
            "missing pipeline ownership")
    root, runs = safe(owner["run_root"]), safe(owner["runs_root"])
    require(root.parent == runs and root.name == owner["run_id"] and root.is_dir()
            and valid_sha(owner["source_sha256"]), "run escapes its ownership root")
    require(json.loads(safe(root / OWNER).read_text()) == owner, "ownership marker changed")
    return root


def checkpoint_name(value):
    p = relative(value)
    require(len(p.parts) == 2 and p.parts[0] == "training"
            and (p.parts[1] == "final" or re.fullmatch(r"checkpoint-[1-9][0-9]*", p.parts[1])),
            "only owned training/final and training/checkpoint-N are eligible")
    return p.as_posix()


def cleanable(value):
    p = relative(value)
    return (len(p.parts) >= 3 and p.parts[0] == "training"
            and (p.parts[1] == "final" or re.fullmatch(r"checkpoint-[1-9][0-9]*", p.parts[1]))
            and (p.suffix == ".safetensors" or p.name == "optimizer.pt"
                 or re.fullmatch(r"pytorch_model(?:-[0-9]+-of-[0-9]+)?\.bin", p.name)))


def check_prefix(prefix, run_id):
    require(isinstance(prefix, str) and prefix.startswith("oss://ss-devbox-stream/")
            and prefix.endswith("/runs/" + run_id + "/")
            and ".." not in PurePosixPath(prefix.split("/", 3)[3]).parts
            and not any(x.isspace() for x in prefix), "archive must use an exact stream run prefix")


def _check_requirement(row, checkpoints):
    require(isinstance(row, dict) and set(row) == {
        "id", "kind", "checkpoint", "protocol_sha256", "expected_units"}, "invalid consumer requirement")
    require(isinstance(row["id"], str) and row["id"] and row["kind"] in KINDS
            and row["checkpoint"] in checkpoints and valid_sha(row["protocol_sha256"])
            and type(row["expected_units"]) is int and row["expected_units"] > 0,
            "invalid consumer binding or coverage")


def checkpoint_inventory(run_root, checkpoint):
    """The same path-sensitive inventory identity used by manifests and consumers."""
    root = safe(run_root)
    checkpoint_name(checkpoint)
    directory = safe(root / checkpoint)
    require(directory.is_dir(), "checkpoint directory missing")
    rows = []
    for path in sorted(directory.rglob("*")):
        safe(path)
        if path.is_file():
            rows.append({**file_pin(path), "path": path.relative_to(root).as_posix()})
    require(rows and any(cleanable(r["path"]) for r in rows), "checkpoint has no model payload")
    return {"files": rows, "inventory_sha256": digest(rows)}


def prepare_manifest(owner, *, archive_prefix, required_consumers, training_completion, guard):
    """Freeze every stable run payload except owned runtime caches/retention state."""
    root = check_owner(owner)
    check_prefix(archive_prefix, owner["run_id"])
    _quiescent(owner, guard)
    files = []
    for directory, dirs, names in os.walk(root, followlinks=False):
        here = Path(directory)
        for name in list(dirs):
            p = safe(here / name)
            if here == root and name in {".retention", "tmp", "cache"}:
                dirs.remove(name)
        for name in sorted(names):
            p = safe(here / name)
            pin = file_pin(p)
            files.append({**pin, "path": p.relative_to(root).as_posix()})
    files.sort(key=lambda x: x["path"])
    checkpoints = {}
    for row in files:
        if cleanable(row["path"]):
            cp = "/".join(relative(row["path"]).parts[:2])
            checkpoints[cp] = digest([r for r in files if r["path"].startswith(cp + "/")])
    require("training/final" in checkpoints, "a completed final checkpoint is required")
    requirements = sorted(required_consumers, key=lambda x: x["id"])
    require(requirements and len({r["id"] for r in requirements}) == len(requirements),
            "all declared checkpoint consumers must be specified without duplicates")
    for row in requirements:
        _check_requirement(row, checkpoints)
    _quiescent(owner, guard)
    value = {"schema_version": 1, "owner": owner, "archive_prefix": archive_prefix,
             "files": files, "checkpoints": checkpoints, "required_consumers": requirements,
             "training_completion": training_completion,
             "excluded": [".retention/", "tmp/", "cache/"]}
    validate_manifest(value)
    verify_training_completion(value)
    write_once(root / ".retention" / "MANIFEST.json", value)
    return value


def validate_manifest(manifest):
    require(isinstance(manifest, dict) and set(manifest) == {
        "schema_version", "owner", "archive_prefix", "files", "checkpoints", "required_consumers",
        "training_completion", "excluded"}
        and manifest["schema_version"] == 1, "invalid archive manifest")
    owner = manifest["owner"]
    check_prefix(manifest["archive_prefix"], owner["run_id"])
    files = manifest["files"]
    require(isinstance(files, list) and files and len({r["path"] for r in files}) == len(files),
            "empty or duplicated archive file inventory")
    for row in files:
        require(set(row) == {"path", "bytes", "sha256"} and type(row["bytes"]) is int
                and row["bytes"] >= 0 and valid_sha(row["sha256"]), "invalid archive file metadata")
        p = relative(row["path"])
        require(p.parts[0] not in {".retention", "tmp", "cache"}
                and row["path"] != REMOTE_MANIFEST, "reserved archive path")
    expected = {"/".join(relative(r["path"]).parts[:2]) for r in files if cleanable(r["path"])}
    require(set(manifest["checkpoints"]) == expected and "training/final" in expected,
            "checkpoint inventory differs")
    for name, pin in manifest["checkpoints"].items():
        checkpoint_name(name)
        require(pin == digest([r for r in files if r["path"].startswith(name + "/")]),
                "checkpoint byte inventory identity changed")
    required = manifest["required_consumers"]
    require(isinstance(required, list) and required and len({r["id"] for r in required}) == len(required),
            "missing or duplicated consumer declaration")
    for row in required:
        _check_requirement(row, manifest["checkpoints"])
    return manifest


def _verify_outputs(receipt):
    outputs = receipt.get("outputs")
    require(isinstance(outputs, list) and outputs and len({p["path"] for p in outputs}) == len(outputs)
            and receipt.get("output_inventory_sha256") == digest(outputs), "missing bound completion outputs")
    for pin in outputs:
        check_pin(pin)


def verify_training_completion(manifest):
    """Intermediate checkpoints cannot be pruned from a merely stopped run."""
    path = check_pin(manifest["training_completion"])
    receipt = json.loads(path.read_text())
    expected = {"schema_version": 1, "status": "SUITE_TRAINING_COMPLETE",
                "run_id": manifest["owner"]["run_id"], "source_sha256": manifest["owner"]["source_sha256"],
                "final_checkpoint_inventory_sha256": manifest["checkpoints"]["training/final"], "exit_code": 0}
    require(all(type(receipt.get(k)) is type(v) and receipt.get(k) == v for k, v in expected.items()),
            "training completion source/checkpoint/status differs")
    require(type(receipt.get("optimizer_steps")) is int and receipt["optimizer_steps"] > 0
            and type(receipt.get("target_optimizer_steps")) is int
            and receipt["optimizer_steps"] == receipt["target_optimizer_steps"],
            "training has not completed its declared update budget")
    _verify_outputs(receipt)
    return receipt


def _quiescent(owner, guard):
    require(callable(guard), "a live no-writer guard is required")
    result = guard()
    require(isinstance(result, dict) and result.get("run_id") == owner["run_id"]
            and result.get("source_sha256") == owner["source_sha256"]
            and result.get("active_writers") == [], "run has a writer or quiescence is unverified")


@contextmanager
def run_lock(owner):
    root = check_owner(owner)
    directory = safe(root / ".retention")
    directory.mkdir(exist_ok=True)
    with safe(directory / "run.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _objects(manifest):
    files = list(manifest["files"])
    raw = canonical(manifest) + b"\n"
    files.append({"path": REMOTE_MANIFEST, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    return {manifest["archive_prefix"] + r["path"]: r for r in files}


def _listing(store, prefix):
    values = store.list_prefix(prefix)
    require(isinstance(values, list) and len({r["uri"] for r in values}) == len(values),
            "incomplete or duplicate object list")
    for row in values:
        require(isinstance(row, dict) and row["uri"].startswith(prefix)
                and type(row["bytes"]) is int and row["bytes"] >= 0,
                "object listing escaped the exact prefix")
    return {r["uri"]: r for r in sorted(values, key=lambda x: x["uri"])}


def _object_digest(store, uri):
    h, size = hashlib.sha256(), 0
    for block in store.iter_object(uri):
        require(isinstance(block, bytes), "object reader must yield bytes")
        size += len(block)
        h.update(block)
    return {"bytes": size, "sha256": h.hexdigest()}


def verify_archive(manifest, store):
    """Complete streaming GET/SHA verification, with stable complete prefix lists."""
    validate_manifest(manifest)
    expected = _objects(manifest)
    before = _listing(store, manifest["archive_prefix"])
    require(set(before) == set(expected), "archive object inventory is incomplete or has extras")
    for uri, row in expected.items():
        require(before[uri]["bytes"] == row["bytes"] and _object_digest(store, uri) == {
            "bytes": row["bytes"], "sha256": row["sha256"]}, "full object GET/SHA differs")
    after = _listing(store, manifest["archive_prefix"])
    require(before == after, "archive prefix changed during full verification")
    return {"status": "FULL_ARCHIVE_GET_SHA256_VERIFIED", "utc": utc(),
            "manifest_sha256": digest(manifest), "archive_prefix": manifest["archive_prefix"],
            "objects": len(expected), "bytes": sum(r["bytes"] for r in expected.values()),
            "stable_listing_sha256": digest(before), "all_objects_full_get_sha256": True}


def archive_and_verify(manifest, store, *, guard):
    """Resume only matching partial uploads; never overwrite an existing object."""
    validate_manifest(manifest)
    owner = manifest["owner"]
    with run_lock(owner):
        root = check_owner(owner)
        _quiescent(owner, guard)
        manifest_path = write_once(root / ".retention" / "MANIFEST.json", manifest)
        expected = _objects(manifest)
        existing = _listing(store, manifest["archive_prefix"])
        require(set(existing) <= set(expected), "unexpected object in archive prefix")
        # Check the complete local inventory before any upload. A failed check
        # never deletes local state or overwrites an object.
        for row in manifest["files"]:
            check_pin({**row, "path": str(root / row["path"])})
        for uri, row in expected.items():
            _quiescent(owner, guard)
            source = manifest_path if row["path"] == REMOTE_MANIFEST else root / row["path"]
            if uri in existing:
                require(existing[uri]["bytes"] == row["bytes"] and _object_digest(store, uri) == {
                    "bytes": row["bytes"], "sha256": row["sha256"]}, "existing archive object differs")
            else:
                store.put_if_absent(uri, source, row["sha256"])
            require(file_pin(source)["sha256"] == row["sha256"], "source changed during upload")
        receipt = verify_archive(manifest, store)
        _quiescent(owner, guard)
        path = root / ".retention" / ("archive-verification-" + uuid.uuid4().hex + ".json")
        write_once(path, receipt)
        return {**receipt, "receipt": file_pin(path)}


def verify_consumers(manifest, receipt_pins, checkpoints=None):
    selected = set(manifest["checkpoints"] if checkpoints is None else checkpoints)
    requirements = [r for r in manifest["required_consumers"] if r["checkpoint"] in selected]
    require(isinstance(receipt_pins, dict) and set(receipt_pins) == {
        r["id"] for r in requirements}, "not every declared consumer has completed")
    for row in requirements:
        path = check_pin(receipt_pins[row["id"]])
        receipt = json.loads(path.read_text())
        expected = {"schema_version": 1, "status": "SUITE_CONSUMER_COMPLETE", "consumer_id": row["id"],
                    "kind": row["kind"], "source_sha256": manifest["owner"]["source_sha256"],
                    "checkpoint_inventory_sha256": manifest["checkpoints"][row["checkpoint"]],
                    "protocol_sha256": row["protocol_sha256"], "completed_units": row["expected_units"],
                    "failed_units": 0}
        require(all(type(receipt.get(k)) is type(v) and receipt.get(k) == v for k, v in expected.items()),
                "consumer completion, coverage or input identity differs")
        _verify_outputs(receipt)
    return receipt_pins


def plan_cleanup(manifest, consumer_receipts, *, checkpoints=None):
    validate_manifest(manifest)
    check_owner(manifest["owner"])
    verify_training_completion(manifest)
    selected = sorted(manifest["checkpoints"] if checkpoints is None else checkpoints)
    require(selected and len(set(selected)) == len(selected)
            and set(selected) <= set(manifest["checkpoints"]), "invalid checkpoint cleanup selection")
    verify_consumers(manifest, consumer_receipts, selected)
    targets = [r for r in manifest["files"] if cleanable(r["path"])
               and "/".join(relative(r["path"]).parts[:2]) in selected]
    require(targets, "no eligible checkpoint weights")
    return {"schema_version": 1, "status": "CLEANUP_PLAN_REQUIRES_FRESH_ARCHIVE_VERIFICATION",
            "manifest": manifest, "manifest_sha256": digest(manifest),
            "checkpoints": selected,
            "consumer_receipts": consumer_receipts, "targets": targets,
            "reclaim_bytes": sum(r["bytes"] for r in targets)}


def cleanup(plan, store, *, guard, execute=False):
    """Delete only named weights, after fresh archive/consumer checks; resumable per file.

    All local files are checked before the first unlink. A durable per-file intent
    precedes each unlink, so interruption after unlink but before its completion
    receipt remains distinguishable from an unexplained missing checkpoint.
    """
    require(type(execute) is bool, "execute must be Boolean")
    manifest = validate_manifest(plan["manifest"])
    expected_plan = plan_cleanup(manifest, plan["consumer_receipts"], checkpoints=plan["checkpoints"])
    require(plan == expected_plan, "cleanup plan changed")
    if not execute:
        return {"status": "CLEANUP_PLAN_ONLY", "plan_sha256": digest(plan),
                "reclaim_bytes": plan["reclaim_bytes"], "files_deleted": False}
    owner = manifest["owner"]
    with run_lock(owner):
        root = check_owner(owner)
        _quiescent(owner, guard)
        verify_training_completion(manifest)
        verify_consumers(manifest, plan["consumer_receipts"], plan["checkpoints"])
        transaction = safe(root / ".retention" / "cleanup" / digest(plan))
        transaction.mkdir(parents=True, exist_ok=True)
        write_once(transaction / "PLAN.json", plan)
        final_path = transaction / "DELETION.json"
        # A completed transaction is reusable only if deleted files stayed absent
        # and every non-target archived file retained its exact original bytes.
        for row in manifest["files"]:
            path = root / row["path"]
            if cleanable(row["path"]) and not os.path.lexists(path):
                require(_recorded_deletion(root, row, manifest),
                        "checkpoint file missing without a recorded deletion intent")
            else:
                check_pin({**row, "path": str(path)})
        archive = verify_archive(manifest, store)
        write_once(transaction / ("archive-check-" + uuid.uuid4().hex + ".json"), archive)
        _quiescent(owner, guard)
        if final_path.is_file():
            require(all(not os.path.lexists(root / r["path"]) for r in plan["targets"]),
                    "deleted checkpoint unexpectedly reappeared")
            return {**json.loads(final_path.read_text()), "receipt": file_pin(final_path)}
        removed, recovered, prior = [], [], []
        for row in plan["targets"]:
            _quiescent(owner, guard)
            path = safe(root / row["path"])
            intent = transaction / (digest(row) + ".intent.json")
            if not os.path.lexists(path):
                if intent.is_file():
                    require(json.loads(safe(intent).read_text()) == row,
                            "recorded deletion intent differs")
                    recovered.append(row["path"])
                else:
                    require(_recorded_deletion(root, row, manifest),
                            "missing file was not covered by a verified prior transaction")
                    prior.append(row["path"])
            else:
                before = path.lstat()
                check_pin({**row, "path": str(path)})
                write_once(intent, row)
                _quiescent(owner, guard)
                require(fingerprint(path.lstat()) == fingerprint(before), "target changed before unlink")
                path.unlink()
                fsync_directory(path.parent)
                removed.append(row["path"])
            write_once(transaction / (digest(row) + ".done.json"), row)
        receipt = {"schema_version": 1, "status": "OWNED_CHECKPOINT_WEIGHTS_REMOVED",
                   "utc": utc(), "suite_id": owner["suite_id"], "run_id": owner["run_id"],
                   "plan_sha256": digest(plan), "manifest_sha256": digest(manifest),
                   "archive_verification": archive, "files": plan["targets"],
                   "removed_this_invocation": removed, "absent_after_durable_intent": recovered,
                   "absent_after_prior_transaction": prior,
                   "reclaimed_target_bytes": plan["reclaim_bytes"],
                   "non_weight_files_preserved": True, "consumer_receipts": plan["consumer_receipts"]}
        write_once(final_path, receipt)
        return {**receipt, "receipt": file_pin(final_path)}


def _recorded_deletion(root, row, manifest):
    """Allow prior checkpoint-subset cleanup while retaining the final model."""
    for directory in (root / ".retention" / "cleanup").iterdir():
        if not directory.is_dir():
            continue
        safe(directory)
        plan_path, intent = directory / "PLAN.json", directory / (digest(row) + ".intent.json")
        if not plan_path.is_file() or not intent.is_file():
            continue
        previous = json.loads(safe(plan_path).read_text())
        if (directory.name == digest(previous) and previous.get("manifest_sha256") == digest(manifest)
                and digest(previous.get("manifest")) == digest(manifest)
                and row in previous.get("targets", [])
                and json.loads(safe(intent).read_text()) == row):
            return True
    return False


def restore_checkpoint(manifest, checkpoint, destination, store, *, execute=False):
    """Stream one archived checkpoint into a new owned restore directory, without copies."""
    validate_manifest(manifest)
    require(checkpoint in manifest["checkpoints"], "checkpoint not in the archive manifest")
    destination = safe(destination)
    rows = [{**r, "path": r["path"][len(checkpoint) + 1:]} for r in manifest["files"]
            if r["path"].startswith(checkpoint + "/")]
    identity = {"manifest_sha256": digest(manifest), "checkpoint": checkpoint,
                "checkpoint_inventory_sha256": manifest["checkpoints"][checkpoint]}
    if not execute:
        return {"status": "RESTORE_PLAN_ONLY", **identity, "bytes": sum(r["bytes"] for r in rows)}
    if destination.exists():
        require((destination / ".suite-restore.json").is_file()
                and json.loads((destination / ".suite-restore.json").read_text()) == identity,
                "restore destination is not owned by this checkpoint")
    else:
        destination.mkdir(parents=True)
        write_once(destination / ".suite-restore.json", identity)
    before = _listing(store, manifest["archive_prefix"])
    require(set(before) == set(_objects(manifest)), "restore archive inventory differs")
    for row in rows:
        target = safe(destination / row["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            check_pin({**row, "path": str(target)})
            continue
        pending = target.with_name(target.name + ".restore-pending")
        if pending.exists():
            # This exact scratch suffix belongs only to this marked new restore.
            require(pending.is_file() and not pending.is_symlink(), "unsafe partial restore")
            pending.unlink()
        h, count = hashlib.sha256(), 0
        with pending.open("xb") as handle:
            for block in store.iter_object(manifest["archive_prefix"] + checkpoint + "/" + row["path"]):
                count += len(block)
                require(count <= row["bytes"], "restored object exceeds declared length")
                h.update(block)
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
        require(count == row["bytes"] and h.hexdigest() == row["sha256"], "restored object SHA differs")
        os.link(pending, target)
        pending.unlink()
        fsync_directory(target.parent)
    require(before == _listing(store, manifest["archive_prefix"]), "archive changed during restore")
    receipt = {"status": "CHECKPOINT_RESTORED_AND_SHA256_VERIFIED", **identity,
               "files": rows, "bytes": sum(r["bytes"] for r in rows)}
    write_once(destination / "RESTORED.json", receipt)
    return receipt
