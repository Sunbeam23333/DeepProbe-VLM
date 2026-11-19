"""Prepare explicitly reviewed historical recovery inputs without GPU work.

Local checkpoints are archived before import. Archived checkpoints are restored
only into a new suite-owned input directory. Both paths perform complete SHA
readback and native metadata checks; numerical restoration remains the Trainer's
responsibility. No historical payload is deleted or changed here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path

from . import suite_bindings as bindings, suite_runner as runner


def save(path, value):
    path = runner.safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        runner.require(runner.read(path) == value, "Recovery artifact differs; use a new output root")
    else:
        runner.write_once(path, value)
    return runner.artifact(path)


def recovery_request(plan, request):
    fields = {"task_id", "operation", "entry", "review", "output_dir"}
    runner.require(isinstance(request, dict) and set(request) == fields
                   and request["operation"] in {"archive_local", "restore_archived", "verify_archived_accepted"}, "Invalid recovery preparation request")
    tasks = {t["id"]: t for t in plan["tasks"]}
    runner.require(request["task_id"] in tasks, "Recovery task is outside the exact scientific plan")
    task = tasks[request["task_id"]]
    for key in ("entry", "review"):
        bindings._pin_shape(request[key]); runner.verify_pin(request[key])
    entry, review = (runner.read(request[key]["path"]) for key in ("entry", "review"))
    accepted = request["operation"] == "verify_archived_accepted"
    runner.require(entry.get("mode") == ("accepted" if accepted else "resume")
                   and "acceptance" not in entry and "archive_acceptance" not in entry,
                   "Historical preparation requires a new explicit matching entry")
    bindings.validate_history_shape(task, {**entry, "acceptance": request["review"]})
    expected = {"status": "SUITE_HISTORY_PREPARATION_REVIEW_ACCEPTED" if accepted else "SUITE_RECOVERY_PREPARATION_REVIEW_ACCEPTED", "approved": True,
                "production_evidence": True, "task_id": task["id"], "task_sha256": runner.digest(task),
                "entry_sha256": runner.digest(entry), "operation": request["operation"]}
    runner.require(all(review.get(k) == v for k, v in expected.items()), "Independent recovery source review is required")
    evidence = review.get("evidence")
    runner.require(isinstance(evidence, list) and evidence, "Recovery review lacks original independent evidence")
    for pin in evidence:
        bindings._pin_shape(pin); runner.verify_pin(pin)
    facts = review.get("verified_facts", {})
    identities = {"source_root": entry["source_root"], "config_sha256": entry["config"]["sha256"],
        "frozen_source_manifest_sha256": entry["frozen_source_manifest"]["sha256"],
        "checkpoint_inventory_sha256": runner.digest(entry["checkpoint"]["files"]),
        "checkpoint_oss_prefix": entry["checkpoint"]["oss_prefix"], "optimizer_step": entry["checkpoint_step"],
        "formal_training_complete" if accepted else "resume_metadata_verified": True,
        "optimizer_scheduler_rng_deserialized": False}
    runner.require(all(facts.get(k) == v for k, v in identities.items()), "Recovery review does not bind the exact native source")
    for key in ("config", "frozen_source_manifest"):
        runner.verify_pin(entry[key])
    return task, entry, review, identities


def prepare_recovery(plan, environment, request, *, resume=False, store_factory=None, budget_factory=None):
    """Archive/restore, independently read back, then normalize the import.

    A supplied source review authenticates historical identity before mutation;
    this function adds only facts freshly measured by the transfer/readback.
    Explicit resume reconciles matching owned outputs; no blind transfer retry.
    """
    from .suite_actions import verify_inventory
    from .checkpoint_store import OssutilStore
    from .suite_storage import budget_for
    from .resume_checkpoint import verify_resume_checkpoint
    from .formal_integrity import verify_frozen_source_manifest
    task, entry, source_review, identities = recovery_request(plan, request)
    output = runner.safe(request["output_dir"])
    roots = [runner.safe(p) for p in environment["storage"]["roots"]]
    runner.require(any(output != root and output.is_relative_to(root) for root in roots),
                   "Recovery metadata escapes accounted home roots")
    checkpoint = runner.safe(entry["checkpoint"]["path"])
    files, prefix = entry["checkpoint"]["files"], entry["checkpoint"]["oss_prefix"]
    runner.require(any(prefix.startswith(p) for p in environment["storage"]["oss_prefixes"]),
                   "Recovery archive escapes accounted OSS project")
    restoring = request["operation"] == "restore_archived"
    verifying_only = request["operation"] == "verify_archived_accepted"
    if restoring:
        owned = runner.safe(environment["run_root"]) / ".recovery-inputs"
        runner.require(checkpoint != owned and checkpoint.is_relative_to(owned),
                       "Archive restore must target a new suite-owned recovery input directory")
        runner.require(resume or not checkpoint.exists(), "Existing recovery input requires explicit asset resume")
    elif not verifying_only:
        verify_inventory(checkpoint, files)
    source = verify_frozen_source_manifest(runner.safe(entry["source_root"]),
        Path(entry["frozen_source_manifest"]["path"]), entry["frozen_source_manifest"]["sha256"])
    total = sum(row["bytes"] for row in files)
    budget = (budget_factory or budget_for)(environment)
    owner = {"suite_id": environment["suite_id"], "operation": request["operation"],
             "task_id": task["id"], "checkpoint_inventory_sha256": identities["checkpoint_inventory_sha256"]}
    # Multipart transfer stages one bounded part; restored inputs are counted in
    # addition to original source files and all other workers' reservations.
    token = budget.reserve(owner, home_bytes=(total if restoring else 0) + 128 * 1024 * 1024,
                           oss_bytes=0 if restoring or verifying_only else total)
    try:
        output.mkdir(parents=True, exist_ok=True)
        save(output / "RECOVERY_REQUEST.json", request)
        store = (store_factory or OssutilStore)(environment["tools"]["ossutil"], environment["tools"]["oss_config"],
            allowed_prefix=prefix, state_dir=output / "transfer-state")
        expected = {prefix + row["path"]: row for row in files}
        before = store.list_prefix(prefix)
        runner.require(len({r["uri"] for r in before}) == len(before) and {r["uri"] for r in before} <= set(expected),
                       "Unexpected recovery archive objects; refusing to adopt a prefix")
        if restoring or verifying_only:
            runner.require({r["uri"] for r in before} == set(expected), "Recovery archive is incomplete")
        else:
            runner.require(resume or not before, "Existing recovery archive requires explicit asset resume")
        observed = {r["uri"]: r for r in before}
        for uri, row in expected.items():
            if uri in observed:
                runner.require(observed[uri]["bytes"] == row["bytes"], "Recovery archive byte count differs")
            elif not restoring and not verifying_only:
                store.put_if_absent(uri, checkpoint / row["path"], row["sha256"])
        stable = store.list_prefix(prefix)
        runner.require({r["uri"] for r in stable} == set(expected), "Recovery archive inventory is incomplete after transfer")
        readback = []
        for uri, row in expected.items():
            target = runner.safe(checkpoint / row["path"])
            temporary, handle = None, None
            if restoring and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = runner.safe(target.with_name(target.name + ".suite-recovery-partial"))
                if temporary.exists():
                    runner.require(resume, "Partial owned recovery file requires explicit asset resume")
                    temporary.unlink()
                handle = temporary.open("xb")
            digest, size = hashlib.sha256(), 0
            try:
                for block in store.iter_object(uri):
                    size += len(block)
                    runner.require(size <= row["bytes"], "Recovery archive object exceeds its pinned size")
                    digest.update(block)
                    if handle is not None: handle.write(block)
                runner.require(size == row["bytes"] and digest.hexdigest() == row["sha256"],
                               "Recovery archive full GET/SHA differs")
                if handle is not None:
                    handle.flush(); os.fsync(handle.fileno()); handle.close(); handle = None
                    os.link(temporary, target); temporary.unlink()
                if restoring:
                    runner.verify_pin({**row, "path": str(target)})
                readback.append({"uri": uri, "bytes": size, "sha256": digest.hexdigest()})
            finally:
                if handle is not None: handle.close()
        runner.require(store.list_prefix(prefix) == stable, "Recovery archive changed during independent full readback")
        if verifying_only:
            native = {"optimizer_step": entry["checkpoint_step"], "formal_training_complete": True,
                      "validation_scope": "independently_reviewed_completion_and_fresh_archive_full_get_sha",
                      "completion_review": request["review"]}
            for metadata in entry.get("checkpoint_metadata", {}).values(): runner.verify_pin(metadata)
        else:
            verify_inventory(checkpoint, files)
            native = verify_resume_checkpoint(checkpoint, world_size=8, expected_max_steps=task["expected_optimizer_steps"])
            runner.require(native["optimizer_step"] == entry["checkpoint_step"], "Recovery native optimizer step differs")
        archive = save(output / "ARCHIVE_ACCEPTANCE.json", {"schema_version": 1,
            "status": "SUITE_RECOVERY_ARCHIVE_VERIFIED", "production_evidence": True,
            "checkpoint_oss_prefix": prefix, "checkpoint_inventory_sha256": identities["checkpoint_inventory_sha256"],
            "all_objects_full_get_sha256_verified": True, "objects": readback,
            "stable_listing_sha256": runner.digest(stable), "source_review": request["review"],
            "gpu_submitted": False, "historical_sources_changed": False})
        native_pin = save(output / "NATIVE_METADATA_VERIFICATION.json", {"schema_version": 1,
            "status": "SUITE_NATIVE_METADATA_VERIFIED", "native": native, "source": source,
            "checkpoint_inventory_sha256": identities["checkpoint_inventory_sha256"],
            "optimizer_scheduler_rng_deserialized": False, "runtime_resume_verified": False})
        entry["archive_acceptance"] = archive
        normalized_review = save(output / "MEASURED_IMPORT_REVIEW.json", {
            "status": "SUITE_HISTORY_IMPORT_REVIEW_ACCEPTED", "approved": True, "production_evidence": True,
            "task_id": task["id"], "task_sha256": runner.digest(task), "entry_sha256": runner.digest(entry),
            "evidence": [request["review"], *deepcopy(source_review["evidence"]), archive, native_pin],
            "verified_facts": {**identities, "archive_verified": True},
            "independent_identity_review": request["review"], "measurement_scope": "fresh_transfer_and_complete_readback"})
        accepted_path = output / "HISTORY_ACCEPTANCE.json"
        if accepted_path.exists():
            runner.require(resume, "Existing history import requires explicit asset resume")
            history = {**entry, "acceptance": runner.artifact(accepted_path)}
            bindings.verify_history_binding(plan, task, history)
        else:
            history = bindings.import_history_entry(plan, task["id"], entry, normalized_review, accepted_path)
        save(output / "RECOVERY_PREPARED.json", {"status": "SUITE_RECOVERY_PREPARED", "history": history,
            "checkpoint_step": entry["checkpoint_step"], "remaining_optimizer_updates": task["expected_optimizer_steps"] - entry["checkpoint_step"],
            "gpu_submitted": False, "runtime_resume_verified": False})
        return history
    finally:
        budget.release(token, quiescent=lambda value: {"owner": value, "active_writers": [], "active_uploads": []})
