"""Build every serial-suite command from one frozen machine environment.

The builder expands the entire scientific DAG; operators do not hand-edit one
command per experiment. It writes a reviewable binding only. Asset acceptance,
fresh capacity, scheduler admission and GPU startup remain required before any
execution. Credentials, machine paths and historical receipts belong in the
private environment file, never in the public scientific plan.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import stat

from . import experiment_suite
from . import suite_runner as runner


RESERVE_KEYS = {"startup", "gate", "train_alignment", "train_sft", "evaluate", "latency",
                "compare_latency", "archive", "cleanup", "aggregate"}
REQUIRED_ENVIRONMENT_FIELDS = {
    "schema_version", "suite_id", "experiment_root", "python", "storage", "source_files",
    "runtime_environment", "task_reserve_bytes", "assets", "history", "tools", "run_root",
    "allocation_identity", "archive_root",
}


def load_modelscope_token_file(path, *, environ=None):
    """Load an owner-only token into process memory, returning no credential data."""
    target = runner.safe(path)
    info = target.lstat()
    runner.require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600
                   and info.st_uid == os.getuid() and info.st_nlink == 1 and 0 < info.st_size <= 4096,
                   "ModelScope token must be a private owner-only regular file")
    with os.fdopen(os.open(target, os.O_RDONLY | os.O_NOFOLLOW), 'r', encoding='utf-8') as stream:
        runner.require(stream.fileno() >= 0 and os.fstat(stream.fileno()).st_ino == info.st_ino,
                       "ModelScope token file identity changed")
        token = stream.read(4097).strip()
    runner.require(token and len(token) <= 4096 and not any(c.isspace() for c in token),
                   "ModelScope token file has invalid content")
    fingerprint = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_uid, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    runner.require(fingerprint(target.lstat()) == fingerprint(info), "ModelScope token file changed while reading")
    (os.environ if environ is None else environ)['MODELSCOPE_API_TOKEN'] = token
    return {"token_present": True, "credential_source": "private_owner_only_file", "credential_value_recorded": False}


def _pin_shape(value):
    runner.require(isinstance(value, dict) and set(value) == {"path", "bytes", "sha256"}
                   and type(value["bytes"]) is int and value["bytes"] > 0
                   and isinstance(value["sha256"], str) and runner.SHA.fullmatch(value["sha256"]),
                   "history requires complete path/bytes/SHA pins")
    runner.safe(value["path"])


def _checkpoint_shape(checkpoint):
    from .checkpoint_retention import relative
    runner.require(isinstance(checkpoint, dict) and set(checkpoint) == {"path", "files", "oss_prefix"},
                   "history checkpoint needs original path, complete inventory and archive prefix")
    runner.safe(checkpoint["path"])
    prefix = checkpoint["oss_prefix"]
    runner.require(isinstance(prefix, str) and prefix.startswith("oss://ss-devbox-stream/")
                   and "/runs/" in prefix and prefix.endswith("/") and "//" not in prefix[6:]
                   and not any(x in {".", ".."} for x in prefix[6:].split("/")),
                   "history requires an exact stream run archive prefix")
    rows = checkpoint["files"]
    runner.require(isinstance(rows, list) and rows and all(isinstance(r, dict) for r in rows),
                   "history checkpoint inventory is empty")
    for row in rows:
        runner.require(set(row) == {"path", "bytes", "sha256"} and type(row["bytes"]) is int
                       and row["bytes"] > 0 and isinstance(row["sha256"], str)
                       and runner.SHA.fullmatch(row["sha256"]), "invalid checkpoint inventory entry")
        relative(row["path"])
    names = [row["path"] for row in rows]
    runner.require(names == sorted(set(names)), "checkpoint inventory must be sorted and unique")
    required = {"deepprobe_config.json", "training_provenance.json", "projector.safetensors", "language_model/config.json",
                "vision_model/config.json", "tokenizer/tokenizer_config.json", "image_processor/preprocessor_config.json"}
    runner.require(required <= set(names) and all(any(n.startswith(c + "/") and n.endswith(".safetensors") for n in names)
                   for c in ("language_model", "vision_model")), "history checkpoint lacks its complete model components")


def validate_history_shape(task, value):
    """Reject empty historical hints before assigning zero-compute bindings."""
    runner.require(task["kind"] in {"train", "evaluate"} and isinstance(value, dict), "invalid historical task")
    common = {"mode", "acceptance"}
    if task["kind"] == "train":
        required = common | {"config", "frozen_source_manifest", "source_root", "checkpoint", "checkpoint_step"}
        runner.require(required <= set(value) <= required | {"archive_acceptance", "checkpoint_metadata"} and value["mode"] in {"accepted", "resume"},
                       "training history needs an explicit complete accepted/resume binding")
        for key in ("acceptance", "config", "frozen_source_manifest"):
            _pin_shape(value[key])
        if "archive_acceptance" in value:
            _pin_shape(value["archive_acceptance"])
        runner.safe(value["source_root"])
        _checkpoint_shape(value["checkpoint"])
        if "checkpoint_metadata" in value:
            metadata = value["checkpoint_metadata"]
            allowed = {"deepprobe_config.json", "training_provenance.json", "trainer_state.json"}
            runner.require(isinstance(metadata, dict) and metadata and set(metadata) <= allowed,
                           "Historical checkpoint metadata must contain only declared JSON metadata pins")
            rows = {row["path"]: row for row in value["checkpoint"]["files"]}
            for name, pin in metadata.items():
                _pin_shape(pin)
                runner.require(name in rows and all(pin[key] == rows[name][key] for key in ("sha256", "bytes")),
                               "Historical metadata pin differs from exact checkpoint inventory")
        step = value["checkpoint_step"]
        runner.require(type(step) is int and 0 < step <= task["expected_optimizer_steps"], "invalid historical checkpoint step")
        runner.require(value["mode"] != "accepted" or step == task["expected_optimizer_steps"],
                       "accepted training must have its final checkpoint")
        if value["mode"] == "resume":
            names = {r["path"] for r in value["checkpoint"]["files"]}
            native = {"trainer_state.json", "optimizer.pt", "scheduler.pt", "training_args.bin"} | {f"rng_state_{i}.pth" for i in range(8)}
            runner.require(native <= names and {n for n in names if n.startswith("rng_state")} == {f"rng_state_{i}.pth" for i in range(8)},
                           "resume history lacks complete native optimizer/scheduler/eight-rank RNG state")
    else:
        runner.require(set(value) == common | {"evaluation", "checkpoint_inventory_sha256"}
                       and value["mode"] == "accepted", "evaluation history requires an accepted complete result")
        _pin_shape(value["acceptance"]); _pin_shape(value["evaluation"])
        runner.require(isinstance(value["checkpoint_inventory_sha256"], str)
                       and runner.SHA.fullmatch(value["checkpoint_inventory_sha256"]), "evaluation needs its exact parent checkpoint inventory")
    return value


def verify_history_binding(plan, task, value, *, verify_files=True):
    """Authenticate imported independent evidence, without restoring any payload.

    This validates reuse eligibility. It does not admit a GPU, deserialize native
    optimizer/RNG state, or establish current archive availability.
    """
    validate_history_shape(task, value)
    runner.require(any(t == task for t in plan["tasks"]), "history task is outside this exact plan")
    if not verify_files:
        return {"status": "HISTORY_SHAPE_VALID_NOT_VERIFIED", "task_id": task["id"], "eligible": False}
    receipt = runner.read(runner.verify_pin(value["acceptance"])["path"])
    return _verify_history_receipt(task, value, receipt)


def _verify_history_receipt(task, value, receipt):
    expected = {"schema_version": 1, "status": "SUITE_HISTORY_ACCEPTED", "production_evidence": True,
                "task_id": task["id"], "task_sha256": runner.digest(task), "mode": value["mode"]}
    runner.require(all(receipt.get(k) == v for k, v in expected.items()), "history acceptance identity or production scope differs")
    evidence = receipt.get("evidence")
    runner.require(isinstance(evidence, list) and evidence, "history independent source evidence is missing")
    for item in evidence:
        _pin_shape(item); runner.verify_pin(item)
    if "archive_acceptance" in value:
        runner.require(value["archive_acceptance"] in evidence, "archive acceptance is not pinned by history review")
    if task["kind"] == "train":
        for key in ("config", "frozen_source_manifest"):
            runner.verify_pin(value[key])
        for pin in value.get("checkpoint_metadata", {}).values():
            runner.verify_pin(pin)
        expected.update(config_sha256=value["config"]["sha256"],
                        frozen_source_manifest_sha256=value["frozen_source_manifest"]["sha256"],
                        checkpoint_inventory_sha256=runner.digest(value["checkpoint"]["files"]),
                        checkpoint_oss_prefix=value["checkpoint"]["oss_prefix"],
                        optimizer_step=value["checkpoint_step"], source_root=value["source_root"], archive_verified=True)
        expected["formal_training_complete" if value["mode"] == "accepted" else "resume_metadata_verified"] = True
    else:
        result = runner.read(runner.verify_pin(value["evaluation"])["path"])
        qa = 2697 if task["benchmark"] == "videomme899" else 1580
        expected.update(evaluation_sha256=value["evaluation"]["sha256"],
                        checkpoint_inventory_sha256=value["checkpoint_inventory_sha256"],
                        protocol=task["protocol"], qa=qa, failed_units=0)
        runner.require(result.get("status") == "SUITE_EVALUATION_COMPLETE" and result.get("suite_protocol") == task["protocol"]
                       and result.get("task_sha256") == runner.digest(task) and result.get("qa") == qa
                       and result.get("failed_units") == 0, "historical evaluation protocol or coverage differs")
        rows = result.get("rows", [])
        runner.require(isinstance(rows, list) and len(rows) == qa
                       and all(isinstance(row, dict) and isinstance(row.get("sample_id"), str)
                               and type(row.get("correct")) is int and row["correct"] in {0, 1}
                               and isinstance(row.get("question_identity_sha256"), str)
                               and runner.SHA.fullmatch(row["question_identity_sha256"]) for row in rows)
                       and len({r["sample_id"] for r in rows}) == qa, "historical evaluation lacks complete unique scored rows")
    runner.require(all(receipt.get(k) == v for k, v in expected.items()), "history receipt differs from exact source/config/checkpoint/result binding")
    return {"status": "HISTORY_REUSE_ELIGIBLE", "task_id": task["id"], "eligible": True,
            "mode": value["mode"], "acceptance": deepcopy(value["acceptance"]),
            "optimizer_step": value.get("checkpoint_step"), "runtime_resume_verified": False,
            "scheduler_admitted": False}


def import_history_entry(plan, task_id, entry, review_pin, output):
    """Normalize a separately reviewed real historical entry; never infer approval.

    The independent review pins the entry (without acceptance), every original
    audit artifact and the verified facts. A mock review cannot produce a
    production acceptance. Only small metadata is written; no model is copied.
    """
    tasks = {t["id"]: t for t in plan["tasks"]}
    runner.require(task_id in tasks and "acceptance" not in entry, "invalid history import target")
    task = tasks[task_id]
    _pin_shape(review_pin)
    review = runner.read(runner.verify_pin(review_pin)["path"])
    runner.require(review.get("status") == "SUITE_HISTORY_IMPORT_REVIEW_ACCEPTED"
                   and review.get("approved") is True and review.get("production_evidence") is True
                   and review.get("task_id") == task_id and review.get("task_sha256") == runner.digest(task)
                   and review.get("entry_sha256") == runner.digest(entry), "independent production history import review required")
    evidence = review.get("evidence")
    runner.require(isinstance(evidence, list) and evidence, "independent review must bind original acceptance evidence")
    for item in evidence:
        _pin_shape(item); runner.verify_pin(item)
    facts = review.get("verified_facts")
    runner.require(isinstance(facts, dict), "review lacks verified facts")
    receipt = {**facts, "schema_version": 1, "status": "SUITE_HISTORY_ACCEPTED", "production_evidence": True,
               "task_id": task_id, "task_sha256": runner.digest(task), "mode": entry["mode"],
               "evidence": [deepcopy(review_pin), *deepcopy(evidence)],
               "import_scope": "existing_independent_evidence_only", "gpu_submitted": False}
    # Validate all facts and referenced bytes before publishing acceptance.
    candidate = {**deepcopy(entry), "acceptance": {"path": str(runner.safe(output)), "bytes": 1, "sha256": "0" * 64}}
    validate_history_shape(task, candidate)
    expected = {"source_root": entry["source_root"], "config_sha256": entry["config"]["sha256"],
                "frozen_source_manifest_sha256": entry["frozen_source_manifest"]["sha256"],
                "checkpoint_inventory_sha256": runner.digest(entry["checkpoint"]["files"]),
                "checkpoint_oss_prefix": entry["checkpoint"]["oss_prefix"], "optimizer_step": entry["checkpoint_step"],
                "archive_verified": True,
                "formal_training_complete" if entry["mode"] == "accepted" else "resume_metadata_verified": True} if task["kind"] == "train" else {
                "evaluation_sha256": entry["evaluation"]["sha256"], "checkpoint_inventory_sha256": entry["checkpoint_inventory_sha256"],
                "protocol": task["protocol"], "qa": 2697 if task["benchmark"] == "videomme899" else 1580, "failed_units": 0}
    runner.require(all(receipt.get(k) == v for k, v in expected.items()), "reviewed facts do not authenticate the imported entry")
    for key in (("config", "frozen_source_manifest") if task["kind"] == "train" else ("evaluation",)):
        runner.verify_pin(entry[key])
    _verify_history_receipt(task, candidate, receipt)
    target = runner.safe(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        runner.require(runner.read(target) == receipt, "Existing history import differs from its reviewed identity")
    else:
        runner.write_once(target, receipt)
    candidate["acceptance"] = runner.artifact(output)
    verify_history_binding(plan, task, candidate)
    return candidate


def remaining_summary(plan, environment):
    """Account for verified history while preserving every operational DAG task."""
    tasks = {t["id"]: t for t in plan["tasks"]}
    rows, verified, blockers = [], Counter(), []
    for tid, task in tasks.items():
        row = {"task_id": tid, "kind": task["kind"], "disposition": "execute"}
        value = environment.get("history", {}).get(tid)
        if value:
            try:
                proof = verify_history_binding(plan, task, value)
                row.update(disposition="verify_and_reuse" if value["mode"] == "accepted" else "restore_and_resume", evidence=proof)
                verified[task["kind"] if value["mode"] == "accepted" else "resume"] += 1
                if value["mode"] == "resume":
                    row.update(checkpoint_step=value["checkpoint_step"], remaining_optimizer_updates=task["expected_optimizer_steps"]-value["checkpoint_step"])
            except (ValueError, KeyError, OSError) as exc:
                row.update(disposition="blocked_history", reason=str(exc)); blockers.append(tid)
        elif task.get("historical_hint", {}).get("status") in {"reuse_candidate", "resume_candidate"}:
            row.update(disposition="blocked_history", reason="Explicit verified history import is required; hints cannot skip or restart work")
            blockers.append(tid)
        rows.append(row)
    totals = Counter(t["kind"] for t in tasks.values())
    return {"status": "REMAINING_SUITE_METADATA_ONLY", "task_count": len(tasks), "task_counts": dict(totals),
            "verified_historical_counts": dict(verified), "remaining_training_stages": totals["train"]-verified["train"],
            "remaining_evaluations": totals["evaluate"]-verified["evaluate"], "blocked_history_tasks": blockers,
            "tasks": rows, "operational_tasks_omitted": 0, "scheduler_admitted": False,
            "gpu_submitted": False, "current_archive_availability_verified": False}


def validate_environment(environment, plan):
    require = runner.require
    require(isinstance(environment, dict) and REQUIRED_ENVIRONMENT_FIELDS <= set(environment)
            and set(environment) <= REQUIRED_ENVIRONMENT_FIELDS | {"task_reserve_by_id", "execution_profile", "benchmark_preparations", "modelscope_token_file", "machine_identity"},
            "invalid suite machine environment fields")
    require(type(environment["schema_version"]) is int and environment["schema_version"] == 1,
            "unsupported suite environment")
    require(isinstance(environment["suite_id"], str) and runner.ID.fullmatch(environment["suite_id"]),
            "invalid execution suite id")
    runner.safe(environment["experiment_root"])
    if "modelscope_token_file" in environment:
        runner.safe(environment["modelscope_token_file"])
    if "machine_identity" in environment:
        require(set(environment["machine_identity"]) == {"uid", "gid"}
                and all(type(v) is int and v > 0 for v in environment["machine_identity"].values()),
                "machine identity requires explicit nonroot UID/GID")
    if "execution_profile" in environment:
        from .suite_fleet import validate_execution_profile
        validate_execution_profile(environment["execution_profile"], suite_id=environment["suite_id"])
    runner.executable_path(environment["python"])
    archive = environment["archive_root"]
    require(isinstance(archive, str)
            and re.fullmatch(r"oss://ss-devbox-stream/[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*/?", archive)
            and not {".", "..", "runs"}.intersection(archive.removeprefix("oss://ss-devbox-stream/").strip("/").split("/")),
            "archive root must be an exact stream project prefix before its per-run /runs/ suffix")
    root = runner.safe(environment["run_root"])
    storage = environment["storage"]
    require(isinstance(storage, dict), "storage policy must be an object")
    if storage.get("schema_version") == 2:
        from .suite_storage import validate_config
        validate_config(storage)
    else:
        require(isinstance(storage, dict) and isinstance(storage.get("roots"), list)
                and len(storage["roots"]) == 2, "two storage roots required for legacy storage v1")
    roots = [runner.safe(x) for x in storage["roots"]]
    require(any(root != p and root.is_relative_to(p) for p in roots), "suite run root escapes accounted storage")
    allocation = environment["allocation_identity"]
    require(isinstance(allocation, dict) and set(allocation) == {"job_name", "namespace", "project_name_prefixes"},
            "allocation identity needs exact Job name, namespace and project prefixes")
    require(isinstance(allocation["job_name"], str)
            and re.fullmatch(r"[a-z0-9][-a-z0-9.]*", allocation["job_name"]), "invalid exact job name")
    require(allocation["namespace"] == "default", "unexpected job namespace")
    prefixes = allocation["project_name_prefixes"]
    require(isinstance(prefixes, list) and prefixes and len(set(prefixes)) == len(prefixes)
            and all(isinstance(x, str) and x for x in prefixes), "project job prefixes required")
    require(any(allocation["job_name"].startswith(x) for x in prefixes), "exact suite Job not covered by project audit")
    tools = environment["tools"]
    require(isinstance(tools, dict) and set(tools) == {"kubectl", "kubeconfig", "ossutil", "oss_config"},
            "exact kube/OSS tool and configuration paths required")
    for key, value in tools.items():
        (runner.executable_path if key in {"kubectl", "ossutil"} else runner.safe)(value)
    assets = environment["assets"]
    require(isinstance(assets, dict) and {"training", "benchmarks", "backbones", "development"} <= set(assets)
            and all(isinstance(assets[k], dict) for k in ("training", "benchmarks", "backbones", "development")),
            "training, benchmark, backbone and development asset bindings required")
    # The scientific action adapter validates each asset's exact provenance and
    # protocol. Keep their nested structure opaque here so an old accepted run
    # is never reduced to a guessed generic status/manifest layout.
    runner.canonical(assets)
    ids = {task["id"] for task in plan["tasks"]}
    history = environment["history"]
    require(isinstance(history, dict) and set(history) <= ids, "history includes an undeclared task")
    tasks = {task["id"]: task for task in plan["tasks"]}
    for task_id, binding in history.items():
        require(tasks[task_id]["kind"] in {"train", "evaluate"}, "history applies only to training or accepted evaluation")
        require(isinstance(binding, dict) and binding.get("mode") in {"accepted", "resume"},
                "history must explicitly select accepted reuse or exact resume")
        require(not (binding["mode"] == "resume" and tasks[task_id]["kind"] != "train"),
                "historical resume applies to training only")
        validate_history_shape(tasks[task_id], binding)
    reserves = environment["task_reserve_bytes"]
    require(isinstance(reserves, dict) and set(reserves) == RESERVE_KEYS
            and all(type(value) is int and value >= 0 for value in reserves.values()),
            "every task category requires an explicit nonnegative storage reserve")
    overrides = environment.get("task_reserve_by_id", {})
    require(isinstance(overrides, dict) and set(overrides) <= ids
            and all(type(value) is int and value >= 0 for value in overrides.values()),
            "invalid task-specific storage reserve")
    return environment


def reserve_for(task, environment):
    override = environment.get("task_reserve_by_id", {})
    if task["id"] in override:
        return override[task["id"]]
    if environment["history"].get(task["id"], {}).get("mode") == "accepted":
        # Reuse itself is receipt verification. Any actual restore inside the
        # adapter must pass its separate exact-byte storage admission first.
        return 0
    key = "train_" + task["stage"] if task["kind"] == "train" else task["kind"]
    return environment["task_reserve_bytes"][key]


def build_binding(plan, environment, *, plan_path, plan_sha256, environment_path,
                  environment_sha256, config_root=None):
    """Generate complete exact argv/receipt bindings for every declared task."""
    experiment_suite.validate_plan(plan, config_root=config_root)
    validate_environment(environment, plan)
    runner.safe(plan_path)
    runner.safe(environment_path)
    runner.require(isinstance(plan_sha256, str) and runner.SHA.fullmatch(plan_sha256)
                   and isinstance(environment_sha256, str) and runner.SHA.fullmatch(environment_sha256),
                   "exact plan and environment byte hashes required")
    python = environment["python"]
    common = [python, "-B", "-m", "deepprobe_vlm.suite_actions", "--plan", str(plan_path),
              "--plan-sha256", plan_sha256, "--environment", str(environment_path),
              "--environment-sha256", environment_sha256, "--state-root", "{state_root}"]
    input_pins = [{"path": str(plan_path), "sha256": plan_sha256},
                  {"path": str(environment_path), "sha256": environment_sha256}]
    env = deepcopy(environment["runtime_environment"])
    expected_pythonpath = str(Path(environment["experiment_root"]) / "src")
    runner.require(env.get("PYTHONPATH", expected_pythonpath) == expected_pythonpath,
                   "suite action import path must be its frozen experiment source")
    env["PYTHONPATH"] = expected_pythonpath
    startup_receipt = "{invocation_dir}/STARTUP.json"
    startup = {"argv": common + ["--task", "startup", "--output-receipt", startup_receipt],
               "cwd": environment["experiment_root"], "inputs": deepcopy(input_pins),
               "reserve_bytes": environment["task_reserve_bytes"]["startup"],
               "receipts": [{"path": startup_receipt, "checks": [
                   {"pointer": "/status", "equals": "SUITE_STARTUP_ACCEPTED"}]}]}
    task_bindings = {}
    for task in plan["tasks"]:
        receipt = "{task_attempt_dir}/TASK_VERIFIED.json"
        argv = common + ["--task", task["id"], "--output-receipt", receipt]
        task_bindings[task["id"]] = {
            "mode": "execute", "argv": argv, "cwd": environment["experiment_root"],
            "inputs": deepcopy(input_pins), "reserve_bytes": reserve_for(task, environment),
            "receipts": [{"path": receipt, "checks": [
                {"pointer": "/status", "equals": "SUITE_TASK_ACCEPTED"},
                {"pointer": "/task_id", "equals": task["id"]}]}],
            "resume": {"argv": argv + ["--resume"]},
        }
    binding = {"schema_version": 1, "suite_id": environment["suite_id"], "plan_sha256": plan_sha256,
               "allocation": dict(runner.ALLOCATION), "source_files": deepcopy(environment["source_files"]),
               "storage": deepcopy(environment["storage"]), "environment": env,
               "startup": startup, "tasks": task_bindings}
    if "execution_profile" in environment:
        from .suite_fleet import profile_allocation
        binding.update(schema_version=2, execution_profile=deepcopy(environment["execution_profile"]),
                       allocation=profile_allocation(environment["execution_profile"]))
    return runner.validate_binding(plan, binding, plan_sha256=plan_sha256)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--environment-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-root", type=Path)
    args = parser.parse_args(argv)
    runner.verify_pin({"path": str(args.plan), "sha256": args.plan_sha256})
    runner.verify_pin({"path": str(args.environment), "sha256": args.environment_sha256})
    binding = build_binding(runner.read(args.plan), runner.read(args.environment),
                            plan_path=args.plan, plan_sha256=args.plan_sha256,
                            environment_path=args.environment, environment_sha256=args.environment_sha256,
                            config_root=args.config_root)
    runner.write_once(args.output, binding)
    print(json.dumps({"status": "SUITE_BINDING_WRITTEN_NOT_ADMITTED", "task_count": len(binding["tasks"]),
                      "binding": runner.artifact(args.output), "source_identity_sha256": runner.source_identity(binding),
                      "asset_bytes_verified": False, "gpu_used": False, "scheduler_submitted": False},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
