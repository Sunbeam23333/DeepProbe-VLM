"""Build every serial-suite command from one frozen machine environment.

The builder expands the entire scientific DAG; operators do not hand-edit one
command per experiment. It writes a reviewable binding only. Asset acceptance,
fresh capacity, scheduler admission and GPU startup remain required before any
execution. Credentials, machine paths and historical receipts belong in the
private environment file, never in the public scientific plan.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re

from . import experiment_suite
from . import suite_runner as runner


RESERVE_KEYS = {"startup", "gate", "train_alignment", "train_sft", "evaluate", "latency",
                "compare_latency", "archive", "cleanup", "aggregate"}
REQUIRED_ENVIRONMENT_FIELDS = {
    "schema_version", "suite_id", "experiment_root", "python", "storage", "source_files",
    "runtime_environment", "task_reserve_bytes", "assets", "history", "tools", "run_root",
    "allocation_identity", "archive_root",
}


def validate_environment(environment, plan):
    require = runner.require
    require(isinstance(environment, dict) and REQUIRED_ENVIRONMENT_FIELDS <= set(environment)
            and set(environment) <= REQUIRED_ENVIRONMENT_FIELDS | {"task_reserve_by_id"},
            "invalid suite machine environment fields")
    require(type(environment["schema_version"]) is int and environment["schema_version"] == 1,
            "unsupported suite environment")
    require(isinstance(environment["suite_id"], str) and runner.ID.fullmatch(environment["suite_id"]),
            "invalid execution suite id")
    runner.safe(environment["experiment_root"])
    runner.executable_path(environment["python"])
    archive = environment["archive_root"]
    require(isinstance(archive, str)
            and re.fullmatch(r"oss://ss-devbox-stream/[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*/?", archive)
            and not {".", "..", "runs"}.intersection(archive.removeprefix("oss://ss-devbox-stream/").strip("/").split("/")),
            "archive root must be an exact stream project prefix before its per-run /runs/ suffix")
    root = runner.safe(environment["run_root"])
    storage = environment["storage"]
    require(isinstance(storage, dict) and isinstance(storage.get("roots"), list)
            and len(storage["roots"]) == 2, "two storage roots required")
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
        runner.canonical(binding)
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
