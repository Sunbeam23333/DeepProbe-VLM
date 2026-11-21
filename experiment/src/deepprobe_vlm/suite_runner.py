"""Durable dependency execution on already allocated eight-GPU worker nodes.

The scientific plan and the machine binding are separate, byte-pinned inputs.
This module never submits a scheduler job, downloads assets or removes weights.
Task adapters perform scientific acceptance and archive/retention operations;
the runner admits one foreground process group at a time and verifies their
retained completion receipts. Resuming is explicit and never silently reruns a
failed command. A low-priority node can still be preempted externally.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time

from .declared_training import staging_bytes
from .distributed_launch import single_node_torchrun


ALLOCATION = {"cluster": "ali-gpu", "queue": "stream", "priority": "low",
              "nodes": 1, "gpus": 8, "restart_policy": "Never"}
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
           "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
ENVIRONMENT_KEYS = set(OFFLINE) | {
    "PATH", "PYTHONPATH", "HF_HOME", "HF_HUB_CACHE", "HF_MODULES_CACHE", "TRANSFORMERS_CACHE",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TMPDIR", "NCCL_DEBUG",
    "NCCL_IB_TIMEOUT", "NCCL_NET_PLUGIN", "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "NCCL_ASYNC_ERROR_HANDLING", "TOKENIZERS_PARALLELISM", "DP_SUITE_KUBECTL_CACHE",
}
CONTEXT_NAMES = {"state_root", "invocation_dir", "task_id", "task_attempt_dir"}
FORBIDDEN_EXECUTABLES = {"sh", "bash", "zsh", "dash", "fish", "csh", "tcsh",
                         "sslaunch", "kubectl", "ssh", "sudo", "su"}
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
SHA = re.compile(r"[0-9a-f]{64}")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe(path):
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts, "absolute physical path required")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink path is not allowed")
    return path


def executable_path(path):
    # Venv Python executables are commonly symlinks to the installed runtime.
    # Keep that invocation path so Python discovers the intended pyvenv.cfg.
    # Runtime identity is checked by the allocation adapter; data/source paths
    # use the stricter physical-path rule above.
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts, "absolute executable path required")
    return path


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_once(path, value):
    """Publish one complete receipt atomically; never truncate an existing one."""
    path = safe(path)
    pending = path.with_name(path.name + ".pending-" + str(os.getpid()))
    payload = canonical(value) + b"\n"
    descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(pending, path)
        _fsync_directory(path.parent)
    finally:
        # This is only the runner's unpublished temporary hard link. It never
        # removes a task artifact, checkpoint or the published receipt.
        try:
            pending.unlink()
        except FileNotFoundError:
            pass


def read(path):
    return json.loads(safe(path).read_text(encoding="utf-8"))


def artifact(path):
    path = safe(path)
    metadata = path.stat()
    require(stat.S_ISREG(metadata.st_mode) and metadata.st_size > 0, "nonempty regular artifact required")
    return {"path": str(path), "bytes": metadata.st_size, "sha256": sha(path)}


def verify_pin(item):
    require(isinstance(item, dict) and set(item) <= {"path", "bytes", "sha256"}
            and {"path", "sha256"} <= set(item), "invalid artifact pin")
    require(isinstance(item["sha256"], str) and SHA.fullmatch(item["sha256"]), "invalid SHA256 pin")
    observed = artifact(item["path"])
    require(observed["sha256"] == item["sha256"] and
            ("bytes" not in item or type(item["bytes"]) is int and observed["bytes"] == item["bytes"]),
            "pinned artifact changed: " + str(item["path"]))
    return observed


def expand(value, context):
    """Resolve only runner-owned path tokens, never shell/environment expansion."""
    require(isinstance(value, str) and value and "\x00" not in value, "nonempty string required")
    names = re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", value)
    require(all(name in CONTEXT_NAMES and name in context for name in names), "unknown or unavailable context token")
    for name in names:
        value = value.replace("{" + name + "}", str(context[name]))
    return value


def _raw_path(value):
    require(isinstance(value, str), "path must be text")
    context = {key: ("task" if key == "task_id" else "/suite-placeholder/" + key)
               for key in CONTEXT_NAMES}
    safe(expand(value, context))


def validate_argv(argv):
    require(isinstance(argv, list) and argv and all(isinstance(x, str) and x and "\x00" not in x for x in argv),
            "command must be a nonempty argv array")
    executable_path(expand(argv[0], {key: ("task" if key == "task_id" else "/suite-placeholder/" + key)
                                     for key in CONTEXT_NAMES}))
    require(Path(argv[0]).name not in FORBIDDEN_EXECUTABLES,
            "suite tasks must use direct executables, not shells, remote clients or scheduler submitters")
    for value in argv:
        expand(value, {key: ("task" if key == "task_id" else "/suite-placeholder/" + key)
                       for key in CONTEXT_NAMES})


def _validate_receipt(spec, *, reuse=False):
    require(isinstance(spec, dict) and set(spec) <= {"path", "sha256", "checks"}
            and {"path", "checks"} <= set(spec), "invalid completion receipt specification")
    _raw_path(spec["path"])
    if reuse or "sha256" in spec:
        require(isinstance(spec.get("sha256"), str) and SHA.fullmatch(spec["sha256"]),
                "reused completion must have a frozen SHA256")
    require(isinstance(spec["checks"], list) and spec["checks"], "completion requires explicit JSON assertions")
    for check in spec["checks"]:
        require(isinstance(check, dict) and set(check) == {"pointer", "equals"}
                and isinstance(check["pointer"], str) and check["pointer"].startswith("/"),
                "receipt checks use JSON pointers and exact expected values")
        canonical(check["equals"])


def _validate_command(command, *, startup=False):
    require(isinstance(command, dict), "command binding must be an object")
    allowed = {"argv", "cwd", "inputs", "receipts", "timeout_seconds", "reserve_bytes", "oss_reserve_bytes", "mode", "resume"}
    require(set(command) <= allowed, "unknown command binding field")
    mode = "execute" if startup else command.get("mode")
    require(mode in {"execute", "reuse"}, "command mode must be execute or reuse")
    inputs = command.get("inputs", [])
    require(isinstance(inputs, list), "inputs must be pinned artifacts")
    for item in inputs:
        require(isinstance(item, dict) and "path" in item and "sha256" in item, "invalid input pin")
        _raw_path(item["path"])
        require(isinstance(item["sha256"], str) and SHA.fullmatch(item["sha256"]), "invalid input SHA")
    require(isinstance(command.get("receipts"), list) and command["receipts"], "retained acceptance receipts are required")
    for item in command["receipts"]:
        _validate_receipt(item, reuse=mode == "reuse")
    if mode == "reuse":
        require(not any(k in command for k in ("argv", "cwd", "resume", "timeout_seconds")),
                "reuse binding cannot execute commands")
    else:
        validate_argv(command.get("argv"))
        _raw_path(command.get("cwd"))
    reserve = command.get("reserve_bytes", 0)
    require(type(reserve) is int and reserve >= 0, "nonnegative task storage reserve required")
    require(type(command.get("oss_reserve_bytes", 0)) is int and command.get("oss_reserve_bytes", 0) >= 0, "invalid OSS reserve")
    timeout = command.get("timeout_seconds")
    require(timeout is None or type(timeout) is int and timeout > 0, "invalid task timeout")
    if "resume" in command:
        resume = command["resume"]
        require(not startup and mode == "execute" and isinstance(resume, dict)
                and set(resume) <= {"argv", "inputs"} and "argv" in resume, "invalid explicit resume command")
        validate_argv(resume["argv"])
        for item in resume.get("inputs", []):
            require(isinstance(item, dict) and set(item) <= {"path", "bytes", "sha256"}
                    and {"path", "sha256"} <= set(item), "invalid resume pin")
            _raw_path(item["path"])
            require(isinstance(item["sha256"], str) and SHA.fullmatch(item["sha256"]), "invalid resume SHA")


def validate_binding(plan, binding, *, plan_sha256):
    require(isinstance(binding, dict) and set(binding) == {
        "schema_version", "suite_id", "plan_sha256", "allocation", "source_files", "storage",
        "environment", "startup", "tasks"} | ({"execution_profile"} if binding.get("schema_version") == 2 else set())
        | ({"isolated_storage"} if 'isolated_storage' in binding else set()), "invalid execution binding fields")
    require(binding["schema_version"] in {1, 2} and type(binding["schema_version"]) is int,
            "unsupported suite execution binding")
    if 'isolated_storage' in binding:
        require(binding['schema_version'] == 2, 'Isolated storage requires a private fleet binding')
        from .suite_isolated_storage import contract
        contract(binding)
    require(isinstance(binding["suite_id"], str) and ID.fullmatch(binding["suite_id"]), "invalid suite id")
    require(binding["plan_sha256"] == plan_sha256 and SHA.fullmatch(plan_sha256), "binding refers to another plan")
    expected_allocation = ALLOCATION
    if binding["schema_version"] == 2:
        from .suite_fleet import validate_execution_profile, profile_allocation
        validate_execution_profile(binding["execution_profile"], binding["suite_id"])
        expected_allocation = profile_allocation(binding["execution_profile"])
    require(canonical(binding["allocation"]) == canonical(expected_allocation), ("allocation differs from frozen execution policy" if binding["schema_version"] == 2 else "only stream/low/one-node/eight-GPU/Never is allowed"))
    sources = binding["source_files"]
    require(isinstance(sources, list) and sources, "complete source file pins required")
    require(len({item.get("path") for item in sources}) == len(sources), "duplicate source pin")
    for item in sources:
        require(isinstance(item, dict) and set(item) == {"path", "bytes", "sha256"}, "source pins include path/size/SHA")
        safe(item["path"])
        require(type(item["bytes"]) is int and item["bytes"] > 0
                and isinstance(item["sha256"], str) and SHA.fullmatch(item["sha256"]), "invalid source pin")
    storage = binding["storage"]
    if binding["schema_version"] == 2:
        from .suite_storage import validate_config
        validate_config(storage)
    else:
        require(isinstance(storage, dict) and set(storage) == {
            "roots", "limit_bytes", "other_reserve_bytes", "free_guard_bytes"}, "invalid storage policy")
        roots = storage["roots"]
        require(isinstance(roots, list) and len(roots) == 2, "two accounted storage roots required")
        roots = [safe(p) for p in roots]
        require(roots[0] != roots[1] and not roots[0].is_relative_to(roots[1])
                and not roots[1].is_relative_to(roots[0]), "storage roots must be disjoint")
        require(type(storage["limit_bytes"]) is int and storage["limit_bytes"] == 99_000_000_000,
                "the 99 GB storage limit cannot be relaxed")
        require(type(storage["other_reserve_bytes"]) is int and storage["other_reserve_bytes"] >= 0
                and type(storage["free_guard_bytes"]) is int and storage["free_guard_bytes"] >= 1_000_000_000,
                "invalid additional storage reserve/free guard")
    env = binding["environment"]
    require(isinstance(env, dict) and set(env) <= ENVIRONMENT_KEYS
            and all(isinstance(v, str) and "\x00" not in v for v in env.values()), "unreviewed runtime environment")
    require(all(env.get(k, v) == v for k, v in OFFLINE.items()), "offline and isolated Python settings cannot be disabled")
    _validate_command(binding["startup"], startup=True)
    require(len(binding["startup"]["receipts"]) == 1, "one current-allocation startup receipt required")
    require(binding["startup"]["receipts"][0]["path"].startswith("{invocation_dir}/"),
            "startup receipt must be fresh for each invocation")
    require(isinstance(plan, dict) and isinstance(plan.get("tasks"), list) and plan["tasks"],
            "nonempty declared task graph required")
    require(all(isinstance(task, dict) and {"id", "kind", "dependencies"} <= set(task)
                and isinstance(task["id"], str) and ID.fullmatch(task["id"])
                and isinstance(task["dependencies"], list) for task in plan["tasks"]), "invalid task graph")
    task_ids = [task["id"] for task in plan["tasks"]]
    require(len(set(task_ids)) == len(task_ids), "duplicate task id")
    require(all(len(set(task["dependencies"])) == len(task["dependencies"])
                and set(task["dependencies"]) <= set(task_ids) and task["id"] not in task["dependencies"]
                for task in plan["tasks"]), "invalid task dependencies")
    require(isinstance(binding["tasks"], dict) and set(binding["tasks"]) == set(task_ids),
            "every declared task needs exactly one executable or accepted-reuse binding")
    for task in plan["tasks"]:
        require(isinstance(task["id"], str) and ID.fullmatch(task["id"]), "unsafe task id")
        _validate_command(binding["tasks"][task["id"]])
    return binding


def source_identity(binding):
    return digest(binding["source_files"])


def verify_sources(binding):
    return [verify_pin(item) for item in binding["source_files"]]


def _pointer(document, pointer):
    value = document
    for part in pointer[1:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            require(part.isdigit() and int(part) < len(value), "receipt JSON pointer missing")
            value = value[int(part)]
        else:
            require(isinstance(value, dict) and part in value, "receipt JSON pointer missing: " + pointer)
            value = value[part]
    return value


def verify_receipts(specs, context):
    pins = []
    for spec in specs:
        path = safe(expand(spec["path"], context))
        observed = artifact(path)
        if "sha256" in spec:
            require(observed["sha256"] == spec["sha256"], "accepted receipt SHA changed")
        require(observed["bytes"] <= 16 * 1024 * 1024, "acceptance receipt exceeds bounded JSON size")
        value = read(path)
        for check in spec["checks"]:
            require(canonical(_pointer(value, check["pointer"])) == canonical(check["equals"]),
                    "receipt assertion failed: " + check["pointer"])
        # Hash again after parsing to reject concurrent mutation of the receipt.
        require(artifact(path) == observed, "acceptance receipt changed while checking")
        pins.append(observed)
    return pins


def machine_environment(binding):
    argv = binding["startup"]["argv"]
    require(argv.count("--environment") == 1 and argv.count("--environment-sha256") == 1,
            "Parallel binding requires one pinned machine environment")
    item = {"path": argv[argv.index("--environment") + 1],
            "sha256": argv[argv.index("--environment-sha256") + 1]}
    verify_pin(item)
    return read(item["path"])


def capacity(binding, reserve_bytes):
    policy = binding["storage"]
    if policy.get("schema_version") == 2:
        from .suite_storage import budget_for, capacity_violations
        environment = machine_environment(binding)
        require(environment['storage'] == policy, 'Binding and machine storage policy differ')
        value = budget_for(environment).snapshot()
        require(not capacity_violations(value, policy, additional_home_bytes=reserve_bytes),
                "shared home or OSS storage capacity exceeded")
        return value
    roots = [safe(p) for p in policy["roots"]]
    require(all(p.is_dir() for p in roots), "storage roots do not exist")
    require(roots[0].stat().st_dev == roots[1].stat().st_dev, "storage roots must share the budgeted filesystem")
    observed = staging_bytes(roots)
    used = sum(observed.values())
    additional = reserve_bytes + policy["other_reserve_bytes"]
    free = shutil.disk_usage(roots[0]).free
    require(used + additional < policy["limit_bytes"], "99 GB projected storage budget exceeded")
    require(free >= additional + policy["free_guard_bytes"], "filesystem free-space guard failed")
    return {"utc": utc(), "root_bytes": observed, "combined_bytes": used,
            "task_reserve_bytes": reserve_bytes, "other_reserve_bytes": policy["other_reserve_bytes"],
            "projected_bytes": used + additional, "limit_bytes": policy["limit_bytes"],
            "filesystem_free_bytes": free, "free_guard_bytes": policy["free_guard_bytes"]}


def runtime_environment(binding, context):
    env = dict(os.environ)
    for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONINSPECT", "BASH_ENV", "ENV", "LD_PRELOAD"):
        env.pop(key, None)
    env.update({k: expand(v, context) if v else v for k, v in binding["environment"].items()})
    env.update(OFFLINE)
    require(env.get("WORLD_SIZE") in (None, "1") and env.get("RANK") in (None, "0")
            and env.get("LOCAL_RANK") in (None, "0"), "suite controller must be one process, not a torchrun rank")
    require(env.get("CUDA_VISIBLE_DEVICES") in (None, "0,1,2,3,4,5,6,7"), "suite controller cannot hide allocated GPUs")
    return env


class Interrupted(RuntimeError):
    def __init__(self, signal_number):
        self.signal_number = signal_number
        super().__init__("suite received signal " + str(signal_number))


def _process_start(pid):
    try:
        # Linux proc field 22 is process start time; preserve it against PID reuse.
        text = Path(f"/proc/{pid}/stat").read_text()
        return text[text.rfind(")") + 2:].split()[19]
    except (OSError, IndexError):
        return None


def _stop_group(process):
    """Only terminate the task process group created by this runner."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        # EPERM still means the group could be alive. The caller is already
        # failing this task; do not hide that failure during best-effort stop.
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    # A process group can outlive its leader. An adapter must not leave a
    # background GPU worker behind after it has returned success.
    if _group_exists(process.pid):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _group_exists(group):
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # A group we cannot inspect is not evidence of quiescence.
        return True
    return True


def _await_group_exit(group):
    # Python multiprocessing resource trackers can exit just after their owner.
    # Wait briefly for normal shutdown; never start the next task meanwhile.
    deadline = time.monotonic() + 2
    while _group_exists(group) and time.monotonic() < deadline:
        time.sleep(.02)
    return not _group_exists(group)


def run_command(command, binding, context, attempt_dir, *, argv=None, extra_inputs=(), extra_environment=None, storage_snapshot=None):
    attempt_dir = safe(attempt_dir)
    expanded_argv = [expand(x, context) for x in (argv or command["argv"])]
    validate_argv(expanded_argv)
    executable = executable_path(expanded_argv[0])
    require(executable.is_file() and os.access(executable, os.X_OK), "command executable missing")
    cwd = safe(expand(command["cwd"], context))
    require(cwd.is_dir(), "command working directory missing")
    source_pins = verify_sources(binding)
    inputs = [verify_pin({**item, "path": expand(item["path"], context)})
              for item in [*command.get("inputs", []), *extra_inputs]]
    storage = storage_snapshot if storage_snapshot is not None else capacity(binding, command.get("reserve_bytes", 0))
    intent = {"schema_version": 1, "status": "COMMAND_INTENT", "utc": utc(),
              "argv": expanded_argv, "cwd": str(cwd), "inputs": inputs,
              "source_identity_sha256": digest(source_pins), "storage": storage,
              "automatic_retry": False}
    write_once(attempt_dir / "COMMAND_INTENT.json", intent)
    stdout_path, stderr_path = attempt_dir / "stdout.log", attempt_dir / "stderr.log"
    returncode, error, cleanup_verified = None, None, False
    started = time.monotonic()
    from .suite_process import run_group
    def child_started(process):
        write_once(attempt_dir / "CHILD_STARTED.json", {"utc": utc(), "pid": process.pid,
            "process_group": process.pid, "hostname": socket.gethostname(),
            "linux_process_start": _process_start(process.pid), "argv_sha256": digest(expanded_argv)})
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            result = run_group(expanded_argv, cwd=cwd, env={**runtime_environment(binding, context),
                **(extra_environment or {})}, stdout=stdout, stderr=stderr,
                timeout=command.get("timeout_seconds", 30 * 24 * 3600), on_started=child_started)
            returncode = result.returncode
            cleanup_verified = result.group_cleanup_verified
    except BaseException as exc:
        cleanup_verified = getattr(exc, "group_cleanup_verified", False)
        returncode = getattr(exc, "group_returncode", None)
        error = {"type": type(exc).__name__, "message": str(exc)}
        if "command leader exited while owned descendants remained" in str(exc):
            raise ValueError("task left live descendants after its command exited") from exc
        raise
    finally:
        if stdout_path.exists() and stderr_path.exists():
            write_once(attempt_dir / "COMMAND_EXIT.json", {"schema_version": 1, "utc": utc(),
                "returncode": returncode, "elapsed_seconds": time.monotonic() - started, "error": error,
                "group_cleanup_verified": cleanup_verified,
                "stdout": {"path": str(stdout_path), "bytes": stdout_path.stat().st_size, "sha256": sha(stdout_path)},
                "stderr": {"path": str(stderr_path), "bytes": stderr_path.stat().st_size, "sha256": sha(stderr_path)},
                "automatic_retry": False})
    require(returncode == 0, "task command failed with exit " + str(returncode))
    verify_sources(binding)
    return verify_receipts(command["receipts"], context)


def validate_startup_receipt(path, binding):
    value = read(path)
    require(value.get("status") == "SUITE_STARTUP_ACCEPTED", "current GPU allocation startup did not pass")
    require(canonical(value.get("allocation")) == canonical(binding["allocation"]), "actual allocation differs from suite policy")
    require(value.get("source_identity_sha256") == source_identity(binding), "startup source identity differs")
    require(value.get("other_active_project_jobs") == [] and value.get("gpu_count") == 8,
            "startup has another project job or wrong GPU count")
    uuids = value.get("gpu_uuids")
    require(isinstance(uuids, list) and len(uuids) == len(set(uuids)) == 8
            and all(isinstance(x, str) and x for x in uuids), "startup requires eight actual distinct GPU UUIDs")
    require(isinstance(value.get("job_uid"), str) and value["job_uid"]
            and value.get("hardware_preflight_passed") is True, "current job identity/hardware preflight missing")
    if binding.get("execution_profile"):
        require(value.get("gpu_product") == binding["execution_profile"]["gpu_product"],
                "Actual GPU hardware does not match the B300 execution profile")
    return value


def _journal(root, event):
    path = root / "events.jsonl"
    event = {"utc": utc(), **event}
    with path.open("ab") as stream:
        stream.write(canonical(event) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _new_numbered(parent, prefix):
    parent = safe(parent)
    parent.mkdir(exist_ok=True)
    names = [p.name for p in parent.iterdir() if p.name.startswith(prefix)]
    require(all(re.fullmatch(re.escape(prefix) + r"[0-9]{6}", name) for name in names), "unexpected attempt directory")
    number = max([int(name[len(prefix):]) for name in names] or [0]) + 1
    path = parent / f"{prefix}{number:06d}"
    safe(path).mkdir()
    _fsync_directory(parent)
    return path


def completion_path(state_root, task_id):
    """Keep a typed import distinct from a task executed by this fleet."""
    require(isinstance(task_id, str) and ID.fullmatch(task_id), "unsafe task id")
    root = safe(state_root)
    executed = root / "tasks" / task_id / "COMPLETED.json"
    imported = root / "task-imports" / (task_id + ".json")
    require(not (executed.exists() and imported.exists()), "Task has conflicting executed/imported completion records")
    return imported if imported.exists() else executed


def completion_for(state_root, task_id):
    require(isinstance(task_id, str) and ID.fullmatch(task_id), "unsafe task id")
    path = completion_path(state_root, task_id)
    if path == safe(state_root) / "task-imports" / (task_id + ".json"):
        from .suite_task_import import get_task_import
        return get_task_import(state_root, task_id)
    value = read(path)
    require(value.get("status") == "TASK_ACCEPTED" and value.get("task_id") == task_id,
            "invalid task completion")
    for item in value["receipts"]:
        verify_pin(item)
    return value


def _previous_child_quiet(task_root):
    for path in task_root.glob("attempts/attempt-*/CHILD_STARTED.json"):
        if (path.parent / "COMMAND_EXIT.json").exists():
            continue
        value = read(path)
        if value.get("hostname") != socket.gethostname():
            # The fresh startup's complete project-job audit handles prior nodes.
            continue
        pid = value.get("pid")
        require(type(pid) is int and pid > 1, "invalid previous child process identity")
        current = _process_start(pid)
        if current is not None and current == value.get("linux_process_start"):
            raise ValueError("previous task process is still alive; do not launch a duplicate")
        if current is None and value.get("linux_process_start") is None:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError("cannot rule out a still-active previous task PID")


def run_bound_suite(plan, binding, *, plan_sha256, binding_sha256, state_root, resume=False, relocation_pin=None):
    """Run ready DAG tasks serially; only explicitly declared recovery may retry."""
    validate_binding(plan, binding, plan_sha256=plan_sha256)
    require(binding["schema_version"] == 1, "Use a named fleet worker for binding v2")
    require(not relocation_pin or resume, "allocation relocation requires explicit resume")
    require(SHA.fullmatch(binding_sha256), "invalid binding SHA256")
    state_root = safe(state_root)
    roots = [safe(p) for p in binding["storage"]["roots"]]
    require(any(state_root != root and state_root.is_relative_to(root) for root in roots),
            "suite state must live within an accounted storage root")
    lock_root = roots[0] / "locks"
    lock_root.mkdir(exist_ok=True)
    safe(lock_root)
    with (lock_root / "experiment-suite.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another experiment suite is active") from exc
        if resume:
            require(state_root.is_dir(), "resume requires existing suite state")
            from .suite_relocation import resume_bindings
            accepted_binding_shas = resume_bindings(state_root, plan_sha256=plan_sha256,
                binding_sha256=binding_sha256, suite_id=binding["suite_id"], source_sha256=source_identity(binding),
                relocation_pin=relocation_pin, apply=True)
        else:
            accepted_binding_shas = {binding_sha256}
            state_root.mkdir(parents=True, exist_ok=False)
            write_once(state_root / "SUITE_IDENTITY.json", {"schema_version": 1, "suite_id": binding["suite_id"],
                "plan_sha256": plan_sha256, "binding_sha256": binding_sha256,
                "source_identity_sha256": source_identity(binding), "created_utc": utc(),
                "allocation_policy": ALLOCATION, "automatic_requeue": False})
            (state_root / "tasks").mkdir()
        invocation = _new_numbered(state_root / "invocations", "invocation-")
        context = {"state_root": str(state_root), "invocation_dir": str(invocation)}
        _journal(state_root, {"event": "INVOCATION_STARTED", "invocation": invocation.name, "resume": resume})
        completed = {}
        try:
            verify_sources(binding)
            # Always repeat allocation checks after a preemption/reallocation.
            startup = invocation / "startup"
            startup.mkdir()
            pins = run_command(binding["startup"], binding, context, startup)
            startup_value = validate_startup_receipt(pins[0]["path"], binding)
            write_once(invocation / "STARTUP_VERIFIED.json", {"receipt": pins[0],
                "job_uid": startup_value["job_uid"], "source_identity_sha256": source_identity(binding)})
            for task in plan["tasks"]:
                path = state_root / "tasks" / task["id"] / "COMPLETED.json"
                if path.exists():
                    value = completion_for(state_root, task["id"])
                    require(value["plan_sha256"] == plan_sha256 and value["binding_sha256"] in accepted_binding_shas,
                            "completed task belongs to another frozen suite")
                    completed[task["id"]] = value
            pending = {t["id"] for t in plan["tasks"]} - completed.keys()
            while pending:
                ready = [t for t in plan["tasks"] if t["id"] in pending
                         and set(t["dependencies"]) <= completed.keys()]
                require(ready, "no executable task: dependency graph or completion state is inconsistent")
                task = ready[0]
                task_id = task["id"]
                command = binding["tasks"][task_id]
                task_root = safe(state_root / "tasks" / task_id)
                task_root.mkdir(exist_ok=True)
                _previous_child_quiet(task_root)
                prior_attempts = list((task_root / "attempts").glob("attempt-*")) if (task_root / "attempts").exists() else []
                is_recovery = bool(prior_attempts)
                require(not is_recovery or resume and (command["mode"] == "reuse" or "resume" in command),
                        "unfinished task requires an explicit verified resume command: " + task_id)
                attempt = _new_numbered(task_root / "attempts", "attempt-")
                task_context = {**context, "task_id": task_id, "task_attempt_dir": str(attempt)}
                # Completed consumers retain receipt files after weight cleanup.
                dependency_pins = []
                for dependency in task["dependencies"]:
                    completion_for(state_root, dependency)
                    dependency_pins.append(artifact(state_root / "tasks" / dependency / "COMPLETED.json"))
                write_once(attempt / "TASK_INTENT.json", {"schema_version": 1, "utc": utc(), "task_id": task_id,
                    "kind": task["kind"], "plan_sha256": plan_sha256, "binding_sha256": binding_sha256,
                    "is_explicit_recovery": is_recovery, "dependencies": dependency_pins,
                    "mode": command["mode"], "context": task_context})
                _journal(state_root, {"event": "TASK_STARTED", "task_id": task_id,
                    "attempt": attempt.name, "is_explicit_recovery": is_recovery})
                if command["mode"] == "reuse":
                    verify_sources(binding)
                    for item in command.get("inputs", []):
                        verify_pin({**item, "path": expand(item["path"], task_context)})
                    receipt_pins = verify_receipts(command["receipts"], task_context)
                else:
                    recovery = command.get("resume", {}) if is_recovery else {}
                    receipt_pins = run_command(command, binding, task_context, attempt,
                        argv=recovery.get("argv"), extra_inputs=recovery.get("inputs", []))
                completion = {"schema_version": 1, "status": "TASK_ACCEPTED", "utc": utc(),
                    "task_id": task_id, "kind": task["kind"], "plan_sha256": plan_sha256,
                    "binding_sha256": binding_sha256, "attempt": str(attempt), "mode": command["mode"],
                    "receipts": receipt_pins, "dependencies": dependency_pins,
                    "task_intent_sha256": sha(attempt / "TASK_INTENT.json")}
                write_once(task_root / "COMPLETED.json", completion)
                completed[task_id] = completion
                pending.remove(task_id)
                _journal(state_root, {"event": "TASK_ACCEPTED", "task_id": task_id,
                    "completion_sha256": sha(task_root / "COMPLETED.json")})
            result = {"schema_version": 1, "status": "EXPERIMENT_SUITE_COMPLETE", "utc": utc(),
                "suite_id": binding["suite_id"], "plan_sha256": plan_sha256, "binding_sha256": binding_sha256,
                "completed_tasks": len(completed), "completion_inventory": [
                    artifact(state_root / "tasks" / task["id"] / "COMPLETED.json") for task in plan["tasks"]],
                "model_quality_or_speedup_claim": False}
            write_once(invocation / "COMPLETE.json", result)
            _journal(state_root, {"event": "INVOCATION_COMPLETE", "invocation": invocation.name,
                "completed_tasks": len(completed)})
            return result
        except BaseException as exc:
            failure = {"schema_version": 1, "status": "EXPERIMENT_SUITE_STOPPED", "utc": utc(),
                "error_type": type(exc).__name__, "error": str(exc), "completed_tasks_this_readback": len(completed),
                "automatic_retry": False, "automatic_requeue": False,
                "resume_requires_identical_plan_binding_and_explicit_recovery": True}
            write_once(invocation / "STOPPED.json", failure)
            _journal(state_root, {"event": "INVOCATION_STOPPED", "invocation": invocation.name,
                "error_type": type(exc).__name__, "error": str(exc)})
            raise


def _fleet_live(environment, binding, state_root, worker_id):
    from .suite_fleet import audit_fleet_jobs
    from .suite_cluster import list_workloads
    document = list_workloads(environment['tools'],
        project_name_prefixes=environment['allocation_identity']['project_name_prefixes'])
    live = audit_fleet_jobs(binding['execution_profile'], document, state_root, worker_id,
        project_name_prefixes=environment['allocation_identity']['project_name_prefixes'])
    from .suite_actions import terminal_job
    jobs = {item['metadata']['uid']: item for item in document['items'] if item.get('kind') == 'PyTorchJob'}
    if not (safe(state_root) / 'COMPLETE.json').exists():
        exact_names = {w['job_name'] for w in binding['execution_profile']['workers']}
        require(not any(job['metadata']['name'] in exact_names and terminal_job(job) for job in jobs.values()),
                'A fleet worker exited before suite completion; recovery is required')
    for path in (safe(state_root) / 'tasks').glob('*/CLAIM.json'):
        if (path.parent / 'COMPLETED.json').exists():
            continue
        owner = read(path)['owner']
        require(owner['job_uid'] in jobs and not terminal_job(jobs[owner['job_uid']]),
                'An unfinished task owner exited; frozen claim requires explicit recovery')
    return live


def run_fleet_worker(plan, binding, *, plan_sha256, binding_sha256, state_root, worker_id,
                     poll_seconds=15, budget=None, live_check=None):
    """Join one frozen fleet and drain only dependency-ready, budget-admitted work.

    Completed acceptance receipts are the sole skip authority. A failed or lost
    claim stops new work without stealing another worker's allocation or files.
    """
    from .suite_fleet import FleetState, worker_identity, project_transaction
    from .suite_storage import budget_for
    validate_binding(plan, binding, plan_sha256=plan_sha256)
    require(binding.get('schema_version') == 2 and SHA.fullmatch(binding_sha256), 'A fleet binding is required')
    worker_identity(binding['execution_profile'], worker_id)
    state_root = safe(state_root)
    require(any(state_root != safe(root) and state_root.is_relative_to(safe(root))
                for root in binding['storage']['roots']), 'Fleet state escapes accounted home roots')
    environment = machine_environment(binding)
    if 'isolated_storage' in environment:
        from .suite_isolated_storage import require_output, validate_isolation
        require(environment['isolated_storage'] == binding.get('isolated_storage'), 'Isolated environment/binding differ')
        validate_isolation(environment)
        require(read(environment['isolated_storage']['coordinator_hold']['path'])['plan_sha256'] == plan_sha256,
                'Lifetime hold belongs to another scientific plan')
        require_output(environment, state_root)
        from .suite_isolated_pod import inspect_isolated_mounts
        from .suite_scratch import _mounts
        inspect_isolated_mounts(environment, _mounts())
    if environment.get('modelscope_token_file'):
        from .suite_bindings import load_modelscope_token_file
        load_modelscope_token_file(environment['modelscope_token_file'])
    require(environment['storage'] == binding['storage'] and environment['execution_profile'] == binding['execution_profile'],
            'Pinned machine environment differs from execution/storage binding')
    budget = budget or budget_for(environment)
    live_check = live_check or (lambda: _fleet_live(environment, binding, state_root, worker_id))
    fleet = FleetState(plan, binding, state_root, plan_sha256=plan_sha256, binding_sha256=binding_sha256)
    fleet.initialize()
    worker_claim, invocation = fleet.join(worker_id)
    context = {'state_root': str(state_root), 'invocation_dir': str(invocation)}
    worker_environment = {'DP_SUITE_WORKER_ID': worker_id, 'DP_SUITE_FLEET_STATE': str(state_root),
        'DP_SUITE_INVOCATION_DIR': str(invocation), 'DP_SUITE_STORAGE_LEDGER': binding['storage']['ledger_root']}
    def execute(command, task_context, attempt, token):
        tokens = ({'DP_SUITE_STORAGE_TOKEN': token['home'],
                   **({'DP_SUITE_SCRATCH_TOKEN': token['scratch']} if token['scratch'] else {})}
                  if isinstance(token, dict) else {'DP_SUITE_STORAGE_TOKEN': token})
        return run_command(command, binding, task_context, attempt,
            extra_environment={**worker_environment, **tokens,
                               **({'DP_SUITE_TASK_ID': task_context['task_id']} if 'task_id' in task_context else {})},
            storage_snapshot=budget.snapshot())
    released_tokens = set()
    def release_if_quiet(token, attempt):
        token_key = digest(token) if isinstance(token, dict) else token
        if token_key in released_tokens:
            return True
        path = attempt / 'COMMAND_EXIT.json'
        if path.is_file() and read(path).get('group_cleanup_verified') is True:
            budget.release(token, quiescent=lambda owner: {'owner': owner, 'active_writers': [],
                'active_uploads': [], 'command_exit': artifact(path)})
            released_tokens.add(token_key)
            return True
        return False
    try:
        verify_sources(binding)
        if binding['execution_profile'].get('schema_version') == 3:
            from .suite_scratch import inspect_scratch, worker_run_root
            from .suite_scratch_budget import PrivateScratchBudget, TieredTaskBudget
            scratch_live = live_check()
            scratch_proof = inspect_scratch(environment, scratch_live)
            run_root = worker_run_root(binding['execution_profile'], worker_id)
            safe(run_root.parent).mkdir(mode=0o700, exist_ok=True)
            safe(run_root).mkdir(mode=0o700)
            require(not any(run_root.iterdir()), 'A new private worker must have an empty run container')
            _fsync_directory(run_root.parent)
            write_once(state_root / 'workers' / worker_id / 'PRIVATE_SCRATCH_PREPARED.json', {
                'live': scratch_live, 'inspection': scratch_proof, 'run_root': str(run_root),
                'profile_sha256': digest(binding['execution_profile'])})
            private_budget = PrivateScratchBudget(binding['execution_profile'], worker_id, scratch_live, state_root)
            if 'isolated_storage' in environment:
                from .suite_isolated_storage import IsolatedWorkerBudget
                budget = IsolatedWorkerBudget(environment, scratch_live, state_root)
                worker_environment['DP_SUITE_STORAGE_LEDGER'] = str(budget.root)
            budget = TieredTaskBudget(budget, private_budget, {task['id']: task for task in plan['tasks']})
            caches = safe(binding['execution_profile']['scratch']['mount_path']) / 'runtime-cache' / worker_id
            for key, name in {'TMPDIR': 'tmp', 'CUDA_CACHE_PATH': 'cuda', 'TRITON_CACHE_DIR': 'triton',
                              'TORCH_EXTENSIONS_DIR': 'torch-extensions'}.items():
                path = safe(caches / name)
                path.mkdir(parents=True, mode=0o700, exist_ok=True)
                worker_environment[key] = str(path)
        startup = invocation / 'startup'
        startup.mkdir()
        startup_owner = {'fleet_id': binding['suite_id'], 'worker_id': worker_id, 'task_id': 'startup',
                         'claim_id': worker_claim['claim_id'], 'job_name': worker_identity(binding['execution_profile'], worker_id)['job_name']}
        token = budget.reserve(startup_owner, binding['startup'].get('reserve_bytes', 0),
                               binding['startup'].get('oss_reserve_bytes', 0))
        try:
            pins = execute(binding['startup'], context, startup, token)
            startup_value = validate_startup_receipt(pins[0]['path'], binding)
            require(startup_value.get('worker_id') == worker_id, 'Startup belongs to another fleet worker')
            fleet.register(worker_id, startup_value)
            write_once(invocation / 'STARTUP_VERIFIED.json', {'receipt': pins[0], 'job_uid': startup_value['job_uid'],
                                                            'worker_id': worker_id})
        finally:
            release_if_quiet(token, startup)
        while not fleet.workers_ready():
            live_check()
            time.sleep(poll_seconds)
        while True:
            live_check()
            def reserve(task, owner):
                command = binding['tasks'][task['id']]
                return budget.reserve(owner, command.get('reserve_bytes', 0), command.get('oss_reserve_bytes', 0))
            def readiness(task):
                from .suite_progressive import task_readiness
                return task_readiness(environment, plan, task)
            selected = fleet.claim(worker_id, startup_value['job_uid'], reserve, readiness=readiness)
            if selected['status'] == 'complete':
                result = fleet.complete(worker_id)
                write_once(invocation / 'COMPLETE.json', result)
                return result
            if selected['status'] == 'wait':
                time.sleep(poll_seconds)
                continue
            task, claim = selected['task'], selected['claim']
            attempt = safe(claim['attempt'])
            task_id, command = task['id'], binding['tasks'][task['id']]
            task_context = {**context, 'task_id': task_id, 'task_attempt_dir': str(attempt)}
            dependency_pins = [artifact(completion_path(state_root, item)) for item in task['dependencies']]
            write_once(attempt / 'TASK_INTENT.json', {'schema_version': 2, 'utc': utc(), 'task_id': task_id,
                'kind': task['kind'], 'plan_sha256': plan_sha256, 'binding_sha256': binding_sha256,
                'dependencies': dependency_pins, 'mode': command['mode'], 'context': task_context,
                'claim': claim, 'is_explicit_recovery': False})
            try:
                if command['mode'] == 'reuse':
                    verify_sources(binding)
                    for item in command.get('inputs', []):
                        verify_pin({**item, 'path': expand(item['path'], task_context)})
                    pins = verify_receipts(command['receipts'], task_context)
                    budget.release(claim['storage_token'], quiescent=lambda owner: {
                        'owner': owner, 'active_writers': [], 'active_uploads': [], 'receipt_only_reuse': True})
                else:
                    pins = execute(command, task_context, attempt, claim['storage_token'])
                    require(release_if_quiet(claim['storage_token'], attempt), 'Task processes were not verified quiescent')
                completion = {'schema_version': 2, 'status': 'TASK_ACCEPTED', 'utc': utc(), 'task_id': task_id,
                    'kind': task['kind'], 'plan_sha256': plan_sha256, 'binding_sha256': binding_sha256,
                    'attempt': str(attempt), 'mode': command['mode'], 'receipts': pins, 'dependencies': dependency_pins,
                    'task_intent_sha256': sha(attempt / 'TASK_INTENT.json'), 'claim_id': claim['owner']['claim_id']}
                fleet.finish(task, claim, completion)
            except BaseException as exc:
                fleet.fail(task, claim, exc)
                # An uncertain process cleanup deliberately leaves its durable reservation occupied.
                if command['mode'] != 'reuse':
                    release_if_quiet(claim['storage_token'], attempt)
                raise
    except BaseException as exc:
        from .suite_isolated_storage import coordination_storage
        with project_transaction(coordination_storage(binding)):
            write_once(invocation / 'STOPPED.json', {'status': 'EXPERIMENT_SUITE_STOPPED', 'worker_id': worker_id,
                'error_type': type(exc).__name__, 'error': str(exc), 'utc': utc(), 'automatic_retry': False,
                'automatic_requeue': False, 'claims_and_uncertain_reservations_preserved': True})
        raise


def build_training_argv(python, *, config, train_manifest, eval_manifest, media_root, output_dir,
                        formal_ready, formal_training_data_dir, frozen_source_manifest,
                        frozen_source_manifest_sha256, declared=False, init_checkpoint=None,
                        resume_checkpoint=None, no_alignment=False):
    """Direct Trainer/DDP entry: no historical stage wrapper or weight cleanup."""
    require(type(declared) is bool and type(no_alignment) is bool, "training entry choices must be Boolean")
    require(not (init_checkpoint and resume_checkpoint), "stage initialization and exact resume are exclusive")
    require(not no_alignment or declared, "no-alignment requires the declared training entry")
    require(isinstance(frozen_source_manifest_sha256, str) and SHA.fullmatch(frozen_source_manifest_sha256),
            "frozen training source SHA required")
    paths = {"config": config, "train-manifest": train_manifest, "eval-manifest": eval_manifest,
             "media-root": media_root, "output-dir": output_dir, "formal-ready": formal_ready,
             "formal-training-data-dir": formal_training_data_dir,
             "frozen-source-manifest": frozen_source_manifest}
    executable_path(python)
    for path in paths.values():
        safe(path)
    argv = single_node_torchrun(python) + ["-m",
            "deepprobe_vlm.train_declared" if declared else "deepprobe_vlm.train"]
    for flag, path in paths.items():
        argv.extend(["--" + flag, str(path)])
    argv.extend(["--frozen-source-manifest-sha256", frozen_source_manifest_sha256])
    if init_checkpoint:
        argv.extend(["--init-checkpoint", str(safe(init_checkpoint))])
    if resume_checkpoint:
        argv.extend(["--resume", str(safe(resume_checkpoint))])
    if no_alignment:
        argv.append("--no-alignment")
    return argv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--relocation", type=Path)
    parser.add_argument("--relocation-sha256")
    args = parser.parse_args(argv)
    verify_pin({"path": str(args.plan), "sha256": args.plan_sha256})
    verify_pin({"path": str(args.binding), "sha256": args.binding_sha256})
    from .experiment_suite import validate_plan
    plan = validate_plan(read(args.plan), config_root=args.config_root)
    binding = validate_binding(plan, read(args.binding), plan_sha256=args.plan_sha256)
    require(not args.resume or args.execute, "--resume requires explicit --execute")
    require(bool(args.relocation) == bool(args.relocation_sha256) and (not args.relocation or args.resume),
            "--relocation and --relocation-sha256 require explicit --resume")
    verify_sources(binding)
    if not args.execute:
        print(json.dumps({"status": "SUITE_BINDING_VALID_NO_EXECUTION", "task_count": len(plan["tasks"]),
            "plan_sha256": args.plan_sha256, "binding_sha256": args.binding_sha256,
            "source_identity_sha256": source_identity(binding), "gpu_used": False,
            "scheduler_submission_performed": False, "input_media_or_weights_verified": False}))
        return 0
    relocation_pin = {"path": str(args.relocation), "sha256": args.relocation_sha256} if args.relocation else None
    if binding.get("execution_profile"):
        require(args.worker_id and not args.resume and not args.relocation,
                "Parallel execution requires --worker-id; claims never automatically retry or relocate")
        result = run_fleet_worker(plan, binding, plan_sha256=args.plan_sha256,
            binding_sha256=args.binding_sha256, state_root=args.state_root, worker_id=args.worker_id)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    require(not args.worker_id, "--worker-id requires a fleet execution profile")
    result = run_bound_suite(plan, binding, plan_sha256=args.plan_sha256,
                            binding_sha256=args.binding_sha256, state_root=args.state_root, resume=args.resume,
                            relocation_pin=relocation_pin)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
