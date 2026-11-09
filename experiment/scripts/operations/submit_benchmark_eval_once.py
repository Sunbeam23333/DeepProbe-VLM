#!/usr/bin/env python3
"""One-shot, fail-closed Ali benchmark admission and optional submission.

Run on the authorized Ali devbox through haopengjin2ali. Without --execute this
only preflights and generates/validates a scheduler dry-run. A global durable
intent permanently blocks another real submission for the same run. Ambiguous
client outcomes trigger read-only discovery, never a submit retry.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import getpass
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_benchmark_eval as runner

BASE = Path("/home/jinhaopeng/.cache/Paper/DeepProbe-VLM")
SSLAUNCH = "/home/jinhaopeng/haopengjin/.sslaunch-venv/bin/sslaunch"
OSSUTIL = "/home/jinhaopeng/.local/bin/ossutil"
OSS_CONFIG = "/home/jinhaopeng/.ossutilconfig"
KUBECTL = "/home/jinhaopeng/.local/bin/kubectl"
KUBECONFIG = "/mnt/vepfs/base/infra/.kube/aliconfig"
JOB_SHORT = "dpvlm-videomme-r1r2-s17-a1"
JOB_PREFIX = "jinhaopeng-" + JOB_SHORT + "-n1-"
PROJECT_PREFIX = "jinhaopeng-dpvlm-"
TERMINAL = {"Succeeded", "Failed", "Cancelled", "Canceled"}
# Exact shell setup prefix from the independently accepted 2026-10-12
# sslaunch dry-run, ending at the final su "$SSLAUNCH_USER" -c delimiter.
WRAPPER_PREFIX_SHA256 = "3e6c66918bdcac343cfdc0c26da7e5864563a59af9169dc9e832c5cd3daf79cd"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_exclusive(path, value):
    """Intent uses O_EXCL plus file and directory fsync, never replacement."""
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def capture(directory, label, argv, timeout=180):
    value = {"argv": argv, "started_utc": utc(), "returncode": None, "error": None}
    try:
        result = subprocess.run(argv, capture_output=True, timeout=timeout, check=False)
        stdout, stderr = result.stdout, result.stderr
        value["returncode"] = result.returncode
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = exc.stdout or b"", exc.stderr or b""
        value["error"] = "TimeoutExpired"
    except OSError as exc:
        stdout, stderr = b"", str(exc).encode()
        value["error"] = type(exc).__name__
    for suffix, payload in (("stdout", stdout), ("stderr", stderr)):
        path = directory / f"{label}.{suffix}"
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        value[suffix + "_sha256"] = runner.sha256_file(path)
    value["completed_utc"] = utc()
    write_exclusive(directory / f"{label}.json", value)
    return {**value, "stdout": stdout.decode(errors="replace"), "stderr": stderr.decode(errors="replace")}


def jobs_in_list(output):
    """Validate every row/count, then return this project's exact states."""
    lines = output.splitlines()
    titles = [index for index, line in enumerate(lines) if line.strip() == "Job List:"]
    require(len(titles) == 1, "sslaunch job list title missing/repeated")
    title = titles[0]
    summary = [line.strip().split() for line in lines[:title] if line.strip().startswith("jinhaopeng")]
    require(len(summary) == 1 and len(summary[0]) == 8 and summary[0][:2] == ["jinhaopeng", "ali-gpu"]
            and summary[0][3].isdigit(), "Malformed scheduler summary")
    jobs, details, seen = {}, {}, set()
    for line in lines[title + 2:]:
        if not line.strip():
            continue
        fields = line.strip().split(maxsplit=7)
        require(len(fields) in {7, 8} and re.fullmatch(r"jinhaopeng-[a-z0-9-]+", fields[0])
            and fields[1].isdigit() and int(fields[1]) >= 0 and fields[2].isdigit() and int(fields[2]) >= 1
            and fields[3] in {"Created", "Pending", "Running", "Restarting", "Succeeded", "Failed", "Cancelled", "Canceled", "Unknown"}
            and fields[4] in {"low", "normal", "high", "-"}
            and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z|-", fields[5])
            and re.fullmatch(r"-|(?=\d)(?:\d+d)?(?:\d+h)?(?:\d+m)?(?:\d+s)?", fields[6])
            and fields[0] not in seen, f"Malformed scheduler row: {line}")
        seen.add(fields[0])
        if len(fields) == 8:
            require(len(fields[7]) <= 1024 and fields[7].isprintable(), "Unsafe scheduler detail")
            details[fields[0]] = fields[7]
        if fields[0].startswith(PROJECT_PREFIX):
            jobs[fields[0]] = fields[3]
    require(len(seen) == int(summary[0][3]), "Scheduler row count differs from total")
    return jobs, details


def kube_states(output, *, pods=False):
    document = json.loads(output)
    require(isinstance(document, dict) and isinstance(document.get("items"), list), "Invalid Kubernetes list")
    jobs = {}
    for item in document["items"]:
        name = item["metadata"]["name"]
        if not name.startswith(PROJECT_PREFIX):
            continue
        require(item["metadata"].get("namespace") == "default" and name not in jobs, "Unexpected/duplicate project Kubernetes object")
        if pods:
            state = item.get("status", {}).get("phase")
            jobs[name] = state if state in {"Succeeded", "Failed"} else "ActiveOrUnknown"
        else:
            conditions = {}
            for row in item.get("status", {}).get("conditions", []):
                require(row["type"] not in conditions and row["status"] in {"True", "False", "Unknown"}, "Malformed Kubernetes condition")
                conditions[row["type"]] = row["status"]
            active = conditions.get("Running") == "True" or conditions.get("Restarting") == "True" or "Unknown" in conditions.values()
            succeeded, failed = conditions.get("Succeeded") == "True", conditions.get("Failed") == "True"
            jobs[name] = "Succeeded" if succeeded and not failed and not active else "Failed" if failed and not succeeded and not active else "ActiveOrUnknown"
    return jobs


def verify_oss_empty(output):
    totals = re.findall(r"^Object Number is:\s*(\d+)\s*$", output, re.M)
    require(totals == ["0"] and not re.search(r"\soss://\S+", output), "Exact evaluation OSS prefix is not proved empty")


def runner_argv(plan_path, plan_sha):
    return ["bash", str(Path(__file__).with_name("run_benchmark_eval.sh")), "--plan", str(plan_path), "--plan-sha256", plan_sha]


def scheduler_argv(plan_path, plan_sha, *, dry_run):
    return [SSLAUNCH, "submit", "--dry-run" if dry_run else "--no-log", "-j", JOB_SHORT,
            "-c", "ali-gpu", "-q", "stream", "-n", "1", "-p", "low", "-w", "/home/jinhaopeng",
            "--", *runner_argv(plan_path, plan_sha)]


def validate_yaml(document, expected_argv):
    require(document.get("apiVersion") == "kubeflow.org/v1" and document.get("kind") == "PyTorchJob", "Unexpected scheduler object")
    meta, spec = document["metadata"], document["spec"]
    require(meta.get("namespace") == "default" and re.fullmatch(re.escape(JOB_PREFIX) + r"[a-z0-9-]+", meta.get("name", "")), "Unexpected scheduler job identity")
    replicas = spec.get("pytorchReplicaSpecs", {})
    require(set(replicas) == {"Master"}, "Expected only one Master replica type")
    master = replicas["Master"]
    require(type(master.get("replicas")) is int and master["replicas"] == 1 and master.get("restartPolicy") == "Never", "Expected one Master with restart Never")
    pod = master["template"]["spec"]
    require(pod.get("nodeSelector") == {"sslaunch-pool": "stream"} and pod.get("priorityClassName") == "low"
            and "nodeName" not in pod and "affinity" not in pod and pod.get("restartPolicy", "Never") == "Never", "Expected stream/low with no forced node")
    containers = pod.get("containers", [])
    require(len(containers) == 1 and not pod.get("initContainers") and not pod.get("ephemeralContainers"), "Unexpected extra container")
    container = containers[0]
    resources = container.get("resources", {})
    require(resources.get("limits", {}).get("nvidia.com/gpu") == 8
            and resources.get("requests", {}).get("nvidia.com/gpu", 8) == 8,
            "Expected exactly eight requested/limited GPUs")
    require(container.get("workingDir") == "/home/jinhaopeng" and not container.get("args"), "Unexpected container workdir/args")
    cuda_env = [item for item in container.get("env", []) if item.get("name") == "CUDA_VISIBLE_DEVICES"]
    require(not cuda_env or cuda_env == [{"name": "CUDA_VISIBLE_DEVICES", "value": "0,1,2,3,4,5,6,7"}], "CUDA_VISIBLE_DEVICES hides allocated evaluation GPUs")
    command = container.get("command")
    require(isinstance(command, list) and len(command) == 3 and command[:2] == ["/bin/bash", "-c"], "Unexpected shell wrapper")
    tokens = shlex.split(command[2])
    require(len(tokens) >= 4 and tokens[-4:-1] == ["su", "$SSLAUNCH_USER", "-c"], "Expected sslaunch submit-user wrapper")
    outer_prefix = command[2][:command[2].rfind(" -c ") + 4]
    import hashlib
    require(hashlib.sha256(outer_prefix.encode()).hexdigest() == WRAPPER_PREFIX_SHA256, "Scheduler shell setup differs from accepted wrapper")
    expected_inner = ["cd", "/home/jinhaopeng", "&&", "export", "HOME=/home/jinhaopeng", "&&", *expected_argv]
    require(shlex.split(tokens[-1]) == expected_inner, "Container evaluation argv differs from frozen plan")
    require(tokens[-1] == "cd /home/jinhaopeng && export HOME=/home/jinhaopeng && " + shlex.join(expected_argv), "Unexpected runner shell quoting/control operators")
    require(sum(command[2].count(str(item)) for item in expected_argv[1:2]) == 1, "Runner occurs more than once")
    return {"job_name": meta["name"], "stream_low_one_node_eight_gpu_restart_never": True,
            "runner_argv": expected_argv, "no_forced_node": True}


def live_snapshot(directory, label, plan, *, before_submission=True):
    prefix = "oss://ss-devbox-stream/jinhaopeng/Paper/DeepProbe-VLM/evaluations/" + plan["run_id"] + "/"
    argv = {"jobs": [SSLAUNCH, "job", "list", "-c", "ali-gpu", "-d"],
            "kubernetes": [KUBECTL, "--kubeconfig", KUBECONFIG, "get", "pytorchjobs", "-A", "-o", "json"],
            "pods": [KUBECTL, "--kubeconfig", KUBECONFIG, "get", "pods", "-A", "-o", "json"]}
    if before_submission:
        argv["oss"] = [OSSUTIL, "ls", prefix, "-c", OSS_CONFIG]
    captures = {name: capture(directory, label + "_" + name, command) for name, command in argv.items()}
    require(all(value["returncode"] == 0 for value in captures.values()), "Read-only scheduler/Kubernetes/OSS query failed")
    jobs, details = jobs_in_list(captures["jobs"]["stdout"])
    kube = kube_states(captures["kubernetes"]["stdout"])
    pods = kube_states(captures["pods"]["stdout"], pods=True)
    if before_submission:
        require(not any(state not in TERMINAL for state in jobs.values()), "Active project scheduler job exists")
        require(not any(state == "ActiveOrUnknown" for state in list(kube.values()) + list(pods.values())), "Active/unknown project Kubernetes job or pod exists")
        require(not any(name.startswith(JOB_PREFIX) for name in set(jobs) | set(kube) | set(pods)), "This benchmark attempt prefix already exists")
        verify_oss_empty(captures["oss"]["stdout"])
        require(not os.path.lexists(plan["run_root"]), "Evaluation run root already exists")
    return {"observed_utc": utc(), "jobs": jobs, "details": details, "kubernetes": kube, "pods": pods,
            "exact_oss_prefix": prefix, "run_root_absent": not os.path.lexists(plan["run_root"])}


def inspect_after(directory, plan, client_exact):
    result = {"never_resubmitted": True, "exact_job": None}
    try:
        snapshot = live_snapshot(directory, "after_submit", plan, before_submission=False)
        candidates = sorted(name for name in set(snapshot["jobs"]) | set(snapshot["kubernetes"]) if name.startswith(JOB_PREFIX))
        result.update(snapshot, matching_jobs=candidates)
        if len(candidates) == 1 and client_exact in {None, candidates[0]}:
            exact = candidates[0]
            info = capture(directory, "after_exact_job", [SSLAUNCH, "job", "info", "-c", "ali-gpu", exact])
            require(info["returncode"] == 0 and re.search(r"^Job:\s*" + re.escape(exact) + r"\s*$", info["stdout"], re.M), "Exact-job info unavailable")
            state = re.search(r"^State:\s*(\w+)\s*$", info["stdout"], re.M)
            require(state is not None, "Exact-job state missing")
            result.update(exact_job=exact, scheduler_state=state.group(1), inspection_complete=True)
            no_other_active = (not any(name != exact and state not in TERMINAL for name, state in snapshot["jobs"].items())
                and not any(name != exact and state == "ActiveOrUnknown" for name, state in snapshot["kubernetes"].items())
                and not any(not name.startswith(exact + "-") and state == "ActiveOrUnknown" for name, state in snapshot["pods"].items()))
            result["no_other_active_deepprobe_job"] = no_other_active
        else:
            result.update(inspection_complete=False, error="No unique exact benchmark job discovered")
    except Exception as exc:
        result.update(inspection_complete=False, error=f"{type(exc).__name__}: {exc}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    require(sys.platform == "linux" and getpass.getuser() == "jinhaopeng" and BASE.is_dir(), "Run only on authorized Ali devbox as jinhaopeng")
    runner.verify_artifact({"path": str(args.plan), "sha256": args.plan_sha256})
    plan = runner.read_json(args.plan)
    helper_rows = [item for item in plan["source_files"] if Path(item["path"]).resolve() == Path(__file__).resolve()]
    require(len(helper_rows) == 1, "Frozen plan must pin this single-submission helper")
    runner.verify_artifact(helper_rows[0])
    require(re.fullmatch(r"[a-z0-9][a-z0-9-]+", plan["run_id"]), "Unsafe run ID")
    require(args.output_dir.is_absolute() and not os.path.lexists(args.output_dir), "Submission evidence requires a new absolute output directory")
    lock_dir = BASE / "locks"
    lock_dir.mkdir(exist_ok=True)
    intent = lock_dir / ("benchmark-submit-" + plan["run_id"] + ".json")
    with (lock_dir / "benchmark-submit.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not os.path.lexists(intent), "Submission intent already exists: inspect exact job; never retry")
        args.output_dir.mkdir(parents=True, mode=0o700)
        preflight = runner.preflight(plan, args.plan_sha256)
        write_exclusive(args.output_dir / "CPU_PREFLIGHT.json", preflight)
        before = live_snapshot(args.output_dir, "before_dryrun", plan)
        dry = capture(args.output_dir, "scheduler_dryrun", scheduler_argv(args.plan, args.plan_sha256, dry_run=True))
        require(dry["returncode"] == 0 and "[dry-run] YAML generated, job not submitted." in dry["stdout"], "Scheduler dry-run failed")
        for key, value in (("cluster/queue", "ali-gpu/stream"), ("nnodes", "1"), ("priority", "low")):
            require(re.search(r"^\s*" + re.escape(key) + r"\s+" + re.escape(value) + r"\s*$", dry["stdout"], re.M), "Dry-run scheduler contract differs")
        matches = re.findall(r"^\s*yaml\s+(\S+\.yaml)\s*$", dry["stdout"], re.M)
        require(len(matches) == 1, "Dry-run YAML path missing/repeated")
        path = Path(matches[0])
        require(path.is_file() and not path.is_symlink() and path.parent == Path("/home/jinhaopeng/.kube/job_yamls"), "Unexpected generated YAML path")
        import yaml
        yaml_bytes = path.read_bytes()
        document = yaml.safe_load(yaml_bytes)
        checks = validate_yaml(document, runner_argv(args.plan, args.plan_sha256))
        with (args.output_dir / "scheduler_dryrun.generated.yaml").open("xb") as handle:
            handle.write(yaml_bytes)
        final = live_snapshot(args.output_dir, "final_gate", plan)
        require(before["jobs"] == final["jobs"] and before["kubernetes"] == final["kubernetes"] and before["pods"] == final["pods"], "Project job state changed during submission admission")
        runner.verify_artifact({"path": str(args.plan), "sha256": args.plan_sha256})
        for item in plan["source_files"]:
            runner.verify_artifact(item)
        admission = {"status": "PASSED_NO_GPU_SUBMISSION", "plan_sha256": args.plan_sha256,
                     "cpu_preflight_sha256": runner.sha256_file(args.output_dir / "CPU_PREFLIGHT.json"),
                     "dryrun_yaml_sha256": runner.sha256_file(args.output_dir / "scheduler_dryrun.generated.yaml"),
                     "checks": checks, "final_snapshot": final,
                     "submit_argv": scheduler_argv(args.plan, args.plan_sha256, dry_run=False)}
        write_exclusive(args.output_dir / "SUBMISSION_ADMISSION.json", admission)
        if not args.execute:
            print(json.dumps(admission, sort_keys=True), flush=True)
            return 0
        # One irreversible submit attempt, only after the global fsynced intent.
        write_exclusive(intent, {"created_utc": utc(), "run_id": plan["run_id"], "plan_sha256": args.plan_sha256,
            "job_prefix": JOB_PREFIX, "evidence_dir": str(args.output_dir), "at_most_one_submit": True,
            "admission_sha256": runner.sha256_file(args.output_dir / "SUBMISSION_ADMISSION.json"),
            "submit_argv": admission["submit_argv"]})
        client = capture(args.output_dir, "SUBMIT_CLIENT", admission["submit_argv"], timeout=300)
        combined = client["stdout"] + "\n" + client["stderr"]
        submitted = re.findall(r"^\s*Submitted\s+(" + re.escape(JOB_PREFIX) + r"[a-z0-9-]+)\s*$", combined, re.M)
        exact = submitted[0] if len(submitted) == 1 else None
        inspection = inspect_after(args.output_dir, plan, exact)
        clean = (client["returncode"] == 0 and inspection.get("inspection_complete") is True
                 and inspection.get("no_other_active_deepprobe_job") is True)
        outcome = {"status": "SUBMITTED_EXACT_JOB_OBSERVED" if clean else "SUBMIT_OUTCOME_UNCERTAIN_NO_RETRY",
            "run_id": plan["run_id"], "intent": str(intent), "intent_sha256": runner.sha256_file(intent),
            "client_returncode": client["returncode"], "client_error": client["error"],
            "exact_job_from_client": exact, "inspection": inspection, "never_retry_this_run_id": True}
        write_exclusive(args.output_dir / "SUBMISSION_OUTCOME.json", outcome)
        print(json.dumps(outcome, sort_keys=True), flush=True)
        return 0 if clean else 2


if __name__ == "__main__":
    raise SystemExit(main())
