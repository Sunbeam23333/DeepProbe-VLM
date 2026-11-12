"""Real tiny CPU subprocess tests for serial control, never GPU/model acceptance."""
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from deepprobe_vlm import suite_runner as runner


WORKER = '''import argparse,json,os,time,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--receipt',required=True);p.add_argument('--trace',required=True)
p.add_argument('--task',required=True);p.add_argument('--source');p.add_argument('--mode',default='success')
a=p.parse_args()
with Path(a.trace).open('a') as f:f.write(a.task+'\\n')
if a.mode=='fail':raise SystemExit(9)
if a.mode=='sleep':time.sleep(10)
if a.mode=='background':subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
r={'status':'ACCEPTED','task':a.task,'units':1,'offline':all(os.environ.get(k)=='1' for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'])}
if a.task=='startup':
 r={'status':'SUITE_STARTUP_ACCEPTED','source_identity_sha256':a.source,
    'allocation':{'cluster':'ali-gpu','queue':'stream','priority':'low','nodes':1,'gpus':8,'restart_policy':'Never'},
    'other_active_project_jobs':[],'gpu_count':8,'gpu_uuids':['fixture-gpu-'+str(i) for i in range(8)],
    'job_uid':'synthetic-allocation','hardware_preflight_passed':True}
if a.mode=='bad-receipt':r['status']='NOT_ACCEPTED'
out=Path(a.receipt);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x') as f:json.dump(r,f)
'''


class SuiteRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.a, self.b = self.root / "project", self.root / "staging"
        self.a.mkdir(); self.b.mkdir()
        self.worker = self.a / "worker.py"
        self.worker.write_text(WORKER)
        self.trace = self.a / "trace.txt"
        self.plan = {"tasks": [
            {"id": "first", "kind": "train", "dependencies": []},
            {"id": "second", "kind": "evaluate", "dependencies": ["first"]}]}
        self.plan_sha = runner.digest(self.plan)
        self.binding = {
            "schema_version": 1, "suite_id": "synthetic-suite", "plan_sha256": self.plan_sha,
            "allocation": dict(runner.ALLOCATION), "source_files": [runner.artifact(self.worker)],
            "storage": {"roots": [str(self.a), str(self.b)], "limit_bytes": 99_000_000_000,
                        "other_reserve_bytes": 0, "free_guard_bytes": 1_000_000_000},
            "environment": {"OMP_NUM_THREADS": "1"}, "startup": {}, "tasks": {},
        }
        self.binding["startup"] = self.command("startup", "{invocation_dir}/STARTUP.json")
        self.binding["startup"].pop("mode")
        self.binding["startup"]["argv"] += ["--source", runner.source_identity(self.binding)]
        self.binding["startup"]["receipts"][0]["checks"] = [
            {"pointer": "/status", "equals": "SUITE_STARTUP_ACCEPTED"}]
        for task in self.plan["tasks"]:
            self.binding["tasks"][task["id"]] = self.command(task["id"], "{task_attempt_dir}/VERIFIED.json")
        self.state = self.a / "suite-state"

    def command(self, task, receipt):
        return {"mode": "execute", "argv": [sys.executable, "-B", str(self.worker),
                    "--receipt", receipt, "--trace", str(self.trace), "--task", task],
                "cwd": str(self.a), "inputs": [], "reserve_bytes": 1024,
                "timeout_seconds": 20,
                "receipts": [{"path": receipt, "checks": [
                    {"pointer": "/status", "equals": "ACCEPTED"},
                    {"pointer": "/task", "equals": task}, {"pointer": "/offline", "equals": True}]}]}

    def execute(self, *, resume=False):
        return runner.run_bound_suite(self.plan, self.binding, plan_sha256=self.plan_sha,
                                      binding_sha256=runner.digest(self.binding), state_root=self.state,
                                      resume=resume)

    def test_real_serial_subprocesses_dependencies_and_retained_completion(self):
        result = self.execute()
        self.assertEqual(result["status"], "EXPERIMENT_SUITE_COMPLETE")
        self.assertEqual(result["completed_tasks"], 2)
        self.assertEqual(self.trace.read_text().splitlines(), ["startup", "first", "second"])
        second = runner.completion_for(self.state, "second")
        self.assertEqual(second["dependencies"][0]["sha256"], runner.sha(self.state / "tasks/first/COMPLETED.json"))
        self.assertEqual(runner.read(Path(second["attempt"]) / "COMMAND_EXIT.json")["returncode"], 0)
        with self.assertRaises(FileExistsError):
            self.execute()
        self.execute(resume=True)
        self.assertEqual(self.trace.read_text().splitlines(), ["startup", "first", "second", "startup"])
        self.assertTrue((self.state / "invocations/invocation-000002/STARTUP_VERIFIED.json").is_file())

    def test_failure_stops_before_consumer_and_only_explicit_resume_runs(self):
        first = self.binding["tasks"]["first"]
        first["resume"] = {"argv": list(first["argv"])}
        first["argv"] += ["--mode", "fail"]
        with self.assertRaisesRegex(ValueError, "exit 9"):
            self.execute()
        self.assertFalse((self.state / "tasks/first/COMPLETED.json").exists())
        self.assertFalse((self.state / "tasks/second").exists())
        stopped = runner.read(self.state / "invocations/invocation-000001/STOPPED.json")
        self.assertFalse(stopped["automatic_retry"])
        self.execute(resume=True)
        self.assertEqual(self.trace.read_text().splitlines(), ["startup", "first", "startup", "first", "second"])
        latest = runner.completion_for(self.state, "first")
        self.assertIn("attempt-000002", latest["attempt"])
        self.assertTrue(runner.read(Path(latest["attempt"]) / "TASK_INTENT.json")["is_explicit_recovery"])

    def test_missing_explicit_resume_refuses_retry_of_failed_task(self):
        self.binding["tasks"]["first"]["argv"] += ["--mode", "fail"]
        with self.assertRaises(ValueError):
            self.execute()
        with self.assertRaisesRegex(ValueError, "explicit verified resume"):
            self.execute(resume=True)
        self.assertEqual(self.trace.read_text().splitlines(), ["startup", "first", "startup"])

    def test_exit_zero_without_accepted_output_stops(self):
        self.binding["tasks"]["first"]["argv"] += ["--mode", "bad-receipt"]
        with self.assertRaisesRegex(ValueError, "receipt assertion"):
            self.execute()
        attempt = self.state / "tasks/first/attempts/attempt-000001"
        self.assertEqual(runner.read(attempt / "COMMAND_EXIT.json")["returncode"], 0)
        self.assertFalse((self.state / "tasks/first/COMPLETED.json").exists())
        self.assertFalse((self.state / "tasks/second").exists())

    def test_resume_rejects_changed_retained_evidence_and_binding(self):
        self.execute()
        altered = deepcopy(self.binding)
        altered["environment"]["OMP_NUM_THREADS"] = "2"
        original = self.binding
        self.binding = altered
        with self.assertRaisesRegex(ValueError, "resume plan/binding"):
            self.execute(resume=True)
        self.binding = original
        completion = runner.completion_for(self.state, "first")
        Path(completion["receipts"][0]["path"]).write_text('{"status":"ACCEPTED","units":99}')
        with self.assertRaisesRegex(ValueError, "pinned artifact changed"):
            self.execute(resume=True)

    def test_source_change_and_pin_change_are_not_accepted(self):
        self.worker.write_text(WORKER + "\n# changed\n")
        with self.assertRaisesRegex(ValueError, "pinned artifact changed"):
            self.execute()
        self.assertFalse(self.trace.exists())

    def test_reuse_is_hash_verified_without_task_execution(self):
        accepted = self.a / "accepted.json"
        accepted.write_text('{"status":"ACCEPTED","coverage":7}')
        self.binding["tasks"]["first"] = {"mode": "reuse", "inputs": [], "receipts": [
            {"path": str(accepted), "sha256": runner.sha(accepted), "checks": [
                {"pointer": "/status", "equals": "ACCEPTED"}, {"pointer": "/coverage", "equals": 7}]}]}
        self.execute()
        self.assertEqual(self.trace.read_text().splitlines(), ["startup", "second"])
        self.assertEqual(runner.completion_for(self.state, "first")["mode"], "reuse")

    def test_whole_suite_lock_prevents_duplicate_controller(self):
        locks = self.a / "locks"
        locks.mkdir()
        with (locks / "experiment-suite.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "another experiment suite"):
                self.execute()
        self.assertFalse(self.state.exists())

    def test_capacity_guard_stops_before_spawning_task(self):
        self.binding["startup"]["reserve_bytes"] = 99_000_000_000
        with self.assertRaisesRegex(ValueError, "99 GB"):
            self.execute()
        self.assertFalse(self.trace.exists())

    def test_bound_timeout_stops_child_and_retains_exit(self):
        self.binding["tasks"]["first"]["argv"] += ["--mode", "sleep"]
        self.binding["tasks"]["first"]["timeout_seconds"] = 1
        with self.assertRaises(subprocess.TimeoutExpired):
            self.execute()
        exit_record = runner.read(self.state / "tasks/first/attempts/attempt-000001/COMMAND_EXIT.json")
        self.assertEqual(exit_record["error"]["type"], "TimeoutExpired")
        self.assertLess(exit_record["returncode"], 0)
        self.assertFalse((self.state / "tasks/second").exists())

    def test_successful_adapter_cannot_leave_background_workers(self):
        self.binding["tasks"]["first"]["argv"] += ["--mode", "background"]
        with self.assertRaisesRegex(ValueError, "live descendants"):
            self.execute()
        attempt = self.state / "tasks/first/attempts/attempt-000001"
        self.assertEqual(runner.read(attempt / "COMMAND_EXIT.json")["returncode"], 0)
        self.assertFalse((self.state / "tasks/first/COMPLETED.json").exists())
        self.assertFalse((self.state / "tasks/second").exists())

    def test_binding_rejects_unsafe_or_incomplete_contracts(self):
        for change, error in [
            (lambda b: b["allocation"].update(priority="normal"), "stream/low"),
            (lambda b: b["storage"].update(limit_bytes=100_000_000_000), "99 GB"),
            (lambda b: b["environment"].update(HF_HUB_OFFLINE="0"), "offline"),
            (lambda b: b["environment"].update(LD_PRELOAD="arbitrary.so"), "environment"),
            (lambda b: b["tasks"].pop("second"), "every declared task"),
            (lambda b: b["tasks"]["first"].update(argv=["/bin/bash", "-c", "echo no"]), "direct executables"),
            (lambda b: b["startup"]["receipts"][0].update(path=str(self.a / "stale-startup.json")), "fresh"),
        ]:
            b = deepcopy(self.binding)
            change(b)
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                runner.validate_binding(self.plan, b, plan_sha256=self.plan_sha)

    def test_startup_receipt_rejects_fake_count_duplicate_devices_and_active_job(self):
        self.execute()
        p = self.state / "invocations/invocation-000001/STARTUP.json"
        original = runner.read(p)
        for change in (lambda x: x.update(gpu_count=7),
                       lambda x: x.update(gpu_uuids=["same"] * 8),
                       lambda x: x.update(other_active_project_jobs=["another"]),
                       lambda x: x.update(source_identity_sha256="0" * 64),
                       lambda x: x.update(hardware_preflight_passed=False)):
            doc = deepcopy(original); change(doc); p.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):
                runner.validate_startup_receipt(p, self.binding)

    def test_training_helper_uses_one_torchrun_and_no_historical_cleanup(self):
        values = {name: self.a / name for name in (
            "config", "train_manifest", "eval_manifest", "media_root", "output_dir", "formal_ready",
            "formal_training_data_dir", "frozen_source_manifest")}
        kwargs = {**values, "frozen_source_manifest_sha256": "a" * 64}
        argv = runner.build_training_argv(sys.executable, **kwargs, declared=True, no_alignment=True)
        self.assertEqual(argv.count("torch.distributed.run"), 1)
        self.assertIn("--nproc_per_node=8", argv)
        self.assertIn("--max_restarts=0", argv)
        self.assertIn("deepprobe_vlm.train_declared", argv)
        self.assertIn("--no-alignment", argv)
        self.assertFalse(any("run_p1_stage" in item for item in argv))
        with self.assertRaisesRegex(ValueError, "exclusive"):
            runner.build_training_argv(sys.executable, **kwargs, init_checkpoint=self.a / "init",
                                       resume_checkpoint=self.a / "checkpoint-300")
        with self.assertRaisesRegex(ValueError, "declared"):
            runner.build_training_argv(sys.executable, **kwargs, no_alignment=True)

    def test_receipt_json_assertions_are_type_strict_and_paths_do_not_expand_shell(self):
        p = self.a / "receipt.json"
        p.write_text('{"status":true}')
        with self.assertRaisesRegex(ValueError, "receipt assertion"):
            runner.verify_receipts([{"path": str(p), "checks": [{"pointer": "/status", "equals": 1}]}], {})
        self.assertEqual(runner.expand("$(touch nope)", {}), "$(touch nope)")
        with self.assertRaisesRegex(ValueError, "context token"):
            runner.expand("{untrusted_value}", {})


if __name__ == "__main__":
    unittest.main()
