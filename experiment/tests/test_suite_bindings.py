"""All scientific tasks receive executable bindings; no real assets or GPU used."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from deepprobe_vlm import experiment_suite
from deepprobe_vlm import suite_bindings as bindings
from deepprobe_vlm import suite_runner as runner


class SuiteBindingsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.a, self.b = self.root / "project", self.root / "staging"
        self.a.mkdir(); self.b.mkdir()
        self.source = self.a / "source.py"
        self.source.write_text("# synthetic source pin\n")
        self.plan = experiment_suite.default_plan()
        self.plan_path, self.environment_path = self.a / "PLAN.json", self.a / "ENVIRONMENT.json"
        self.plan_path.write_text(json.dumps(self.plan))
        self.environment = {
            "schema_version": 1, "suite_id": "test-serial-suite", "experiment_root": str(self.a),
            "python": sys.executable, "run_root": str(self.a / "runs/test-serial-suite"),
            "archive_root": "oss://ss-devbox-stream/DeepProbe-VLM/serial-suite",
            "source_files": [runner.artifact(self.source)], "runtime_environment": {},
            "storage": {"roots": [str(self.a), str(self.b)], "limit_bytes": 99_000_000_000,
                        "other_reserve_bytes": 1_000_000_000, "free_guard_bytes": 1_000_000_000},
            "task_reserve_bytes": {key: 1_000_000 for key in bindings.RESERVE_KEYS},
            "assets": {key: {} for key in ("training", "benchmarks", "backbones", "development")},
            "history": {},
            "tools": {"kubectl": "/usr/bin/kubectl", "kubeconfig": str(self.a / "kubeconfig"),
                      "ossutil": "/usr/bin/ossutil", "oss_config": str(self.a / "ossconfig")},
            "allocation_identity": {"job_name": "jinhaopeng-dpvlm-suite-example-n1-fixed",
                "namespace": "default", "project_name_prefixes": ["jinhaopeng-dpvlm-", "deepprobe-"]},
        }
        self.save_environment()

    def save_environment(self):
        self.environment_path.write_text(json.dumps(self.environment))

    def build(self):
        self.save_environment()
        return bindings.build_binding(self.plan, self.environment, plan_path=self.plan_path,
            plan_sha256=runner.sha(self.plan_path), environment_path=self.environment_path,
            environment_sha256=runner.sha(self.environment_path))

    def test_complete_matrix_has_every_task_and_explicit_resume_command(self):
        result = self.build()
        self.assertEqual(set(result["tasks"]), {task["id"] for task in self.plan["tasks"]})
        self.assertEqual(len(result["tasks"]), 185)
        self.assertEqual(result["allocation"], runner.ALLOCATION)
        for task in self.plan["tasks"]:
            value = result["tasks"][task["id"]]
            self.assertEqual(value["mode"], "execute")
            self.assertIn("deepprobe_vlm.suite_actions", value["argv"])
            self.assertEqual(value["argv"][value["argv"].index("--task") + 1], task["id"])
            self.assertEqual(value["resume"]["argv"], value["argv"] + ["--resume"])
            self.assertEqual(value["receipts"][0]["checks"][-1], {"pointer": "/task_id", "equals": task["id"]})
            self.assertNotIn("sslaunch", value["argv"])
            self.assertNotIn("bash", value["argv"])
        self.assertEqual(result["environment"]["PYTHONPATH"], str(self.a / "src"))
        self.assertTrue(result["startup"]["receipts"][0]["path"].startswith("{invocation_dir}/"))

    def test_binding_rebuild_is_deterministic_without_rendezvous_ids(self):
        with mock.patch("deepprobe_vlm.distributed_launch.uuid.uuid4",
                        side_effect=AssertionError("rendezvous ID generated during planning")):
            first = runner.canonical(self.build())
            self.assertEqual(first, runner.canonical(self.build()))
        self.assertNotIn(b"--rdzv-id", first)

    def test_accepted_history_has_no_training_reserve_and_override_is_explicit(self):
        task = next(x for x in self.plan["tasks"] if x["kind"] == "train")
        self.environment["history"][task["id"]] = {"mode": "accepted"}
        result = self.build()
        self.assertEqual(result["tasks"][task["id"]]["reserve_bytes"], 0)
        self.environment["task_reserve_by_id"] = {task["id"]: 12345}
        self.assertEqual(self.build()["tasks"][task["id"]]["reserve_bytes"], 12345)
        self.environment["history"][task["id"]] = {"mode": "resume"}
        self.environment.pop("task_reserve_by_id")
        self.assertEqual(self.build()["tasks"][task["id"]]["reserve_bytes"],
                         self.environment["task_reserve_bytes"]["train_" + task["stage"]])

    def test_builder_writes_only_fresh_binding_and_reports_no_admission(self):
        output = self.a / "BINDING.json"
        args = ["--plan", str(self.plan_path), "--plan-sha256", runner.sha(self.plan_path),
                "--environment", str(self.environment_path), "--environment-sha256", runner.sha(self.environment_path),
                "--output", str(output)]
        text = io.StringIO()
        with redirect_stdout(text):
            self.assertEqual(bindings.main(args), 0)
        receipt = json.loads(text.getvalue())
        self.assertEqual(receipt["status"], "SUITE_BINDING_WRITTEN_NOT_ADMITTED")
        self.assertFalse(receipt["asset_bytes_verified"])
        self.assertFalse(receipt["gpu_used"])
        self.assertFalse(receipt["scheduler_submitted"])
        with self.assertRaises(FileExistsError):
            bindings.main(args)

    def test_planning_full_graph_does_not_execute_startup_or_create_state(self):
        output = self.a / "BINDING.json"
        runner.write_once(output, self.build())
        state = self.a / "state"
        text = io.StringIO()
        with redirect_stdout(text):
            self.assertEqual(runner.main(["--plan", str(self.plan_path), "--plan-sha256", runner.sha(self.plan_path),
                "--binding", str(output), "--binding-sha256", runner.sha(output), "--state-root", str(state)]), 0)
        result = json.loads(text.getvalue())
        self.assertEqual(result["status"], "SUITE_BINDING_VALID_NO_EXECUTION")
        self.assertFalse(result["gpu_used"])
        self.assertFalse(state.exists())

    def test_bad_environment_does_not_produce_partial_matrix(self):
        cases = [
            lambda e: e["task_reserve_bytes"].pop("cleanup"),
            lambda e: e.update(task_reserve_by_id={"invented-task": 1}),
            lambda e: e["allocation_identity"].update(project_name_prefixes=["unrelated-"]),
            lambda e: e["allocation_identity"].update(namespace="other"),
            lambda e: e["tools"].update(kubectl="relative-kubectl"),
            lambda e: e["runtime_environment"].update(PYTHONPATH="/unreviewed/source"),
            lambda e: e["runtime_environment"].update(HF_HUB_OFFLINE="0"),
            lambda e: e.update(history={"invented-task": {"mode": "accepted"}}),
            lambda e: e.update(history={"gpu_gate": {"mode": "accepted"}}),
            lambda e: e.update(run_root="/outside-accounted-budget/runs"),
            lambda e: e.pop("archive_root"),
            lambda e: e.update(archive_root="oss://ss-devbox/DeepProbe-VLM"),
            lambda e: e.update(archive_root="oss://ss-devbox-stream/"),
            lambda e: e.update(archive_root="oss://ss-devbox-stream/project/runs/old"),
            lambda e: e.update(archive_root="oss://ss-devbox-stream/project/../other"),
            lambda e: e.update(archive_root="oss://ss-devbox-stream/project//other"),
        ]
        original = deepcopy(self.environment)
        for case in cases:
            self.environment = deepcopy(original)
            case(self.environment)
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.build()

    def test_changed_scientific_plan_is_refused(self):
        self.plan["tasks"] = self.plan["tasks"][:-1]
        with self.assertRaises(ValueError):
            self.build()


if __name__ == "__main__":
    unittest.main()
