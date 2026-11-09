"""a3 changes only the single-node rendezvous environment and attempt identity."""
import copy
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "scripts/operations"
spec = importlib.util.spec_from_file_location("loopback_submit", OPS / "submit_videomme_loopback_eval_once.py")
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)


def write(path, data):
    path.write_text(json.dumps(data, sort_keys=True) + "\n")


def pin(path):
    return {"path": str(path), "sha256": s.sha(path)}


class LoopbackSubmissionTests(unittest.TestCase):
    def fixture(self, root):
        p = root / "frozen-source.py"; p.write_text("# unchanged inference source")
        original = {"run_id": "dp-benchmark-videomme-r1-r2-s17-a2", "run_root": str(root / "dp-benchmark-videomme-r1-r2-s17-a2"),
                    "protocol": "videomme-899-no-subtitles-no-audio-v1", "manifest": {"sha256": "a" * 64},
                    "checkpoint_restore_receipt": {"sha256": "b" * 64},
                    "benchmark_score_based_model_selection": False, "source_files": [pin(p)]}
        original_path = root / "a2.json"; write(original_path, original)
        evidence = []
        for index in range(3):
            path = root / f"recovery-proof-{index}.json"; write(path, {"fixture": index}); evidence.append(pin(path))
        plan = copy.deepcopy(original)
        plan.update(run_id=s.RUN_ID, run_root=str(root / s.RUN_ID), distributed_runtime_environment=s.ENVIRONMENT,
                    recovery_of={"run_id": original["run_id"], "plan": pin(original_path),
                                 "exact_job": "jinhaopeng-dpvlm-videomme-r1r2-s17-a2-n1-261021164247",
                                 "observed_state": "Failed", "prediction_count": 0}, recovery_evidence=evidence)
        plan["source_files"] += [pin(Path(s.__file__).resolve()), pin(original_path), *evidence]
        return plan, original_path

    def test_only_transport_and_attempt_delta_is_accepted(self):
        with tempfile.TemporaryDirectory() as name:
            plan, original = self.fixture(Path(name).resolve())
            with mock.patch.multiple(s, BASE_PLAN=original, BASE_PLAN_SHA=s.sha(original)):
                self.assertEqual(s.validate_transport_only_delta(plan)["run_id"], plan["recovery_of"]["run_id"])
                for key, value in (("protocol", "changed"), ("manifest", {"sha256": "c" * 64}),
                                   ("checkpoint_restore_receipt", {}), ("benchmark_score_based_model_selection", True),
                                   ("distributed_runtime_environment", {"PET_LOCAL_ADDR": "0.0.0.0"}),
                                   ("run_id", "old-attempt")):
                    changed = copy.deepcopy(plan); changed[key] = value
                    with self.subTest(key=key), self.assertRaises(ValueError): s.validate_transport_only_delta(changed)

    def test_original_pins_and_exact_recovery_source_additions_are_required(self):
        with tempfile.TemporaryDirectory() as name:
            plan, original = self.fixture(Path(name).resolve())
            with mock.patch.multiple(s, BASE_PLAN=original, BASE_PLAN_SHA=s.sha(original)):
                for mutate in (lambda p: p["source_files"].pop(0),
                               lambda p: p["source_files"].append(p["source_files"][0]),
                               lambda p: p["recovery_of"].update(prediction_count=1),
                               lambda p: p["recovery_evidence"].pop(),
                               lambda p: p.update(unexpected_protocol_option=True)):
                    changed = copy.deepcopy(plan); mutate(changed)
                    with self.assertRaises(ValueError): s.validate_transport_only_delta(changed)
                Path(plan["recovery_evidence"][0]["path"]).write_text("changed after pin")
                with self.assertRaises(ValueError): s.validate_transport_only_delta(plan)

    def load_real_layers(self):
        adapter_path = OPS / "submit_videomme_reviewed_eval_once.py"
        with mock.patch.multiple(s, ADAPTER=adapter_path, ADAPTER_SHA=s.sha(adapter_path)):
            adapter = s.load_adapter()
        helper_path = OPS / "submit_benchmark_eval_once.py"
        adapter.FROZEN_HELPER = helper_path
        adapter.FROZEN_HELPER_SHA = s.sha(helper_path)
        with mock.patch.dict(sys.modules, {"run_benchmark_eval": types.ModuleType("run_benchmark_eval")}):
            helper = adapter.load_frozen_helper()
        return adapter, helper

    def test_actual_three_layer_load_preserves_paths_and_wraps_each_command_once(self):
        adapter, helper = self.load_real_layers()
        self.assertEqual(Path(adapter.__file__).resolve(), OPS / "submit_videomme_reviewed_eval_once.py")
        self.assertEqual(Path(helper.__file__).resolve(), OPS / "submit_benchmark_eval_once.py")
        self.assertEqual(helper.JOB_SHORT, "dpvlm-videomme-r1r2-s17-a3")
        self.assertEqual(helper.JOB_PREFIX, "jinhaopeng-dpvlm-videomme-r1r2-s17-a3-n1-")
        expected = ["env", "PET_LOCAL_ADDR=127.0.0.1", "bash", str(OPS / "run_benchmark_eval.sh"),
                    "--plan", "/frozen/a3.json", "--plan-sha256", "d" * 64]
        self.assertEqual(helper.runner_argv("/frozen/a3.json", "d" * 64), expected)
        for dry in (False, True):
            argv = helper.scheduler_argv("/frozen/a3.json", "d" * 64, dry_run=dry)
            self.assertEqual(argv[argv.index("--") + 1:], expected)
            self.assertEqual(argv.count("PET_LOCAL_ADDR=127.0.0.1"), 1)
            self.assertEqual(argv[argv.index("-c") + 1], "ali-gpu")
            self.assertEqual(argv[argv.index("-q") + 1], "stream")
            self.assertEqual(argv[argv.index("-p") + 1], "low")
            self.assertEqual(argv[argv.index("-n") + 1], "1")

    def test_original_a2_self_pin_still_validates_after_a3_run_id_patch(self):
        from test_submit_videomme_reviewed_eval_once import ReviewedSubmitTests
        with tempfile.TemporaryDirectory() as name:
            plan, raw_sha = ReviewedSubmitTests().fixture(Path(name).resolve())
            adapter, _ = self.load_real_layers()
            plan["run_id"] = s.RUN_ID; plan["run_root"] = str(Path(plan["run_root"]).with_name(s.RUN_ID))
            adapter.ORIGINAL_AUDIT_SHA = raw_sha
            adapter.validate_review_plan(plan)
            plan["source_files"] = [p for p in plan["source_files"] if p["path"] != str(Path(adapter.__file__).resolve())]
            with self.assertRaises(ValueError): adapter.validate_review_plan(plan)

    def test_real_yaml_validator_accepts_only_exact_wrapped_command_and_resources(self):
        adapter, helper = self.load_real_layers()
        with mock.patch.dict(sys.modules, {"run_benchmark_eval": types.ModuleType("run_benchmark_eval")}):
            from test_submit_benchmark_eval_once import make_yaml
        expected = helper.runner_argv("/frozen/a3.json", "d" * 64)
        document = make_yaml(expected); document["metadata"]["name"] = helper.JOB_PREFIX + "12345"
        self.assertTrue(helper.validate_yaml(document, expected)["stream_low_one_node_eight_gpu_restart_never"])
        for change in (lambda master, pod, c: pod.update(priorityClassName="normal"),
                       lambda master, pod, c: master.update(restartPolicy="OnFailure"),
                       lambda master, pod, c: c["resources"]["limits"].update({"nvidia.com/gpu": 4}),
                       lambda master, pod, c: c["command"].__setitem__(2, c["command"][2].replace("127.0.0.1", "0.0.0.0"))):
            bad = copy.deepcopy(document); master = bad["spec"]["pytorchReplicaSpecs"]["Master"]
            pod = master["template"]["spec"]; change(master, pod, pod["containers"][0])
            with self.assertRaises(ValueError): helper.validate_yaml(bad, expected)

    def test_changed_delta_stops_before_loading_submit_layer_and_ambiguous_result_propagates(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); plan, original = self.fixture(root)
            path = root / "a3.json"; write(path, plan)
            adapter = types.SimpleNamespace(main=mock.Mock(return_value=2))
            with mock.patch.multiple(s, BASE_PLAN=original, BASE_PLAN_SHA=s.sha(original)), mock.patch.object(s, "load_adapter", return_value=adapter):
                argv = ["--plan", str(path), "--plan-sha256", s.sha(path), "--execute"]
                self.assertEqual(s.main(argv), 2); adapter.main.assert_called_once_with(argv)
                plan["protocol"] = "changed"; write(path, plan)
                adapter.main.reset_mock()
                with self.assertRaises(ValueError): s.main(["--plan", str(path), "--plan-sha256", s.sha(path)])
                adapter.main.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
