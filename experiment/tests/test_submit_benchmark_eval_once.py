import copy
import importlib.util
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("submit_benchmark_eval_once", Path(__file__).resolve().parents[1] /
                                           "scripts/operations/submit_benchmark_eval_once.py")
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)

# The actual accepted sslaunch setup wrapper, independent of evaluation args.
PREFIX = ('cp /mnt/vepfs/base/infra/linux-bcecmd-0.5.6/bcecmd /usr/sbin && echo "switching to submit user: jinhaopeng" && export SSLAUNCH_USER=jinhaopeng SSLAUNCH_UID=83980 SSLAUNCH_GID=83980 && getent group "$SSLAUNCH_GID" >/dev/null 2>&1 || groupadd -g "$SSLAUNCH_GID" "$SSLAUNCH_USER" && if id "$SSLAUNCH_USER" >/dev/null 2>&1; then   usermod -d /home/jinhaopeng "$SSLAUNCH_USER" 2>/dev/null || true; else   useradd -u "$SSLAUNCH_UID" -g "$SSLAUNCH_GID"     -d /home/jinhaopeng -s /bin/bash -M "$SSLAUNCH_USER"; fi && su "$SSLAUNCH_USER" -c ')


def make_yaml(expected):
    inner = "cd /home/jinhaopeng && export HOME=/home/jinhaopeng && " + shlex.join(expected)
    return {"apiVersion": "kubeflow.org/v1", "kind": "PyTorchJob",
        "metadata": {"name": submit.JOB_PREFIX + "261012230000", "namespace": "default"},
        "spec": {"pytorchReplicaSpecs": {"Master": {"replicas": 1, "restartPolicy": "Never",
            "template": {"spec": {"nodeSelector": {"sslaunch-pool": "stream"}, "priorityClassName": "low",
                "containers": [{"name": "pytorch", "workingDir": "/home/jinhaopeng",
                    "resources": {"limits": {"nvidia.com/gpu": 8}},
                    "command": ["/bin/bash", "-c", PREFIX + shlex.quote(inner)]}]}}}}}}


class SingleSubmitTest(unittest.TestCase):
    def test_scheduler_parser_checks_total_and_optional_details(self):
        text = ("Summary:\njinhaopeng ali-gpu - 2 1 0 1 0\n\nJob List:\nJob Name GPUs Nodes State Priority Created At Running Time\n"
                "jinhaopeng-dpvlm-old-n1-123 8 1 Succeeded low 2026-10-12T00:00:00Z 1h\n"
                "jinhaopeng-other-n1-123 8 1 Created low 2026-10-12T00:00:00Z - NodeScheduling: FailedScheduling\n")
        jobs, details = submit.jobs_in_list(text)
        self.assertEqual(jobs, {"jinhaopeng-dpvlm-old-n1-123": "Succeeded"})
        self.assertEqual(details["jinhaopeng-other-n1-123"], "NodeScheduling: FailedScheduling")
        with self.assertRaisesRegex(ValueError, "row count"):
            submit.jobs_in_list(text.replace("ali-gpu - 2", "ali-gpu - 3"))
        with self.assertRaises(ValueError):
            submit.jobs_in_list(text + text.splitlines()[-1] + "\n")

    def test_kubernetes_unknown_and_conflicting_terminal_conditions_block(self):
        name = submit.PROJECT_PREFIX + "old"
        item = {"metadata": {"name": name, "namespace": "default"}, "status": {}}
        self.assertEqual(submit.kube_states(json.dumps({"items": [item]}))[name], "ActiveOrUnknown")
        item["status"]["conditions"] = [{"type": "Succeeded", "status": "True"}, {"type": "Running", "status": "True"}]
        self.assertEqual(submit.kube_states(json.dumps({"items": [item]}))[name], "ActiveOrUnknown")
        item["status"]["conditions"] = [{"type": "Succeeded", "status": "True"}]
        self.assertEqual(submit.kube_states(json.dumps({"items": [item]}))[name], "Succeeded")
        item["status"] = {"phase": "Pending"}
        self.assertEqual(submit.kube_states(json.dumps({"items": [item]}), pods=True)[name], "ActiveOrUnknown")

    def test_only_exact_empty_oss_listing_is_accepted(self):
        submit.verify_oss_empty("Object Number is: 0\n\n0.03(s) elapsed\n")
        for value in ("", "Object Number is: 1\n", "Object Number is: 0\nObject Number is: 0\n",
                      "2026 1 oss://ss-devbox-stream/object\nObject Number is: 0\n"):
            with self.assertRaises(ValueError):
                submit.verify_oss_empty(value)

    def test_yaml_rejects_resource_priority_node_and_command_drift(self):
        expected = ["bash", "/frozen/run_benchmark_eval.sh", "--plan", "/frozen/plan.json", "--plan-sha256", "a" * 64]
        document = make_yaml(expected)
        self.assertTrue(submit.validate_yaml(document, expected)["stream_low_one_node_eight_gpu_restart_never"])
        mutations = [
            lambda master, pod, container: pod.update(priorityClassName="normal"),
            lambda master, pod, container: pod.update(nodeName="forced-node"),
            lambda master, pod, container: pod.update(nodeSelector={"sslaunch-pool": "embody"}),
            lambda master, pod, container: master.update(replicas=2),
            lambda master, pod, container: master.update(restartPolicy="OnFailure"),
            lambda master, pod, container: container["resources"]["limits"].update({"nvidia.com/gpu": 4}),
            lambda master, pod, container: container.update(env=[{"name": "CUDA_VISIBLE_DEVICES", "value": ""}]),
            lambda master, pod, container: container["command"].__setitem__(2, container["command"][2].replace("/frozen/plan.json", "/other.json")),
            lambda master, pod, container: container["command"].__setitem__(2, "echo unexpected && " + container["command"][2]),
        ]
        for mutate in mutations:
            bad = copy.deepcopy(document)
            master = bad["spec"]["pytorchReplicaSpecs"]["Master"]
            pod = master["template"]["spec"]
            mutate(master, pod, pod["containers"][0])
            with self.assertRaises(ValueError):
                submit.validate_yaml(bad, expected)

    def test_intent_cannot_be_overwritten_or_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "intent.json"
            submit.write_exclusive(path, {"attempt": 1})
            with self.assertRaises(FileExistsError):
                submit.write_exclusive(path, {"attempt": 2})
            self.assertEqual(json.loads(path.read_text()), {"attempt": 1})

    def test_uncertain_client_is_only_followed_by_readonly_exact_discovery(self):
        exact = submit.JOB_PREFIX + "261012230000"
        snapshot = {"jobs": {exact: "Created"}, "kubernetes": {exact: "ActiveOrUnknown"},
                    "pods": {exact + "-master-0": "ActiveOrUnknown"}}
        calls = []
        def capture(directory, label, argv, timeout=180):
            calls.append(argv)
            return {"returncode": 0, "stdout": f"Job: {exact}\nState: Created\n"}
        with patch.object(submit, "live_snapshot", return_value=snapshot), patch.object(submit, "capture", side_effect=capture):
            result = submit.inspect_after(Path("/unused"), {}, None)
        self.assertTrue(result["inspection_complete"])
        self.assertTrue(result["no_other_active_deepprobe_job"])
        self.assertEqual(calls, [[submit.SSLAUNCH, "job", "info", "-c", "ali-gpu", exact]])
        snapshot["jobs"][submit.JOB_PREFIX + "duplicate"] = "Created"
        with patch.object(submit, "live_snapshot", return_value=snapshot), patch.object(submit, "capture") as command:
            result = submit.inspect_after(Path("/unused"), {}, None)
            command.assert_not_called()
        self.assertFalse(result["inspection_complete"])


if __name__ == "__main__":
    unittest.main()
