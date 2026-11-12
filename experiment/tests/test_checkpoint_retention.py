"""Destructive retention is tested only against synthetic owned temporary runs."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from deepprobe_vlm import checkpoint_retention as r
from deepprobe_vlm.checkpoint_store import OssutilStore, _field


class MemoryStore:
    def __init__(self):
        self.objects = {}
        self.puts = []
        self.gets = []
        self.list_hook = None

    def list_prefix(self, prefix):
        if self.list_hook:
            self.list_hook(self)
        return [{"uri": key, "bytes": len(value), "etag": hashlib.sha256(value).hexdigest()}
                for key, value in sorted(self.objects.items()) if key.startswith(prefix)]

    def iter_object(self, uri):
        self.gets.append(uri)
        value = self.objects[uri]
        yield value[:3]
        yield value[3:]

    def put_if_absent(self, uri, source, sha):
        if uri in self.objects:
            raise FileExistsError(uri)
        value = Path(source).read_bytes()
        assert hashlib.sha256(value).hexdigest() == sha
        self.puts.append(uri)
        self.objects[uri] = value


class RetentionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.runs = self.base / "runs"
        self.runs.mkdir()
        self.owner = r.claim_owned_run(self.runs, "run-a", "suite-a", "a" * 64)
        self.root = Path(self.owner["run_root"])
        self.guard = lambda: {"run_id": "run-a", "source_sha256": "a" * 64, "active_writers": []}
        values = {"training/final/language_model/model.safetensors": b"final-model",
                  "training/final/projector.safetensors": b"projector",
                  "training/final/config.json": b'{"model":"tiny"}',
                  "training/checkpoint-3/language_model/model.safetensors": b"checkpoint-model",
                  "training/checkpoint-3/optimizer.pt": b"optimizer-state",
                  "training/checkpoint-3/rng_state_0.pth": b"rng-must-remain",
                  "training/checkpoint-3/scheduler.pt": b"scheduler-must-remain",
                  "training/train_results.json": b'{"train_loss":1}',
                  "stdout.log": b"completed\n"}
        for name, data in values.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.requirements = [{"id": name, "kind": kind, "checkpoint": "training/final",
                              "protocol_sha256": char * 64, "expected_units": count}
                             for name, kind, char, count in (("sft", "training", "b", 718),
                                 ("videomme", "evaluation", "c", 2697),
                                 ("tempcompass", "evaluation", "d", 7540),
                                 ("latency", "latency", "e", 416))]
        final = r.checkpoint_inventory(self.root, "training/final")
        outputs = [r.file_pin(self.root / "training/train_results.json")]
        training = {"schema_version": 1, "status": "SUITE_TRAINING_COMPLETE", "run_id": "run-a",
                    "source_sha256": "a" * 64, "final_checkpoint_inventory_sha256": final["inventory_sha256"],
                    "optimizer_steps": 3, "target_optimizer_steps": 3, "exit_code": 0,
                    "outputs": outputs, "output_inventory_sha256": r.digest(outputs)}
        training_path = self.root / "SUITE_TRAINING_COMPLETE.json"
        training_path.write_text(json.dumps(training))
        self.manifest = r.prepare_manifest(self.owner,
            archive_prefix="oss://ss-devbox-stream/project/runs/run-a/",
            required_consumers=self.requirements, training_completion=r.file_pin(training_path), guard=self.guard)
        self.receipts = {}
        for requirement in self.requirements:
            proof = self.base / (requirement["id"] + "-result.json")
            proof.write_text(json.dumps({"complete": True, "count": requirement["expected_units"]}))
            outputs = [r.file_pin(proof)]
            receipt = {"schema_version": 1, "status": "SUITE_CONSUMER_COMPLETE",
                "consumer_id": requirement["id"], "kind": requirement["kind"],
                "source_sha256": "a" * 64,
                "checkpoint_inventory_sha256": self.manifest["checkpoints"]["training/final"],
                "protocol_sha256": requirement["protocol_sha256"],
                "completed_units": requirement["expected_units"], "failed_units": 0,
                "outputs": outputs, "output_inventory_sha256": r.digest(outputs)}
            path = self.base / (requirement["id"] + "-receipt.json")
            path.write_text(json.dumps(receipt))
            self.receipts[requirement["id"]] = r.file_pin(path)
        self.store = MemoryStore()

    def archive(self):
        return r.archive_and_verify(self.manifest, self.store, guard=self.guard)

    def plan(self):
        return r.plan_cleanup(self.manifest, self.receipts)

    def assert_weights_present(self):
        self.assertTrue(all((self.root / row["path"]).is_file()
                            for row in self.manifest["files"] if r.cleanable(row["path"])))

    def test_complete_archive_reads_every_object_and_cleanup_preserves_small_evidence(self):
        accepted = self.archive()
        self.assertEqual(accepted["objects"], len(self.manifest["files"]) + 1)
        self.assertTrue(accepted["all_objects_full_get_sha256"])
        self.assertEqual(set(self.store.gets), set(self.store.objects))
        plan = self.plan()
        result = r.cleanup(plan, self.store, guard=self.guard)
        self.assertEqual(result["status"], "CLEANUP_PLAN_ONLY")
        self.assert_weights_present()
        before_gets = len(self.store.gets)
        result = r.cleanup(plan, self.store, guard=self.guard, execute=True)
        self.assertEqual(result["status"], "OWNED_CHECKPOINT_WEIGHTS_REMOVED")
        self.assertEqual(len(self.store.gets) - before_gets, len(self.store.objects))
        for row in self.manifest["files"]:
            path = self.root / row["path"]
            if r.cleanable(row["path"]):
                self.assertFalse(path.exists())
            else:
                r.check_pin({**row, "path": str(path)})
        self.assertEqual(r.cleanup(plan, self.store, guard=self.guard, execute=True)["receipt"], result["receipt"])

    def test_archive_resume_reuses_only_matching_partial_objects(self):
        first = self.manifest["files"][0]
        uri = self.manifest["archive_prefix"] + first["path"]
        self.store.objects[uri] = (self.root / first["path"]).read_bytes()
        self.archive()
        self.assertNotIn(uri, self.store.puts)
        self.assert_weights_present()

    def test_wrong_partial_archive_or_extra_key_never_overwrites_or_deletes(self):
        uri = self.manifest["archive_prefix"] + self.manifest["files"][0]["path"]
        self.store.objects[uri] = b"bad"
        with self.assertRaisesRegex(ValueError, "existing archive object differs"):
            self.archive()
        self.assertEqual(self.store.objects[uri], b"bad")
        self.assert_weights_present()
        self.store.objects = {self.manifest["archive_prefix"] + "foreign": b"foreign"}
        with self.assertRaisesRegex(ValueError, "unexpected object"):
            self.archive()
        self.assert_weights_present()

    def test_dependency_and_coverage_gates_require_all_consumers(self):
        incomplete = {k: v for k, v in self.receipts.items() if k != "latency"}
        with self.assertRaisesRegex(ValueError, "every declared consumer"):
            r.plan_cleanup(self.manifest, incomplete)
        path = Path(self.receipts["videomme"]["path"])
        value = json.loads(path.read_text())
        value["completed_units"] -= 1
        path.write_text(json.dumps(value))
        self.receipts["videomme"] = r.file_pin(path)
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.plan()
        self.assert_weights_present()

    def test_completed_training_allows_intermediate_cleanup_but_final_waits_for_consumers(self):
        self.archive()
        first = r.plan_cleanup(self.manifest, {}, checkpoints=["training/checkpoint-3"])
        result = r.cleanup(first, self.store, guard=self.guard, execute=True)
        self.assertTrue(result["files"])
        self.assertFalse((self.root / "training/checkpoint-3/optimizer.pt").exists())
        self.assertTrue((self.root / "training/final/projector.safetensors").is_file())
        with self.assertRaisesRegex(ValueError, "every declared consumer"):
            r.plan_cleanup(self.manifest, {}, checkpoints=["training/final"])
        final = r.plan_cleanup(self.manifest, self.receipts, checkpoints=["training/final"])
        result = r.cleanup(final, self.store, guard=self.guard, execute=True)
        self.assertEqual(result["status"], "OWNED_CHECKPOINT_WEIGHTS_REMOVED")
        self.assertFalse((self.root / "training/final/projector.safetensors").exists())
        self.assertTrue((self.root / "training/checkpoint-3/rng_state_0.pth").is_file())

    def test_stopped_training_cannot_prune_any_checkpoint(self):
        path = Path(self.manifest["training_completion"]["path"])
        receipt = json.loads(path.read_text())
        receipt["target_optimizer_steps"] = 4
        path.write_text(json.dumps(receipt))
        manifest = deepcopy(self.manifest)
        manifest["training_completion"] = r.file_pin(path)
        with self.assertRaisesRegex(ValueError, "declared update budget"):
            r.plan_cleanup(manifest, {}, checkpoints=["training/checkpoint-3"])
        self.assert_weights_present()

    def test_all_checkpoint_cleanup_reuses_prior_intermediate_deletion(self):
        self.archive()
        first = r.plan_cleanup(self.manifest, {}, checkpoints=["training/checkpoint-3"])
        r.cleanup(first, self.store, guard=self.guard, execute=True)
        result = r.cleanup(self.plan(), self.store, guard=self.guard, execute=True)
        self.assertEqual(set(result["absent_after_prior_transaction"]),
                         {row["path"] for row in first["targets"]})
        self.assertEqual(set(result["removed_this_invocation"]),
                         {row["path"] for row in self.manifest["files"]
                          if r.cleanable(row["path"]) and row["path"].startswith("training/final/")})
        self.assertEqual(r.cleanup(self.plan(), self.store, guard=self.guard, execute=True)["receipt"],
                         result["receipt"])

    def test_success_marker_is_insufficient_when_bound_outputs_changed(self):
        (self.base / "videomme-result.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "pinned file changed"):
            self.plan()

    def test_wrong_checkpoint_and_source_receipts_fail(self):
        for field in ("source_sha256", "checkpoint_inventory_sha256", "protocol_sha256"):
            with self.subTest(field=field):
                path = Path(self.receipts["sft"]["path"])
                original = path.read_text()
                value = json.loads(original)
                value[field] = "f" * 64
                path.write_text(json.dumps(value))
                altered = {**self.receipts, "sft": r.file_pin(path)}
                with self.assertRaisesRegex(ValueError, "identity differs"):
                    r.plan_cleanup(self.manifest, altered)
                path.write_text(original)

    def test_archive_corruption_stability_and_active_writer_block_all_deletion(self):
        self.archive()
        plan = self.plan()
        key = next(iter(self.store.objects))
        old = self.store.objects[key]
        self.store.objects[key] = b"x" * len(old)
        with self.assertRaisesRegex(ValueError, "GET/SHA"):
            r.cleanup(plan, self.store, guard=self.guard, execute=True)
        self.assert_weights_present()
        self.store.objects[key] = old
        active = lambda: {"run_id": "run-a", "source_sha256": "a" * 64, "active_writers": [123]}
        with self.assertRaisesRegex(ValueError, "writer"):
            r.cleanup(plan, self.store, guard=active, execute=True)
        self.assert_weights_present()
        count = [0]
        def mutate(store):
            count[0] += 1
            if count[0] == 2:
                store.objects[self.manifest["archive_prefix"] + "new"] = b"new"
        self.store.list_hook = mutate
        with self.assertRaisesRegex(ValueError, "prefix changed"):
            r.cleanup(plan, self.store, guard=self.guard, execute=True)
        self.assert_weights_present()

    def test_all_local_targets_checked_before_first_unlink(self):
        self.archive()
        plan = self.plan()
        last = self.root / plan["targets"][-1]["path"]
        last.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "pinned file changed"):
            r.cleanup(plan, self.store, guard=self.guard, execute=True)
        self.assert_weights_present()

    def test_interruption_after_unlink_resumes_from_durable_file_intent(self):
        self.archive()
        plan = self.plan()
        first = self.root / plan["targets"][0]["path"]
        original = Path.unlink
        def interrupted(path, *args, **kwargs):
            original(path, *args, **kwargs)
            if path == first:
                raise KeyboardInterrupt("simulated process interruption")
        with patch.object(Path, "unlink", interrupted), self.assertRaises(KeyboardInterrupt):
            r.cleanup(plan, self.store, guard=self.guard, execute=True)
        result = r.cleanup(plan, self.store, guard=self.guard, execute=True)
        self.assertIn(plan["targets"][0]["path"], result["absent_after_durable_intent"])

    def test_unexplained_missing_target_or_changed_plan_is_rejected(self):
        self.archive()
        plan = self.plan()
        (self.root / plan["targets"][0]["path"]).unlink()
        with self.assertRaisesRegex(ValueError, "without a recorded"):
            r.cleanup(plan, self.store, guard=self.guard, execute=True)
        changed = deepcopy(plan)
        changed["targets"].append({"path": "stdout.log", "bytes": 10, "sha256": "a" * 64})
        with self.assertRaisesRegex(ValueError, "plan changed"):
            r.cleanup(changed, self.store, guard=self.guard, execute=True)

    def test_existing_unowned_runs_and_symlinks_are_not_adopted(self):
        old = self.runs / "old"
        old.mkdir()
        (old / "weights.safetensors").write_bytes(b"user-data")
        with self.assertRaisesRegex(ValueError, "existing nonempty"):
            r.claim_owned_run(self.runs, "old", "suite-a", "a" * 64)
        (self.root / "training/final/projector.safetensors").unlink()
        (self.root / "training/final/projector.safetensors").symlink_to(old / "weights.safetensors")
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.archive()
        self.assertEqual((old / "weights.safetensors").read_bytes(), b"user-data")

    def test_restore_streams_exact_checkpoint_and_reuses_completed_files(self):
        self.archive()
        destination = self.base / "new-restore"
        self.assertEqual(r.restore_checkpoint(self.manifest, "training/final", destination,
                                             self.store)["status"], "RESTORE_PLAN_ONLY")
        self.assertFalse(destination.exists())
        result = r.restore_checkpoint(self.manifest, "training/final", destination, self.store, execute=True)
        self.assertEqual(result["status"], "CHECKPOINT_RESTORED_AND_SHA256_VERIFIED")
        gets = len(self.store.gets)
        self.assertEqual(r.restore_checkpoint(self.manifest, "training/final", destination,
                                             self.store, execute=True), result)
        self.assertEqual(gets, len(self.store.gets))
        (destination / "projector.safetensors").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "pinned file changed"):
            r.restore_checkpoint(self.manifest, "training/final", destination, self.store, execute=True)

    def test_corrupted_restore_never_publishes_a_complete_checkpoint(self):
        self.archive()
        uri = self.manifest["archive_prefix"] + "training/final/projector.safetensors"
        self.store.objects[uri] = b"X" * len(self.store.objects[uri])
        destination = self.base / "bad-restore"
        with self.assertRaisesRegex(ValueError, "restored object SHA"):
            r.restore_checkpoint(self.manifest, "training/final", destination, self.store, execute=True)
        self.assertFalse((destination / "projector.safetensors").exists())
        self.assertFalse((destination / "RESTORED.json").exists())


class StoreContractTest(unittest.TestCase):
    def test_actual_api_xml_json_shapes(self):
        self.assertEqual(_field({"UploadId": "new-id"}, "uploadid"), "new-id")
        self.assertEqual(_field({"Header": {"Etag": ['"hash"']}}, "etag"), '"hash"')
        with self.assertRaises(ValueError):
            _field({"a": {"ETag": "one"}, "b": {"Etag": "two"}}, "etag")

    def store(self, root):
        executable, config = root / "ossutil", root / "config"
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
        config.write_text("not-a-credential")
        return OssutilStore(executable, config, allowed_prefix="oss://ss-devbox-stream/project/runs/run-a/",
                            state_dir=root / "state")

    def test_list_pagination_counts_and_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            store = self.store(Path(temp).resolve())
            prefix = store.allowed_prefix
            key = prefix.removeprefix("oss://ss-devbox-stream/")
            def page(name, truncated, next_token=None):
                result = {"Name": "ss-devbox-stream", "Prefix": key, "KeyCount": "1",
                          "IsTruncated": truncated, "Contents": {"Key": key + name,
                          "Size": "7", "ETag": '"etag"', "LastModified": "now"}}
                if next_token:
                    result["NextContinuationToken"] = next_token
                return result
            with patch.object(store, "_json", side_effect=[page("a", "true", "next"), page("b", "false")]) as call:
                self.assertEqual([x["uri"] for x in store.list_prefix(prefix)], [prefix + "a", prefix + "b"])
                self.assertIn("next", call.call_args_list[1].args[0])
            with patch.object(store, "_json", return_value=page("a", "true", "same")):
                with self.assertRaises(ValueError):
                    store.list_prefix(prefix)

    def test_small_and_multipart_uploads_use_atomic_overwrite_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            store = self.store(root)
            store.PART_BYTES = 4  # tiny fixture exercises exactly the large-object state machine
            source = root / "source"
            source.write_bytes(b"abcdefghij")
            calls = []
            def api(args):
                calls.append(args)
                if args[0] == "initiate-multipart-upload":
                    return {"UploadId": "owned-upload"}
                if args[0] == "upload-part":
                    path = Path(args[args.index("--body") + 1].removeprefix("file://"))
                    self.assertLessEqual(path.stat().st_size, 4)
                    return {"Header": {"Etag": ['"' + hashlib.md5(path.read_bytes()).hexdigest() + '"']}}
                return {"ETag": "complete"}
            with patch.object(store, "list_prefix", return_value=[]), patch.object(store, "_json", side_effect=api):
                store.put_if_absent(store.allowed_prefix + "large", source, r.file_pin(source)["sha256"])
            self.assertEqual([x[0] for x in calls], ["initiate-multipart-upload", "upload-part", "upload-part",
                                                   "upload-part", "complete-multipart-upload"])
            for command in (calls[0], calls[-1]):
                self.assertEqual(command[command.index("--forbid-overwrite") + 1], "true")
            self.assertFalse(list((root / "state").rglob("part.pending")))
            small = root / "small"
            small.write_bytes(b"tiny")
            with patch.object(store, "list_prefix", return_value=[]), patch.object(store, "_json", return_value={}) as call:
                store.put_if_absent(store.allowed_prefix + "small", small, r.file_pin(small)["sha256"])
            self.assertEqual(call.call_args.args[0][0], "put-object")
            self.assertIn("--forbid-overwrite", call.call_args.args[0])

    def test_unknown_multipart_initiation_is_never_blindly_repeated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            store = self.store(root)
            store.PART_BYTES = 4
            source = root / "source"
            source.write_bytes(b"long-file")
            with patch.object(store, "list_prefix", return_value=[]), patch.object(store, "_json", side_effect=TimeoutError):
                with self.assertRaises(TimeoutError):
                    store.put_if_absent(store.allowed_prefix + "large", source, r.file_pin(source)["sha256"])
            with patch.object(store, "list_prefix", return_value=[]), patch.object(store, "_json") as call:
                with self.assertRaisesRegex(ValueError, "unknown initiation"):
                    store.put_if_absent(store.allowed_prefix + "large", source, r.file_pin(source)["sha256"])
                call.assert_not_called()

    def test_streaming_cat_has_no_object_sized_readback_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            store = self.store(root)
            store.executable.write_text("#!" + sys.executable + "\nimport sys\nsys.stdout.buffer.write(b'payload' * 200000)\n")
            actual = b"".join(store.iter_object(store.allowed_prefix + "object"))
            self.assertEqual(actual, b"payload" * 200000)
            self.assertEqual(list(store.state_dir.rglob("*")), [])

    def test_multipart_resume_reuses_same_upload_and_completed_parts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            store = self.store(root)
            store.PART_BYTES = 4
            source = root / "source"
            source.write_bytes(b"abcdefghij")
            calls = []
            fail = [True]
            def api(args):
                calls.append(args)
                if args[0] == "initiate-multipart-upload":
                    return {"UploadId": "only-upload"}
                if args[0] == "upload-part":
                    number = args[args.index("--part-number") + 1]
                    if number == "2" and fail[0]:
                        fail[0] = False
                        raise RuntimeError("simulated interrupted part")
                    return {"Header": {"Etag": ['"part-' + number + '"']}}
                return {"ETag": "complete"}
            with patch.object(store, "list_prefix", return_value=[]), patch.object(store, "_json", side_effect=api):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    store.put_if_absent(store.allowed_prefix + "large", source, r.file_pin(source)["sha256"])
                store.put_if_absent(store.allowed_prefix + "large", source, r.file_pin(source)["sha256"])
            self.assertEqual(sum(c[0] == "initiate-multipart-upload" for c in calls), 1)
            part_numbers = [c[c.index("--part-number") + 1] for c in calls if c[0] == "upload-part"]
            self.assertEqual(part_numbers, ["1", "2", "2", "3"])


if __name__ == "__main__":
    unittest.main()
