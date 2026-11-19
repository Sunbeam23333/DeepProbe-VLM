"""Exercise action/retention boundaries with tiny owned files and a fake bucket.

These tests do not attest real models, metrics, a scheduler, or a remote archive.
"""
from copy import deepcopy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from deepprobe_vlm import experiment_suite
from deepprobe_vlm import suite_actions as actions
from deepprobe_vlm import suite_runner as runner


class MemoryStore:
    def __init__(self):
        self.objects = {}
        self.gets = []
        self.puts = []

    def list_prefix(self, prefix):
        return [{"uri": key, "bytes": len(value), "etag": hashlib.sha256(value).hexdigest()}
                for key, value in sorted(self.objects.items()) if key.startswith(prefix)]

    def iter_object(self, uri):
        self.gets.append(uri)
        value = self.objects[uri]
        yield value[:3]
        yield value[3:]

    def put_if_absent(self, uri, source, sha256):
        if uri in self.objects:
            raise FileExistsError(uri)
        value = Path(source).read_bytes()
        assert hashlib.sha256(value).hexdigest() == sha256
        self.objects[uri] = value
        self.puts.append(uri)

    def delete_exact_verified(self, uri, expected):
        value = self.objects[uri]
        assert len(value) == expected['bytes'] and hashlib.sha256(value).hexdigest() == expected['sha256']
        del self.objects[uri]


class ActionRetentionTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.project, self.staging = self.base / "project", self.base / "staging"
        self.project.mkdir(); self.staging.mkdir()
        self.runs = self.project / "runs"
        self.runs.mkdir()
        self.state = self.project / "state"
        self.state.mkdir()
        self.plan = experiment_suite.default_plan()
        self.tasks = {t["id"]: t for t in self.plan["tasks"]}
        self.stage = "train__ouro_r1_s17__alignment"
        self.archive = self.stage.replace("train__", "archive__", 1)
        self.cleanup = self.stage.replace("train__", "cleanup__", 1)
        self.consumer = "train__ouro_r1_s17__sft"
        self.experiment = Path(__file__).resolve().parents[1]
        source = self.write("pins/source.py", "# tiny source identity\n")
        frozen = self.write("pins/frozen.json", "{\"fixture\":true}\n")
        self.old = self.base / "historical-final"
        for name, value in {"language_model/model.safetensors": "tiny-old-weight",
                            "projector.safetensors": "tiny-old-projector",
                            "deepprobe_config.json": "{\"fixture\":true}\n"}.items():
            path = self.old / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value)
        self.old_files = actions.inventory(self.old)
        raw_audit = self.write("pins/raw-audit.json", "{\"synthetic\":true}\n")
        config = runner.artifact(self.experiment / self.tasks[self.stage]["config"]["path"])
        evidence = {"status": "SUITE_HISTORY_ACCEPTED", "task_id": self.stage, "mode": "accepted",
                    "config_sha256": config["sha256"], "frozen_source_manifest_sha256": frozen["sha256"],
                    "checkpoint_inventory_sha256": runner.digest(self.old_files),
                    "optimizer_step": 718, "formal_training_complete": True, "evidence": [raw_audit]}
        accepted = self.write("pins/history-acceptance.json", json.dumps(evidence))
        self.environment = {
            "suite_id": "tiny-suite", "experiment_root": str(self.experiment), "python": sys.executable,
            "run_root": str(self.runs), "source_files": [source], "runtime_environment": {},
            "archive_root": "oss://ss-devbox-stream/tiny-suite-fixture",
            "storage": {"roots": [str(self.project), str(self.staging)], "limit_bytes": 99_000_000_000,
                        "other_reserve_bytes": 0, "free_guard_bytes": 1_000_000_000},
            "assets": {"training": {"train_manifest": raw_audit, "eval_manifest": raw_audit,
                "formal_ready": raw_audit, "frozen_source_manifest": frozen}},
            "history": {self.stage: {"mode": "accepted", "acceptance": accepted, "config": config,
                "frozen_source_manifest": frozen, "checkpoint": {"path": str(self.old), "files": self.old_files}}},
        }
        self.worker = actions.Actions(self.plan, self.environment, self.state)
        self.store = MemoryStore()
        self.worker.store = lambda owner: (self.store, self.environment["archive_root"] + "/runs/" + owner["run_id"] + "/")
        self.worker.guard = lambda owner: {"run_id": owner["run_id"],
            "source_sha256": owner["source_sha256"], "active_writers": []}
        self.worker.live = lambda: {"synthetic_scheduler": True}
        self.accept_fixture_dependency("gpu_gate")

    def write(self, name, value):
        path = self.project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
        return runner.artifact(path)

    def publish(self, task_id, output):
        root = self.state / "tasks" / task_id
        root.mkdir(parents=True, exist_ok=True)
        runner.write_once(root / "COMPLETED.json", {"status": "TASK_ACCEPTED", "task_id": task_id,
                                                   "receipts": [runner.artifact(output)]})

    def accept_fixture_dependency(self, task_id):
        proof = self.write("synthetic-dependencies/" + task_id + ".json", "{\"synthetic\":true}\n")
        output = self.state / (task_id + ".json")
        actions.save(output, {"status": "SUITE_TASK_ACCEPTED", "task_id": task_id,
                              "task_sha256": runner.digest(self.tasks[task_id]), "outputs": [proof]})
        self.publish(task_id, output)

    def execute(self, task_id):
        output = self.state / (task_id + ".json")
        self.worker.run(task_id, output)
        self.publish(task_id, output)
        return runner.read(output)

    def accept_consumer(self):
        proof = self.write("synthetic-consumer/result.json", "{\"units\":718,\"synthetic\":true}\n")
        consumed = self.worker.consume(self.stage, self.tasks[self.consumer], [proof])
        output = self.state / (self.consumer + ".json")
        actions.save(output, {"status": "SUITE_TASK_ACCEPTED", "task_id": self.consumer,
                              "task_sha256": runner.digest(self.tasks[self.consumer]),
                              "outputs": [proof], "consumer_receipts": [consumed]})
        self.publish(self.consumer, output)
        return consumed

    def test_accepted_history_archive_consumers_then_cleanup_preserves_old_run(self):
        trained = self.execute(self.stage)
        final = Path(trained["checkpoint"])
        self.assertNotEqual(final, self.old)
        self.assertEqual(actions.inventory(final), self.old_files)
        archived = self.execute(self.archive)
        self.assertTrue(self.store.puts)
        self.assertEqual(set(self.store.gets), set(self.store.objects))
        self.assertTrue((final / "language_model/model.safetensors").is_file())
        with self.assertRaises(FileNotFoundError):
            self.worker.cleanup(self.tasks[self.cleanup])
        self.accept_consumer()
        self.accept_fixture_dependency("archive__ouro_r1_s17__results")
        before_gets = len(self.store.gets)
        result = self.execute(self.cleanup)
        self.assertGreater(len(self.store.gets), before_gets)
        self.assertFalse((final / "language_model/model.safetensors").exists())
        self.assertFalse((final / "projector.safetensors").exists())
        self.assertTrue((final / "deepprobe_config.json").is_file())
        self.assertEqual(actions.inventory(self.old), self.old_files)
        self.assertEqual(runner.read(result["outputs"][0]["path"])["status"], "OWNED_CHECKPOINT_WEIGHTS_REMOVED")
        # Retained small stage evidence remains usable after its weights go.
        self.worker.completed(self.stage)
        self.worker.completed(self.archive)
        self.assertTrue(Path(archived["manifest"]["path"]).is_file())

    def test_corrupt_remote_payload_blocks_cleanup_and_preserves_checkpoint(self):
        trained = self.execute(self.stage)
        self.execute(self.archive)
        self.accept_consumer()
        uri = next(key for key in self.store.objects if key.endswith("projector.safetensors"))
        self.store.objects[uri] = b"bad-object"
        with self.assertRaisesRegex(ValueError, "GET/SHA differs"):
            self.worker.cleanup(self.tasks[self.cleanup])
        self.assertEqual(actions.inventory(Path(trained["checkpoint"])), self.old_files)
        self.assertEqual(actions.inventory(self.old), self.old_files)

    def test_archive_resume_after_intermediate_cleanup_before_task_receipt(self):
        trained = self.execute(self.stage)
        training = Path(trained["checkpoint"]).parent
        intermediate = training / "checkpoint-718"
        intermediate.mkdir()
        (intermediate / "model.safetensors").write_bytes(b"intermediate")
        (intermediate / "optimizer.pt").write_bytes(b"optimizer")
        (intermediate / "scheduler.pt").write_bytes(b"small-evidence")
        first = self.worker.archive(self.tasks[self.archive])
        self.assertFalse((intermediate / "model.safetensors").exists())
        self.assertTrue((intermediate / "scheduler.pt").is_file())
        self.worker.resume = True
        second = self.worker.archive(self.tasks[self.archive])
        self.assertEqual(first["manifest"], second["manifest"])
        self.assertEqual(actions.inventory(Path(trained["checkpoint"])), self.old_files)
        for item in second["outputs"]:
            runner.verify_pin(item)

    def test_missing_intermediate_without_deletion_intent_is_never_adopted(self):
        trained = self.execute(self.stage)
        training = Path(trained["checkpoint"]).parent
        intermediate = training / "checkpoint-718"
        intermediate.mkdir()
        weight = intermediate / "model.safetensors"
        weight.write_bytes(b"intermediate")
        owner = trained["owner"]
        prefix = self.worker.store(owner)[1]
        manifest = actions.retention.prepare_manifest(owner, archive_prefix=prefix,
            required_consumers=self.worker.requirements(self.stage),
            training_completion=trained["training_complete"], guard=lambda: self.worker.guard(owner))
        actions.retention.archive_and_verify(manifest, self.store, guard=lambda: self.worker.guard(owner))
        weight.unlink()  # Deliberately emulate damage, not an authorized transaction.
        self.worker.resume = True
        with self.assertRaisesRegex((ValueError, FileNotFoundError), "recorded deletion intent|No such file"):
            self.worker.archive(self.tasks[self.archive])
        self.assertEqual(actions.inventory(Path(trained["checkpoint"])), self.old_files)

    def test_v2_consumed_final_backs_up_then_cleans_both_domains_and_resumes(self):
        from deepprobe_vlm.suite_storage import SharedBudget
        from test_checkpoint_backup import FakeHub
        configuration = {'schema_version': 2, 'roots': [str(self.project), str(self.staging)],
            'ledger_root': str(self.project / 'shared-storage'), 'home_limit_bytes': 150_000_000_000,
            'oss_limit_bytes': 150_000_000_000, 'home_safety_bytes': 1_000_000,
            'oss_safety_bytes': 1_000_000, 'free_guard_bytes': 0,
            'oss_prefixes': [self.environment['archive_root'] + '/']}
        self.worker.env['storage'] = configuration
        budget = SharedBudget(configuration, lambda: {'complete': True, 'multipart': [],
                              'objects': self.store.list_prefix(configuration['oss_prefixes'][0])})
        self.worker._shared_budget = budget
        token = budget.reserve({'worker': 'synthetic-worker'}, 1_000_000, 1_000_000)
        api = FakeHub()
        @contextmanager
        def configured(state):
            yield api
        with patch.dict(os.environ, {'DP_SUITE_STORAGE_TOKEN': token,
                                     'DP_SUITE_STORAGE_LEDGER': configuration['ledger_root']}), \
             patch('deepprobe_vlm.checkpoint_backup.configured_api', configured):
            trained = self.execute(self.stage)
            self.execute(self.archive)
            self.accept_consumer()
            self.accept_fixture_dependency('archive__ouro_r1_s17__results')
            original_delete = self.store.delete_exact_verified
            interrupted = []
            def delete_then_interrupt(uri, expected):
                original_delete(uri, expected)
                if not interrupted:
                    interrupted.append(uri)
                    raise RuntimeError('synthetic interruption after exact remote deletion')
            self.store.delete_exact_verified = delete_then_interrupt
            with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                self.worker.cleanup(self.tasks[self.cleanup])
            final = Path(trained['checkpoint'])
            self.assertFalse((final / 'projector.safetensors').exists())
            self.assertTrue(any(name.endswith('projector.safetensors') for name in api.objects))
            self.store.delete_exact_verified = original_delete
            self.worker.resume = True
            result = self.worker.cleanup(self.tasks[self.cleanup])
            statuses = [runner.read(item['path'])['status'] for item in result['outputs']]
            self.assertIn('MODELSCOPE_COMPLETE_FINAL_FULL_GET_SHA256_VERIFIED', statuses)
            self.assertIn('CONSUMED_REMOTE_WEIGHTS_REMOVED_WITH_EXTERNAL_BACKUP', statuses)
            self.assertTrue(all(not uri.endswith('.safetensors') for uri in self.store.objects))
            self.assertTrue((final / 'deepprobe_config.json').is_file())
            self.assertEqual(actions.inventory(self.old), self.old_files)

    def test_torn_checkpoint_resident_bytes_do_not_reduce_future_training_reservation(self):
        task = self.tasks[self.stage]
        family = next(model['family'] for model in self.plan['models'] if model['id'] == task['model_id'])
        source_pin = self.environment['source_files'][0]
        self.worker.env['assets']['backbones'] = {family: {'files': [source_pin],
            'language_root': '/synthetic/language', 'vision_root': '/synthetic/vision'}}
        output = self.project / 'torn-stage'
        (output / 'checkpoint-3').mkdir(parents=True)
        (output / 'checkpoint-3' / 'optimizer.pt').write_bytes(b'x' * 900)
        reserved = []
        self.worker.budget = lambda count: reserved.append(count) or {'synthetic': True}
        estimator = SimpleNamespace(estimate_stage_storage=lambda *args: {'required_additional_bytes': 1000})
        with patch.object(actions, 'load_operation', return_value=estimator):
            result = self.worker.training_storage(task, output)
        self.assertEqual(result['owned_stage_resident_bytes'], 900)
        self.assertEqual(reserved, [1000])

    def test_modern_history_restores_reviewed_cp350_not_stale_cp300_without_numerical_claim(self):
        task = next(t for t in self.plan['tasks'] if t['id'] == 'train__smollm2_s29__alignment')
        self.assertEqual(task['historical_hint']['checkpoint_step'], 300)
        declared = runner.read(self.experiment / task['config']['path'])
        historical = deepcopy(declared)
        historical.pop('experiment_protocol', None)
        historical['model']['language_model'] = '/synthetic/relocated/smollm2'
        config = self.write('pins/smol-history.json', json.dumps(historical))
        old = self.project / 'historical-smol' / 'checkpoint-350'
        names = {'deepprobe_config.json', 'training_provenance.json', 'projector.safetensors',
                 'language_model/config.json', 'vision_model/config.json', 'language_model/model.safetensors',
                 'vision_model/model.safetensors', 'tokenizer/tokenizer_config.json',
                 'image_processor/preprocessor_config.json', 'training_args.bin', 'trainer_state.json',
                 'optimizer.pt', 'scheduler.pt'} | {f'rng_state_{i}.pth' for i in range(8)}
        for name in names:
            path = old / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(json.dumps({'global_step': 350, 'max_steps': 718}).encode()
                             if name == 'trainer_state.json' else b'synthetic-state')
        files = actions.inventory(old)
        prefix = 'oss://ss-devbox-stream/tiny-suite-fixture/runs/smol-a3/training/checkpoint-350/'
        for row in files:
            self.store.objects[prefix + row['path']] = (old / row['path']).read_bytes()
        source = self.environment['source_files'][0]
        frozen = self.environment['assets']['training']['frozen_source_manifest']
        value = {'mode': 'resume', 'checkpoint_step': 350, 'source_root': str(self.experiment), 'config': config,
                 'frozen_source_manifest': frozen, 'checkpoint': {'path': str(old), 'files': files, 'oss_prefix': prefix}}
        acceptance = {'schema_version': 1, 'status': 'SUITE_HISTORY_ACCEPTED', 'production_evidence': True,
            'task_id': task['id'], 'task_sha256': runner.digest(task), 'mode': 'resume', 'evidence': [source],
            'config_sha256': config['sha256'], 'frozen_source_manifest_sha256': frozen['sha256'],
            'checkpoint_inventory_sha256': runner.digest(files), 'checkpoint_oss_prefix': prefix,
            'optimizer_step': 350, 'source_root': str(self.experiment), 'archive_verified': True,
            'resume_metadata_verified': True, 'optimizer_scheduler_rng_deserialized': False}
        value['acceptance'] = self.write('pins/smol-history-acceptance.json', json.dumps(acceptance))
        self.worker.env['history'][task['id']] = value
        self.worker.env['execution_profile'] = {'schema_version': 2, 'fleet_id': 'tiny-suite', 'priority': 'normal',
            'gpu_product': 'NVIDIA-B300', 'node_selector': {'sslaunch-pool': 'stream',
                'aliyun.accelerator/nvidia_name': 'NVIDIA-L20D'}, 'workers': [{'worker_id': 'worker-01',
                'job_name': 'jinhaopeng-dpvlm-fixture', 'namespace': 'default'}]}
        self.worker.env['tools'] = {'ossutil': '/bin/true', 'oss_config': str(self.project / 'fixture-config')}
        owner = self.worker.owner(task['id'])
        with patch('deepprobe_vlm.checkpoint_store.OssutilStore', return_value=self.store):
            _, target = self.worker.history(task, owner)
        self.assertEqual(target.name, 'checkpoint-350')
        self.assertEqual(actions.inventory(target), files)
        self.assertNotIn('exact_resume_accepted', acceptance)
        from deepprobe_vlm.resume_checkpoint import verify_resume_checkpoint
        observed = verify_resume_checkpoint(target, world_size=8, expected_max_steps=718)
        self.assertEqual(observed['optimizer_step'], 350)
        self.assertEqual(observed['validation_scope'], 'file_presence_and_trainer_state_structure_only')
        changed = deepcopy(historical)
        changed['training']['learning_rate'] *= 2
        with self.assertRaisesRegex(ValueError, 'effective model/training recipe'):
            actions.verify_historical_recipe(changed, declared)


class TerminalJobTest(unittest.TestCase):
    def test_only_unambiguous_inactive_terminal_jobs_are_terminal(self):
        job = {"metadata": {}, "status": {"conditions": [{"type": "Succeeded", "status": "True"}],
               "replicaStatuses": {"Master": {"active": 0, "succeeded": 1}}}}
        self.assertTrue(actions.terminal_job(job))
        cases = [lambda j: j["status"]["replicaStatuses"]["Master"].update(active=1),
                 lambda j: j["status"]["conditions"].append({"type": "Running", "status": "True"}),
                 lambda j: j["status"]["conditions"].append({"type": "Failed", "status": "True"}),
                 lambda j: j["metadata"].update(deletionTimestamp="2026-11-12T00:00:00Z"),
                 lambda j: j["status"].update(conditions=[])]
        for mutate in cases:
            changed = deepcopy(job)
            mutate(changed)
            self.assertFalse(actions.terminal_job(changed))


if __name__ == "__main__":
    unittest.main()
