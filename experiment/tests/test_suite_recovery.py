"""Synthetic complete native layouts; these tests never certify production work."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from deepprobe_vlm import experiment_suite, formal_integrity, suite_recovery, suite_runner


class MemoryStore:
    def __init__(self, objects=None):
        self.objects = objects or {}; self.uploaded = []
    def list_prefix(self, prefix):
        return [{"uri": k, "bytes": len(v), "etag": hashlib.sha256(v).hexdigest()}
                for k, v in sorted(self.objects.items()) if k.startswith(prefix)]
    def put_if_absent(self, uri, path, sha):
        assert uri not in self.objects
        raw = Path(path).read_bytes(); assert hashlib.sha256(raw).hexdigest() == sha
        self.objects[uri] = raw; self.uploaded.append(uri)
    def iter_object(self, uri):
        yield self.objects[uri]


class Budget:
    def __init__(self): self.reservations = []; self.releases = []
    def reserve(self, owner, **kw):
        self.reservations.append((owner, kw)); return {"owner": owner}
    def release(self, token, *, quiescent):
        assert quiescent(token["owner"])["active_writers"] == []
        self.releases.append(token)


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve(); self.plan = experiment_suite.default_plan()
        self.source = self.root / "source"
        for name in formal_integrity.SOURCE_DIRS: (self.source / name).mkdir(parents=True)
        for name in {*formal_integrity.SOURCE_FILES, *formal_integrity.REQUIRED_P1_SOURCE}:
            path = self.source / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("synthetic source\n")
        self.manifest = self.pin("manifest.json", {"schema_version": 1, "purpose": "DEEPPROBE_FROZEN_P1_SOURCE",
                                                   "files": formal_integrity.source_inventory(self.source)})
        self.config = self.pin("config.json", {"scope": "SYNTHETIC_UNIT_TEST_ONLY"})
        self.env = {"suite_id": "synthetic", "run_root": str(self.root / "runs"),
                    "storage": {"roots": [str(self.root)], "oss_prefixes": ["oss://ss-devbox-stream/project/"]},
                    "tools": {"ossutil": "/synthetic/ossutil", "oss_config": "/synthetic/config"}}
        self.budget = Budget()

    def pin(self, name, value):
        p = self.root / name; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value)); return suite_runner.artifact(p)

    def request(self, step, *, restore=False):
        task_id = "train__smollm2_s29__alignment" if step == 350 else "train__ouro_r4_s17__sft"
        task = next(t for t in self.plan["tasks"] if t["id"] == task_id)
        root = self.root / "old-run" / ("checkpoint-" + str(step)); root.mkdir(parents=True)
        names = {"deepprobe_config.json", "training_provenance.json", "projector.safetensors", "language_model/config.json",
            "vision_model/config.json", "language_model/model.safetensors", "vision_model/model.safetensors",
            "tokenizer/tokenizer_config.json", "image_processor/preprocessor_config.json", "training_args.bin", "optimizer.pt", "scheduler.pt"}
        names |= {f"rng_state_{i}.pth" for i in range(8)} | {"trainer_state.json"}
        files, objects = [], {}
        prefix = "oss://ss-devbox-stream/project/runs/historical/checkpoint-" + str(step) + "/"
        for name in sorted(names):
            path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
            raw = (json.dumps({"global_step": step, "max_steps": 718}).encode() if name == "trainer_state.json"
                   else b'{}' if name in {"deepprobe_config.json", "training_provenance.json"}
                   else b"synthetic state bytes")
            path.write_bytes(raw); files.append({**suite_runner.artifact(path), "path": name}); objects[prefix + name] = raw
        checkpoint = self.root / "runs/.recovery-inputs/model" / root.name if restore else root
        entry = {"mode": "resume", "checkpoint_step": step, "source_root": str(self.source), "config": self.config,
            "frozen_source_manifest": self.manifest, "checkpoint": {"path": str(checkpoint), "files": files, "oss_prefix": prefix}}
        operation = "restore_archived" if restore else "archive_local"
        review = {"status": "SUITE_RECOVERY_PREPARATION_REVIEW_ACCEPTED", "approved": True, "production_evidence": True,
            "task_id": task_id, "task_sha256": suite_runner.digest(task), "entry_sha256": suite_runner.digest(entry), "operation": operation,
            "evidence": [self.config], "verified_facts": {"source_root": str(self.source), "config_sha256": self.config["sha256"],
                "frozen_source_manifest_sha256": self.manifest["sha256"], "checkpoint_inventory_sha256": suite_runner.digest(files),
                "checkpoint_oss_prefix": prefix, "optimizer_step": step, "resume_metadata_verified": True,
                "optimizer_scheduler_rng_deserialized": False}}
        request = {"task_id": task_id, "operation": operation, "entry": self.pin("entry.json", entry),
            "review": self.pin("review.json", review), "output_dir": str(self.root / "control")}
        return request, objects, root

    def execute(self, request, store, **kwargs):
        return suite_recovery.prepare_recovery(self.plan, self.env, request,
            store_factory=lambda *a, **kw: store, budget_factory=lambda env: self.budget, **kwargs)

    def test_cp350_archives_all_native_files_then_imports_and_idempotently_resumes(self):
        request, objects, old = self.request(350); store = MemoryStore()
        value = self.execute(request, store)
        self.assertEqual(store.objects, objects); self.assertEqual(len(store.uploaded), len(objects))
        report = suite_runner.read(Path(request["output_dir"]) / "RECOVERY_PREPARED.json")
        self.assertEqual(report["remaining_optimizer_updates"], 368)
        self.assertFalse(report["runtime_resume_verified"]); self.assertFalse(report["gpu_submitted"])
        self.assertEqual(self.execute(request, store, resume=True), value)
        self.assertTrue((old / "optimizer.pt").exists()); self.assertEqual(len(self.budget.releases), 2)

    def test_cp600_full_restore_into_new_owned_input_preserves_original_and_imports(self):
        request, objects, old = self.request(600, restore=True); store = MemoryStore(objects)
        value = self.execute(request, store); restored = Path(value["checkpoint"]["path"])
        self.assertEqual(len(list(restored.rglob("*.pth"))), 8); self.assertTrue((old / "optimizer.pt").exists())
        self.assertEqual(suite_runner.read(Path(request["output_dir"]) / "RECOVERY_PREPARED.json")["remaining_optimizer_updates"], 118)
        self.assertEqual(store.uploaded, [])

    def test_corrupt_archived_payload_never_publishes_acceptance(self):
        request, objects, _ = self.request(600, restore=True)
        uri = next(k for k in objects if k.endswith("optimizer.pt")); objects[uri] = b"x" * len(objects[uri])
        with self.assertRaisesRegex(ValueError, "full GET/SHA differs"): self.execute(request, MemoryStore(objects))
        self.assertFalse((Path(request["output_dir"]) / "HISTORY_ACCEPTANCE.json").exists())
        self.assertEqual(len(self.budget.releases), 1)

    def test_unreviewed_candidate_does_no_transfer_or_reservation(self):
        request, objects, _ = self.request(350); review = suite_runner.read(request["review"]["path"])
        review["approved"] = False; request["review"] = self.pin("review-rejected.json", review)
        with self.assertRaisesRegex(ValueError, "Independent recovery source review"): self.execute(request, MemoryStore())
        self.assertEqual(self.budget.reservations, []); self.assertFalse(Path(request["output_dir"]).exists())

    def test_accepted_final_is_full_get_verified_and_imported_without_local_hydration(self):
        request, objects, _ = self.request(600, restore=True)
        entry = suite_runner.read(request['entry']['path']); entry['mode'] = 'accepted'; entry['checkpoint_step'] = 718
        entry['checkpoint']['path'] = str(self.root / 'old-cleaned-run/final')
        native = {'trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'training_args.bin'} | {f'rng_state_{i}.pth' for i in range(8)}
        entry['checkpoint']['files'] = [row for row in entry['checkpoint']['files'] if row['path'] not in native]
        objects = {entry['checkpoint']['oss_prefix'] + row['path']: objects[entry['checkpoint']['oss_prefix'] + row['path']]
                   for row in entry['checkpoint']['files']}
        request['entry'] = self.pin('accepted-entry.json', entry); request['operation'] = 'verify_archived_accepted'
        review = suite_runner.read(request['review']['path'])
        review.update(status='SUITE_HISTORY_PREPARATION_REVIEW_ACCEPTED', operation=request['operation'],
                      entry_sha256=suite_runner.digest(entry))
        review['verified_facts'].pop('resume_metadata_verified')
        review['verified_facts'].update(formal_training_complete=True, optimizer_step=718,
            checkpoint_inventory_sha256=suite_runner.digest(entry['checkpoint']['files']))
        request['review'] = self.pin('accepted-review.json', review)
        store = MemoryStore(objects); history = self.execute(request, store)
        self.assertEqual(history['mode'], 'accepted')
        self.assertFalse(Path(entry['checkpoint']['path']).exists())
        self.assertEqual(store.uploaded, [])
        archive = suite_runner.read(history['archive_acceptance']['path'])
        self.assertTrue(archive['all_objects_full_get_sha256_verified'])
        self.assertEqual(len(archive['objects']), len(objects))
        self.assertEqual(self.budget.reservations[0][1]['oss_bytes'], 0)

    def deferred_request(self):
        request, objects, old = self.request(600, restore=True)
        request['operation'] = 'verify_archived_resume'
        review = suite_runner.read(request['review']['path'])
        review['operation'] = request['operation']
        request['review'] = self.pin('deferred-review.json', review)
        return request, objects, old

    def test_deferred_resume_verifies_every_byte_without_duplicate_home_checkpoint(self):
        request, objects, old = self.deferred_request()
        store = MemoryStore(objects); history = self.execute(request, store)
        self.assertFalse(Path(history['checkpoint']['path']).exists())
        self.assertEqual(store.uploaded, [])
        self.assertEqual(set(history['checkpoint_metadata']),
                         {'deepprobe_config.json', 'training_provenance.json', 'trainer_state.json'})
        for name, pin in history['checkpoint_metadata'].items():
            self.assertEqual(Path(pin['path']).read_bytes(), objects[history['checkpoint']['oss_prefix'] + name])
        archive = suite_runner.read(history['archive_acceptance']['path'])
        self.assertEqual(len(archive['objects']), len(objects))
        self.assertTrue(archive['all_objects_full_get_sha256_verified'])
        self.assertEqual(self.budget.reservations[0][1], {'home_bytes': 128 * 1024 * 1024, 'oss_bytes': 0})
        self.assertTrue((old / 'optimizer.pt').exists())
        self.assertEqual(self.execute(request, store, resume=True), history)

    def test_deferred_resume_still_rejects_corrupted_optimizer_archive(self):
        request, objects, _ = self.deferred_request()
        uri = next(k for k in objects if k.endswith('optimizer.pt'))
        objects[uri] = b'x' * len(objects[uri])
        with self.assertRaisesRegex(ValueError, 'full GET/SHA differs'):
            self.execute(request, MemoryStore(objects))
        self.assertFalse((Path(request['output_dir']) / 'HISTORY_ACCEPTANCE.json').exists())
        self.assertEqual(len(self.budget.releases), 1)

    def test_deferred_resume_requires_matching_operation_review_before_any_transfer(self):
        request, objects, _ = self.request(600, restore=True)
        request['operation'] = 'verify_archived_resume'
        with self.assertRaisesRegex(ValueError, 'Independent recovery source review'):
            self.execute(request, MemoryStore(objects))
        self.assertEqual(self.budget.reservations, [])


if __name__ == "__main__": unittest.main()
