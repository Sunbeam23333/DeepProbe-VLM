"""Formal source gating must be fixed, auditable, and fail closed."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "prepare_formal_source_gate", ROOT / "scripts/operations/prepare_formal_source_gate.py"
)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def videomme_rows():
    rows = []
    for index in range(1, 901):
        group = f"{index:03d}"
        raw_source = gate.EXCLUDED_RAW_VIDEO_ID if index == 490 else f"{index:011d}"
        for suffix in range(1, 4):
            rows.append({"video_id": group, "videoID": raw_source,
                         "question_id": f"{group}-{suffix}", "question": "secret",
                         "answer": "A"})
    return rows


def manifest_row(sample, source, split, video):
    return {"sample_id": sample, "source_video_id": source, "video_path": video,
            "question": "What happens?", "answer": "An event.", "split": split}


class FormalSourceGateTests(unittest.TestCase):
    def test_fixed_subset_excludes_group_490_without_fabricating_identity(self):
        registry, exclusions, coverage = gate.build_videomme_subset(
            videomme_rows(), "a" * 40, "b" * 64
        )
        self.assertEqual(len(registry), 2697)
        self.assertEqual(len({row["source_video_id"] for row in registry}), 899)
        self.assertEqual({row["question_id"] for row in exclusions},
                         {"490-1", "490-2", "490-3"})
        self.assertEqual({row["source_video_id"] for row in exclusions}, {None})
        self.assertNotIn("youtube:" + gate.EXCLUDED_RAW_VIDEO_ID,
                         {row["source_video_id"] for row in registry})
        self.assertEqual(len(coverage), 2700)

    def test_any_other_unresolved_source_or_changed_exclusion_fails_closed(self):
        rows = videomme_rows()
        for row in rows[:3]:
            row["videoID"] = "invalid-invalid"
        with self.assertRaisesRegex(ValueError, "unresolved original source"):
            gate.build_videomme_subset(rows, "a" * 40, "b" * 64)
        rows = videomme_rows()
        for row in rows:
            if row["question_id"].startswith("490-"):
                row["videoID"] = "abcdefghijk"
        with self.assertRaisesRegex(ValueError, "raw videoID changed"):
            gate.build_videomme_subset(rows, "a" * 40, "b" * 64)

    def test_training_coverage_binds_manifests_and_omits_unknown_and_benchmark_sources(self):
        training = {
            "train": [manifest_row("train-1", "youtube:train000001", "train", "train.mp4")],
            "validation": [manifest_row("val-1", "youtube:valid000001", "validation", "val.mp4")],
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "coverage.jsonl"
            rows = [
                {"sample_id": "train-1", "source_video_id": "youtube:train000001",
                 "selected_split": "train", "decision": "selected", "reasons": []},
                {"sample_id": "val-1", "source_video_id": "youtube:valid000001",
                 "selected_split": "validation", "decision": "selected", "reasons": []},
                {"sample_id": "unknown", "source_video_id": None, "selected_split": None,
                 "decision": "omitted", "reasons": ["unresolved_original_training_source"]},
                {"sample_id": "test-overlap", "source_video_id": "youtube:00000000001",
                 "selected_split": None, "decision": "omitted", "reasons": ["benchmark_source_excluded"]},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            report = gate.load_training_coverage(path, training, {"youtube:00000000001"})
            self.assertTrue(report["valid"])
            self.assertEqual(report["counts"]["unresolved_source_qa_omitted"], 1)
            self.assertEqual(report["counts"]["benchmark_source_qa_omitted"], 1)
            rows[2].update(decision="selected", selected_split="train")
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            report = gate.load_training_coverage(path, training, {"youtube:00000000001"})
            self.assertFalse(report["valid"])

    def test_training_requires_attested_media_while_final_evaluation_can_remain_blocked(self):
        training_report = {"provided": True, "valid": True, "errors": []}
        coverage_report = {"provided": True, "valid": True, "errors": []}
        attestation = {"provided": True, "valid": True, "errors": []}
        media_attestation = {"checked": True, "passed": True, "errors": []}
        leakage = {"checked": True, "passed": True, "benchmark_overlaps": [],
                   "cross_train_validation_sources": []}
        unavailable = {"checked": False, "complete": False, "present_files": 0,
                       "expected_files": 899, "entries": []}
        exact = {"checked": False, "passed": False, "duplicate_pairs": []}
        audit = {"provided": False, "passed": False, "errors": ["required"]}
        training_gates, evaluation_gates = gate.readiness_gates(
            training_report, coverage_report, attestation, media_attestation, leakage,
            unavailable, unavailable, exact, audit
        )
        self.assertTrue(all(value["status"] == "passed" for value in training_gates.values()))
        self.assertEqual(evaluation_gates["benchmark_media"]["status"], "blocked")
        self.assertEqual(evaluation_gates["perceptual_content_audit"]["status"], "blocked")
        media_attestation = {"checked": False, "passed": False,
                             "errors": ["--training-media-root is required"]}
        training_gates, _ = gate.readiness_gates(
            training_report, coverage_report, attestation, media_attestation, leakage,
            unavailable, unavailable, exact, audit
        )
        self.assertEqual(training_gates["training_media_integrity"]["status"], "blocked")

    def _write_receipt(self, root, dataset, files):
        spec = gate._dataset_spec(dataset)
        entries = []
        for relative, content in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            entries.append({"path": relative, "size_bytes": len(content),
                            "local_sha256": gate.sha256_file(path)})
        receipt = root / "ASSET_COMMITTED.json"
        receipt.write_text(json.dumps({"repo_id": spec["repo_id"], "revision": spec["revision"],
                                       "downloaded": True, "files": entries}))
        return receipt

    def test_generator_blocks_unattested_training_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vm_root, mv_root = root / "videomme", root / "mvbench"
            vm_receipt = self._write_receipt(vm_root, "videomme", {
                gate.VIDEOMME_ANNOTATION: b"fixture parquet bytes"
            })
            mv_files = {}
            for task in gate.convert.MV_TASKS:
                mv_files[f"json/{task}.json"] = json.dumps([{"video": f"clip_{task}.mp4"}]).encode()
            mv_receipt = self._write_receipt(mv_root, "mvbench", mv_files)
            train_path, val_path = root / "train.jsonl", root / "validation.jsonl"
            train = manifest_row("train-1", "youtube:train000001", "train", "train.mp4")
            val = manifest_row("val-1", "youtube:valid000001", "validation", "val.mp4")
            train_path.write_text(json.dumps(train) + "\n")
            val_path.write_text(json.dumps(val) + "\n")
            coverage_path = root / "training-coverage.jsonl"
            coverage_path.write_text("".join(json.dumps(row) + "\n" for row in [
                {"sample_id": "train-1", "source_video_id": train["source_video_id"],
                 "selected_split": "train", "decision": "selected", "reasons": []},
                {"sample_id": "val-1", "source_video_id": val["source_video_id"],
                 "selected_split": "validation", "decision": "selected", "reasons": []},
                {"sample_id": "unknown", "source_video_id": None, "selected_split": None,
                 "decision": "omitted", "reasons": ["unresolved_original_training_source"]},
                {"sample_id": "benchmark", "source_video_id": "youtube:00000000001",
                 "selected_split": None, "decision": "omitted", "reasons": ["benchmark_source_excluded"]},
            ]))
            output = root / "formal"
            real_rows_from = gate.convert.rows_from

            def rows_from(path):
                return iter(videomme_rows()) if path.suffix == ".parquet" else real_rows_from(path)

            arguments = ["--videomme-root", str(vm_root), "--videomme-plan", str(vm_receipt),
                         "--mvbench-root", str(mv_root), "--mvbench-plan", str(mv_receipt),
                         "--train-manifest", str(train_path), "--validation-manifest", str(val_path),
                         "--training-coverage", str(coverage_path), "--output-dir", str(output)]
            with mock.patch.object(gate.convert, "rows_from", side_effect=rows_from):
                self.assertEqual(gate.main(arguments), 2)
            ready = json.loads((output / "FORMAL_READY.json").read_text())
            self.assertFalse(ready["formal_training_ready"])
            self.assertFalse(ready["formal_evaluation_ready"])
            self.assertEqual(ready["benchmark_source_identity_gate"], "passed_on_declared_subset")
            self.assertEqual(ready["benchmark_media_gate"], "blocked")
            self.assertEqual(ready["content_duplicate_gate"], "blocked")
            self.assertEqual(ready["training_readiness_gates"]["training_data_attestation"]["status"], "blocked")
            for name, digest in ready["artifact_sha256"].items():
                self.assertEqual(digest, gate.sha256_file(output / name))
            registry = json.loads((output / "benchmark_source_registry.json").read_text())
            self.assertEqual(registry["datasets"]["videomme"]["questions"], 2697)
            self.assertEqual(registry["datasets"]["videomme"]["source_videos"], 899)
            self.assertEqual(registry["datasets"]["mvbench"]["tasks"], 20)
            self.assertNotIn("youtube:" + gate.EXCLUDED_RAW_VIDEO_ID, registry["sources"])

            data_dir = root / "attested-training"
            data_dir.mkdir()
            (data_dir / "train.jsonl").write_bytes(train_path.read_bytes())
            (data_dir / "validation.jsonl").write_bytes(val_path.read_bytes())
            (data_dir / "coverage.jsonl").write_bytes(coverage_path.read_bytes())
            (data_dir / "source_groups.jsonl").write_text(json.dumps({"fixture": True}) + "\n")
            media_root = root / "training-media"
            media_root.mkdir()
            (media_root / "train.mp4").write_bytes(b"train-media")
            (media_root / "val.mp4").write_bytes(b"validation-media")
            media_entries = [
                {"relative_path": "train.mp4", "source_video_id": train["source_video_id"],
                 "size_bytes": (media_root / "train.mp4").stat().st_size,
                 "sha256": gate.sha256_file(media_root / "train.mp4")},
                {"relative_path": "val.mp4", "source_video_id": val["source_video_id"],
                 "size_bytes": (media_root / "val.mp4").stat().st_size,
                 "sha256": gate.sha256_file(media_root / "val.mp4")},
            ]
            media_inventory = {
                "schema_version": 1, "kind": "formal_training_media_sha256_inventory",
                "media_root": str(media_root.resolve()), "file_count": len(media_entries),
                "total_bytes": sum(row["size_bytes"] for row in media_entries),
                "entries_sha256": gate.canonical_sha(media_entries), "entries": media_entries,
            }
            gate.write_json(data_dir / "training_media_inventory.json", media_inventory)
            registry_sha = gate.sha256_file(output / "benchmark_source_registry.json")
            output_hashes = {path.name: gate.sha256_file(path) for path in data_dir.iterdir()}
            annotation_sha = "c" * 64
            data_report = {
                "status": "formal_training_data_ready", "formal_data_ready": True,
                "dataset_revision": gate._dataset_spec("llava_video_178k")["revision"],
                "annotation": {"dataset": "llava_video_178k",
                               "relative_path": "0_30_s_youtube_v0_1/0_30_s_youtube_oe_v0_1_qa_processed.json",
                               "sha256": annotation_sha},
                "benchmark_registry": {"sha256": registry_sha},
                "expected_pinned_counts": {"annotation_rows": 2, "candidate_qa": 4,
                                           "unresolved_source_qa": 1},
                "scan": {"annotation_rows": 2, "candidate_qa": 4, "unresolved_source_qa": 1},
                "counts": {"candidate_qa": 4, "selected_qa": 2},
                "media_root": str(media_root.resolve()),
                "training_media_inventory": {
                    "relative_path": "training_media_inventory.json",
                    "sha256": output_hashes["training_media_inventory.json"],
                    "file_count": media_inventory["file_count"],
                    "total_bytes": media_inventory["total_bytes"],
                    "entries_sha256": media_inventory["entries_sha256"],
                },
                "output_sha256": output_hashes,
            }
            gate.write_json(data_dir / "preparation_report.json", data_report)
            data_ready = {
                "purpose": "FORMAL_TRAINING_DATA", "formal_data_ready": True,
                "report_sha256": gate.sha256_file(data_dir / "preparation_report.json"),
                "benchmark_registry_sha256": registry_sha,
                "manifest_sha256": {"train.jsonl": output_hashes["train.jsonl"],
                                    "validation.jsonl": output_hashes["validation.jsonl"]},
                "coverage_sha256": output_hashes["coverage.jsonl"],
                "source_groups_sha256": output_hashes["source_groups.jsonl"],
                "training_media_inventory_sha256": output_hashes["training_media_inventory.json"],
                "training_media_entries_sha256": media_inventory["entries_sha256"],
                "training_media_file_count": media_inventory["file_count"],
                "training_media_total_bytes": media_inventory["total_bytes"],
                "annotation_sha256": annotation_sha,
            }
            gate.write_json(data_dir / "READY.json", data_ready)
            verified_output = root / "formal-verified"
            verified_arguments = ["--videomme-root", str(vm_root), "--videomme-plan", str(vm_receipt),
                                  "--mvbench-root", str(mv_root), "--mvbench-plan", str(mv_receipt),
                                  "--training-data-dir", str(data_dir),
                                  "--training-media-root", str(media_root),
                                  "--output-dir", str(verified_output)]
            with mock.patch.object(gate.convert, "rows_from", side_effect=rows_from), \
                    mock.patch.multiple(gate, LLAVA_EXPECTED_ANNOTATION_ROWS=2,
                                        LLAVA_EXPECTED_CANDIDATE_QA=4,
                                        LLAVA_EXPECTED_UNRESOLVED_QA=1):
                self.assertEqual(gate.main(verified_arguments), 0)
            verified_ready = json.loads((verified_output / "FORMAL_READY.json").read_text())
            self.assertTrue(verified_ready["formal_training_ready"])
            self.assertFalse(verified_ready["formal_evaluation_ready"])
            self.assertEqual(verified_ready["training_readiness_gates"]["training_data_attestation"]["status"],
                             "passed")
            self.assertEqual(verified_ready["training_readiness_gates"]["training_media_integrity"]["status"],
                             "passed")

            (media_root / "train.mp4").write_bytes(b"changed-media")
            rejected_output = root / "formal-rejected-media"
            rejected_arguments = verified_arguments[:-2] + ["--output-dir", str(rejected_output)]
            with mock.patch.object(gate.convert, "rows_from", side_effect=rows_from), \
                    mock.patch.multiple(gate, LLAVA_EXPECTED_ANNOTATION_ROWS=2,
                                        LLAVA_EXPECTED_CANDIDATE_QA=4,
                                        LLAVA_EXPECTED_UNRESOLVED_QA=1):
                self.assertEqual(gate.main(rejected_arguments), 2)
            rejected_ready = json.loads((rejected_output / "FORMAL_READY.json").read_text())
            self.assertFalse(rejected_ready["formal_training_ready"])
            self.assertEqual(
                rejected_ready["training_readiness_gates"]["training_media_integrity"]["status"],
                "failed")


if __name__ == "__main__":
    unittest.main()
