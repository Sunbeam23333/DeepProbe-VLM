import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from deepprobe_vlm.evaluate import (VIDEOMME_PROTOCOL, canonical_sha256, record_metadata,
                                   score_answer, sha256_file, shard_output, shard_records,
                                   write_json_once)
from test_model_evaluate_contract import videomme_records
from unified_evaluation_fixtures import fixture

spec = importlib.util.spec_from_file_location("run_benchmark_eval", Path(__file__).resolve().parents[1] /
                                           "scripts/operations/run_benchmark_eval.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class BenchmarkRunnerTest(unittest.TestCase):
    def test_qa_binding_rejects_changed_annotation_mapping(self):
        records = videomme_records()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation, registry, cache = (root / name for name in ("annotation.parquet", "registry.json", "cache.json"))
            for path in (annotation, registry, cache):
                path.write_bytes(b"pinned-fixture")
            manifest = root / "videomme.jsonl"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in records))
            qa = [{"sample_id": row["sample_id"], "source_video_id": row["source_video_id"],
                   "annotation_sha256": sha256_file(annotation), "output_record_sha256": canonical_sha256(row)} for row in records]
            write_json_once(root / "QA_BINDINGS.json", qa)
            write_json_once(root / "EXCLUSIONS.json", ["490-1", "490-2", "490-3"])
            manifest_item = {"path": str(manifest), "sha256": sha256_file(manifest)}
            binding_path = root / "MANIFEST_BINDING.json"
            write_json_once(binding_path, {"schema_version": 1, "status": "EVALUATION_MANIFEST_BOUND",
                "protocol": VIDEOMME_PROTOCOL, "repo_id": "lmms-lab/Video-MME",
                "revision": "ead1408f75b618502df9a1d8e0950166bf0a2a0b", "raw_questions": 2700,
                "selected_questions": 2697, "selected_sources": 899,
                "excluded_question_ids": ["490-1", "490-2", "490-3"], "subtitles": False, "audio": False,
                "all_selected_qa_bound_to_original_annotation": True,
                "annotation": {"path": str(annotation), "sha256": sha256_file(annotation)},
                "source_registry": {"path": str(registry), "sha256": sha256_file(registry)},
                "manifest": manifest_item,
                "media_binding": {"path": str(cache), "sha256": sha256_file(cache), "frame_count": 7192, "media_root": str(root)},
                "qa_bindings_sha256": sha256_file(root / "QA_BINDINGS.json"),
                "exclusions_sha256": sha256_file(root / "EXCLUSIONS.json")})
            with patch.multiple(runner, VIDEOMME_ANNOTATION_SHA256=sha256_file(annotation), VIDEOMME_REGISTRY_SHA256=sha256_file(registry)):
                args = (binding_path, manifest_item, records, root, {"frame_cache_receipt_sha256": sha256_file(cache)})
                self.assertTrue(runner.verify_manifest_binding(*args)["all_output_records_bound"])
                records[0]["question"] = "silently changed"
                with self.assertRaisesRegex(ValueError, "QA binding differs"):
                    runner.verify_manifest_binding(*args)

    def test_media_gate_binds_current_frames_order_and_ready_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            media = parent / "media"
            media.mkdir()
            gate = parent / "gate"
            gate.mkdir()
            receipt = parent / "FRAME_CACHE_RECEIPT.json"
            receipt.write_text('{"complete": true}\n')
            records = videomme_records()
            mappings, entries = [], []
            for source in dict.fromkeys(row["source_video_id"] for row in records):
                paths = []
                for slot in range(8):
                    relative = source.replace(":", "-") + f"-{slot}.png"
                    path = media / relative
                    path.write_bytes(b"fixture-bytes")
                    paths.append(relative)
                    entries.append({"relative_path": relative, "source_video_id": source,
                                    "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
                mappings.append({"source_video_id": source, "image_paths": paths})
            by_source = {row["source_video_id"]: row for row in mappings}
            for row in records:
                row["image_paths"] = by_source[row["source_video_id"]]["image_paths"]
            inventory_path = gate / "benchmark_media_inventory.json"
            write_json_once(inventory_path, {"complete": True, "kind": "uniform8_lossless_frame_cache",
                "scope": "uniform8_sampled_frames", "root": str(media), "receipt_path": str(receipt),
                "receipt_sha256": sha256_file(receipt), "entries": entries,
                "original_entries": [{}] * 899, "records": mappings})
            ready_path = gate / "FORMAL_READY.json"
            write_json_once(ready_path, {"formal_evaluation_ready": True, "benchmark_media_gate": "passed",
                "content_duplicate_gate": "passed", "artifact_sha256": {inventory_path.name: sha256_file(inventory_path)}})
            self.assertTrue(runner.verify_media_gate(ready_path, records, media)["current_frame_sha256_verified"])
            records[0]["image_paths"] = list(reversed(records[0]["image_paths"]))
            with self.assertRaisesRegex(ValueError, "image order"):
                runner.verify_media_gate(ready_path, records, media)
            records[0]["image_paths"] = by_source[records[0]["source_video_id"]]["image_paths"]
            (media / entries[0]["relative_path"]).write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "differs from accepted"):
                runner.verify_media_gate(ready_path, records, media)

    def test_capacity_accounts_internal_and_external_links_and_rejects_broken(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            roots = [parent / "base", parent / "p1"]
            for root in roots:
                root.mkdir()
            (roots[0] / "weights").write_bytes(b"12345")
            (roots[1] / "hf-link").symlink_to(roots[0] / "weights")
            (roots[1] / "internal-dir").symlink_to(roots[0], target_is_directory=True)
            external = parent / "python"
            external.write_bytes(b"123")
            (roots[0] / "python-link").symlink_to(external)
            self.assertEqual(sum(runner.staging_bytes(roots).values()), 8)
            (roots[0] / "broken").symlink_to(parent / "missing")
            with self.assertRaises(FileNotFoundError):
                runner.staging_bytes(roots)
            (roots[0] / "broken").unlink()
            (roots[0] / "escaped-dir").symlink_to(parent, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "escapes"):
                runner.staging_bytes(roots)

    def test_checkpoint_full_inventory_and_seed_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            binding = fixture(root, loops=1)
            self.assertTrue(runner.verify_checkpoint(binding)["all_checkpoint_files_sha256_verified"])
            binding["seed"] = 29
            with self.assertRaisesRegex(ValueError, "checkpoint_run_not_matching_sft"):
                runner.verify_checkpoint(binding)
            binding["seed"] = 17
            (root / "final/projector.safetensors").write_bytes(b"tamper")
            with self.assertRaisesRegex(ValueError, "final_graph_bytes_or_sha_changed"):
                runner.verify_checkpoint(binding)

    def test_merge_requires_complete_identical_eight_shards(self):
        records = videomme_records()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provenance_path = root / "CHECKPOINT_PROVENANCE.json"
            provenance = {"checkpoint": "/frozen/r1/final", "config": {"num_loops": 1},
                          "manifest_sha256": "a" * 64}
            write_json_once(provenance_path, provenance)
            for rank in range(8):
                assigned = shard_records(records, rank, 8)
                output = shard_output(root / "predictions.jsonl", rank, 8)
                protocol = {"protocol": VIDEOMME_PROTOCOL, "max_new_tokens": 16, "cache": True,
                            "decoding": "greedy_argmax", "subtitles": False, "audio": False,
                            "checkpoint": provenance["checkpoint"], "checkpoint_config": provenance["config"],
                            "manifest_sha256": provenance["manifest_sha256"], "shard_index": rank, "num_shards": 8,
                            "run_provenance_sha256": sha256_file(provenance_path)}
                write_json_once(output.with_suffix(".protocol.json"), protocol)
                rows = []
                for index, record in enumerate(assigned):
                    prediction = "B" if index % 2 else "ambiguous B or A"
                    row = {key: record[key] for key in ("sample_id", "source_video_id", "split", "answer")}
                    row.update(prediction=prediction, record_sha256=canonical_sha256(record),
                               protocol_sha256=canonical_sha256(protocol), shard_index=rank, num_shards=8)
                    row.update(record_metadata(record))
                    row.update(score_answer(prediction, "B", record["choices"]))
                    rows.append(row)
                output.write_text("".join(json.dumps(row) + "\n" for row in rows))
                write_json_once(output.with_suffix(".summary.json"),
                                {"num_examples": len(rows), "predictions_sha256": sha256_file(output)})
            summary = runner.merge_shards(root, records, provenance_path)
            self.assertEqual(summary["num_examples"], 2697)
            self.assertEqual(summary["num_sources"], 899)
            self.assertGreater(summary["by_metric"]["strict_mcq_accuracy"]["num_invalid"], 0)
            self.assertEqual(runner.merge_shards(root, records, provenance_path), summary)
            shard = shard_output(root / "predictions.jsonl", 1, 8)
            shard.write_text("\n".join(shard.read_text().splitlines()[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "Incomplete shard"):
                runner.merge_shards(root, records, provenance_path)


if __name__ == "__main__":
    unittest.main()
