"""Formal training data preparation must cover every pinned QA exactly once."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "prepare_formal_training_data", ROOT / "scripts/operations/prepare_formal_training_data.py"
)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def annotation_row(video, pairs=1, row_id=None):
    conversations = []
    for index in range(pairs):
        conversations.extend([{"from": "human", "value": f"<video> Question {index}?"},
                              {"from": "gpt", "value": f"Answer {index}."}])
    return {"id": row_id or video, "video": video, "conversations": conversations}


class FormalTrainingDataTests(unittest.TestCase):
    def test_streaming_json_array_handles_chunk_boundaries_and_rejects_trailing_data(self):
        rows = [annotation_row("youtube/aaaaaaaaaaa.mp4"),
                annotation_row("youtube/bbbbbbbbbbb.mp4", pairs=2)]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.json"
            path.write_text(json.dumps(rows, ensure_ascii=False))
            self.assertEqual(list(prep.iter_json_array(path, chunk_chars=7)), rows)
            path.write_text(json.dumps(rows) + " x")
            with self.assertRaisesRegex(ValueError, "unexpected data"):
                list(prep.iter_json_array(path, chunk_chars=11))

    def test_split_is_stable_and_source_level(self):
        source = "youtube:abcdefghijk"
        self.assertEqual(prep.split_for_source(source), prep.split_for_source(source))
        self.assertIn(prep.split_for_source(source), {"train", "validation"})

    def _write_receipt(self, root, files):
        spec = prep.source_gate._dataset_spec("llava_video_178k")
        entries = []
        for relative, content in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            entries.append({"path": relative, "size_bytes": len(content),
                            "local_sha256": prep.source_gate.sha256_file(path)})
        receipt = root / "ASSET_COMMITTED.json"
        receipt.write_text(json.dumps({"repo_id": spec["repo_id"], "revision": spec["revision"],
                                       "downloaded": True, "files": entries}))
        return receipt

    def test_full_generator_and_gate_attestation_on_bounded_fixture(self):
        train_source = val_source = None
        for index in range(1, 10000):
            source = f"youtube:{index:011d}"
            split = prep.split_for_source(source)
            if split == "train" and train_source is None:
                train_source = source
            if split == "validation" and val_source is None:
                val_source = source
            if train_source and val_source:
                break
        self.assertIsNotNone(train_source)
        self.assertIsNotNone(val_source)
        benchmark_source = "youtube:99999999999"
        missing_source = "88888888888"
        unresolved_video = "youtube/25a880129e57bb2e1d433b940226bc8e"
        rows = [
            annotation_row(f"youtube/{train_source.removeprefix('youtube:')}.mp4", row_id="train"),
            annotation_row(f"youtube/{val_source.removeprefix('youtube:')}.mp4", row_id="validation"),
            annotation_row("youtube/99999999999.mp4", row_id="benchmark"),
            annotation_row(f"youtube/{missing_source}_0_1.mp4", row_id="missing-present"),
            annotation_row(f"youtube/{missing_source}_1_2.mp4", row_id="missing-absent"),
            annotation_row(unresolved_video, row_id="unresolved"),
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, media = root / "raw", root / "media"
            annotation_bytes = json.dumps(rows, ensure_ascii=False).encode()
            receipt = self._write_receipt(raw, {
                prep.LLAVA_ANNOTATION: annotation_bytes,
                prep.LLAVA_ARCHIVE: b"fixture archive",
            })
            present = [rows[0]["video"], rows[1]["video"], rows[2]["video"],
                       rows[3]["video"], unresolved_video]
            for relative in present:
                path = media / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(("media:" + relative).encode())
            registry = {
                "schema_version": 1,
                "kind": "annotation_only_benchmark_source_registry",
                "all_source_identities_resolved": True,
                "datasets": {"videomme": {"questions": 2697, "source_videos": 899},
                             "mvbench": {"tasks": 20}},
                "sources": [benchmark_source],
                "entries": [{"source_video_id": benchmark_source}],
            }
            registry_path = root / "benchmark_source_registry.json"
            prep.source_gate.write_json(registry_path, registry)
            output = root / "formal-training"
            arguments = ["--llava-root", str(raw), "--llava-plan", str(receipt),
                         "--media-root", str(media), "--benchmark-source-registry", str(registry_path),
                         "--output-dir", str(output)]
            patched = dict(EXPECTED_ANNOTATION_ROWS=len(rows), EXPECTED_CANDIDATE_QA=len(rows),
                           EXPECTED_UNRESOLVED_QA=1)
            with mock.patch.multiple(prep, **patched), \
                    mock.patch.multiple(prep.source_gate,
                                        LLAVA_EXPECTED_ANNOTATION_ROWS=len(rows),
                                        LLAVA_EXPECTED_CANDIDATE_QA=len(rows),
                                        LLAVA_EXPECTED_UNRESOLVED_QA=1):
                self.assertEqual(prep.main(arguments), 0)
                train, training_report = prep.source_gate.load_training_manifests(
                    output / "train.jsonl", output / "validation.jsonl"
                )
                coverage_report = prep.source_gate.load_training_coverage(
                    output / "coverage.jsonl", train, {benchmark_source}
                )
                attestation = prep.source_gate.validate_training_data_attestation(
                    output, prep.source_gate.sha256_file(registry_path), training_report, coverage_report
                )
                original_coverage = (output / "coverage.jsonl").read_text()
                coverage_lines = original_coverage.splitlines(keepends=True)
                (output / "coverage.jsonl").write_text("".join(coverage_lines[:-1]))
                shortened = prep.source_gate.load_training_coverage(
                    output / "coverage.jsonl", train, {benchmark_source}
                )
                self.assertTrue(shortened["valid"], "omitted-row deletion is not detectable from manifests alone")
                tampered = prep.source_gate.validate_training_data_attestation(
                    output, prep.source_gate.sha256_file(registry_path), training_report, shortened
                )
                self.assertFalse(tampered["valid"])
                (output / "coverage.jsonl").write_text(original_coverage)
            self.assertTrue(training_report["valid"])
            self.assertTrue(coverage_report["valid"])
            self.assertTrue(attestation["valid"], attestation["errors"])
            report = json.loads((output / "preparation_report.json").read_text())
            self.assertEqual(report["counts"]["candidate_qa"], 6)
            self.assertEqual(report["counts"]["selected_qa"], 2)
            self.assertEqual(report["counts"]["train"], {"qa": 1, "source_videos": 1})
            self.assertEqual(report["counts"]["validation"], {"qa": 1, "source_videos": 1})
            coverage = [json.loads(line) for line in (output / "coverage.jsonl").read_text().splitlines()]
            self.assertEqual(len(coverage), 6)
            by_id = {row["sample_id"]: row for row in coverage}
            missing_rows = [row for row in coverage if row["source_video_id"] == "youtube:" + missing_source]
            self.assertEqual(len(missing_rows), 2)
            self.assertEqual({tuple(row["reasons"]) for row in missing_rows},
                             {("source_group_has_missing_media",)})
            self.assertEqual({row["media_present"] for row in missing_rows}, {True, False})
            unresolved = next(row for row in coverage if row["source_video_id"] is None)
            self.assertEqual(unresolved["reasons"], ["unresolved_original_training_source"])
            benchmark = next(row for row in coverage if row["source_video_id"] == benchmark_source)
            self.assertEqual(benchmark["reasons"], ["benchmark_source_excluded"])
            self.assertEqual(len(by_id), 6)
            inventory = json.loads((output / "training_media_inventory.json").read_text())
            self.assertEqual(inventory["file_count"], 2)
            self.assertEqual(inventory["entries_sha256"],
                             prep.source_gate.canonical_sha(inventory["entries"]))
            current = prep.source_gate.media_inventory(
                train["train"] + train["validation"], media
            )
            verified_media = prep.source_gate.validate_attested_training_media(attestation, current)
            self.assertTrue(verified_media["passed"], verified_media["errors"])

            selected_path = media / inventory["entries"][0]["relative_path"]
            original_media = selected_path.read_bytes()
            selected_path.write_bytes(original_media + b"tampered")
            changed = prep.source_gate.media_inventory(
                train["train"] + train["validation"], media
            )
            rejected = prep.source_gate.validate_attested_training_media(attestation, changed)
            self.assertFalse(rejected["passed"])
            self.assertIn("differs from attested bytes", rejected["errors"][0])


if __name__ == "__main__":
    unittest.main()
