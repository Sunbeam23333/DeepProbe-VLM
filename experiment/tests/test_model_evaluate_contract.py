import json
from pathlib import Path
import tempfile
import unittest

from deepprobe_vlm.evaluate import (check_provenance, record_metadata,
                                    summarise_metrics, validate_output_paths)


class EvaluationContractTest(unittest.TestCase):
    def test_missing_provenance_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [{"source_video_id": "a", "split": "test"}]
            with self.assertRaisesRegex(ValueError, "lacks training_provenance"):
                check_provenance(root, records)
            with self.assertWarnsRegex(RuntimeWarning, "UNVERIFIED CHECKPOINT"):
                status = check_provenance(root, records, allow_unverified=True)
            self.assertEqual(status, "UNVERIFIED_MISSING_TRAINING_PROVENANCE")

    def test_test_cannot_reuse_training_or_tuning_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"training_provenance.json").write_text(json.dumps({"training_sources": ["train"], "tuning_sources": ["val"]}))
            for source in ("train", "val"):
                with self.assertRaises(ValueError):
                    check_provenance(root, [{"source_video_id": source, "split": "test"}])
            self.assertEqual(check_provenance(root, [{"source_video_id": "val", "split": "validation"}]),
                             "passed_recorded_training_source_check")
            (root/"training_provenance.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Invalid checkpoint provenance"):
                check_provenance(root, [], allow_unverified=True)

    def test_all_related_output_paths_are_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"predictions.jsonl"
            for path in (output, output.with_suffix(".summary.json"), output.with_suffix(".trace.json")):
                path.write_text("existing")
                with self.assertRaises(FileExistsError):
                    validate_output_paths(output)
                path.unlink()

    def test_converter_metadata_aliases_are_preserved(self):
        metadata = record_metadata({"duration_category": "long", "task_type": "action_sequence"})
        self.assertEqual(metadata["duration"], "long")
        self.assertEqual(metadata["task"], "action_sequence")
        self.assertEqual(metadata["duration_category"], "long")
        self.assertEqual(metadata["task_type"], "action_sequence")

    def test_metrics_are_not_mixed_into_a_single_score(self):
        result = summarise_metrics([{"metric": "strict_mcq_accuracy", "correct": 1},
                                    {"metric": "normalised_exact_match", "correct": 0}])
        self.assertEqual(result["strict_mcq_accuracy"]["score"], 1)
        self.assertEqual(result["normalised_exact_match"]["score"], 0)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
