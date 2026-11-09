import json
from pathlib import Path
import tempfile
import unittest

from deepprobe_vlm.evaluate import (check_provenance, record_metadata,
                                    summarise_metrics, validate_output_paths,
                                    canonical_sha256, read_prediction_rows, score_answer,
                                    shard_records, summarise_groups, validate_videomme_subset,
                                    write_json_once)


def videomme_records():
    return [{"sample_id": f"videomme:{group:03d}-{question}", "source_video_id": f"youtube:id{group}",
             "split": "test", "dataset": "videomme", "question": "What happens?",
             "choices": ["one", "two", "three", "four"], "answer": "B",
             "duration_category": "short" if group <= 300 else "medium" if group <= 600 else "long"}
            for group in range(1, 901) if group != 490 for question in range(1, 4)]


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

    def test_strict_mc_parser_rejects_ambiguous_out_of_range_and_unbalanced(self):
        for value in ("E", "A or B", "The answer might be B", "(B", "B)", ""):
            result = score_answer(value, "B", ["one", "two", "three", "four"])
            self.assertEqual(result["correct"], 0)
            self.assertIsNone(result["parsed_prediction"])
            self.assertTrue(result["invalid_prediction"])
        for value in ("B", "Answer: (B).", "two"):
            self.assertEqual(score_answer(value, "B", ["one", "two", "three", "four"])["correct"], 1)

    def test_invalid_answers_remain_in_overall_and_duration_denominators(self):
        rows = [{**score_answer(text, "B", ["one", "two"]), "duration": duration}
                for text, duration in (("B", "short"), ("E", "short"), ("A", "long"))]
        overall = summarise_metrics(rows)["strict_mcq_accuracy"]
        self.assertEqual(overall["num_examples"], 3)
        self.assertEqual(overall["num_invalid"], 1)
        self.assertEqual(overall["score"], 1 / 3)
        self.assertEqual(summarise_groups(rows, "duration")["short"]["strict_mcq_accuracy"]["score"], 0.5)

    def test_fixed_videomme_subset_does_not_silently_shrink(self):
        records = videomme_records()
        self.assertEqual(validate_videomme_subset(records)["num_questions"], 2697)
        for invalid in (records[:-1], records + [records[0]]):
            with self.assertRaises(ValueError):
                validate_videomme_subset(invalid)
        records[0]["subtitle_path"] = "subtitle.srt"
        with self.assertRaisesRegex(ValueError, "Subtitle/audio"):
            validate_videomme_subset(records)

    def test_whole_source_shards_cover_every_question_once(self):
        records = videomme_records()
        shards = [shard_records(records, rank, 8) for rank in range(8)]
        self.assertEqual(sum(len(shard) for shard in shards), 2697)
        owners = {}
        for rank, shard in enumerate(shards):
            for row in shard:
                self.assertEqual(owners.setdefault(row["source_video_id"], rank), rank)
        self.assertEqual(len(owners), 899)
        with self.assertRaises(ValueError):
            shard_records(records, 8, 8)

    def test_resume_checks_prefix_identity_and_preserves_torn_tail(self):
        records = videomme_records()[:3]
        protocol = "a" * 64
        record = records[0]
        row = {key: record[key] for key in ("sample_id", "source_video_id", "answer", "split")}
        row.update(prediction="B", record_sha256=canonical_sha256(record), protocol_sha256=protocol)
        row.update(record_metadata(record))
        row.update(score_answer("B", record["answer"], record["choices"]))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.jsonl"
            complete_line = (json.dumps(row) + "\n").encode()
            output.write_bytes(complete_line + b'{"torn":')
            with self.assertRaisesRegex(ValueError, "incomplete"):
                read_prediction_rows(output, records, protocol)
            self.assertEqual(read_prediction_rows(output, records, protocol, recover_tail=True), [row])
            self.assertEqual(output.read_bytes(), complete_line)
            self.assertEqual(next(Path(directory).glob("*.bin")).read_bytes(), b'{"torn":')
            with self.assertRaisesRegex(ValueError, "mismatch"):
                read_prediction_rows(output, records, "b" * 64)
            output.write_bytes(complete_line + complete_line)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                read_prediction_rows(output, records, protocol)

    def test_receipts_are_idempotent_and_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            write_json_once(path, {"id": "one"})
            write_json_once(path, {"id": "one"})
            with self.assertRaisesRegex(ValueError, "differs"):
                write_json_once(path, {"id": "two"})


if __name__ == "__main__":
    unittest.main()
