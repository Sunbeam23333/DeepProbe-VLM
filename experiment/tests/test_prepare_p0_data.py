"""P0 preparation must preserve source boundaries and fail closed on bad assets."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_p0_data", ROOT / "scripts/operations/prepare_p0_data.py")
p0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p0)


def candidate(sample, source, video):
    return {"sample_id": sample, "source_video_id": source, "video_path": video,
            "question": "What happens?", "answer": "A person walks.", "split": "train",
            "annotation_row_1based": 1, "qa_index_0based": 0}


class PrepareP0Tests(unittest.TestCase):
    def test_official_converter_disambiguates_questions_with_reused_video_id(self):
        from types import SimpleNamespace
        row = {"id": "abcdefghijk", "video": "youtube/ytb_abcdefghijk.mp4",
               "conversations": [{"from": "human", "value": "First question"}, {"from": "gpt", "value": "Answer"}]}
        args = SimpleNamespace(input=Path("annotations.json"), split="train")
        first = list(p0.convert.convert_llava(row, args, {}))[0]
        row["conversations"][0]["value"] = "Second question"
        second = list(p0.convert.convert_llava(row, args, {}))[0]
        self.assertNotEqual(first["sample_id"], second["sample_id"])
        self.assertEqual(first["source_video_id"], second["source_video_id"])

    def test_reused_upstream_ids_get_distinct_traceable_row_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "annotations.json"
            row = {"id": "reused", "video": "youtube/ytb_abcdefghijk.mp4",
                   "conversations": [{"from": "human", "value": "Question"}, {"from": "gpt", "value": "Answer"}]}
            source.write_text(json.dumps([row, row]))
            records = p0.llava_candidates(source, source.name, "a" * 40, {})
            self.assertNotEqual(records[0]["sample_id"], records[1]["sample_id"])
            self.assertEqual([r["annotation_row_1based"] for r in records], [1, 2])
            self.assertEqual({r["upstream_annotation_id"] for r in records}, {"reused"})

    def test_unidentified_training_sources_are_excluded_with_explicit_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "annotations.json"
            video = "youtube/25a880129e57bb2e1d433b940226bc8e"
            source.write_text(json.dumps([{"id": "opaque", "video": video,
                "conversations": [{"from": "human", "value": "Question"}, {"from": "gpt", "value": "Answer"}]}]))
            with self.assertRaises(ValueError):
                p0.llava_candidates(source, source.name, "a" * 40, {})
            omitted = []
            with mock.patch.object(p0, "validate_records"):
                rows = p0.llava_candidates(source, source.name, "a" * 40, {},
                                           unresolved_omissions=omitted, media_root=root)
            self.assertEqual(rows, [])
            self.assertEqual(len(omitted), 1)
            self.assertIsNone(omitted[0]["source_video_id"])
            self.assertEqual(omitted[0]["reasons"], ["unresolved_original_training_source"])

    def test_explicit_engineering_exception_preserves_malformed_upstream_id(self):
        rows = [{"videoID": "azZZZbSwLQght", "question_id": "490-1"}]
        with self.assertRaises(ValueError):
            p0.benchmark_entries("videomme", "test.parquet", rows, "a" * 40)
        entry = p0.benchmark_entries("videomme", "test.parquet", rows, "a" * 40,
                                     allow_unresolved=True)[0]
        self.assertEqual(entry["source_video_id"], "youtube:azZZZbSwLQght")
        self.assertTrue(entry["source_identity_unresolved"])

    def test_media_existence_cache_still_rejects_escaping_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "media"
            root.mkdir()
            outside = Path(temp) / "outside.mp4"
            outside.write_bytes(b"fixture")
            (root / "escape.mp4").symlink_to(outside)
            with self.assertRaises(ValueError):
                p0.select_source_groups([candidate("x", "youtube:a", "escape.mp4")], root, set(),
                                        seed=17, train_max_qa=2, validation_max_qa=1)

    def test_group_caps_missing_media_exclusion_and_order_independence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("a.mp4", "b.mp4", "c.mp4", "d.mp4", "e.mp4"):
                (root / name).write_bytes(b"media fixture")
            rows = [candidate(f"a{i}", "youtube:a", "a.mp4") for i in range(3)]
            rows += [candidate(f"b{i}", "youtube:b", "b.mp4") for i in range(2)]
            rows += [candidate("c0", "youtube:c", "c.mp4"), candidate("c1", "youtube:c", "missing.mp4")]
            rows += [candidate("d0", "youtube:d", "d.mp4")]
            rows += [candidate(f"e{i}", "youtube:e", "e.mp4") for i in range(6)]
            options = dict(seed=17, train_max_qa=3, validation_max_qa=2)
            first = p0.select_source_groups(rows, root, {"youtube:d"}, **options)
            second = p0.select_source_groups(list(reversed(rows)), root, {"youtube:d"}, **options)
            self.assertEqual(first, second)
            selected, coverage = first
            self.assertEqual(len(selected["train"]), 3)
            self.assertEqual(len(selected["validation"]), 2)
            self.assertEqual({r["source_video_id"] for r in selected["train"]}, {"youtube:a"})
            self.assertEqual({r["source_video_id"] for r in selected["validation"]}, {"youtube:b"})
            self.assertEqual(len(coverage), len(rows))
            omitted = {r["sample_id"]: r for r in coverage if r["decision"] == "omitted"}
            self.assertIn("source_group_has_missing_media", omitted["c0"]["reasons"])
            self.assertTrue(omitted["c0"]["media_present"])
            self.assertFalse(omitted["c1"]["media_present"])
            self.assertIn("benchmark_source_excluded", omitted["d0"]["reasons"])
            self.assertIn("source_group_exceeds_both_qa_caps", omitted["e0"]["reasons"])

    def test_registry_has_original_source_without_gold_or_media_manifest_fields(self):
        vm = p0.benchmark_entries("videomme", "test.parquet", [{"video_id": "001", "videoID": "abcdefghijk",
                                  "question_id": "001-1", "question": "Q", "answer": "secret gold"}], "a" * 40)
        self.assertEqual(vm[0]["source_video_id"], "youtube:abcdefghijk")
        for key in ("question", "answer", "split", "video_path", "image_paths"):
            self.assertNotIn(key, vm[0])
        mv = p0.benchmark_entries("mvbench", "action_sequence.json", [{"video": "ABC12_1_2.mp4"},
                                  {"video": "ABC12_7_9.mp4"}], "b" * 40, "action_sequence")
        self.assertEqual({row["source_video_id"] for row in mv}, {"charades:ABC12"})
        with self.assertRaisesRegex(ValueError, "videoID"):
            p0.benchmark_entries("videomme", "test.parquet", [{"videoID": "001"}], "a" * 40)

    def test_pinned_receipt_rejects_dry_run_changed_revision_and_tampered_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "annotation.json"
            source.write_text("[]")
            configs = json.loads((ROOT / "configs/datasets.json").read_text())["datasets"]
            receipt = {"repo_id": configs["videomme"]["repo_id"], "revision": configs["videomme"]["revision"],
                       "downloaded": True, "files": [{"path": source.name, "size_bytes": 2,
                                                       "local_sha256": p0.sha256_file(source)}]}
            plan = root / "plan.json"
            plan.write_text(json.dumps(receipt))
            verified = p0.read_receipt(plan, "videomme")
            p0.verify_local_file(root, source.name, verified)
            source.write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                p0.verify_local_file(root, source.name, verified)
            for altered, message in (({**receipt, "revision": "0" * 40}, "immutable"),
                                     ({**receipt, "downloaded": False}, "downloaded:true")):
                plan.write_text(json.dumps(altered))
                with self.assertRaisesRegex(ValueError, message):
                    p0.read_receipt(plan, "videomme")

    def make_inputs(self, root):
        datasets = json.loads((ROOT / "configs/datasets.json").read_text())["datasets"]
        media = root / "media"
        media.mkdir()
        llava_rows = []
        for index, video_id in enumerate(("aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc", "ddddddddddd", "zzzzzzzzzzz")):
            video = f"youtube/{video_id}.mp4"
            path = media / video
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"synthetic-media-for-existence-only")
            llava_rows.append({"id": index, "video": video,
                               "conversations": [{"from": "human", "value": "<video>What happens?"},
                                                 {"from": "gpt", "value": "An answer which must never be filtered."}]})
        files = {"llava_video_178k": {p0.LLAVA_ANNOTATION: json.dumps(llava_rows).encode(),
                                      p0.LLAVA_ARCHIVE: b"synthetic archive receipt fixture"},
                 "videomme": {p0.VIDEOMME_ANNOTATION: json.dumps([{"videoID": "zzzzzzzzzzz", "question_id": "q1"}]).encode()},
                 "mvbench": {f"json/{task}.json": json.dumps([{"video": "clip_1.mp4"}]).encode()
                             for task in p0.convert.MV_TASKS}}
        arguments = []
        for dataset, contents in files.items():
            local = root / dataset
            entries = []
            for relative, content in contents.items():
                path = local / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                entries.append({"path": relative, "size_bytes": len(content), "local_sha256": p0.sha256_file(path)})
            plan = local / "ASSET_COMMITTED.json"
            plan.write_text(json.dumps({"repo_id": datasets[dataset]["repo_id"], "revision": datasets[dataset]["revision"],
                                        "downloaded": True, "files": entries}))
            name = "llava" if dataset == "llava_video_178k" else dataset
            arguments += [f"--{name}-root", str(local), f"--{name}-plan", str(plan)]
        output = root / "output"
        arguments += ["--media-root", str(media), "--output-dir", str(output), "--config", str(ROOT / "configs/train/ouro_pilot.json"),
                      "--train-max-qa", "2", "--validation-max-qa", "1"]
        return arguments, output

    def test_token_failure_preserves_selected_rows_and_full_coverage_without_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            args, output = self.make_inputs(Path(temp))
            real_run = subprocess.run
            called = []

            def subprocess_with_failed_token(command, **kwargs):
                called.append(command)
                if "check_token_budget.py" in command[1]:
                    self.assertIn("--local-files-only", command)
                    self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
                    return subprocess.CompletedProcess(command, 2, "overflow: do not truncate", "")
                return real_run(command, **kwargs)

            # The fixture substitutes JSON for parquet bytes, without pyarrow/network.
            with mock.patch.object(p0.convert, "rows_from", side_effect=lambda path: iter(json.loads(path.read_text()))), \
                    mock.patch.object(p0.subprocess, "run", side_effect=subprocess_with_failed_token):
                self.assertEqual(p0.main(args), 2)
            report = json.loads((output / "preparation_report.json").read_text())
            self.assertEqual(report["status"], "failed_token_budget")
            self.assertEqual(report["counts"]["train"]["qa"], 2)
            self.assertEqual(report["counts"]["validation"]["qa"], 1)
            self.assertEqual(len((output / "coverage.jsonl").read_text().splitlines()), 5)
            self.assertEqual(len(called), 2)
            self.assertFalse((output / "READY.json").exists())
            registry = json.loads((output / "benchmark_source_registry.json").read_text())
            self.assertFalse(registry["is_evaluation_manifest"])
            self.assertFalse(registry["benchmark_media_checked"])
            self.assertEqual(len(registry["annotation_tasks"]), 20)
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                p0.main(args)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_empty_existing_output_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            args, output = self.make_inputs(Path(temp))
            output.mkdir()
            with self.assertRaises(FileExistsError):
                p0.main(args)

    def test_success_requires_explicit_token_report_and_ready_fingerprints(self):
        with tempfile.TemporaryDirectory() as temp:
            args, output = self.make_inputs(Path(temp))
            real_run = subprocess.run

            def subprocess_with_token_report(command, **kwargs):
                if "check_token_budget.py" in command[1]:
                    report = Path(command[command.index("--report") + 1])
                    report.write_text(json.dumps({"passed": True, "samples": 3, "fixture_only": True}))
                    return subprocess.CompletedProcess(command, 0, "synthetic token report", "")
                return real_run(command, **kwargs)

            with mock.patch.object(p0.convert, "rows_from", side_effect=lambda path: iter(json.loads(path.read_text()))), \
                    mock.patch.object(p0.subprocess, "run", side_effect=subprocess_with_token_report):
                self.assertEqual(p0.main(args), 0)
            ready = json.loads((output / "READY.json").read_text())
            self.assertEqual(ready["report_sha256"], p0.sha256_file(output / "preparation_report.json"))
            self.assertFalse(ready["gpu_validated"])
            for name, digest in ready["manifest_sha256"].items():
                self.assertEqual(digest, p0.sha256_file(output / name))
            report = json.loads((output / "preparation_report.json").read_text())
            self.assertEqual(report["output_sha256"]["token_budget.json"], p0.sha256_file(output / "token_budget.json"))


if __name__ == "__main__":
    unittest.main()
