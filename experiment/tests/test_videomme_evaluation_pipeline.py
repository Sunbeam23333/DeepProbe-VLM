import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("vm_pipeline", Path(__file__).resolve().parents[1] /
                                             "scripts/operations/run_videomme_evaluation_pipeline.py")
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


def test_media_success_requires_atomic_sentinel_binding(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.time, "sleep", lambda _: None)
    media, training, done = [tmp_path / name for name in ("FRAME_CACHE_RECEIPT.json", "TRAINING_FINGERPRINTS.json", "FRAME_CACHE_COMPLETE.json")]
    media.write_text(json.dumps({"complete": True}))
    training.write_text(json.dumps({"passed": True}))
    sentinel = {"complete": True, "source_count": 899, "frame_count": 7192, "receipt_sha256": "0" * 64}
    done.write_text(json.dumps(sentinel))
    with pytest.raises(ValueError, match="completion"):
        pipeline.await_complete_inputs([media, training, done], tmp_path, 1)
    assert not (tmp_path / "INPUTS_COMPLETE.json").exists()
    sentinel["receipt_sha256"] = pipeline.digest(media)
    done.write_text(json.dumps(sentinel))
    pipeline.await_complete_inputs([media, training, done], tmp_path, 1)
    assert (tmp_path / "INPUTS_COMPLETE.json").is_file()


def test_media_failure_stops_before_downstream_stages(tmp_path):
    (tmp_path / "COORDINATOR_FAILURE.json").write_text('{"complete": false}')
    missing = [tmp_path / name for name in ("FRAME_CACHE_RECEIPT.json", "TRAINING_FINGERPRINTS.json", "FRAME_CACHE_COMPLETE.json")]
    with pytest.raises(RuntimeError, match="Media acquisition failed"):
        pipeline.await_complete_inputs(missing, tmp_path, 1)
    assert not (tmp_path / "INPUTS_COMPLETE.json").exists()


def test_resume_preserves_old_failure_and_waits_on_new_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.time, "sleep", lambda _: None)
    old = tmp_path / "COORDINATOR_FAILURE.json"
    old.write_text('{"error": "URLError"}')
    old_bytes = old.read_bytes()
    marker = tmp_path / "resume-20261013-v5/a1/FAILED.json"
    marker.parent.mkdir(parents=True)
    paths = [tmp_path / name for name in
             ("FRAME_CACHE_RECEIPT.json", "TRAINING_FINGERPRINTS.json", "FRAME_CACHE_COMPLETE.json")]
    paths[0].write_text(json.dumps({"complete": True}))
    paths[1].write_text(json.dumps({"passed": True}))
    paths[2].write_text(json.dumps({"complete": True, "source_count": 899, "frame_count": 7192,
                                  "receipt_sha256": pipeline.digest(paths[0])}))
    pipeline.await_complete_inputs(paths, tmp_path, 1, failure_marker=marker)
    receipt = json.loads((tmp_path / "INPUTS_COMPLETE.json").read_text())
    assert receipt["media_failure_marker"] == str(marker)
    assert old.read_bytes() == old_bytes
    marker.write_text('{"error": "URLError"}')
    with pytest.raises(RuntimeError, match="Media acquisition failed"):
        pipeline.await_complete_inputs(paths, tmp_path, 1, failure_marker=marker)
    assert old.read_bytes() == old_bytes


@pytest.mark.parametrize("candidate", ["relative/FAILED.json", "outside", "traversal", "root"])
def test_failure_marker_rejects_paths_outside_media_root(tmp_path, candidate):
    root = tmp_path / "media"
    root.mkdir()
    marker = {"outside": tmp_path / "FAILED.json", "traversal": root / "../FAILED.json",
              "root": root}.get(candidate, candidate)
    with pytest.raises(ValueError, match="inside media_root"):
        pipeline.media_failure_marker(root, marker)


def test_failure_marker_rejects_symlink_and_directory(tmp_path):
    root = tmp_path / "media"
    root.mkdir()
    target = root / "target"
    target.mkdir()
    (root / "alias").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        pipeline.media_failure_marker(root, root / "alias/FAILED.json")
    with pytest.raises(ValueError, match="identify a file"):
        pipeline.media_failure_marker(root, target)


def test_failed_audit_stage_preserves_exit_and_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "STAGE_CWD", str(tmp_path))
    with pytest.raises(RuntimeError, match="content_audit failed"):
        pipeline.run_stage(tmp_path, "content_audit", [pipeline.sys.executable, "-c", "raise SystemExit(2)"], {})
    receipt = json.loads((tmp_path / "content_audit.exit.json").read_text())
    assert receipt["returncode"] == 2
    assert not (tmp_path / "submit_once.argv.json").exists()
