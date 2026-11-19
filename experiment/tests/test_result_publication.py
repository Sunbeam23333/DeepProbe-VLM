from pathlib import Path
from types import SimpleNamespace

import pytest

from deepprobe_vlm.result_publication import MANIFEST, publish, seal, verify


def bundle(tmp_path):
    root = tmp_path.resolve() / "bundle"
    root.mkdir()
    (root / "results.json").write_text('{"correct": 1, "total": 2}\n')
    seal(root, release_id="test-release")
    return root


class FakeHub:
    def __init__(self, tmp):
        self.tmp = tmp
        self.files = {}
        self.uploads = 0
        self.downloads = []

    def list_repo_files(self, repo_id, repo_type, **kwargs):
        assert repo_id == "owner/results" and repo_type == "dataset"
        return [SimpleNamespace(path=k, size=len(v), type="blob") for k, v in self.files.items()]

    def upload_folder(self, repo_id, repo_type, folder, **kwargs):
        assert kwargs["sync_remote_repo"] is False
        self.uploads += 1
        for name in kwargs["allow_patterns"]:
            self.files[kwargs["path_in_repo"] + "/" + name] = (folder / name).read_bytes()

    def download_file(self, repo_id, repo_type, name, **kwargs):
        assert kwargs["force"] is True
        self.downloads.append(name)
        path = Path(kwargs["local_dir"]) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.files[name])
        return path


def test_publish_verifies_each_file_and_preserves_unrelated_content(tmp_path):
    root = bundle(tmp_path)
    hub = FakeHub(tmp_path)
    hub.files["README.md"] = b"Existing repository card"
    result = publish(hub, root, repo_id="owner/results", prefix="releases/test",
                     receipt_dir=tmp_path / "receipt")
    assert result["file_count"] == 2 and len(hub.downloads) == 2
    assert hub.files["README.md"] == b"Existing repository card"
    assert result["remote_files_deleted"] is False
    assert (tmp_path / "receipt/VERIFIED.json").is_file()


def test_existing_release_is_reverified_without_overwrite(tmp_path):
    root = bundle(tmp_path)
    hub = FakeHub(tmp_path)
    for name in ("results.json", MANIFEST):
        hub.files["releases/test/" + name] = (root / name).read_bytes()
    result = publish(hub, root, repo_id="owner/results", prefix="releases/test",
                     receipt_dir=tmp_path / "receipt")
    assert hub.uploads == 0 and result["already_present"] is True


def test_modified_release_is_refused_before_network(tmp_path):
    root = bundle(tmp_path)
    (root / "results.json").write_text('{"correct": 2, "total": 2}\n')
    hub = FakeHub(tmp_path)
    with pytest.raises(ValueError, match="changed after sealing"):
        publish(hub, root, repo_id="owner/results", prefix="releases/test",
                receipt_dir=tmp_path / "receipt")
    assert hub.uploads == 0


def test_existing_remote_wrong_bytes_never_overwritten(tmp_path):
    root = bundle(tmp_path)
    hub = FakeHub(tmp_path)
    for name in ("results.json", MANIFEST):
        hub.files["releases/test/" + name] = (root / name).read_bytes()
    hub.files["releases/test/results.json"] = b"x" * len(hub.files["releases/test/results.json"])
    with pytest.raises(ValueError, match="Full readback failed"):
        publish(hub, root, repo_id="owner/results", prefix="releases/test",
                receipt_dir=tmp_path / "receipt")
    assert hub.uploads == 0
    assert not (tmp_path / "receipt/VERIFIED.json").exists()


def test_partial_remote_requires_reconciliation(tmp_path):
    root = bundle(tmp_path)
    hub = FakeHub(tmp_path)
    hub.files["releases/test/results.json"] = (root / "results.json").read_bytes()
    with pytest.raises(ValueError, match="partial or differs"):
        publish(hub, root, repo_id="owner/results", prefix="releases/test",
                receipt_dir=tmp_path / "receipt")
    assert hub.uploads == 0


def test_credentials_and_weights_are_rejected(tmp_path):
    root = tmp_path.resolve() / "bundle"
    root.mkdir()
    (root / "result.json").write_text('"hf_' + "a" * 30 + '"')
    with pytest.raises(ValueError, match="Credential pattern"):
        seal(root, release_id="test")
    (root / "result.json").unlink()
    (root / "model.safetensors").write_bytes(b"weights")
    with pytest.raises(ValueError, match="Not a result/document asset"):
        seal(root, release_id="test")


def test_added_file_and_symlink_are_refused(tmp_path):
    root = bundle(tmp_path)
    (root / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="changed after sealing"):
        verify(root)
    (root / "extra.json").unlink()
    (root / "extra.json").symlink_to(root / "results.json")
    with pytest.raises(ValueError, match="symlinks"):
        verify(root)
