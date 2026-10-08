"""Keep the experiment/paper split, runnable paths and documentation in sync."""
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiment"


def test_experiment_is_self_contained():
    for relative in ("pyproject.toml", "README.md", "requirements/train.txt",
                     "src/deepprobe_vlm/modeling.py", "configs/train/ouro_pilot.json",
                     "scripts/launch/train.sh", "docs/REQUIREMENTS.md", "docker/Dockerfile"):
        assert (EXPERIMENT / relative).is_file(), relative
    assert not (ROOT / "src").exists()
    assert not (ROOT / "pyproject.toml").exists()


def test_public_scan_still_covers_entire_repository():
    path = EXPERIMENT / "scripts/check_public_release.py"
    spec = importlib.util.spec_from_file_location("public_release_layout", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.ROOT == ROOT


def test_paper_evidence_sources_resolve():
    manifest = json.loads((ROOT / "paper/figures/manifest.json").read_text())
    for source in manifest["sources"]:
        assert source.startswith("experiment/src/")
        assert (ROOT / source).is_file(), source


def test_repository_markdown_local_links_resolve():
    files = [ROOT / "README.md", EXPERIMENT / "README.md"]
    files += list((EXPERIMENT / "docs").glob("*.md"))
    files += list((ROOT / "paper").rglob("*.md"))
    for file in files:
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", file.read_text()):
            if re.match(r"[a-z]+:", target) or target.startswith("#"):
                continue
            target = target.split("#", 1)[0].strip("<>")
            assert (file.parent / target).exists(), f"{file.relative_to(ROOT)} -> {target}"
