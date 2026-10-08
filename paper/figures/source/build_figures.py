"""Rebuild the vector figure set. Requires XeLaTeX and Poppler on PATH.

On macOS, prepend /Library/TeX/texbin and /opt/homebrew/bin to PATH.
Optional gallery requires Pillow. No generated or benchmark raster imagery is used.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run(*args, cwd=None):
    result = subprocess.run(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(result.stdout.decode("utf-8", errors="replace"))


def build(tex, out):
    out.mkdir(parents=True, exist_ok=True)
    run("xelatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={out}", tex.name, cwd=tex.parent)
    pdf = out / f"{tex.stem}.pdf"
    run("pdftoppm", "-singlefile", "-png", "-r", "240", str(pdf), str(out / tex.stem))
    run("pdftocairo", "-svg", str(pdf), str(out / f"{tex.stem}.svg"))
    for suffix in ("aux", "log"):
        p = out / f"{tex.stem}.{suffix}"
        target = ROOT / "qa" / "logs"
        target.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), target / p.name)
    return pdf


def main():
    for tex in sorted((ROOT / "candidates").glob("*.tex")):
        build(tex, ROOT / "candidates" / "rendered")
    for name in ("main_overview", "training_protocol", "cuda_runtime"):
        build(ROOT / "source" / f"{name}.tex", ROOT / "final")
    proof = ROOT / "source" / "include_size_proof.tex"
    qa = ROOT / "qa"
    qa.mkdir(exist_ok=True)
    run("xelatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={qa}", proof.name, cwd=proof.parent)
    run("pdftoppm", "-png", "-r", "160", str(qa / "include_size_proof.pdf"), str(qa / "include_size_proof"))
    try:
        from PIL import Image, ImageOps, ImageDraw
        sources = sorted((ROOT / "candidates" / "rendered").glob("*.png"))
        gallery = Image.new("RGB", (1800, 1170), "#f1f1f1")
        for i, source in enumerate(sources):
            im = Image.open(source).convert("RGB")
            im.thumbnail((880, 370))
            gallery.paste(im, ((i % 2) * 900 + 10, (i // 2) * 390 + 10))
        gallery.save(ROOT / "candidates" / "gallery.png")
    except ImportError:
        print("Pillow unavailable; gallery skipped.")
    paths = [p for p in ROOT.rglob("*") if p.is_file() and p.suffix in (".pdf", ".svg", ".png", ".tex")]
    manifest = {
        "evidence_status": "illustrative", "visible_status": "ILLUSTRATIVE · NOT MEASURED",
        "date": "2026-10-07", "target_width_inches": 6.75,
        "native_width_cm": 17.145, "minimum_meaningful_font_pt": 7.5,
        "labels": "Comic Sans MS when installed; TeX Gyre Heros fallback",
        "formulas": "XeLaTeX-compiled vector math", "raster_assets": [],
        "selected_candidate": "R2 ribbon spacious, refined with compact cache invariant",
        "claims": {
            "main_overview": "Encode visual inputs once; reuse the native Ouro decoder across rounds; visual LM states still update and loop caches remain distinct.",
            "training_protocol": "Separate projector alignment and full-LM SFT from frozen held-out inference; source-video splits isolate test evidence.",
            "cuda_runtime": "Establish native baseline and hardware profiles before any custom sparse runtime; all selective-update or fused-kernel work remains future.",
        },
        "sources": ["experiment/src/deepprobe_vlm/modeling.py", "experiment/src/deepprobe_vlm/train.py", "experiment/src/deepprobe_vlm/evaluate.py", "experiment/src/deepprobe_vlm/data.py"],
        "boundary": "No benchmark accuracy, convergence, throughput or speedup is claimed. A runnable scaffold is not a validated scientific result.",
        "files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Built six candidates and three final figures.")


if __name__ == "__main__":
    main()
