# Figures and experimental protocol

`experiment_protocol.pdf` is a five-page **pre-experiment note**, not a completed
results paper. It aligns the new diagrams with the native-looped training code.
The previous local frozen-model manuscript remains untouched and is not copied
into this public repository with incompatible claims or private correspondence.

Final figures: `figures/final/main_overview`, `training_protocol`, `cuda_runtime`
in PDF, PNG and outlined SVG. Editable diagram source is XeLaTeX/TikZ. Comic Sans
MS is used when installed; no proprietary font file is redistributed. Math is
compiled by LaTeX. Use the figures at **6.75 inches / full width**, not single-column
width, to retain readable labels. All figures are illustrative, not result plots.

Build with XeLaTeX/latexmk plus Poppler and Python Pillow:

```bash
python paper/figures/source/build_figures.py
cd paper
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build experiment_protocol.tex
cp build/experiment_protocol.pdf experiment_protocol.pdf
```

See [FIGURE_AUDIT.md](../docs/FIGURE_AUDIT.md) for layout selection, evidence
boundaries and QA. Six candidate sources and a gallery are retained. Ignored
build/QA renders can be regenerated; their hashes in the figure manifest are
build evidence, not a promise that temporary files are committed.
