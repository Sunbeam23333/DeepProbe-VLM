# Clear figure set — native recurrent baseline

All three figures are **illustrative, not measured**. They describe the current
native Ouro research scaffold and its experiment plan. They intentionally do
not reuse the older ridge/RLS visual-depth-adaptation mechanism.

## Publication files

- final/main_overview.pdf — one visual pass, native shared-stack recurrence.
- final/training_protocol.pdf — separate training and held-out evaluation lanes.
- final/cuda_runtime.pdf — baseline first; dashed future sparse/CUDA work.

Each figure has a 240-DPI PNG and vector SVG next to its PDF. Labels and formulas
are editable in source/*.tex; SVG glyphs are outlined vectors rather than text.
Use the PDF at **6.75 inches wide**, not single-column width.

## Rebuild

Requires XeLaTeX, Poppler (pdftoppm, pdftocairo) and optional Pillow for the
candidate contact sheet. Comic Sans MS is used when installed, with an explicit
TeX Gyre Heros fallback; the delivered PDFs embed Comic Sans MS. No font files,
external images, benchmark frames, private identifiers or remote assets are
redistributed.

~~~sh
python3 paper/figures/source/build_figures.py
~~~

On macOS add /Library/TeX/texbin and /opt/homebrew/bin to PATH if needed.
Six grayscale candidates are preserved in candidates/ and
candidates/gallery.png. The refined main figure selects R2 (spacious ribbon)
because it exposes one reading direction and one recurrent edge.

source/include_size_proof.tex is an independent full-width paper-page proof.
The enclosing paper's final compile must still be inspected after integration.
manifest.json records evidence status and output hashes.
