# DeepProbe-VLM / paper

本目录同步论文材料与图，实验代码和运行说明在 [../experiment/](../experiment/README.md)。
当前版本是 **5 页实验协议**，不是已有实验结果的完整研究论文。
源码、编译 PDF、绘图源码与三张最终图均已纳入 Git，可直接同步。

| 内容 | 位置 |
|---|---|
| 当前文稿源码 / PDF | [experiment_protocol.tex](experiment_protocol.tex) / [experiment_protocol.pdf](experiment_protocol.pdf) |
| 主图、训练评测图、GPU 流程图 | [figures/final/](figures/final/) |
| 可编辑绘图源码 | [figures/source/](figures/source/) |
| 布局候选 | [figures/candidates/](figures/candidates/) |
| 图形审查与证据边界 | [docs/FIGURE_AUDIT.md](docs/FIGURE_AUDIT.md) |

`experiment_protocol.pdf` is a five-page **pre-experiment note**, not a completed
results paper. It aligns the new diagrams with the native-looped training code.
The previous local frozen-model manuscript remains untouched and is not copied
into this public repository with incompatible claims or private correspondence.

Final figures: `figures/final/main_overview`, `training_protocol`, `cuda_runtime`
in PDF, PNG and outlined SVG. Editable diagram source is XeLaTeX/TikZ. Comic Sans
MS is used when installed; no proprietary font file is redistributed. Math is
compiled by LaTeX. Use the figures at **6.75 inches / full width**, not single-column
width, to retain readable labels. All figures are illustrative, not result plots.

Build with XeLaTeX/latexmk plus Poppler and Python Pillow, starting at the
repository root (not inside `experiment/`):

```bash
python paper/figures/source/build_figures.py
cd paper
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build experiment_protocol.tex
cp build/experiment_protocol.pdf experiment_protocol.pdf
```

See [FIGURE_AUDIT.md](docs/FIGURE_AUDIT.md) for layout selection, evidence
boundaries and QA. Six candidate sources and a gallery are retained. Ignored
build/QA renders can be regenerated; their hashes in the figure manifest are
build evidence, not a promise that temporary files are committed.
