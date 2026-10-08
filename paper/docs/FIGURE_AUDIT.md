# Figure refresh audit — 7 October 2026

## Why the previous overview was hard to read

The former figure combined data acquisition, visual-depth probes, fast ridge/RLS
updates, control selection, a stopping rule, safe evidence construction and the
answer model in one frame. Multiple return paths and formula callouts competed
for attention. More importantly, it described the superseded depth-probe design,
not the requested native recurrent language model.

The new visual system uses three independent claims:

1. **Overview:** encode frames once, then run the same native Ouro stack for R
   rounds. Question embeddings bypass the vision tower. Visual states are not
   skipped during recurrent prefill; native per-loop KV is preserved.
2. **Protocol:** separate offline alignment/SFT from frozen held-out inference.
   Checkpoint/configuration selection belongs to validation, not test.
3. **Runtime:** obtain faithful reference outputs and profiles before designing
   kernels. Selective visual refresh and loop-aware custom CUDA remain future
   work, in a visibly dashed group.

## Source/evidence contract

All three figures have one primary status: illustrative. No accuracy, latency,
memory reduction or speedup value is shown. Benchmark and GPU gates are pending.

| Visual element | Current source | Boundary |
|---|---|---|
| Uniform video sampling | experiment/src/deepprobe_vlm/data.py: decode_video, uniform_indices | Reference two-pass decoder, not an optimized decoder |
| SigLIP once; bounded FG² prefix | modeling.py: encode_visual, SpatialPoolProjector | No selective vision encoding claim |
| Native shared-stack recurrence | DeepProbeVLM.__init__, _language_forward | Uses native total_ut_steps; no wrapper-loop approximation |
| Per-loop KV | DeepProbeVLM.greedy_generate | Token-to-token decode caching does not permit cross-loop KV substitution |
| Alignment and full-LM SFT | DeepProbeVLM.set_stage, train.py | Adaptive exit gate is unused/frozen; vision unfreezing is configurable |
| Test labels excluded from inference | MultimodalCollator(include_answer=False), evaluate.py | Labels may be used by the scorer only |
| Source-video isolation | manifest.py: validate_disjoint | Protocol still requires honest source IDs and dataset provenance |
| Future sparse CUDA boxes | Explicit proposed extension | Not implemented or measured |

## Candidate selection

Six editable grayscale roughs test three families at two density levels.

| ID | Layout | Decision |
|---|---|---|
| R1 | Compact ribbon | Good reading order, explanatory sublabels crowd the mechanism |
| R2 | Spacious ribbon | Selected; one left-to-right path and one obvious recurrence |
| S1 | Compact layered stack | Clear hierarchy but consumes excessive vertical space |
| S2 | Spacious layered stack | Clear but separates input sources from the core too strongly |
| C1 | Compact cutaway | Model boundary useful, interior has competing corners |
| C2 | Spacious cutaway | Clean but additional enclosure adds little semantic value |

The final R2 refinement retains only one cache invariant as a subordinate note.
Training and hardware details are moved to their own figures rather than
compressed into the overview.

## Rendering and inspection

- Native width: 17.145 cm / 6.75 in. Meaningful labels: at least 7.5 pt.
- Body labels: Comic Sans MS; formulas: XeLaTeX-compiled math.
- Outputs: one-page vector PDF, outlined-vector SVG, 240-DPI PNG, editable TeX.
- Main/protocol palette: cream, aubergine, sage, coral.
- Runtime palette: white with alternating charcoal and **filled green** boxes;
  white labels. Green is darkened from the NVIDIA brand accent for contrast.
- No ImageGen, external icons, real-video samples or invented outputs are used.
- Initial label crowding was identified by pixel inspection and fixed by
  shortening labels, separating heading rows and moving details to captions.
- All final standalone PNGs and three 6.75-inch paper-page proof PNGs were
  inspected. Proofs are under paper/figures/qa/. The enclosing experiment note
  also needs a final include-size inspection after any integration changes.
- PDF quality checks pass for all three figures: one page each, embedded fonts,
  no Type 3 fonts, no LaTeX warning/error findings. Edge audit finds no clipping.

Single-column placement is unsupported because it would reduce 7.5 pt labels to
about 3.6 pt. Do not shrink the figures to fit a single-column float.
