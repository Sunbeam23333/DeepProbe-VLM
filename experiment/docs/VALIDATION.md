# Verification history

## Experiment results and code synchronization — 2026-11-09

R1/R2 seed17 alignment and SFT have completed, with accepted eight-GPU execution
and archived Video-MME899/TempCompass results. The exact evaluation scope,
denominators, invalid-response rates and paired source-cluster confidence
intervals are in [the accepted results](../results/RESULTS.md).

R4 alignment completed and passed full archive acceptance. R4 SFT subsequently
failed after 602/718 updates; the last durable checkpoint is step 600, with no
R4 held-out result. See [the dated status](../results/STATUS.md).

This source update includes deterministic spatial pooling, native Ouro token-0
embedding-gradient preservation across reloads, formal data/source and resume
checks, distributed preflight, guarded operational tools, TempCompass scoring
and the prepared latency runner. These implementations and CPU tests do not
establish multi-seed robustness or a warmed GPU performance comparison.

The reviewed unified-evaluation candidate has also been integrated: explicit
model/backend/loop/seed and checkpoint acceptance bindings replace model-name
inference. Its additional R4/Smol/seed interfaces have CPU validation only;
they are not completed evaluations or admission for resumed-SFT variants.

Validation of the published source tree:

- A clean export of the staged Git files, with no private evidence directory:
  **448 passed / 79 skipped**. Skips comprise 75 private-evidence integration
  checks, three opt-in pinned native remote-code checks and one Ali-only original
  artifact check.
- The 75 private-evidence integration checks passed separately against the
  original local artifacts. Those inputs are not distributed in this release.
- CPU attention forward/backward preflight passed with `status=cpu_only`.
  Python 3.12.14, macOS CPU, Torch 2.13.0, Transformers 4.57.6,
  Accelerate 1.15.0, huggingface-hub 0.36.0, NumPy 2.5.4, Pillow 11.3.0,
  PyAV 16.1.0. This environment is not a GPU runtime recommendation.
- Python AST checks, 11 shell syntax checks, whitespace checks and staged-file
  credential/artifact scans passed. No private evidence, media or weights are
  present in the exported source tree.

These are local checks; GitHub Actions results are recorded separately by CI.

The sections below describe checks performed on their stated dates. Their
historical “not verified” lists are not the current experiment status.

## Directory migration check — 2026-10-08

Experiment installation, configs, scripts, tests and documentation now live in
`experiment/`; paper sources and figures remain in `paper/`. Run experiment
commands from `experiment/` and reinstall the editable package after pulling
this layout change. No model algorithm or paper figure content changed.

After the move: **53 tests passed**, including the opt-in tiny native Ouro
eager/SDPA tests and four new checks for directory layout, paper evidence paths,
Markdown links, and whole-repository public-file scanning. The same two benign
Trainer checkpoint-order warnings remained. This is still CPU validation,
not H20/B300 execution. The GitHub workflow was updated to use `experiment/`.

## Initial verification — 2026-10-07

Date: 2026-10-07. This records software checks, **not benchmark or GPU results**.

## Passed locally

- **49 tests passed** with `DEEPPROBE_TEST_NATIVE_OURO=1 python -m pytest -q`.
- Tiny pinned native Ouro with random weights, in both eager and SDPA attention:
  full active-component gradients, non-reentrant checkpointing, distinct
  per-loop KV slots, cached/no-cache generation parity, native save/reload.
- Actual HF Trainer alignment → SFT → checkpoint resume → reference evaluation.
  Resumed and uninterrupted tiny runs produced bitwise-identical final weights.
- Source-disjoint manifest validation, unsafe-path/archive rejection, bounded
  sampling, frame-cache provenance, exact token-budget preflight, masked labels,
  metric separation and evaluation contamination checks.
- CLI help, launcher shell syntax, Python compilation, dependency consistency
  (`pip check`) and tracked public-file privacy/size scan.
- CPU-only attention forward/backward preflight. Output explicitly says CPU-only.
- Metadata-only pinned Video-MME download plan. No video/model-weight download.
- Three vector figure PDFs plus five-page protocol compiled and visually
  inspected at final include width. All fonts embedded; no Type 3 fonts or
  overfull/underfull boxes in the final protocol build.

Two benign Trainer warnings about filesystem modification-time ordering
appeared during checkpoint rotation; Trainer used checkpoint-number ordering.

Environment: Python 3.12.14, macOS CPU, PyTorch 2.14.1, Transformers 4.57.6,
Accelerate 1.12.0, huggingface-hub 0.36.0, NumPy 2.2.6, Pillow 11.3.0,
PyAV 16.0.1. Do not use this CPU Torch build as a GPU deployment recommendation.

## Not verified here

- Full pretrained Ouro/SigLIP training or convergence on real videos.
- Eight-device DDP/NCCL, H20/B300 memory fit, throughput or inference latency.
- Docker image build against a particular host-driver configuration.
- Official benchmark reproduction, missing licensed MVBench media, semantic
  judging of open-ended answers, or upstream pretraining contamination.
- Any novel selective-update policy, test-time training, shared cross-loop KV,
  or custom CUDA/PTX/CUTLASS kernel.

CI runs the offline CPU suite by default (the two remote-code native Ouro tests
are opt-in). CI status is available in the repository's Actions tab; local test
success is not a substitute for the server-side gates in [HARDWARE.md](HARDWARE.md).
