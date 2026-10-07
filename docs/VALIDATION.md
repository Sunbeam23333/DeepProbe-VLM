# Initial release verification

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
