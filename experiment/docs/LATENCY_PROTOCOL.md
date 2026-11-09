# Development latency protocol v1

`deepprobe_vlm.benchmark_latency` is an opt-in, per-device latency runner. It does
not submit jobs. The October 26 preparation has passed CPU tests with simulated
CUDA/model adapters; real GPU validation remains pending. Existing evaluator
timings retain their original cold/mixed interpretation.

The predeclared settings are batch size 1, BF16, greedy generation with a
16-token maximum, seed 17, three warmup requests followed by ten measured
requests for each of 32 fixed examples. The sample order is the ascending
SHA-256 of `17`, a NUL byte, and the sample ID, with sample ID breaking ties.
The runner rejects test splits and known benchmark datasets, requires all input
rows to belong to train or development validation, and does not select using
answers, quality scores, predictions, lengths, or measured speed. A manifest
with fewer than 32 examples fails rather than silently reducing coverage.

These numerical defaults fill gaps in the existing experiment documents.
They establish a fixed warmup procedure, not evidence that every hardware
configuration has reached a convergence threshold. The prepared input is the
historical formal LLaVA-Video validation manifest: 4,472 QA over 185 source
videos, SHA-256 `533e427260266f6a426b5c4486f16e5ec41881000e1a9b6e01e1e225499d2ab1`.
Its deterministic selection contains 32 QA over 30 videos. This is development
data, so its reference answer scores support diagnostics and no held-out
benchmark claim. The fixed 16-token cap can truncate free-form answers.

Every request repeats media decoding, image processing, tokenization and input
transfer. The runner records four wall-clock intervals, with explicit CUDA
synchronization at each boundary:

- Preparation: media decoding, image processing, tokenization and CPU-to-GPU transfer.
- Model: the unchanged `greedy_generate` call, including vision encoding, prefill and autoregressive decode.
- Postprocessing: generated-token GPU-to-CPU transfer and tokenizer text decoding.
- Request: the sum of those three intervals.

Model/tokenizer/processor loading, input hashes, answer scoring and artifact
writing are outside those intervals. Vision/prefill/decode are not individually
instrumented, and no time-to-first-token field is claimed. OS file caches and
the CUDA allocator remain warmed; the runner does not clear them between
requests. Absolute allocated and reserved memory peaks include the resident
model and request tensors. Baseline allocated/reserved bytes are also retained.

Warmup requests are saved and excluded from all measured summaries. All ten
measured repeats remain, including inconsistent predictions. Results contain
per-request and per-sample timings, mean, median, and p95 using linear
interpolation at `(n-1)*0.95` (R7), absolute memory peaks, prompt and tensor
sizes, actual generated token IDs/counts (including generated EOS), and the
first measured repeat's reference quality diagnostic for each sample.
Repeated requests are correlated observations; these summaries supply no
confidence interval or serving-throughput claim.

Cache choice is explicit in each plan. Cache-on and cache-off plans are separate
runs and share a comparison identity only when every other plan field agrees.
Prepared R1 and R2 pairs use the same sample identities and their own frozen
checkpoint configurations. Compare each pair on the same hardware/software
configuration and inspect token-output agreement before interpreting timing
differences. Never choose which pair to report from its quality score.

Plans bind the complete checkpoint file inventory, selected media bytes, full
manifest, development-source metadata, model configuration, and six executable
source dependencies. `freeze_plan` constructs such a plan from local files
using byte hashes without loading weights. Schema-only CLI validation is the
default; only `--execute --output NEW_DIRECTORY` invokes the GPU path. Execution
preserves the original plan bytes and rechecks all bound input bytes before
model construction. Caught Python exceptions retain intent, runtime metadata
when available, the durable request prefix, and `FAILURE.json`; no completed
summary or automatic retry is produced. Process interruptions, including
SIGTERM, KeyboardInterrupt and SIGKILL, may leave only the durable prefix and
no failure receipt. The outer job must retain its exit and scheduler evidence;
absence of `FAILURE.json` is never proof of success. Existing output directories
are refused.

Receipts are written and fsynced under a pending filename, then published with
an exclusive atomic hard link. A failed flush or publication leaves no new final
receipt. Pending files do not certify success. Cleanup of a pending hard link
after successful publication is best effort and cannot reverse that success.

The prepared candidate is in
`operations/evidence/latency-preparation-20261026/runtime-v2`, with four plans
under `plans/`. It has not been deployed. Root-owned admission must independently
enforce the allocated single-node/eight-GPU/low-priority policy and fresh storage,
identity, source and data checks. The runner measures one process on logical
`cuda:0`; an eight-GPU allocation does not make this an eight-GPU throughput
measurement. Concurrent work and hardware isolation require external review.
