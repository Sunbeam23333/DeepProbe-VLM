# Training and inference optimization — 2026-11-13

The code audit identified avoidable inference work and a missing recovery guard.
This change implements bounded improvements and the measurement tools needed to
evaluate them. No CUDA speedup, H20/B300 ratio or optimized held-out score has
been accepted. Historical releases, checkpoints and benchmark results retain
their frozen implementations.

## Implemented

- **Recovery integrity:** both training entries check native optimizer,
  scheduler, rank-specific RNG, required scaler and Trainer progress before
  model loading. Missing state now fails instead of allowing a fresh optimizer
  or RNG state. This is a structure/presence guard, not tensor authentication or
  distributed numerical recovery acceptance.
- **Last-position logits:** `greedy_generate(last_token_logits=True)` uses the
  backend's explicit `logits_to_keep=1` interface. This avoids projecting all
  prefix positions only to discard them. The default and training loss continue
  to use full logits. Native cache checks and EOS behavior are retained; no
  next-token embedding/mask preparation is done after the last allowed token.
- **Fused spatial pooling:** `pool_backend="triton"` replaces weighted-feature
  materialization and separable reduction with a CUDA kernel. The learned gate,
  sigmoid rounding, FP32 clamp, adaptive bin boundaries and final projection
  are retained. Accumulation is FP32; different reduction order can change
  rounding. This path is inference-only, has no backward, and rejects unsupported
  inputs or a missing runtime rather than falling back silently.
- **Measurement and comparison:** explicit v2 implementation plans, separate
  diagnostic ranges/traces, and an artifact comparator that requires matching
  hardware/software, inputs and exact generated token sequences before timing
  ratios are emitted. See [latency protocol](LATENCY_PROTOCOL.md).

The model-shaped `[8,729,1152]` weighted-feature FP32 intermediate occupies
26,873,856 bytes by shape arithmetic. Avoiding that intermediate is a design
property, not a measured peak-memory saving or a speedup. Projection, gate,
vision encoding and the language model still perform their usual work.

## Validation and commands

CPU tests cover malformed recovery state, an actual tiny alignment → SFT →
resume sequence, untouched training logits/loss/gradients, cached and uncached
generation, EOS boundaries, backend refusal, pooling bins and evidence
comparison gates. They do not execute the Triton kernel. The optional native
Ouro tests instantiate tiny random weights from the pinned upstream Python
code; no pretrained weights are downloaded.

On November 12, the full public CPU suite passed **512 tests**, with **80 skipped**
(private evidence, native-code or GPU opt-ins). The separate pinned native Ouro
suite passed **3 tests**, including real language-head input length and token
parity across cache/head-slicing combinations in eager and SDPA modes. The local
runtime was Python 3.12, Torch 2.13.0, Transformers 4.57.6 and Accelerate 1.15.0.
These counts do not certify CUDA, DDP or pretrained-checkpoint performance.

The earlier bounded engineering run `dp-cuda-pool-qa-20261112-a1`, binding
commit `bec405e8802232cf4815f1c6dfea968318a12aa9`, progressed from the retained
**20:47 Pending** observation to Running at **20:55:46**, then failed at
**20:56:31 on November 12**. Exact Pod-UID events establish external preemption
at **20:56:29**. It produced no GPU startup acceptance, completed CUDA numerical
test or microbenchmark result. The **21:15** observation found its original Pod
absent and its exact archive prefix empty. This complete failed-attempt history
is retained; it is not an accepted OSS archive or a CUDA performance result.

A separately identified attempt 2 binds source commit
`c03ffefa485682d2c737124c07b83473442cf6f4`. Its package/control CPU review passed
**18 tests, one skipped**, followed by CPU admission, independent review and
the exact submission dry-run. The job
`jinhaopeng-dpvlm-cuda-pool-qa-20261113-a2-n1-772f8`, UID
`ca956db5-7166-4452-93b8-d89fecd6cc33`, was created once at **00:26:08 Beijing
time on November 13** (UTC November 12, 16:26:08). Independent submission
review checked the intent/create/UID and actual resource-contract linkage.

At **00:31:21 Beijing time**, the exact attempt-2 Pod remained **Pending**,
with no assigned node or run directory. The scheduler could not satisfy the
combined CPU/memory/eight-GPU request on one permitted node. Its contract is
stream/low, one node, eight GPUs, restart Never and no forced node. Eight-rank
BF16/NCCL startup acceptance is required before the numerical tests and
separate one-device BF16/FP32 microbenchmarks. None of those GPU results exists
at this observation. [Status evidence](../results/status.json) binds both
attempts and the individual admission/submission/readback receipts by file
path, bytes and SHA256. These are dated observations, not a background monitor
or an accepted acceleration result.

Run from `experiment/`, using a compatible existing CUDA-enabled PyTorch/Triton
environment and a freshly admitted GPU allocation:

```bash
# Planning only: does not import Torch/Triton or initialize CUDA.
python -m deepprobe_vlm.benchmark_cuda_pool

# Explicit real-kernel numerical coverage and isolated operator measurement.
DEEPPROBE_TEST_TRITON_POOL=1 python -m pytest -q tests/test_cuda_pool.py
python -m deepprobe_vlm.benchmark_cuda_pool --execute --profile \
  --output runs/cuda-pool-micro-NEW
```

The default microbenchmark uses BF16 features of shape `[8,729,1152]`, grid 4,
25 warmups and 100 alternating-order measured calls per backend. It retains
the first possible JIT call separately, input/source hashes, numerical error,
CUDA-event and synchronized wall durations, allocated/reserved memory peaks
and an independent profiler pass with actual kernel-event verification. Gate
and projector are outside its measured scope. It measures one device even when
the required scheduler allocation contains eight. It submits no job itself.

Before promoting an optimization, execute real GPU numerical tests, measure
each switch alone and together against a new reference plan, inspect exact
generated-token agreement and measure checkpoint-level latency/memory. Assess
held-out quality under a separately frozen protocol. A fast pooling kernel may
have little effect if recurrent decode or media preparation dominates.

## Remaining work and thesis scope

The complete serial suite is scripted but **has not been submitted**. Real
F4/F16 caches sampled from original benchmark videos, independently accepted
R4/Smol recovery inputs and fresh storage/source admission remain outstanding.
The November 13 **00:15:52** read-only capacity snapshot measured
78,345,490,305 bytes in the two accounted roots; conservative new-copy recovery
projections remain above the 99 GB limit. Small-QA admission is separate from
these training gates. See [suite operating runbook](EXPERIMENT_SUITE.md) and
the separately dated capacity evidence in [status.json](../results/status.json).

The paper's selective visual-state refresh, learned compute budget, token
compaction and algorithm/kernel co-design remain research hypotheses. An
ordinary pooling fusion and head slicing are engineering improvements; they do
not establish the proposed novel mechanism or a quality–compute trade-off.
The existing paper protocol/figures describe the default dense baseline and
those future branches.

Further optimization should follow real vision/pooling/prefill/decode profiles.
Potential targets include redundant checkpoint weight loading, final checkpoint
serialization, repeated preprocessing and visual-prefix reuse across questions.
They need their own I/O or timing evidence and protocol design. Skipping the
last Trainer save naively can lose final RNG/progress state, and cross-question
reuse changes the request workload. These are intentionally deferred.

Training currently retains full backpropagation through every recurrent step,
BF16 autocast, FP32 optimizer state, non-reentrant activation checkpointing and
replicated DDP. No fused training backward, selective refresh, static Ouro KV,
cross-loop cache substitution or lower-precision training is introduced.
