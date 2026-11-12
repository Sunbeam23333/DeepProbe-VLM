# Training and inference optimization — 2026-11-12

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
