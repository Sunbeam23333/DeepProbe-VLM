# Model and training contract — experimental baseline

This repository implements a **native looped-language-model video SFT baseline**.
It does not implement the previous selective visual-depth/ridge-RLS proposal.
It does not yet implement a novel video-specific recurrent architecture,
cross-loop KV sharing, token early exit, test-time weight updates, or custom CUDA.
No pretrained-model accuracy, convergence or H20/B300 throughput result is claimed.

## Architecture

1. Deterministically sample `num_frames` chronological frames from a video, or
   an explicit image list. The reference PyAV decoder scans a video twice and
   keeps only selected frames in RAM. Optional `start_seconds` / `end_seconds`
   restrict decoding to the half-open interval `[start, end)`. It is a reference
   correctness path, not an optimized video input pipeline. Cache frames offline
   for high-throughput training and account for that preprocessing separately.
2. SigLIP produces a full square patch grid for every sampled frame. A learned
   sigmoid gate weights patch features within adaptive spatial pooling bins;
   normalized pooling preserves bin order. A two-layer GELU projector maps to
   language hidden size. `pool_grid=4`, `num_frames=8` yields 128 visual tokens.
3. The sequence is: short textual video prefix, chronological visual tokens,
   question and optional choices, answer delimiter, answer tokens and EOS.
   Visual tokens replace explicit placeholder embeddings. No gold answer enters
   the inference collator. Text over the configured limit fails explicitly;
   there is no silent removal of an answer or its supervision.
4. Ouro performs its native shared-weight loop internally. `total_ut_steps` is
   set in both the HF config and the live core. One model call runs all loops.
   Ordinary `language_backend="causal_lm"` is supported only with `num_loops=1`
   for the non-looped, shared-interface comparison.

The visual prefix precedes the question under causal masking. Visual states
therefore cannot attend to later question tokens; the text states can read the
visual prefix at each loop. This baseline is not a query-conditioned visual
update mechanism. Such a change requires an explicit architectural experiment.

The default public checkpoint pins are:

- `ByteDance/Ouro-1.4B` at `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
- `google/siglip-so400m-patch14-384` at
  `9fdffc58afc957d1a03a25b10dba0329ab15c2a3`.

Both pins were resolved from the providers' Hugging Face model API when this
release was assembled. Native Ouro loading executes its pinned custom Python
code (`trust_remote_code=True`); review it before running. Local checkpoint
directories are supported for offline loading and synthetic contract tests.
The actual parameter counts are written per component in each run manifest;
the model is not advertised as exactly 2B merely from its language-model name.

## Loss and gradient boundary

For sequence labels, the prefix, visual positions, question and padding are
masked with `-100`; only the answer and EOS are supervised. The wrapper calls
Ouro with `labels=None`, `exit_at_step=R-1`, `use_weighted_exit=False`, and
`use_cache=False`. It computes ordinary next-token cross entropy on final-loop
logits. This deliberately avoids the upstream `labels` path, which computes a
gate-weighted multi-exit objective rather than this fixed-loop baseline loss.

There is **full backpropagation through all native recurrent steps**: no loop
state is detached, no stop-gradient is inserted, and no model-level loop is
replayed by the wrapper. Activation checkpointing uses `use_reentrant=False`.

- `stage="alignment"`: only learned pooling and the projector train; the vision
  tower and language model remain frozen. Gradient flow through the frozen LM
  is still retained so the projector can learn.
- `stage="sft"`: all active language backbone parameters train. The vision
  encoder also trains when `freeze_vision=false`; the projector always trains.
- Unused auxiliary heads are explicitly frozen: Ouro's early-exit gate and
  SigLIP's global contrastive pooling head. They are not part of this fixed-loop
  answer loss. This avoids unused-parameter failure under ordinary DDP and is
  reflected in reported trainable counts. It is not LoRA/router-only training.

Training loads FP32 parameters and uses BF16 autocast on supported CUDA devices,
preserving FP32 Adam states. CPU tests explicitly disable BF16. The first recipe
targets single GPU or replicated DDP through HF Trainer/Accelerate; FSDP and
DeepSpeed/ZeRO are deliberately rejected until their checkpoint paths are added
and tested. H20/B300 activation memory still requires an actual pilot.

## Cache and decoding

The reference greedy decoder handles one unpadded example. It retains Ouro's
native distinct `(recurrence, physical layer)` KV entries across generated
tokens. It does **not** reuse an earlier recurrence's KV as a deeper recurrence's
KV. `--no-cache` provides a full-prefix recomputation reference. In either case
the vision encoder runs once per request, not again at every decoded token.

`--profile` records the first request using PyTorch profiler. Output timings
separate media/preprocessing (`prepare_seconds`) and model execution
(`model_seconds`), with their sum in `end_to_end_seconds` / `latency_s`.
These first-pass timings are explicitly cold/mixed, with **no warmup guarantee**.
The trace-bearing example is marked `profiled_not_benchmark`. They cannot be
presented as warmed p50/p95, isolated kernel speedup, or an official benchmark.

## Checkpoint and provenance

Every checkpoint includes language and vision weights/configuration, learned
pool/projector weights, tokenizer, image processor, `deepprobe_config.json`,
and portable `training_provenance.json`. Trainer checkpoint directories also
contain optimizer, scheduler, trainer progress and RNG state.

- `--init-checkpoint`: load weights for a new phase/run, with a new optimizer.
  It preserves the union of ancestor training and validation/tuning sources.
- `--resume`: continue the same model, stage, dataset hashes, world/global batch,
  optimizer and schedule contract. Changed contracts fail rather than silently
  restoring an incompatible optimizer. For a changed training schedule, use a
  new output directory and `--init-checkpoint` instead.
- A nonempty output directory is refused unless `--resume` was requested.
- Validation/test examples sharing recorded training source IDs are rejected.
  Held-out test examples sharing prior validation/tuning sources are rejected.
  Missing checkpoint provenance is a hard error unless the explicit
  `--allow-unverified-provenance` import override is set; such predictions and
  summaries remain visibly unverified. Separate metrics receive separate scores,
  never a combined MCQ/exact-match accuracy. Existing predictions, summary and
  trace artifact names are protected against accidental overwrite.
  Source checks cannot discover overlap hidden behind incorrect source IDs or
  prove freedom from the upstream model's original pretraining contamination.

## Validation actually performed

The CPU test suite runs meaningful small Torch models without downloading any
pretrained weights. It covers final-loop loss equivalence, gradient flow into
all active components and earlier loop states, projector-only alignment,
deterministic image/video sampling, bounded clips, answer masking, strict MCQ
parsing, cached versus uncached generation, component save/reload, and an actual
HF Trainer alignment → SFT → optimizer-resume → evaluation sequence.

The optional `DEEPPROBE_TEST_NATIVE_OURO=1` test downloads only the pinned Ouro
configuration/Python code and constructs tiny **random** weights. It tests
native shared recurrence, non-reentrant gradient checkpointing, generation
parity, distinct per-loop cache entries and local native-model save/reload,
in both eager attention and the shipping SDPA configuration.
Passing this test is not evidence that the pretrained 1.4B model has converged
or that a full distributed GPU run works.

Initial verification environment: Python 3.12, Torch 2.14.1 (macOS CPU),
Transformers 4.57.6, Accelerate 1.12.0, PyAV 16.0.1, Pillow 11.3.0,
Safetensors 0.6.2, NumPy 2.2.6. GPU server Torch/CUDA compatibility must be checked
independently; this document is not a recommendation to replace a working
GPU-specific Torch wheel with the local CPU-test version.

The pinned Ouro model card contains an older `transformers<4.56.0` warning,
while its pinned model code includes a subsequently changed custom cache API.
This release therefore records a **specific tested combination**, rather than
inferring compatibility solely from that card: tiny native Ouro contracts pass
on Transformers 4.57.6. This does not prove full-checkpoint GPU compatibility;
retain the pinned code and run the pilot before a long training job.

## Evidence links

- [Pinned Ouro model implementation](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py)
- [Pinned SigLIP configuration](https://huggingface.co/google/siglip-so400m-patch14-384/blob/9fdffc58afc957d1a03a25b10dba0329ab15c2a3/config.json)
- [Transformers Trainer documentation](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer)
