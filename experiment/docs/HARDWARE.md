# H20 and B300 execution gates

These are launch templates, **not measured support/performance claims**. No H20 or
B300 was connected during the initial local release. Do not reserve a long run
until the gates below pass on your machine.

All commands run from `experiment/`. Build the optional container from that
directory with `docker build -f docker/Dockerfile -t deepprobe-vlm:local .`;
the repository root is not its Docker build context.

## Environment

Use Linux + a CUDA-enabled PyTorch >=2.8 and Transformers 4.57.6. Keep PyTorch
provided by the GPU server/container; install `requirements/train.txt`, then
`python -m pip install --no-deps -e .`. Do not install a CPU-only torch wheel or
blindly upgrade the provider's CUDA/driver stack.

`docker/Dockerfile` starts from NVIDIA's documented `25.09-py3` image (CUDA 13.0.1,
PyTorch 2.9 prerelease). It is an optional reproducible starting point, not a claim
that its build was run here. Check the host driver against NVIDIA's compatibility
guide and record the resolved image digest. If the container's pip constraints
reject a dependency, keep the error and resolve versions in a separate tested
environment rather than removing constraints in the system installation.

B300 is compute capability **10.3**, not consumer Blackwell 12.0. Query the
actual GPU; do not hard-code B200/RTX architecture flags for B300. This release
does not build custom CUDA extensions. H20's actual reported capability, GPU
count, driver, torch CUDA runtime and supported architectures are logged by the
preflight. Successful matrix multiplication alone does not certify the entire
model path, so the next gates test attention and the real model.

Sources, checked 2026-10-07:
- https://developer.nvidia.com/cuda/gpus
- https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-09.html
- https://docs.nvidia.com/deploy/cuda-compatibility/
- https://pytorch.org/blog/pytorch-2-7/

## Run in this order

1. Hardware kernel gate:
   `python scripts/preflight.py --expected-gpu H20 --output runs/preflight.json`
   (replace H20 with B300 when appropriate).
2. Communication gate:
   `torchrun --standalone --nproc-per-node=8 scripts/preflight.py --output runs/preflight_8gpu.json`
3. Unit tests: `python -m pytest -q`. These use tiny local fixtures, not benchmark
   scores and not a substitute for real checkpoint validation.
4. Prepare source-disjoint train/validation manifests and run the 20-step SFT
   connectivity pilot. It starts with a random connector and is not a quality
   result. Use one GPU first (`DEEP_PROBE_GPUS=1`). For actual quality training,
   complete projector alignment, then initialize SFT from that checkpoint.
5. Confirm finite loss, inspect its trend, and check all intended parameter groups receive finite
   gradients and checkpoint reload works. Generation/inference prompts must not
   contain gold answers; teacher-forced validation loss uses masked answer supervision.
6. Run the same pilot on eight GPUs. Check effective batch size and resume state.
7. Profile a fixed validation subset on **both** GPU types before launching the
   full experiment matrix.

The first 8-GPU SFT preset uses microbatch 1 x accumulation 16 x 8 GPUs = global
batch 128. On one GPU the same configuration is global batch 16; for a strict
training comparison set accumulation to preserve global batch explicitly.
H20/B300 launchers intentionally use the same frame/token/loop workload. Faster
hardware does not justify changing evaluation frames only for one model.

## What is and is not optimized

Baseline: native shared decoder layers, independent per-loop KV, PyTorch SDPA,
BF16, activation checkpointing, and distributed data parallel training.

Not yet implemented: visual-token early halting, cross-loop KV sharing, custom
CUDA/PTX kernels, FP8/FP4 training, and architecture-specific CUTLASS kernels.
The CUDA figure marks these as future work. A lower-precision or cache-sharing
variant requires output/gradient parity or an explicitly measured quality loss.

Measure video decode/vision encoding/prefill/generation separately where possible,
as well as total latency and allocated/reserved peak memory. Preserve warmup,
synchronization, batch size, frames, sequence lengths, output length and software
versions. Never infer a B300/H20 speed ratio from peak TFLOPS.

For a pilot with K optimizer steps, S measured seconds per step and E evaluation
seconds, estimate `wall_hours = (K*S + E)/3600`, then add measured data/checkpoint
overhead. The repository does not advertise an unmeasured training ETA.
