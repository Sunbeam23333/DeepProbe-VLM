---
license: Apache License 2.0
---

# DeepProbe-VLM experiment results

Verified results and reproducibility records for DeepProbe-VLM. The repository visibility is unchanged.

## Recovery update — November 21, 17:11 Beijing time

Replacement **a9** was submitted once at 17:03:10 and entered Running at
17:03:15. The exact Job/Pod identities, startup and eight-GPU gate are verified.
The accepted historical R4 alignment checkpoint has been restored; its archive
is in progress. **No new optimizer update or scientific completion is claimed.**
The next training operation resumes R4 SFT checkpoint600 to step718.

Code `d5d6e3d` fixes the reproduced numeric-UID TorchInductor cache import error
without changing the scientific recipe or legacy training source. 112 tests
and 7 subtests passed, including actual frozen-source cache compatibility.
Both F4 and F8 benchmark shapes passed READY checks; F16 preparation continues.
The original a6 training exception is still unconfirmed. Accepted result totals
and the November 19 scientific release are unchanged.

See [recovery evidence](https://modelscope.ai/datasets/JasonJin/DeepProbe-VLM/files?branch=master&path=progress%2F2026-11-21-recovery-a9.json).

## Latest release

[2026-11-19 result package](https://modelscope.ai/datasets/JasonJin/DeepProbe-VLM/files?branch=master&path=releases%2F2026-11-19) · [GitHub source and experiment scripts](https://github.com/Sunbeam23333/DeepProbe-VLM)

The release contains four scored prediction sets (8,554 rows), paired source-cluster statistics, 400 raw CUDA operator timing records, numerical checks, exact SHA256 manifests, and a separate progress snapshot.

- **Video-MME 899-video subset**: R1 6.3404%, R2 18.4650%; paired difference +12.1246 percentage points, 95% CI [10.6044, 13.5716]. Invalid answers remain in the denominator. This is not the official full 900-video result.
- **TempCompass**: R1 42.5316%, R2 50.4430%; paired difference +7.9114 percentage points, 95% CI [5.1876, 10.6785]. The accepted official `disable_llm` multiple-choice rules are used.
- **Synthetic weighted pooling**: reference/Triton median CUDA-event ratios are 2.4096× for BF16 and 2.2835× for FP32 at shape [8,729,1152], G4. These measurements do not establish full-model or training speedup.

The accepted quality comparison uses training seed17 only. Bootstrap intervals represent test-source sampling uncertainty, not training-seed robustness. Overall intervals were independently recomputed; subgroup intervals were not part of that independent recomputation. The high Video-MME invalid-response rates limit the absolute results.

The `releases/2026-11-19/README.md` file explains the artifact layout and limitations. `PUBLICATION_MANIFEST.json` pins all published payloads, `SOURCE_MAP.json` maps derivatives to original evidence digests, and `VALIDATION.json` records count and metric checks. Private model filesystem names in Video-MME predictions were replaced with public experiment identifiers. No scores were changed.

R4 SFT, SmolLM2, the remaining seeds and ablations, and formal full-model latency measurements remain incomplete in this snapshot. The current small result release is not a backup of model weights. Future fully verified model backups, if present, use a separate `artifacts/` prefix.

## Historical a6 terminal snapshot

The isolated **a6 suite worker failed**: its container exited 1 at
**08:59:43 Beijing time on November 21** (00:59:43 UTC), and the Job's Failed
condition was recorded two seconds later. The exact Job/Pod identities match
the single submission, with zero restarts. The terminal observation ended at
09:01:02 Beijing time. **3/185 operational tasks** completed: the GPU engineering
gate, restoration of accepted historical R4 seed17 alignment, and its archive.
This is not fresh alignment training or a scientific completion percentage.
The R4 SFT task failed; no new optimizer update or scientific completion was
verified. The original training exception remains unconfirmed because its
private training stderr was not recovered. This publication uses dated stored
evidence and does not claim a replacement worker has started.

Separate terminal-metadata preservation accepted **458 objects / 1,946,568 bytes**
by full GET/SHA verification. The restored historical alignment archive retains
**23 objects / 7,482,820,366 bytes**. At **09:23:36 Beijing time**, only the old
**48,000,000,000-byte reservation** transitioned to RELEASED. The other 42
reservations, one artifact entry, all 91 prior events and three active CPU
reservation owners were preserved. The old namespace and checkpoint files were
not deleted. This releases an accounting reservation, not 48 GB of disk files.
The independent local audit passed 30 checks and rehashed 17 source artifacts.

The **09:24:53** capacity snapshot recorded **70,172,344,228 bytes** of actual
project home use and a **72,442,876,836-byte** projection including active
reservations and safety, under the 150,000,000,000-byte limit. It precedes the
9,504-byte retirement receipt and is not current capacity or new admission.
OSS has no configured capacity limit. A CPU reservation still being ACTIVE at
retirement does not establish that its process is alive now.

Public diagnostic-retention code is synchronized at `1b3af01`; its 39 scoped
tests passed. It preserves bounded failure records (768 KiB payload plus a
32 KiB report per attempt). These tests establish diagnostic handling, not
successful training execution or the cause of the original a6 failure.

Accepted scientific totals remain **6 training stages, 2 complete two-stage
models, 4 model/benchmark evaluations and 0 full-model latency runs**. Smol
alignment a6 completed 718 steps and its independent archive recovery/history
import was accepted; its original archive-phase Job failure is preserved.
The full plan retains 18 models, 34 training stages, 36 evaluations, 15 latency
runs, 12 comparisons and 185 operational tasks. Remaining work is **27 fresh
training stages plus R4's 118 updates, 32 new evaluations, 15 latency runs and
12 comparisons**. Checkpoint600 archive/native metadata acceptance does not
prove runtime optimizer, scheduler and RNG restoration or the remaining updates.

At **09:02:04 Beijing time**, Video-MME899 F4 cache preparation had **677 metadata
entries, 2,708 PNGs and 3,386 objects / 1,072,916,095 bytes**. No complete marker,
READY or PREPARED_ASSET receipt was present. The other three F4/F16 shapes had
zero observed objects and no READY. Both original F8 caches have accepted READY
records. These dated counts are not accepted clips or completed questions;
each evaluation requires its own verified READY receipt.

The [progress-v7 package](https://github.com/Sunbeam23333/DeepProbe-VLM/blob/main/experiment/results/releases/2026-11-21-progress-v7/README.md) records these
separate observation times and opaque evidence hashes. The sealed
[progress-v6 package](https://github.com/Sunbeam23333/DeepProbe-VLM/blob/main/experiment/results/releases/2026-11-21-progress-v6/README.md) remains an unchanged
00:38:25 UTC Running snapshot and retains the previously reported synthetic
pooling measurements (BF16 2.5714×, FP32 2.4768×). They do not establish full-model
or training speedup. No new held-out score or operator measurement is added here.
The accepted [November 19 results](https://github.com/Sunbeam23333/DeepProbe-VLM/blob/main/experiment/results/releases/2026-11-19/README.md) are unchanged.

Benchmark media is not redistributed here. Access and reuse of benchmark assets remain subject to their upstream licenses and terms.
