---
license: Apache License 2.0
---

# DeepProbe-VLM experiment results

Verified results and reproducibility records for DeepProbe-VLM. The repository visibility is unchanged.

## R4 SFT accepted — November 21, 21:28 Beijing time

R4 seed17 SFT completed all **718 optimizer steps** and was accepted at
**21:08:16 Beijing time**, after resuming checkpoint 600 and replaying 118 updates.
Native training exited 0. The native checkpoint is now durable in OSS:
**31 payload files / 22,317,165,602 bytes**, plus MANIFEST and READY (33 objects).
The export receipt confirms full GET/SHA256 verification of every payload;
this follow-up independently read back both metadata objects and checked the
exact object names and sizes. The parent-process cache fix passed the actual
checkpoint export path.

Accepted totals are now **7/34 training stages, 3/18 complete two-stage models,
4/36 benchmark evaluations, 0/15 full-model latency runs and 0/12 comparisons**.
The full SFT stage archive is still running on the original worker. Its archive
acceptance and downstream evaluation remain pending; no R4 benchmark score is
claimed. The suite is incomplete. The November 19 scored release is unchanged.

See [training acceptance evidence](https://modelscope.ai/datasets/JasonJin/DeepProbe-VLM/files?branch=master&path=progress%2F2026-11-21-r4-sft-accepted.json).

## Recovery update — November 21, 19:52 Beijing time

**a10 completed the 718-step training calculation and exited 0**, with formal
completion and process-quiescence records retained. Its outer checkpoint export
then failed at **19:09:48**: importing Transformers triggered a numeric-UID
passwd lookup because only the training child had an explicit Inductor cache.
No native checkpoint 718 payload reached OSS. A read-only CPU probe on the exact
original node found the original private run directory unavailable. This attempt
is therefore **not accepted as a recoverable completed model**.

Fix `9beeee6` binds the cache in the action process as well. 91 tests passed; an
actual production-runtime parent-import regression failed before the fix and
passed after it. The failed attempt's 108 metadata files plus archive manifest
were verified by full GET/SHA. Its original Failed Pod record is retained; only
the missing-status controller was retired with orphan propagation, then its
48 GB reservation was released. No checkpoint files were deleted by recovery.

Replacement **a11 is Running and passed the eight-GPU gate**. It restores the
accepted checkpoint 600 and must repeat 118 updates to reconstruct the unavailable
weights. The full 185-task plan and accepted scientific recipes remain unchanged.
No new accepted model, benchmark score or full-model latency result is claimed.
Admission accounts for 120.56 GB including reservations and safety, below 150 GB.
F4/F8 caches remain READY; F16 is deferred. Scientific release totals remain
6 accepted training stages, 2 complete models, 4 evaluations and 0 full-model
latency runs. The November 19 scored release is unchanged.

See [recovery evidence](https://modelscope.ai/datasets/JasonJin/DeepProbe-VLM/files?branch=master&path=progress%2F2026-11-21-recovery-a11.json).

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
