# DeepProbe-VLM terminal and archive update — 2026-11-21

This is a dated records update, compiled from already collected local evidence.
It supersedes the current-status interpretation of progress-v6, while retaining
that sealed Running snapshot unchanged. It contains no new training, evaluation,
or CUDA measurements, and makes no replacement-worker launch claim.

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

The [progress-v7 package](../2026-11-21-progress-v7/README.md) records these
separate observation times and opaque evidence hashes. The sealed
[progress-v6 package](../2026-11-21-progress-v6/README.md) remains an unchanged
00:38:25 UTC Running snapshot and retains the previously reported synthetic
pooling measurements (BF16 2.5714×, FP32 2.4768×). They do not establish full-model
or training speedup. No new held-out score or operator measurement is added here.
The accepted [November 19 results](../2026-11-19/README.md) are unchanged.

[progress.json](progress.json) records exact byte counts, separate observation
times and SHA256 references. The private source records are not distributed;
their hashes do not imply public access. Media, weights, private identifiers,
cluster snapshots and credentials are excluded.
