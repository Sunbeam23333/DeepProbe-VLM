# Experiment status — 2026-11-12

Latest training Job/Pod observation: **November 12, 20:06 Beijing time
(UTC+08:00)**. Smol alignment attempt 2 is **Failed**, with its failure transition
at **November 11, 21:34:31**. Hardware startup passed and training reached
317/718 updates in the captured log; checkpoint-300 is preserved but full
recovery has not been validated. A complete exact OSS-prefix listing at
**20:08:15 on November 12** found zero objects. These are dated observations,
not a background monitor. No new benchmark result is claimed.

## Completed

- R1 and R2 seed17: alignment and SFT completed; Video-MME899 and TempCompass
  multiple-choice evaluations and their source-cluster paired statistics were
  independently accepted. See [results and limitations](RESULTS.md).
- R4 seed17 alignment: completed 718 optimizer steps. Its full archive and final
  checkpoint passed independent acceptance.

## Smol attempt-2 training interruption

SmolLM2-1.7B-Instruct seed29 alignment attempt 2 was submitted once on
November 11 at **19:49**, using stream/low, one node, eight GPUs, restart policy
Never and no forced node. The **19:52 Pending** observation is retained as
history. The Job subsequently transitioned to Running at **20:16:13**, then
Failed at **21:34:31**. Its original Pod was absent at the November 12 readback,
so the original container exit code and termination source remain unknown.

The deployed hardware identity repair passed on all eight NVIDIA L20D devices:
each reported compute capability [10,3] and 267.69 GiB of memory. The separate
eight-rank BF16 attention forward/backward and NCCL all-reduce preflight also
passed. These validate startup on the allocated devices; they do not establish
training completion or CUDA performance.

The log reaches **317/718 optimizer updates**. The preserved checkpoint
directory and its Trainer state both identify **step 300**, leaving **418
updates** if recovery is accepted. Its complete weight/optimizer/RNG payload
has not been validated, so its presence is not a resumability guarantee. No
final checkpoint or training-completion receipt was observed. Captured NCCL
environment deprecation warnings do not establish the failure cause.

The exact run archive prefix was completely listed once and contained **0
objects / 0 bytes**; no local archive manifest or success receipt was present.
Failure archival, full recovery validation and fresh submission admission
remain outstanding. No retry was submitted in this observation.

Fresh CPU/source/input/storage admission and independent review passed before
the November 11 submission. Its submission-side capacity projection was
**95,109,611,082 bytes**, below the 99,000,000,000-byte limit. The submitted
control and source/input binding were read back with the exact Job/Pod evidence.

The scientific recipe remains F8/G4/R1 and global batch 128, with 91,826 train
and 4,472 dev QA and a target of 718 optimizer updates. No Smol training
completion or held-out result has been accepted.

## Smol attempt-1 failure retained

Attempt 1 was submitted once at 14:36. Its container started at **16:01:50**,
exited with code **1** at **16:03:33**, and the Job became Failed at
**16:03:35**, with zero container restarts. The **19:26** exact Job/Pod
readback confirmed those terminal facts.

The inventory contained eight NVIDIA L20D devices, but an outer name-only
check rejected them before actual torch capability/memory validation and the
training runner. This was inconsistent with the already-frozen identity rule.
No optimizer update or checkpoint was produced. The exact allocated device
capabilities were not measured by attempt 1; its names cannot establish that
hardware validation passed.

At **19:53:42**, independent acceptance verified its complete startup-failure
archive: **78 objects / 6,055,444 bytes**, including 76 payload objects /
5,968,562 bytes. Every object was fully read back and SHA256-checked; the exact
object inventory remained stable, and the source copies were preserved and
reverified. This archive contains failure evidence, not a training checkpoint.
The earlier **19:28** empty-prefix observation and **14:38 Pending** state remain
in the status record as history. Attempt 2 has a separate identity and does not
overwrite or resubmit attempt 1.

## R4 SFT interruption and recovery

R4 SFT attempt 1 failed on **November 9 at 17:04:27 Beijing time**. Its log last
reached **602/718 updates**, while the preserved durable checkpoint is at step
600. SIGTERM was observed, but the original process exit code and termination
source remain unknown. There is no final checkpoint or R4 held-out score.

On November 11 at **14:38**, independent readback accepted the complete failure
archive: **147 objects, 23,234,938,760 bytes**, including 145 payload objects.
Every object was fully read back and SHA256-checked. The original run and all
31 checkpoint files (22,317,153,372 bytes) were preserved and reverified.
This completes failure archival; training and recovery acceptance remain open.

Model headers and optimizer/scheduler/eight-rank RNG presence checks previously
passed. Actual optimizer/RNG deserialization and a GPU resume have not been
accepted. Recovery from step 600 requires 118 further updates. The November 11
capacity analysis retains the checkpoint and projects 122,687,970,220 bytes,
above the 99,000,000,000-byte limit, so the R4 recovery job remains gated.

The sanitized [status record](status.json) preserves the dated R4 failure facts
and records Smol separately, with hashes linking each claim to private evidence.
No new benchmark results are claimed by these training and archive updates.

## CUDA engineering validation

The opt-in inference changes and recovery guard are committed as
`bec405e8802232cf4815f1c6dfea968318a12aa9`. The public CPU suite passed 512 tests
(80 skipped), and three separate pinned native Ouro CPU tests passed.

The synthetic pooling QA job was created at **20:46:10 on November 12**.
At **20:47:41**, its Pod was **Pending**: the combined single-node CPU, memory
and eight-GPU request could not yet be placed. Its actual contract is
stream/low, one node, eight GPUs and restart Never. No GPU startup, CUDA
numerical result, microbenchmark or archive had been produced. It measures one
device if allocated; no model/training speedup or held-out claim follows.

The initial submission client failed before starting `kubectl` because its
PATH lacked that executable. Complete and exact-name reads confirmed no job
existed. A separate reviewed repair used the absolute executable and a single
`create` of the original validated YAML, retaining the original intent and
failure. The actual Job UID and resource contract were then read back.
See [optimization scope](../docs/OPTIMIZATION_STATUS.md) and the evidence hashes
in [status.json](status.json). These are dated facts, not a background monitor.

## Remaining work

1. Archive Smol attempt-2 failure evidence and validate checkpoint-300 recovery;
   after fresh admission finish the remaining 418 alignment updates, then run
   matched SFT and benchmark evaluations.
2. Finish R4 checkpoint recovery and storage admission, run the remaining SFT,
   then archive and evaluate the completed model.
3. Complete the additional training seeds and declared ablations, controlled
   cost/latency experiments and updated comparative statistics.

MVBench has no accepted result in this release. Its authorization/media status
was not reverified in this check. Existing benchmark timings do not constitute
a warmed latency comparison. No canceled automation was recreated.
