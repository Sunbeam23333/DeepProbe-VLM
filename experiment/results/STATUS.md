# Experiment status — 2026-11-11

Latest exact Job/Pod observation: **19:52 Beijing time (UTC+08:00)**. Smol
alignment attempt 2 was submitted at **19:49** and is **Pending**. The GPU
identity repair is deployed, but allocation and training have not started.
Attempt 1 failed during startup at **16:03**; its small failure archive passed
independent acceptance at **19:53:42**. These are dated observations,
not a background monitor. No new benchmark result is claimed.

## Completed

- R1 and R2 seed17: alignment and SFT completed; Video-MME899 and TempCompass
  multiple-choice evaluations and their source-cluster paired statistics were
  independently accepted. See [results and limitations](RESULTS.md).
- R4 seed17 alignment: completed 718 optimizer steps. Its full archive and final
  checkpoint passed independent acceptance.

## Smol repaired attempt submitted

SmolLM2-1.7B-Instruct seed29 alignment attempt 2 has one confirmed submission.
The submission client exited successfully, and the actual Job resource contract
passed: stream/low, one node, eight GPUs, restart policy Never and no forced
node. At **19:52**, its Pod was Pending because the scheduler could not satisfy
the combined CPU, memory and GPU request on one node. No allocated node,
running container, startup/run receipt or optimizer update was observed;
training has not yet started.

The replacement control package and every deployed file passed independent
source and readback review. It now reuses the frozen preflight's complete GPU
identity rule: each of the eight devices must have an accepted B300-family name
(the cluster's NVIDIA L20D alias is allowed), torch compute capability [10,3]
and at least 240 GiB of memory. Names alone cannot pass. The original eight-rank
BF16/attention/NCCL startup preflight remains required. Neither hardware check
has run while the new job is Pending.

Fresh CPU/source/input/storage admission and independent review passed before
this separate submission. The actual submission-side capacity projection was
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

## Remaining work

1. Accept Smol attempt-2 hardware startup, complete alignment and its archive,
   then run matched SFT and benchmark evaluations.
2. Finish R4 checkpoint recovery and storage admission, run the remaining SFT,
   then archive and evaluate the completed model.
3. Complete the additional training seeds and declared ablations, controlled
   cost/latency experiments and updated comparative statistics.

MVBench has no accepted result in this release. Its authorization/media status
was not reverified in this check. Existing benchmark timings do not constitute
a warmed latency comparison. No canceled automation was recreated.
