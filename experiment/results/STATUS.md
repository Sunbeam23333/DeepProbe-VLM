# Experiment status — 2026-11-11

Latest observation: **14:38 Beijing time (UTC+08:00)**. The Smol baseline was
submitted at 14:36 and is **Pending**. R4 SFT failure archival passed independent
acceptance. These are dated observations, not a background monitor.

## Completed

- R1 and R2 seed17: alignment and SFT completed; Video-MME899 and TempCompass
  multiple-choice evaluations and their source-cluster paired statistics were
  independently accepted. See [results and limitations](RESULTS.md).
- R4 seed17 alignment: completed 718 optimizer steps. Its full archive and final
  checkpoint passed independent acceptance.

## Smol baseline submitted

SmolLM2-1.7B-Instruct seed29 alignment has one confirmed submitted GPU job. At
14:38, its Pod was Pending because the scheduler could not satisfy the combined
CPU, memory and GPU request on one node. It occupies the sole project GPU job
slot. No node, running container, GPU startup receipt, eight-rank preflight or
optimizer update had been observed; training has not yet started.

The baseline uses F8/G4/R1 and global batch 128, with 91,826 train and 4,472 dev
QA; the target is 718 optimizer updates. Its fresh CPU/source/input/storage
admission and actual Job contract passed. The submission-time storage projection
was 95,095,371,398 bytes, below the 99,000,000,000-byte project limit.
The resource contract is stream/low, one node, eight GPUs and restart policy
Never; GPU startup additionally requires eight B300 devices. That hardware check
has not run while the job is Pending. No Smol held-out result is available.

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

1. Complete and accept Smol alignment, then its matched SFT and benchmark runs.
2. Finish R4 checkpoint recovery and storage admission, run the remaining SFT,
   then archive and evaluate the completed model.
3. Complete the additional training seeds and declared ablations, controlled
   cost/latency experiments and updated comparative statistics.

MVBench has no accepted result in this release. Its authorization/media status
was not reverified in this check. Existing benchmark timings do not constitute
a warmed latency comparison. No canceled automation was recreated.
