# Experiment status — 2026-11-09

Last remote checks: **19:09–19:18 Beijing time (UTC+08:00)**. This is a dated
observation, not a background monitor.

## Completed

- R1 and R2 seed17: alignment and SFT completed; Video-MME899 and TempCompass
  multiple-choice evaluations and their source-cluster paired statistics were
  independently accepted. See [results and limitations](RESULTS.md).
- R4 seed17 alignment: completed 718 optimizer steps. Its full archive and final
  checkpoint passed independent acceptance.

## R4 SFT interruption

R4 SFT attempt 1 changed to **Failed at 17:04:27 Beijing time**. The training log
last reached **602/718 updates** and recorded **SIGTERM at 17:04:23**. A durable
`checkpoint-600/trainer_state.json` reports step 600 of 718. The failed attempt
has no final checkpoint or training-complete receipt; there is no R4 held-out
score.

The original Pod is no longer present. Exact Job/Pod UID event queries returned
no events, so the source of SIGTERM and the actual process exit code remain
unknown. This observation does not establish scheduler preemption. The exact
SFT archive prefix was empty (0 objects) at the check; failure archival and
recovery acceptance are still pending. No retry was submitted during this check.

At 19:18, all 31 checkpoint files (22,317,153,372 bytes) were SHA256-hashed on
the server, with stable file-tree checks and matching model/training contracts.
The model headers and optimizer/scheduler/eight-rank RNG presence checks passed.
Optimizer/RNG deserialization and an actual GPU resume were not performed, so
this is structural verification rather than full recovery acceptance.

The two project roots account for 67,803,479,292 bytes. Reusing the previous
54,884,490,928-byte SFT reserve projects 122,687,970,220 bytes, above the
99,000,000,000-byte project limit. This reserve has not yet been recomputed for
a new recovery attempt; it is a conservative planning check, not final admission.

The sanitized [status record](status.json) binds these observations to the
original private evidence hashes. Historical Running observations are retained
as history and do not override this later failed state.

## Remaining sequence

1. Archive and verify the interrupted R4 SFT attempt, validate the complete
   step-600 checkpoint and recovery contract, then finish the remaining SFT.
   Capacity, exact destination prefix, absence of duplicate work, fresh CPU
   admission and the resource contract must pass before a new GPU job.
2. Run the prepared Smol baseline and evaluate completed new models under the
   same accepted protocols.
3. Complete the additional training seeds and declared ablations, then the
   controlled cost/latency experiments and updated comparative statistics.

MVBench has no accepted result in this release. Its current authorization/media
status was not reverified in this check. Existing benchmark timings do not
constitute a warmed latency comparison. No canceled automation was recreated.
