# Experiment status — 2026-11-21

Smol attempt 6 **completed alignment training at 718/718 optimizer steps**,
adding all 368 remaining updates from checkpoint350. The training child exited 0
at about **11:25:46 Beijing time** after 100 min 15 s; both preflights also exited 0.
Formal and stage completion receipts, Trainer state and the saved final/cp718
inventories agree on completion. This completes alignment training only.

The enclosing Job **Failed at 11:27:04 Beijing time**, with container exit1,
after the controller could not read `/proc/1687474/environ` during archival.
The archive child received SIGTERM; external preemption is not established.
The original archive manifest lists 239 payloads / 18,149,819,512 bytes. The exact
prefix at the November 20 14:57 UTC readback contained 82 objects /
916,219,811 bytes, leaving 157 payloads / 17,233,599,701 bytes absent.
There is no `ARCHIVED.json` or independent archive
acceptance. Final 18 files / 8,589,657,473 bytes and cp718 29 files / 8,642,280,705 bytes
remained locally; names and lengths matched the manifest, with fresh full hashes
still pending in that observation. The later recovery precheck passed at
**01:18:26 Beijing time on November 21**: all 240 local files passed SHA checks,
and all 82 existing objects passed full GET/SHA. The supplement-upload and
independent-audit client started at **01:19:54** and is running, with no client,
execute or audit completion receipt yet. Completion receipts and root acceptance
remain required; client startup does not establish a completed upload.
The original Job remains Failed, and no native `ARCHIVED.json` is fabricated.
**Do not repeat alignment training.** Complete and verify its
archive before admitting Smol SFT and evaluations. No new held-out result exists.
Current source identities and evidence hashes are recorded in [status.json](status.json).

The full **185-task** source is synchronized at `4034666`; **190 scoped CPU
tests** passed, including synthetic full-DAG execution with 1–4 claimants.
The synthetic completions are not experiment results. Checkpoint-chain
backpressure covers 17 components, permits at most one active training claim,
waits for the previous stage's archive and holds a component until all its
consumer/cleanup barriers finish. The source-v4/control-v8 candidate is locally
frozen and has not been deployed or submitted to the GPU queue. The current
plan is **one normal-priority worker, one node and eight GPUs** under the
shared 150 GB home budget; fresh capacity and submission admission remain required.

At **01:13 Beijing time on November 21**, independently checked cleanup removed
six already archived old checkpoint weight files, releasing **17,116,975,648 bytes**.
All a6 files were preserved and its old 42 GB reservation was released.
The accounted home roots measured **62,504,292,995 bytes**; all other active
reservations were retained. This dated snapshot is not GPU admission.

The recovery precheck's **01:18:26** workload snapshot found all **five project
Jobs and three Pods terminal**. It does not measure available GPU nodes or
authorize a new submission.

The following attempt 5 paragraph is retained as history, superseded by the
a6 training completion and separate archive failure above.

Smol attempt 5 **Failed on November 18 at 02:37:32 Beijing time (UTC+08:00)**
after actually resuming from checkpoint300 and reaching logged step
**354/718**, or **54 new optimizer updates**. Its actual GPU submission
handshake, eight-device identity checks, eight-rank BF16/NCCL checks and
training preflight passed. The controller records interruption by SIGTERM;
the signal source is unconfirmed. The November 19 **18:29:22** exact readback
found its original Pod absent and all **six project Jobs and three Pods
terminal**, with no active or queued project workload at that time. Checkpoint350's complete
29-file payload passed full SHA and native CPU metadata/contract checks; it was
a candidate for the remaining **368 updates** at that observation. Its later
authenticated continuation is a6, which completed those updates. Attempt 5
itself has no final model, training-completion receipt or new held-out result.

CUDA pooling QA attempt 3 **Succeeded on November 14 at 13:51:28 Beijing
time (UTC+08:00)**, with container exit code 0 and no restarts. The November
17 readback confirms that all eight ranks passed BF16 attention
forward/backward and NCCL all-reduce. The pooling test suite passed **13 tests,
including one explicit GPU numerical test**; both BF16 and FP32 operator
microbenchmarks completed with real Triton kernel events recorded.

For this single-device synthetic weighted-pooling workload, the ratios of
reference to Triton median CUDA-event time were **2.41× for BF16** and
**2.28× for FP32**. These measure the pooling operator only. They do not measure
full-model inference, training speed or held-out quality. The complete
**56-object / 2,518,835-byte** archive passed independent full GET/SHA checks,
and the timing statistics were recomputed from every recorded call.

The actual submission and atomically published READY receipt passed independent
review. Earlier attempt-3 and attempt-4 failure archives are independently
accepted and remain distinct from this new run; original source/checkpoint
files were preserved. The historical attempt 5 continuation did not complete alignment or establish
numerical resume equivalence; the newer a6 training completion is recorded above.

The full experiment suite remains unsubmitted. R4 SFT still needs its final 118
updates from the authenticated cp600 archive binding. Smol SFT and new-model
evaluation remain pending; the a6 alignment result is not a complete two-stage
model or a held-out quality result.

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
updates** if recovery is accepted. The later November 17 verification accepted
all 29 checkpoint files by content hash. Actual optimizer/RNG deserialization
and GPU recovery remain separate runtime checks. No final checkpoint or
training-completion receipt was observed. Captured NCCL
environment deprecation warnings do not establish the failure cause.

The earlier exact-prefix observations contained **0 objects / 0 bytes** and no
archive receipts. Those dated snapshots remain in the history. A separate
one-time failure archive was completed and independently accepted on November
17; it does not change the original failed Job or establish training success.

Fresh CPU/source/input/storage admission and independent review passed before
the November 11 submission. Its submission-side capacity projection was
**95,109,611,082 bytes**, below the 99,000,000,000-byte limit. The submitted
control and source/input binding were read back with the exact Job/Pod evidence.

The scientific recipe remains F8/G4/R1 and global batch 128, with 91,826 train
and 4,472 dev QA and a target of 718 optimizer updates. Attempt 2 did not
complete training or produce a held-out result; a6 completion is recorded above.

The initial November 17 readback found the same failed Job UID, log step 317
and checkpoint step 300. A subsequent complete inventory freshly hashed all
29 checkpoint files / **8,642,273,132 bytes**. The failure archive contains
**268 objects / 9,561,086,447 bytes**, including 266 payload objects /
9,560,833,055 bytes. Every object passed independent full GET/SHA verification,
with a stable exact object set and unchanged original run and control files.
The original checkpoint was neither copied for recovery nor deleted.

## Smol attempt-3 failed after GPU checks, before formal training

The new recovery control is configured to resume the original step-300 checkpoint with the
same scientific recipe and final-step selection. CPU admission v1 stopped
because the unchanged native safetensors PT metadata reader imports torch,
which the earlier no-framework guard rejected. No GPU Job was created.

Control v2 permits this specific CPU metadata import while still rejecting
model/training imports and CUDA initialization. The metadata checks did not
materialize tensors or deserialize optimizer/RNG state. A real Linux probe
and independent source review passed; the frozen scientific validator and
training source remain unchanged.
Fresh source/data/checkpoint/capacity/live checks completed at **12:29:02**,
and independent admission review passed at **12:31:57**. The two roots total
**48,594,794,547 bytes**; the additional 27,284,546,264-byte reservation yields
**75,879,340,811 bytes**, without crediting future cleanup or copying the
checkpoint.

The new exact OSS prefix was empty, all five project Jobs and three Pods were
terminal, and client/server dry-runs verified stream/low, one node, eight GPUs,
restart policy Never and no forced node. The server dry-run object was **not
created**; its temporary UID is not an actual Job identity.

After fresh submission checks, attempt 3 was submitted once at **12:36:14** as
`jinhaopeng-dpvlm-p1-smollm2-s29-align-a3-n1-bdff2`, UID
`ab061a99-000f-4283-b806-249ea5287fdf`. The client exited 0, and the actual Job
contract matches stream/low, one node, eight GPUs, restartNever and no forced
node. The final submission-side capacity projection was **75,879,399,133
bytes**, still below 99 GB. This actual Job is distinct from the earlier
dry-run name ending in `792fb`.

At **12:37:10**, the Job had `Created=True` and its exact Pod was
**Pending/Unschedulable** because matching nodes lacked sufficient CPU, memory
and GPU capacity. That project inventory contained this single nonterminal Job
and Pod; all other project workloads were terminal. At that observation there
was no allocated node, container, startup-admission or completion receipt.
The later **12:44:59 Pending** exact readback is also retained as history.

The **23:16:12** readback verifies the same exact Job UID and its subsequent
Running transition at **14:20:58**, followed by **Failed at 14:25:50**. Its
original Pod is now NotFound, so the original container exit code and the
initiating source of termination are unavailable from this snapshot. All six
project Jobs and three remaining Pods are terminal.

`STARTUP_ADMISSION.json` binds fresh source/input, storage and live workload
checks, plus eight distinct NVIDIA L20D devices with compute capability 10.3
and 267.69 GiB each. The run's `preflight-before.json` records
`gpu_gate_passed`, world size eight, and passing BF16 attention
forward/backward and NCCL all-reduce for every rank. These checks completed,
but their enclosing preflight process did not exit normally: the complete log
ends with `END preflight-before: code=None, elapsed=58.5s` and signal-15
tracebacks. The controller's failure receipt records SIGTERM. Its source is
unconfirmed; SIGTERM alone does not establish external preemption.

The configured preflight timeout was 600 seconds and the outer stage budget
194,400 seconds. The observed failure is not a recorded timeout or capacity
guard failure. No `START formal-train` marker, optimizer recovery, new update
or training-completion receipt is observed. The earlier CPU admission remains
accepted as a historical gate, and the accepted recovery source remains
checkpoint 300, with **418 alignment updates** still to run before matched SFT
and evaluation. Attempt-3 failure archival and any later recovery require
their own evidence; the accepted attempt-2 archive does not cover this run.

At **23:18:48**, successful event queries for the exact Job and original Pod
UIDs returned no retained events. The exact attempt-3 OSS run prefix contained
**0 objects / 0 bytes**, so its failure archive was incomplete at that observation. The local
startup directory contains **36 files / 6,567,659 bytes** and the run contains
**200 files / 1,297,930,247 bytes**, mostly copied inputs and caches; the full
inventory contains no training-directory files. Source checkpoint 300 still
contains **29 files / 8,642,273,132 bytes**. This later check measured its file
count and sizes without rehashing the large payloads; it does not replace the
earlier accepted full-byte binding. Small terminal evidence files were read
back completely. Neither the original attempt-2 archive nor its acceptance
was repeated or replaced.

## Smol attempt-4 submitted, then failed before GPU checks

On November 18 at **00:40:07 Beijing time**, the separate attempt-3 failure
archive passed independent acceptance: **337 objects / 940,846,953 bytes**,
including 335 payload objects / 940,509,935 bytes. Every archived object passed
full GET/SHA verification against local sources and the exact stable listing.
Its original run/control/admission records were preserved. The archive excludes
18 regenerable CUDA-cache files / 380,794,397 bytes; those originals remain
local. The already accepted attempt-2 archive and its checkpoint were not
archived again. This completes failure archival, not training or recovery.

Attempt-4 control is sealed to resume the same accepted attempt-2 checkpoint
300 to step 718. The manifest binds 37 payload files / 819,382 bytes, SHA256
`93c962a60465ff72952c2592ab7ddb9bbd33eec9792eb650ca85ed8343e881e8`.
Its scientific recipe, native optimizer, scheduler and eight-rank RNG recovery
requirements remain unchanged. Fresh CPU/live admission completed at
**00:47:33**, and independent review accepted all 32 collected admission files
at **00:48:09**. The two roots contain **49,916,338,246 bytes**; the additional
**27,284,546,264-byte** reserve gives a **77,200,884,510-byte** projection, below
99 GB without crediting future cleanup. The exact new OSS prefix is empty,
all six project Jobs and three Pods were terminal, and client/server dry-runs
verified stream/low, one node, eight GPUs, restart Never and no forced node.
The server dry-run object was not created; its name ending `8abcf` is not an
actual Job identity.

The one-shot submit client exited0 and confirmed the actual Job
`jinhaopeng-dpvlm-p1-smollm2-s29-align-a4-n1-0aece`, UID
`01f3dce0-998d-47c3-95c0-98d7c5c833d2`, created at **00:52:31**. The actual
contract matches stream/low, one node, eight GPUs, restart Never and no forced
node. Submission freshly rechecked a **77,200,944,069-byte** capacity projection
and bound its durable intent before the single create.

The Job became Running at **00:52:59** and Failed at **00:53:02**. Its container
ran from **00:52:58 to 00:53:00**, exit code1, with zero restarts. The **00:53:30**
readback verifies both exact Job and Pod UIDs. Startup recorded a
`FileNotFoundError` when reading the exact attempt-4 `SUBMISSION.json` receipt;
that file is present in the later submission collection. The failure occurred
before startup admission, hardware validation or eight-rank preflight. There
is no formal-training start, native checkpoint recovery or new optimizer step.
The all-project inventory contains seven Jobs and four Pods, all terminal.

The earlier CPU admission and checkpoint/archive bindings remain valid as
historical evidence. On November 18 at **01:06:57**, independent full GET/SHA
verification accepted its separate metadata/startup failure archive:
**124 objects / 15,310,756 bytes**, including 122 payload objects /
15,148,208 bytes. Exact listings remained stable and all original source trees
were preserved. No run directory or training checkpoint was created; the
external attempt-2 checkpoint and its archive were not repeated. This is
failure archival, not training or quality acceptance.

## Smol attempt-5 resumed to step354, then interrupted

Attempt 5 adds an atomic READY receipt after the real submission record is
complete. GPU startup waits up to 600 seconds for that publication before
validating its exact Job, UID, package and intent; the wait holds no shared
lock. Ninety CPU tests and a real Linux shared-filesystem synthetic smoke
passed, including a reader started before its writer and rejection of partial
publication. The smoke did not submit a Job, call the scheduler or initialize
CUDA. Fresh CPU/live admission completed at **01:14:19 Beijing time**, followed
by independent acceptance at **01:14:54**. The two roots total
**49,932,002,041 bytes**, plus **27,284,546,264 bytes** reserve, for a
**77,216,548,305-byte** projection below 99 GB. Seven project Jobs and four Pods
were terminal, the exact new OSS prefix was empty, and client/server dry-runs
created no actual Job. The scientific recipe and checkpoint300 recovery
source remain unchanged.

After fresh submission checks, the one-shot client exited0 and created
`jinhaopeng-dpvlm-p1-smollm2-s29-align-a5-n1-6e7fe`, UID
`f9761d5d-7e4e-4e0b-8e45-78dd658f580a`, at **01:19:04**. The actual allocation
contract matches stream/low, one node, eight GPUs, restart Never and no forced
node. Submission remeasured **49,932,063,109 bytes** plus the same reserve,
for a **77,216,609,373-byte** projection. The durable intent preceded the one
create. Root confirmed the exact Job at **01:19:36**, then atomically published
`SUBMISSION_READY.json` binding the real Job UID, package, intent and complete
submission receipt. Independent submission acceptance verified the complete
**45-file** readback and these bindings.

At **01:22:49 on November 18**, the exact Job had `Created=True` and its Pod was
**Pending/Unschedulable** because eligible nodes lacked sufficient CPU, memory
and GPU resources. That dated inventory contained eight project Jobs and five
Pods, with only this attempt nonterminal. This Pending snapshot and the accepted
CPU/submission records are retained as history.

The same Job became Running at **02:18:19** and Failed at **02:37:32 Beijing
time on November 18**. The November 19 **18:29:22** exact readback confirms
the original UID and terminal condition. Its original Pod is `NotFound`, so
the original container exit code is unavailable. The current project inventory
contains six Jobs and three Pods, all terminal; earlier attempt-1/2 Jobs are
no longer listed, which does not change their historical outcomes.

The actual GPU startup handshake recorded `ROOT_SUBMISSION_HANDSHAKE_ACCEPTED`
at **02:18:20**, binding the exact Job/UID, package, intent, READY and complete
submission receipt after one poll. Startup admission then passed all eight
device identities. All eight ranks passed BF16 attention forward/backward and
NCCL all-reduce; the enclosing `preflight-before` command exited0 after
**53.84 seconds**. Formal training initialized from the accepted attempt-2
checkpoint300. Its initializer receipt and native run manifest agree on the
source and selected alignment/world8/R1/F8/G4/seed29 recipe fields, and the
training log reaches **354/718**. This supports actual native continuation and
54 new updates, without proving bitwise or numerical resume equivalence.

The initial local `checkpoint-350/trainer_state.json` observation identifies
step350. A separate native CPU verification completed on November 19 at
**18:34:11 Beijing time** and passed all **29 files / 8,642,274,062 bytes**, with
full SHA hashes and stable before/after file metadata. The original native
source, runtime configuration and formal training contract match the accepted
bindings. The canonical checkpoint file-map SHA256 is
`821ee7b3970ddb0b3c08718898b131382a53b7571db44592b51f7a1ce1a2bba9`.
Checkpoint350 is therefore a recovery candidate with **368 updates** to 718.
This CPU check neither deserializes optimizer/scheduler/RNG state nor proves
GPU recovery or numerical equivalence; exact archival and fresh independent
admission remain required. There is no
final model, `FORMAL_TRAIN_COMPLETE.json` or `P1_STAGE_COMPLETE.json`. The runner
`EXIT_STATUS.json` records exit1, separately from the unavailable original
container exit code. `startup/FAILED.json` records
`QA controller interrupted by signal 15` at **02:37:30** with automatic retry
disabled. The SIGTERM source is unknown and is not evidence of external
preemption. At **18:31:29 Beijing time on November 19**, event queries for the
exact Job and original Pod UIDs both returned empty lists. The exact attempt-5
OSS run prefix contained **0 objects / 0 bytes**, and no local `ARCHIVED.json`
was observed. Automatic failure archival did not complete.

The terminal Job/Pod and inventory readback is pinned at
`job-readback-20261119-v1/stdout.json`, **3,437,645 bytes**, SHA256
`0e08059970a17d5b069ebaae1aecfc9f1a37c69c56dcb88d3c9c2b224a5b2164`.
The separate stable file readback is
`stage-readback-20261119-v1/stdout.json`, **937,918 bytes**, SHA256
`4bc6e5e9efd457326064084f479ed96034350af11df7dbbb17526cb5420acc4f`;
its **17,974-byte** summary has SHA256
`6c7813b3513ef614188fd1bb791776a2e086fc0e6d4a3fb0e97cda07fd3046f6`.
The separate exact events/OSS readback is
`terminal-events-oss-20261119-v1/stdout.json`, **6,145 bytes**, SHA256
`29d84c8b6ea354e30ee2d7b6d2c02956a2bd5268d1d7b59fd058a3b162b41b1d`.
The independent outcome review, **7,369 bytes**, SHA256
`5fd524cf097bb4f7811faf617131c0220fcfcb5ca337fb7e34141ea403c22b82`,
verifies the real handshake, training progress and terminal evidence; its
checkpoint conclusion retains the earlier metadata-only scope. The separate
native checkpoint verification is
`checkpoint350-readonly-native-verify-20261119-v1/stdout.json`, **17,713 bytes**,
SHA256 `e73d8c02a2c153f184b511b36fe802044a3ba8c55bbc3b2474ab01797a1188b9`.
These private artifacts are under the attempt-5 operations evidence directory;
[status.json](status.json) binds their paths and observation times.
The full 185-task suite and R4 restoration retain separate admission gates.

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
reached **602/718 updates**, while its checkpoint Trainer metadata remains at step
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

The November 17 readback found the exact R4 Job and original Pod **NotFound**;
this does not change its previously observed failure. Five model/optimizer
payloads totaling **22,312,029,341 bytes** are now missing locally, leaving
26 metadata/state files / 5,124,031 bytes. The cause of their absence is unknown.
The log and Trainer state are unchanged at 602 and 600 respectively.

The corresponding archived keys remain present with matching sizes, and the
147-object archive listing is unchanged. This check did not download or rehash
those large payloads. Restore them against the original SHA256 evidence, then
validate complete optimizer/RNG recovery and fresh admission before resuming
the remaining 118 updates. Local Trainer metadata alone is insufficient.

The sanitized [status record](status.json) preserves the dated R4 failure facts
and records Smol separately, with hashes linking each claim to private evidence.
No new benchmark results are claimed by these training and archive updates.

## CUDA engineering validation

### Attempt 3 completed: synthetic CUDA operator QA

Run `dp-cuda-pool-qa-20261114-a3` binds source commit
`5a14d3e7bd65a5869f21cb32d79221a03d695af9` and the reviewed v3 package.
Its exact job is `jinhaopeng-dpvlm-cuda-pool-qa-20261114-a3-n1-abd5f`, UID
`152670ef-e80f-4ec0-8359-1060e0db32dc`. The earlier November 14 Pending
observations remain in history. The Job became Running at **13:48:11** and
Succeeded at **13:51:28**; its container ran from **13:48:10 to 13:51:26**,
exit code 0, with no restarts. The contract was stream/low, one node, eight
GPUs, restart Never and no forced node.

Actual eight-rank BF16/NCCL startup passed before the pooling suite and
measurements. Of the **13 passing pooling tests**, 12 cover CPU/reference or
guards and one explicitly exercises CUDA numerical cases. Both dtype
microbenchmarks passed their numerical gates and recorded the real
`_weighted_spatial_pool_kernel` in separate profiler passes.

The measurement protocol uses one device, synthetic `[8,729,1152]` inputs,
G4 and seed 17, with 25 warmups and 100 measured calls per backend. Alternating
backend order, synchronized CUDA events and separate profiling are retained.
The timed operation includes output allocation and excludes the gate,
projector, vision encoder and language model. Ratios are ratios of backend
medians; one run and one synthetic shape do not establish model-level gains.
See [operator measurements and scope](../docs/OPTIMIZATION_STATUS.md).

Independent archive acceptance on November 17 verified **56 objects /
2,518,835 bytes**, including 54 payload objects / 2,494,061 bytes, against
local source bytes and the stable exact OSS inventory. Recalculation of all
200 rows per dtype matched the recorded mean, median, R7 P95 and other
summary fields. Original run files were preserved. [status.json](status.json)
binds the complete success, archive and measurement evidence; no model
checkpoint or held-out result is part of this QA run.

### Attempt 2 failure and retained validation

The earlier opt-in inference changes and recovery guard were committed as
`bec405e8802232cf4815f1c6dfea968318a12aa9`. The public CPU suite passed 512 tests
(80 skipped), and three separate pinned native Ouro CPU tests passed.

Attempt 2 binds source commit `c03ffefa485682d2c737124c07b83473442cf6f4`.
Its package/control CPU review passed **18 tests, with one skipped**; the
skipped GPU test is not CUDA acceptance. CPU admission, independent admission
review, an exact server-side dry-run and independent submission review passed
before/after the one-shot create as recorded in their individual receipts.

The attempt-2 exact job
`jinhaopeng-dpvlm-cuda-pool-qa-20261113-a2-n1-772f8`, UID
`ca956db5-7166-4452-93b8-d89fecd6cc33`, was created at **00:26:08 on November
13** (UTC November 12, 16:26:08). The **00:26**, **00:27** and **00:31 Pending**
observations remain in history. It subsequently became Running at **01:11:09**
and Failed at **01:14:39**. The original Pod is still present; its container
finished at **01:14:37**, exit code **1**, with zero restarts. The actual
contract remains stream/low, one node, eight GPUs, restart Never and no forced
node.

All eight NVIDIA L20D devices passed the complete identity gate: compute
capability [10,3] and 267.69 GiB per device. The next `torchrun --standalone`
preflight repeatedly failed to resolve its rendezvous hostname, then hit its
**120-second timeout**. Its output contains no accepted eight-rank BF16/NCCL
result, and the pooling numerical tests and BF16/FP32 microbenchmarks were not
reached. The diagnosis is bound to the actual command, stderr and installed
Torch source; hardware identity alone does not establish distributed startup
or CUDA correctness/performance.

The automatic failure archive contains **26 objects / 2,172,940 bytes**,
including **24 payload objects / 2,161,309 bytes**. At **11:53:04**, independent
full GET/SHA verification accepted every object against local source bytes and
the exact stable object inventory. This archives startup/failure evidence,
not a model checkpoint or performance result. The original run was preserved;
no files were deleted and no replacement job was submitted in this check.

The submission-side small-QA capacity projection was **86,352,050,339 bytes**,
including the 1 GB QA and 7 GB existing media/CPU reserves, below the 99 GB
limit. This does not admit a training recovery or the full experiment suite.
Attempt 2 supplies no accepted CUDA result. The later successful attempt 3
has its own source/package and Job identity; the frozen attempt-2 source and
failure evidence remain intact.

### Public launcher repair

The five future training, startup-preflight and evaluation launch sites now
use explicit c10d loopback rendezvous, with OS-allocated ports and a fresh ID
for each invocation. Child-process environment guards prevent inherited
standalone/store-sharing flags from overriding that path. The one-node,
eight-rank and zero-restart defaults remain unchanged.

At **12:03:48 Beijing time on November 13**, the final repair receipt bound all
seven changed code/test files and the completed CPU checks. The public targeted
run passed **48 tests, with 10 private-evidence tests skipped**. Two simultaneous
real Gloo groups, each with two CPU ranks, passed all-reduce while hostname
resolution was deliberately broken and conflicting flags were inherited; the
groups used distinct live ports. This ran on local Torch **2.13.0**, not the
remote Torch **2.10.0+cu130** GPU runtime.

An additional run with private-artifact tests enabled recorded **57 passed and
one failed**. The failed legacy TempCompass check compares current model and
generation source with the October 21 frozen Video-MME snapshot. SHA evidence
shows that difference predates this repair, and all four checked scientific
source files are unchanged from the pre-repair baseline. The old test and frozen
snapshot were preserved. These scoped results are not a full-suite pass, new
eight-rank BF16/NCCL acceptance or a CUDA correctness/performance result. The
repair, both test outputs and baseline proof are pinned separately under
`public_launcher_repair` in [status.json](status.json).

### Attempt 1 retained as failed history

The earlier synthetic pooling QA job
`jinhaopeng-dpvlm-cuda-pool-qa-20261112-a1-n1-bb1e7` was created at **20:46:10
on November 12**, became Running at **20:55:46**, and Failed at **20:56:31**.
Exact Pod-UID events confirm **external preemption at 20:56:29**, followed by
container termination. The earlier **20:47:41 Pending** observation is retained
as history. Its contract was stream/low, one node, eight GPUs and restart Never.

The **21:15** readback found the original Pod absent, so its container exit code
is unavailable. Startup checks were retained, but GPU startup acceptance,
CUDA numerical tests and microbenchmark results were not produced. The exact
archive prefix contained **0 objects / 0 bytes**. No retry, archive or deletion
was performed in that dated read-only check. Its full record remains under
`engineering_qa_history` in [status.json](status.json); retaining that history
does not mean its OSS archive was completed. No CUDA speedup or held-out claim
follows from attempt 1.

The initial submission client failed before starting `kubectl` because its
PATH lacked that executable. Complete and exact-name reads confirmed no job
existed. A separate reviewed repair used the absolute executable and a single
`create` of the original validated YAML, retaining the original intent and
failure. The actual Job UID and resource contract were then read back.
See [optimization scope](../docs/OPTIMIZATION_STATUS.md) and the evidence hashes
in [status.json](status.json). That record now separately binds attempt 2's
package, CPU admission, independent reviews, submission intent/outcome and exact
readbacks, terminal command/error evidence, diagnosis and independent archive
acceptance by file path, byte count and SHA256. Raw cluster responses and other
workload identities are excluded from the public summary. These are dated
facts, not a background monitor.

## November 13 recovery-capacity observation

A separate read-only snapshot completed at **00:15:52 Beijing time** measured
**78,345,490,305 bytes** across the two accounted roots. The original R4 and
Smol checkpoint state files are present, but exact numerical recovery remains
independently unaccepted. This snapshot did not perform new full GET/SHA
verification of their large checkpoint payloads or complete CPU admission.

Preserving the old runs and including the new suite's checkpoint copy, that
snapshot's conservative projections were **155,547,134,605 bytes for R4** and
**114,272,309,701 bytes for Smol**, both above the strict 99 GB limit. The later
November 17 dedicated Smol recovery uses its original checkpoint in place and
has a separately accepted 75,879,340,811-byte projection; the older suite-copy
estimate remains historical. Additional initializer/source/media requirements
may increase those figures; future
cleanup is not credited. The older November 9/11 capacity observations remain
intact. Exact facts and the receipt pin are appended to
`recovery_capacity_observations` in [status.json](status.json).

The later **11:51** training inspection verified unchanged checkpoint names,
sizes and Trainer-state SHA values, plus unchanged exact archive keys/sizes/
ETags. It did not rehash the large checkpoint payloads, rerun preflight or
rescan capacity. No new recovery-submission evidence was found. These fresh
observations are separately retained under `training_progress_observations`;
the older training and capacity facts remain unchanged.

The complete suite retains 18 model configurations, 34 training stages,
36 evaluations, 15 latency runs, 12 comparisons and **185 operational tasks**.
It remains unsubmitted. The new locally tested scheduler can run admitted,
ready tasks while benchmark caches prepare; it publishes verified cache
receipts as they become ready and applies checkpoint-chain backpressure.
This does not waive individual task inputs, capacity or startup checks.
The seven November 20 historical/checkpoint preparations are authenticated;
R4 cp600 has full archive GET/SHA and native metadata verification, while
runtime model/optimizer/RNG recovery remains untested. The suite must replace
its old Smol cp350 continuation with the accepted a6 final after archive
recovery, so the completed 368 alignment updates are not repeated.

## Remaining work

1. Obtain completion receipts from the running a6 supplement-upload/audit client
   and root archive acceptance, import its completed final into the suite, then
   admit matched Smol SFT and evaluations. Its 718 alignment updates are complete.
2. Finish R4 runtime checkpoint recovery and storage admission, run its remaining
   118 SFT updates, then archive and evaluate the completed model.
3. Complete the additional training seeds and declared ablations, controlled
   cost/latency experiments and updated comparative statistics.

MVBench has no accepted result in this release. Its authorization/media status
was not reverified in this check. Existing benchmark timings do not constitute
a warmed latency comparison. No canceled automation was recreated.
