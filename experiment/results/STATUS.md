# Experiment status — 2026-11-13

Latest engineering QA readback: **November 13, 11:50:24 Beijing time
(UTC+08:00)**. CUDA pooling QA attempt 2 is **Failed** after an eight-GPU
allocation. GPU identity passed, but the all-rank preflight timed out while
resolving its rendezvous hostname. No accepted BF16/NCCL preflight, CUDA
correctness test or microbenchmark result was produced. Its complete failure
archive passed independent readback at **11:53:04**.

The **11:51:21** training readback found the same R4 and Smol Job UIDs still
Failed, with their original Pods absent. Highest logged steps remain 602 and
317; durable checkpoints remain 600 and 300, without final checkpoints or
training-completion receipts. The **11:51:37** exact OSS listings remain
147 objects / 23,234,938,760 bytes for R4 and zero objects for Smol. These are
dated observations; full numerical recovery remains unaccepted.

At **11:54:54**, a complete project Job/Pod inventory found eight Jobs and four
Pods, all terminal, with no active/Pending/unknown project workload. No new job
or old-file deletion was performed in this check. The full suite remains
unsubmitted, and no new training, held-out quality or GPU-performance result
is claimed.

The future public single-node launchers now use explicit loopback rendezvous.
Their final public targeted CPU regression passed **48 tests, with 10 skipped**;
an additional private-artifact run retained one pre-existing frozen-source
comparison failure. This repair has not been submitted or accepted on GPUs.

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
There is no completed CUDA result or accepted speedup. A new GPU attempt still
needs its own reviewed source, fresh admission and separate identity; the frozen
attempt-2 source and failure evidence remain intact.

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
snapshot's conservative projections are **155,547,134,605 bytes for R4** and
**114,272,309,701 bytes for Smol**, both above the strict 99 GB limit. Additional
initializer/source/media requirements may increase those figures; future
cleanup is not credited. The older November 9/11 capacity observations remain
intact. Exact facts and the receipt pin are appended to
`recovery_capacity_observations` in [status.json](status.json).

The later **11:51** training inspection verified unchanged checkpoint names,
sizes and Trainer-state SHA values, plus unchanged exact archive keys/sizes/
ETags. It did not rehash the large checkpoint payloads, rerun preflight or
rescan capacity. No new recovery-submission evidence was found. These fresh
observations are separately retained under `training_progress_observations`;
the older training and capacity facts remain unchanged.

The complete suite's scripts cover 18 model configurations, 34 training stages,
36 evaluations, 15 latency runs and 12 comparisons. **The suite remains
unsubmitted** while actual F4/F16 original-video caches, independent historical
recovery bindings and fresh storage/source admission are outstanding. The
separate small QA submission does not change those gates.

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
