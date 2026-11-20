# Remaining experiment suite

`scripts/operations/complete_experiment_suite.py` prepares the frozen scientific
DAG and can pass the admitted binding to the fleet launcher. Its default is a
report: it does not build caches, restore checkpoints, initialize task state or
contact the scheduler. The plan remains 18 models, 34 training stages,
36 evaluations, 15 latency runs, 12 comparisons and 185 operational tasks.

## Current preparation on 2026-11-21

The scientific DAG still contains all 185 tasks. Source commit `4034666` adds
checkpoint-chain backpressure: 17 training components share one resident-chain
slot, at most one training claim is active, a following stage waits for its
predecessor's archive, and a new component waits for all consumer/cleanup
barriers of the preceding component. Ready admitted tasks can run while the
CPU cache producer continues, with verified cache receipts published as ready.
The 190 scoped CPU tests passed, including synthetic full-DAG claimants at
counts 1–4; these are scheduling tests, not completed experiments.

The source-v4/control-v8 candidate is frozen locally and has **not been
deployed**. Its source manifest SHA is
`7f6a0d40385d6363845b6f936a161ef9ca02356a587a60a7ff62864981b96208`.
No full-suite GPU worker has been submitted. The current plan uses **one
normal-priority worker, one node and eight GPUs**. The shared 150 GB budget
does not currently admit four workers. Fresh deployment/readback, complete
CPU admission and exact Job/server dry-run checks remain required.

Smol a6 completed all 718 alignment updates; its original Job failed during
archival. The recovery precheck passed at November 21 01:18:26 Beijing time:
240 local files passed SHA checks and 82 existing objects passed full GET/SHA.
The supplement-upload and independent-audit client started at 01:19:54 and is
running. No client, execute or audit completion receipt is present yet; wait
for those receipts and root acceptance without treating startup as a completed
upload. The new candidate must import its accepted final
instead of repeating the old cp350 continuation. Smol SFT, R4's remaining
118 SFT updates, evaluations and the rest of the DAG remain pending.

The November 21 01:13 Beijing-time cleanup acceptance measured
62,504,292,995 bytes in the accounted home roots after deleting six already
archived old checkpoint weight files (17,116,975,648 bytes). All a6 files and
other active reservations were retained; the original a6 42 GB reservation
was released. Home including all active reservations and safety measured
65,049,552,515 bytes before releasing the cleanup operation's own 8 MiB token.
These are dated observations and must be refreshed at GPU admission.
The 01:18:26 recovery precheck also found five project Jobs and three Pods
terminal; this is a dated workload snapshot, not a free-GPU capacity measurement.

## Historical v5 machine discovery and entry point

The following v5 commands document the earlier deployment. They are not the
current launch authorization: their Smol cp350 binding predates completed a6,
and their four-worker example has no current capacity admission. Keep the
immutable candidates as history; use a newly admitted v8 binding for the next
launch after actual deployment and a6 archive acceptance.

The deployed v5 worker-count candidates and command script use
`/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/suite-suite-remaining-20261119/control-v5`.
Their immutable, SHA-verified private inputs remain in the sibling
`control-v3/inputs` directory. V5 adds a separately derived R4 source review and
recovery index under `control-v5/inputs`, retaining the original entry bytes.
Their commands use the new immutable executable source at
`/home/jinhaopeng/.cache/Paper/DeepProbe-VLM/suite-source/20261120-v1/experiment`
and isolated runtime at
`/home/jinhaopeng/.cache/Paper/DeepProbe-VLM/suite-runtime/20261119-v1/bin/python`.
The frozen historical runtimes and old runs are preserved. The isolated runtime
has the training dependencies plus `requirements/suite.txt`.

Choose one worker count for this fleet. Run the selected chain on
`haopengjin3ali`; `prepare` must finish successfully before `launch-prepared`.
The `discover` phase verifies the actual byte-pinned discovery candidate already
produced on that machine, including its four evaluation imports, five accepted
training preparations and two checkpoint recoveries. It reports current source
availability and performs no GPU operation.

One normal-priority worker, one node and eight GPUs:

```bash
set -e
DP_SUITE=/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/suite-suite-remaining-20261119/control-v5/RUN_COMMANDS.sh
bash "$DP_SUITE" 1 discover
bash "$DP_SUITE" 1 plan
bash "$DP_SUITE" 1 prepare
# After independent review of the prepared CPU admission and bindings:
bash "$DP_SUITE" 1 launch-prepared
```

Four normal-priority workers, each one node and eight GPUs:

```bash
set -e
DP_SUITE=/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/suite-suite-remaining-20261119/control-v5/RUN_COMMANDS.sh
bash "$DP_SUITE" 4 discover
bash "$DP_SUITE" 4 plan
bash "$DP_SUITE" 4 prepare
# After independent review of the prepared CPU admission and bindings:
bash "$DP_SUITE" 4 launch-prepared
```

The one-worker and four-worker candidates are alternatives for the same suite;
never run both. `RUN_COMMANDS.sh` pins their candidate hashes and exact reviewed
input indexes. The source review and result evidence were actually copied with
complete SHA verification. `REBASE_RECEIPT.json` records the original and new
paths without changing original evidence bytes. Training/recovery entry bodies
are unchanged; evaluation entries only relocate the same result-byte pin.

The underlying discovery CLI is
`scripts/operations/discover_experiment_suite_machine.py`; it accepts the actual
`--history-import-index`, `--history-preparation-index`,
`--recovery-preparation-index`, `--capacity-snapshot` and
`--capacity-recovery-plan`. Changed inputs or a changed worker profile require a
new candidate and reviewable hash; the saved candidate is never silently edited.

The current authorized policy removes the OSS capacity quota while retaining the
home limit. New candidates must explicitly use `--unlimited-oss`, which writes
`storage.oss_limit_bytes: null`; discovery without this flag keeps the previous
150 GB OSS limit. Unlimited-OSS candidates also omit the ModelScope-token
dependency because verified model archives remain in OSS. The private token
file and previously published results are preserved. The old v4 candidates are
immutable historical artifacts and do not express this new policy. The actual
v5 candidates bind the new policy, all eleven historical inputs and new source.

The v5 command script adds `launch-prepared` for the reviewed preparation path.
After shared-ledger migration, prepare once:

```bash
DP_SUITE=/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/suite-suite-remaining-20261119/control-v5/RUN_COMMANDS.sh
bash "$DP_SUITE" 1 prepare
```

After independently reviewing the latest CPU admission and prepared bindings,
submit those exact prepared files without repeating the full checkpoint GETs:

```bash
DP_SUITE=/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/suite-suite-remaining-20261119/control-v5/RUN_COMMANDS.sh
bash "$DP_SUITE" 1 launch-prepared
```

Use `4` in both commands only when the four-worker candidate was selected.
`launch-prepared` verifies the candidate, its authorization and input indexes,
the saved preparation request, and all five plan/binding/environment/Jobs/CPU
preflight pins before invoking `launch_experiment_fleet.py --submit`. The
launcher still performs fresh capacity, cluster, exact-prefix and server dry-run
checks and rejects expired CPU admission. It does not repeat asset preparation.
The original `submit` phase remains a combined preparation/admission/submission
operation; invoking it after `prepare` repeats the complete asset readback.

Add `--prepare-assets --cpu-admission` to perform preparation and admission.
Add `--dry-run` to also run server-side scheduler dry-run, or `--submit` to
explicitly create the admitted exact worker Jobs. `--resume-assets` reconciles
owned partial asset work; it does not authorize resubmitting a GPU Job. A cache
shape that has not started still uses fresh creation during a suite resume.
Objects without their exact local build plan block for reconciliation.

Immutable candidate, history, source, environment and binding artifacts are
reused only when their contents match. Fresh capacity/preflight observations and
workflow results are appended under `attempts/<unique-id>/`, so a previous
capacity block or dry-run does not collide with new observations. Gated
preparation, incomplete CPU admission, unknown submission outcomes and terminated
workers return exit code 2. A successful plan-only report returns 0 and still
states `production_admitted=false`.

An existing submission intent is reconciled against exact Job identity and UID.
For a partial or uncertain four-worker submission, use the same pinned inputs:

```bash
DP_SUITE=/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/suite-suite-remaining-20261119/control-v5/RUN_COMMANDS.sh
bash "$DP_SUITE" 4 reconcile
bash "$DP_SUITE" 4 continue-unsubmitted
```

Use worker count `1` for a one-worker fleet. `continue-unsubmitted` creates only
workers whose create was never attempted, after fresh capacity, cluster,
exact-prefix and server dry-run checks. An uncertain attempted Job is never
blindly retried. A preflight older than its allowed window is rejected; refresh
CPU admission before continuing.

## Historical reuse and recovery

Historical hints do not skip work or restart it from zero. Accepted training
needs the exact final checkpoint inventory, configuration, frozen source and
archive evidence. Accepted evaluation needs the exact parent inventory and all
2,697 Video-MME or 1,580 TempCompass uniquely identified scored rows, with the
declared protocol and zero failed units. A separately reviewed import binds the
entry and every original evidence pin before producing `SUITE_HISTORY_ACCEPTED`.

For accepted training whose old archive acceptance used CRC/length checks,
`history_preparations` uses `verify_archived_accepted`: independent source and
718-step completion review, then complete streaming GET/SHA of the final model,
then a newly measured archive pin and history import. The five current models
require 37,414,073,398 bytes of readback. This operation does not hydrate the old
final directory, rerun training or claim a GET that has not occurred.

Recovery requests use the same exact source and configuration as the interrupted
training. Their source review has status
`SUITE_RECOVERY_PREPARATION_REVIEW_ACCEPTED`, approved production scope, exact
task/entry identities, original independent evidence and the verified native
metadata facts. A draft with `approved=false` cannot perform any transfer.

* `archive_local` verifies every local file, uploads missing files only into the
  dedicated new exact checkpoint prefix, and then independently GETs and hashes
  every object before importing history. Existing objects require explicit asset
  resume and matching ownership/inventory. The old run is never modified.
* `restore_archived` requires the complete exact archive inventory and writes
  only into `environment.run_root/.recovery-inputs/.../checkpoint-<step>`. Every
  streamed byte is counted and SHA-checked. The complete native eight-rank state
  and Trainer schedule are checked before history import.
* `verify_archived_resume` performs complete streaming GET/SHA and records only
  the three bound JSON metadata files. It checks the checkpoint inventory and
  Trainer step/schedule without hydrating checkpoint weights. The training
  action later hydrates one copy and performs native optimizer/RNG restoration.
  This is the current R4 checkpoint-600 preparation policy and requires its own
  derived source review; the earlier restore request and review stay unchanged.

The accepted preparation includes the archive readback receipt and metadata
verification; it does not claim optimizer/RNG deserialization or numerical
resume equivalence. The training action performs the native runtime restoration
and training continues from the explicit bound step. Smol checkpoint 350 has
368 updates remaining; R4 SFT checkpoint 600 has 118. Never replace an explicit
350 binding with the older 300 candidate hint. Preserve the accepted prior
recovery copy until the newer copy's complete archival acceptance exists.

`checkpoint_metadata` may provide separately retained JSON pins for
`deepprobe_config.json`, `training_provenance.json`, and `trainer_state.json`.
Their sizes and SHA values must match the full checkpoint inventory exactly;
this allows metadata admission without hydrating large weights prematurely.

## Frame caches and storage

Fleet candidates may explicitly select `benchmark_preparation_mode:
external_progressive`. The environment then freezes all six benchmark/frame
requests and their exact producer source and READY paths. CPU admission checks
every training input and records unfinished benchmark caches as deferred. It
uses `SUITE_CPU_PROGRESSIVE_ADMITTED`; this is not an assertion that those
benchmarks are ready or evaluated.

`publish_experiment_suite_caches.py` verifies a completed cache with full PNG
GET/SHA/RGB checks before atomically publishing its READY receipt. A worker
can claim an evaluation only after its own cache is ready, pins that receipt
in the task claim, and checks it again before hydration. Independent training
continues while other cache shapes are being prepared. Invalid readiness
evidence fails closed. The scientific DAG and all 36 evaluations remain intact.

An independently accepted historical evaluation is reused after checking its
complete scored rows, protocol, evidence hashes and the current parent
checkpoint inventory. The adapter emits a reuse receipt and the normal
checkpoint-consumer receipt without inference or cache hydration. These reused
results are not new quality measurements.

F4/F8/F16 cache preparation decodes the SHA-bound original videos. It does not
resample an F8 cache to make F4 or F16. Video-MME legacy archive references are
resolved through the exact pinned original archive audit, SHA, filename, ETag
and verification-receipt SHA. TempCompass uses its official processed clips,
including reverse/speed/concat variants. Cache images go directly to the stream
bucket; only bounded metadata is written on the development machine. CPU
admission verifies complete OSS cache inventories. Evaluation reserves and
hydrates only the currently needed shape on the worker.

All workers and maintenance commands share this one configuration:

* Home roots, in order: `/home/jinhaopeng/.cache/Paper/DeepProbe-VLM`, its `-p1`
  sibling, `/home/jinhaopeng/Paper/DeepProbe-VLM`, and
  `/home/jinhaopeng/.config/deepprobe-vlm`.
* Ledger: `/home/jinhaopeng/.cache/Paper/DeepProbe-VLM/locks/suite-storage-ledger`.
* OSS prefix: `oss://ss-devbox-stream/jinhaopeng/Paper/DeepProbe-VLM/`.
* Home stays below 150,000,000,000 bytes, including its 2 GB safety reserve,
  with a 1 GB filesystem-free guard. Reservations are global across 1–4 workers,
  including caches, transfers and partial uploads.
* Explicit unlimited-OSS candidates set `oss_limit_bytes: null`. Their quota
  snapshots record OSS usage and projected bytes as null and skip project-wide
  quota scans. Exact archive-prefix inventories, payload SHA checks and bounded
  local staging remain mandatory. The default finite policy still enforces the
  independent 150 GB OSS ceiling and 2 GB OSS safety reserve.

The shared ledger must be migrated under its lock before the new policy is used.
The migration requires no active reservations, keeps every other configuration
field identical, and preserves its old configuration, events and artifacts.
Changing a candidate alone does not silently migrate a ledger.

The workflow checks fresh capacity before recovery or benchmark transfers. A
capacity block records actual usage and the pinned exact retirement/migration
plan; it never bypasses the reservation or deletes anything automatically.
Cache shapes are hydrated only while needed and evicted locally after consumers
finish; complete verified OSS copies remain available. Model checkpoints follow
the existing archival and consumer-retention requirements.

For successful models, intermediate checkpoints can be retired only after
complete final/model-result archival and after consumers and resume references
finish. ModelScope release requires the full final model backup and complete
readback SHA verification. Publishing a small result bundle is not a backup of
model weights. Load the ModelScope token from the private owner-only 0600 file;
never put its value in candidate JSON, YAML, logs or Git.

## V5 metadata deployment evidence on 2026-11-20

The actual v5 controls and small-file deployment receipts are retained under
`evidence/suite-machine-readonly-20261120-v5/`. The immutable source manifest SHA
is `073eb1a243b8bbc14feedd7bfb7e69f3ebf824a149509396f5a055f2c3b9efd8`, covering
181 executable source files. Both one-worker and four-worker plans completed
with exit code 0, 185 operational tasks, `production_admitted=false` and
`gpu_submitted=false`.

The candidate SHA values are
`24f99ae5dc5167945bd9fbe38dcd82c3823c7caab1cd335e03e00bb8d3bd834b` for one worker
and `d14a4faf6f9aa4579ffd7d3b2186d9fff38cc0329395290f839a159843ff87f2` for four.
`RUN_COMMANDS.sh` is pinned to
`d944b96dc53232582f08c86179224698579b1a38a75495b06e6133dcddaa1151`.
Freshly generated candidates were compared against the exact v4 candidates;
only the reviewed policy/source/R4 preparation changes and removal of the
superseded quota evidence were permitted.
The independent control audit checked 293 files totaling 27,592,762 bytes,
including all 181 source files, eleven historical inputs and the exact command
template. Its receipt is
`independent-control-audit-v1/audit-20261119T164336Z/INDEPENDENT_V5_CONTROL_AUDIT.json`,
SHA `395965c5b330ce93e3ac20cfe3d56a7aa4dc1eb7557da1bb9662ea6f0b0796f4`.

The copied `control-v5/migrate_policy.py` has SHA
`1d26fff5f90c7e4c8207979b67d035dd87e6eb051244435c7f56eb0bab0ffb5c` and uses
`control-v5/inputs/HUMAN_STORAGE_POLICY.json`, SHA
`eceee9b22dcb5a62c4cf73363151dd453d6d432c42a72013690c4d7e4f63d38d`.
The metadata deployment and plan receipts do not claim that migration, model
readback, cache preparation, CPU admission or GPU submission has occurred.
Those stages require their own actual receipts.

The shared ledger policy was subsequently migrated successfully with
`migrate_policy_v2.py` (SHA
`4a96bbb5afe748b17e416d0addb8c6417424913075851e25b69056dfe0ab04d2`).
This metadata-only wrapper queries Jobs and Pods separately in parallel; its
policy mutation still uses the frozen `disable_oss_quota` implementation.
The new policy SHA is
`13984027010389c7dab3ddccfe0533c5c5c1ccabd7aa9e029d6340812a0e16b8`.
The completed migration receipt SHA is
`6e52607d15c0a79ee4732346a3c70f529f1730a7c8aa381c115ab1ff8995c1d2`;
the exact previous ledger was retained before atomic replacement.

The one-worker CPU preparation driver ran from 2026-11-19 16:52:05 to
17:38:44 UTC (November 20 in Asia/Shanghai), then exited with code 1 and
`group_cleanup_verified=true`. Its invocation is
`8379d12cf0314b3eb4861f5c3149baff`; `CPU_PREPARE_ACTIVE.json` is now absent.
All seven checkpoint/history preparations completed before the failure.
Smol checkpoint350 and R4 SFT checkpoint600 passed complete archive GET/SHA
and native metadata checks; the other five histories bind accepted finals.
These preparations do not prove that a resumed Trainer has loaded its state.

The failure occurred before building the first benchmark cache. The OSS
adapter returns a generator, and the old ownership guard incorrectly treated
an empty generator as a nonempty object listing. A fresh read-only inspection
at 2026-11-20 01:15:49 UTC confirmed all six output directories absent and all
six exact OSS prefixes empty. The fix consumes the complete listing and fails
closed if listing is interrupted; 15 workflow tests passed. The deployed v1
source remains immutable, and cache continuation requires a new source
release. Reuse the seven authenticated histories rather than repeating their
payload transfers. See [the exact driver/cache observation](evidence/suite-cache-ownership-diagnosis-20261120-v1/REMOTE.stdout.json)
and [the recovered history receipts](evidence/smol-a6-launch-20261120-v1/RECOVERY_COMPLETIONS_INDEX.json).

The full suite is not yet admitted or submitted. Successful preparation still
requires a `WORKFLOW_PREPARED.json` with `cpu_admitted=true`; then use
`launch-prepared` with the verified binding and fresh admission. A separate
Smol checkpoint350 continuation can use its own admission without waiting for
the benchmark caches.

That separate continuation was submitted once on November 20 at 01:40:03 UTC
as `jinhaopeng-dpvlm-p1-smollm2-s29-align-a6-n1-16f41`, UID
`58a74e0d-7309-4ad6-8a22-266be9d8d956`, using normal/stream, one node/eight GPUs
and restart-Never. At 01:48:21 UTC its exact Job and Pod were Running with zero
restarts, and the formal training log had advanced from cp350 to **354/718**.
All eight BF16/NCCL preflight ranks and the enclosing preflight process passed.
The shared 42 GB reservation was persisted before Job creation; the independent
submission review confirmed a 104,526,968,806-byte projected home total and no
other active project Job/Pod. This starts only the standalone Smol stage, not
the complete suite. See [the actual running handoff](evidence/smollm2-resume-a6-20261120/ROOT_A6_RUNNING_HANDOFF.json).

The later 14:57 UTC readback supersedes that Running snapshot: a6 completed
718/718 alignment steps with training and both preflight exit codes 0, but its
outer Job Failed at 03:27:04 UTC/container exit1 during archival. A procfs
PermissionError preceded the archive child's SIGTERM. The final/cp718 local
inventories remain; exact OSS has 82 objects and 157 missing manifest payloads.
No archive acceptance or cleanup was recorded in that November 20 check.
Reconcile and verify archival, then bind the completed final for SFT; do not
rerun the 368 alignment updates.
See [terminal status](evidence/smollm2-resume-a6-20261120/ROOT_A6_TERMINAL_STATUS_20261120.json).

The reusable process tracker now rechecks process identity after a denied
procfs environment read. Exited/reused identities may be skipped; known
descendants remain tracked, and a stable unknown live process still blocks
quiescence acceptance. Twenty-one targeted tests passed. The frozen a6 package
was not changed or redeployed, and the denied PID's historical identity is
unknown, so this patch does not establish the exact trigger or recover the
incomplete archive. A local recovery plan identifies the 157 missing payloads
and the original manifest: completing them would produce 240 objects /
18,149,863,235 bytes, without fabricating the original `ARCHIVED.json`.
Fresh local SHA, full GET checks of existing objects and writer quiescence
were required before uploading. No recovery upload or local deletion was
performed in that November 20 status check. The separate November 21 recovery
and accepted old-checkpoint cleanup are described above; the original a6
archive has not yet gained independent recovery acceptance.

The offline helper `scripts/operations/derive_suite_cache_continuation.py`
authenticates the original candidate and 35 captured preparation receipts,
preserves all seven history bindings, four evaluation imports and six cache
requests, and removes completed preparation requests and GPU launch templates.
The local candidate at
`evidence/suite-cache-continuation-20261120-v1/CANDIDATE-CACHE-CONTINUATION.json`
has SHA `d5c93e1639a49fed265b821d0578a1bfd18a40eff246c71718ae2ce4d5af3398`.
Its new source path is `suite-source/20261120-v2/experiment` and workflow output
is `suite-suite-remaining-20261119/control-v6/cache-workflow`; neither is
deployed by this derivation. The workflow/helper regressions passed 24 tests.
After deploying and verifying that new source, the cache-only phase uses
`--prepare-assets --resume-assets`, without CPU admission or GPU submission.
The live workflow still verifies its history pins and exact cache ownership.
Before any later full-suite GPU launch, replace the old Smol cp350 binding
with the separately observed a6 outcome to avoid repeating optimizer work.

The 16:52:57 UTC cluster snapshot had zero unallocated GPUs across the 56
matching stream B300-platform nodes and zero active project GPU Jobs.
The final local regression for this source passed 240 tests across 21 modules.
The fresh preparation capacity gate passed with 60,510,285,511 bytes home
(62,510,285,511 with safety), no blocked domains, and OSS explicitly unmeasured
under the unlimited policy. Its SHA is
`112a3e220be578dae15e94e5bf7a6dff18aababbd521b2381cad2fe031e9f5d1`.
This accepts preparation capacity only; historical transfers, cache completion
and complete CPU admission are still separate requirements.

## Historical finite-policy evidence on 2026-11-19

The final v4 evidence is retained under
`evidence/suite-machine-readonly-20261119-v4/`. On 2026-11-19 at 12:47:41 UTC,
the deployed workflow measured 60,507,240,643 bytes home and
1,023,423,448,230 bytes live OSS, with no active reservations. Its 2 GB safety
projection is 62,507,240,643 bytes home and 1,025,423,448,230 bytes OSS. The
complete OSS inventory contains 17,190 objects and no uploaded multipart parts;
legacy-version bytes remain separately unknown because that API is denied.

Both one-worker and four-worker default plans returned exit code 0. Real
one-worker `prepare` returned exit code 2 with
`SUITE_WORKFLOW_CAPACITY_GATED`, before historical preparation or cache work.
The observed state root and all recovery/cache output directories were absent.
`FRESH_CAPACITY_GATE.json` has SHA
`379f9397f18a56da3816456618125160080794e25de884ffa93e28673024bf14`.
The immutable source and private input indexes are deployed; this is not a CPU
or scheduler admission.

The original v2 candidate preceded source deployment and historical imports.
The later v4 candidates connect all eleven actual historical inputs: four
complete existing evaluations were normalized and independently checked over
all 8,554 rows; five accepted final checkpoints have independent source and
completion reviews; and the two recovery checkpoints have independent source
reviews. Their fresh archive/recovery/cache/CPU gates still execute before any
GPU launch. Historical completion is not counted as freshly restored weights.

The old finite-policy P0 retirement plan covers 640,178,450,792 bytes. Even if every
listed retirement completes, its projection is 383,244,997,438 bytes of live OSS,
still above the former 150 GB OSS ceiling. This quota blocker is superseded by
the user's explicit unlimited-OSS authorization; the old observation does not
block a freshly admitted unlimited-OSS candidate.
The attempted P0 deletions returned `AccessDenied:403`; no completed deletion
or reclaimed byte was credited, and the maintenance processes exited. The
observed OSS total above confirms that capacity was not released by those
attempts. No new OSS deletion is required to satisfy the removed quota. No GPU
training or server dry-run was performed by this preparation work.

The real clean Job template preserves the observed image, volumes, mounts and
resources, removes server state and old startup commands, and has no stdin/tty
hooks. Four Jobs were then materialized on the real Linux machine and passed the Job
validator with every source/state/mount path unchanged. The receipt is
`LINUX_TEMPLATE_CONTRACT.stdout.json`. The probe used an explicitly non-production
binding identity and made zero scheduler calls. It is not scheduler admission
or a GPU submission. Exact current machine
UID/GID is 83980/83980. The observed stream scheduling labels are
`sslaunch-pool=stream` and `aliyun.accelerator/nvidia_name=NVIDIA-L20D`; runtime
startup still requires B300 identity, compute capability 10.3 and at least
240 GiB per GPU. No node is forced by name.
