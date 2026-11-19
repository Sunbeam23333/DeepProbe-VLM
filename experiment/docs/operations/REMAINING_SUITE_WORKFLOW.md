# Remaining experiment suite

`scripts/operations/complete_experiment_suite.py` prepares the frozen scientific
DAG and can pass the admitted binding to the fleet launcher. Its default is a
report: it does not build caches, restore checkpoints, initialize task state or
contact the scheduler. The plan remains 18 models, 34 training stages,
36 evaluations, 15 latency runs, 12 comparisons and 185 operational tasks.

## Machine discovery and one entry point

The final worker-count candidates and command script use
`/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/suite-suite-remaining-20261119/control-v4`.
Their immutable, SHA-verified private inputs are retained in the sibling
`control-v3/inputs` directory.
Their commands use the new immutable executable source at
`/home/jinhaopeng/.cache/Paper/DeepProbe-VLM/suite-source/20261119-v1/experiment`
and isolated runtime at
`/home/jinhaopeng/.cache/Paper/DeepProbe-VLM/suite-runtime/20261119-v1/bin/python`.
The frozen historical runtimes and old runs are preserved. The isolated runtime
has the training dependencies plus `requirements/suite.txt`.

Choose one worker count for this fleet. Run the selected chain on
`haopengjin3ali`; `prepare` must finish successfully before dry-run or submission.
The `discover` phase verifies the actual byte-pinned discovery candidate already
produced on that machine, including its four evaluation imports, five accepted
training preparations and two checkpoint recoveries. It reports current source
availability and performs no GPU operation.

One normal-priority worker, one node and eight GPUs:

```bash
set -e
DP_SUITE=/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/suite-suite-remaining-20261119/control-v4/RUN_COMMANDS.sh
bash "$DP_SUITE" 1 discover
bash "$DP_SUITE" 1 plan
bash "$DP_SUITE" 1 prepare
bash "$DP_SUITE" 1 dry-run
bash "$DP_SUITE" 1 submit
```

Four normal-priority workers, each one node and eight GPUs:

```bash
set -e
DP_SUITE=/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/suite-suite-remaining-20261119/control-v4/RUN_COMMANDS.sh
bash "$DP_SUITE" 4 discover
bash "$DP_SUITE" 4 plan
bash "$DP_SUITE" 4 prepare
bash "$DP_SUITE" 4 dry-run
bash "$DP_SUITE" 4 submit
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
DP_SUITE=/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1/suite-suite-remaining-20261119/control-v4/RUN_COMMANDS.sh
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
* Home and OSS independently stay below 150,000,000,000 bytes, with 2 GB safety
  for each domain and a 1 GB filesystem-free guard. Reservations are global
  across 1–4 workers, including caches, transfers and partial uploads.

The workflow checks fresh capacity before recovery or benchmark transfers. A
capacity block records actual usage and the pinned exact retirement/migration
plan; it never bypasses the reservation or deletes anything automatically.
Original archives and every cache shape need a reviewed hot/cold retention plan;
keeping all of them resident is not an assumed capacity optimization.

For successful models, intermediate checkpoints can be retired only after
complete final/model-result archival and after consumers and resume references
finish. ModelScope release requires the full final model backup and complete
readback SHA verification. Publishing a small result bundle is not a backup of
model weights. Load the ModelScope token from the private owner-only 0600 file;
never put its value in candidate JSON, YAML, logs or Git.

## Actual readiness evidence on 2026-11-19

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

The reviewed P0 retirement plan covers 640,178,450,792 bytes. Even if every
listed retirement completes, its projection is 383,244,997,438 bytes of live OSS,
still above the 150 GB ceiling. It therefore does not by itself admit this suite.
The attempted P0 deletions returned `AccessDenied:403`; no completed deletion
or reclaimed byte was credited, and the maintenance processes exited. The
current OSS total above confirms that capacity has not been released. No GPU
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
