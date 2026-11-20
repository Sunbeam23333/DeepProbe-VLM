# Complete experiment suite: operating runbook

This runbook describes the executable, serial experiment matrix in
`configs/experiment_suite.json`. A prepared script, a passing CPU test, a
historical checkpoint, and a completed scientific experiment are separate
states. Only the real task's verified outputs advance its completion journal.

## Current admission status

As of **2026-11-20 01:48 UTC**, the 185-task suite has **not been submitted**.
Its declared DAG and scientific scope are unchanged. Separately controlled
Smol recovery is tracked independently of the full suite.

- SmolLM2 seed29 alignment **a6 was submitted once** as
  `jinhaopeng-dpvlm-p1-smollm2-s29-align-a6-n1-16f41`, UID
  `58a74e0d-7309-4ad6-8a22-266be9d8d956`, created at **01:40:03 UTC** and
  confirmed by the root controller at **01:40:44 UTC**.
  The observed Job contract is **normal / stream / one node / eight GPUs /
  restart Never**, with no forced node. It targets native recovery from the
  verified checkpoint350 toward step718; it does not submit the full suite.
  [The exact submission receipt](operations/evidence/smollm2-resume-a6-20261120/submit-execution-v1/stdout.json)
  establishes creation and identity. The **01:48:21 UTC**
  [actual training readback](operations/evidence/smollm2-resume-a6-20261120/exact-training-progress-v2/stdout.json)
  confirms the Job remains Running, with eight GPUs allocated and zero
  restarts; its original Running transition was **01:40:05 UTC**. The
  submission startup handshake passed after **36.8 seconds**, startup
  admission passed, and all eight ranks passed BF16 forward/backward and
  NCCL checks. The enclosing preflight ended with code0 and formal alignment
  training started. Logs show steps **351 through 354/718**: actual native
  continuation from checkpoint350 has added **four optimizer updates**. The
  target remains step718; no final checkpoint, completion receipt or new
  held-out result is established by this observation.

- Seven completed historical preparations are now authenticated for suite
  reuse: Ouro R1 seed17 alignment/SFT, R2 seed17 alignment/SFT and R4 seed17
  alignment each have their accepted **718-step final**, with 20 checkpoint
  files per stage. SmolLM2 seed29 alignment has a **step350** native-resume
  binding (29 files / **8,642,274,062 bytes**, 368 updates remaining). R4 seed17
  SFT has a **step600** native-resume binding (31 files /
  **22,317,153,372 bytes**, 118 updates remaining). All seven retain their
  exact original source/config identities and independently accepted archive
  inventories. The R4 binding uses `verify_archived_resume`; it does not depend
  on the previously incomplete old local checkpoint tree.
- The November 20 review authenticated the existing preparation, full GET/SHA
  archive acceptance, history acceptance and native metadata receipts. It made
  **zero new checkpoint payload GETs**. Native CPU metadata verification does
  not deserialize optimizer/scheduler/RNG state or establish successful GPU
  continuation. Five final-stage histories are eligible for accepted reuse;
  the two resume histories still need runtime restoration and completion.
  The concrete reuse proof is
  [PREPARED_REUSE.stdout.json](operations/evidence/suite-cache-ownership-diagnosis-20261120-v1/PREPARED_REUSE.stdout.json),
  SHA256 `bfb16d20dd03e5c3b836a3349ddd3beac48472285d39644df1778a158a74a504`.
- Preparation stopped at the first new benchmark cache because the ownership
  check treated an empty OSS key generator as truthy. The complete read-only
  inspection found **all six exact prefixes empty and all six cache output
  directories absent**: Video-MME899 and TempCompass, each at F4/F8/F16. The
  code now consumes the complete listing before deciding whether the prefix
  is empty; a listing error still stops preparation. The failed CPU driver
  exited1 with process-group cleanup verified. None of these six new caches
  has been built or admitted; the existing benchmark results are unchanged.
  [The continuation handoff](operations/evidence/suite-cache-ownership-diagnosis-20261120-v1/CONTINUATION_HANDOFF.md)
  binds the actual diagnosis and the offline helper. It preserves the seven
  histories and four evaluation imports, and omits GPU submission templates.
  **24 local tests and independent review passed**; these tests are not cache,
  training, performance or GPU-admission evidence. Continue preparation from a
  new immutable source/output after the separate recovery admission finishes.

The following attempt history remains part of the record; these dated
observations do not reinstate superseded preparation blockers:

- CUDA QA attempt 3 completed eight-rank BF16/NCCL startup, the pooling tests
  including a real CUDA numerical test, and BF16/FP32 single-device synthetic
  operator measurements. Its 56-object archive and raw timing statistics were
  independently accepted. These are operator-level results; full-model,
  distributed-training and H20/B300 speed comparisons remain open. See
  [the accepted QA evidence and scope](../results/STATUS.md#cuda-engineering-validation).
- SmolLM2 seed29 alignment a2 failed, and its 268-object failure archive and
  29-file checkpoint300 passed independent full GET/SHA verification. Attempt
  3 retained that checkpoint and original source. Its first control package
  failed a CPU framework-import guard; corrected v2 passed admission and was
  submitted as `jinhaopeng-dpvlm-p1-smollm2-s29-align-a3-n1-bdff2`, UID
  `ab061a99-000f-4283-b806-249ea5287fdf`. On November 17 it became Running at
  **06:20:58 UTC** and Failed at **06:25:50 UTC**. Hardware and eight-rank
  BF16/NCCL checks passed, but preflight received SIGTERM. The signal source
  is unconfirmed; no native training recovery or new optimizer step was
  observed. Its separate failure archive passed independent acceptance on
  November 18 at **00:40:07 Beijing time**: 337 objects / 940,846,953 bytes.
- Attempt 4, `jinhaopeng-dpvlm-p1-smollm2-s29-align-a4-n1-0aece`, UID
  `01f3dce0-998d-47c3-95c0-98d7c5c833d2`, passed CPU/live admission and was
  submitted on November 18 at **00:52:31 Beijing time**. It became Running
  at **00:52:59** and Failed at **00:53:02** after startup could not read its
  exact submission receipt. The later collection contains that receipt.
  No native recovery or optimizer update was observed. Its separate
  metadata/startup archive passed independent full GET/SHA acceptance at
  **01:06:57**: 124 objects / 15,310,756 bytes. Original trees and the accepted
  attempt-2 archive were preserved.
- Attempt 5 added atomic submission READY publication and a bounded startup
  wait. CPU/live admission, independent review and the exact submission
  handshake passed for `jinhaopeng-dpvlm-p1-smollm2-s29-align-a5-n1-6e7fe`, UID
  `f9761d5d-7e4e-4e0b-8e45-78dd658f580a`. It was created at **01:19:04**, became
  Running at **02:18:19** and Failed at **02:37:32 Beijing time on November
  18**. Hardware, eight-rank BF16/NCCL and preflight passed; formal training
  natively continued from step300 to logged **354/718**, adding 54 updates.
  Durable checkpoint350 subsequently passed its full 29-file SHA and native
  CPU metadata checks and is now included in the prepared history above.
  The controller recorded SIGTERM with unknown source; runner exit1 does not
  establish the missing original container's exit code. The November 19
  **18:29:22 Beijing time** readback found six project Jobs and three Pods
  terminal, with the original a5 Pod absent. At **18:31:29**, exact UID event
  lists were empty and the original a5 run prefix contained zero objects.
  The later dedicated checkpoint-preparation archive is distinct from that
  dated incomplete whole-run archive observation. No a5 final model or new
  held-out result was established.

The current model results and curated evidence identities are linked from
[STATUS.md](../results/STATUS.md) and [status.json](../results/status.json).
The Smol attempt-2 failure-archive acceptance SHA256 is
`ee0fdcb1e2c6198e1cb6b10b9c167787a4c9875d545f06ba82a4caa5cb052b6f`;
the corrected a3 v2 control manifest is
`ec16b955ebe27da3971e2290d44447eb01e596552593373af8d555d04eba24f4`.
Original operational artifacts remain private.

These are preparation snapshots, not permanent assumptions. Re-read the
latest human instructions, source artifacts, exact scheduler Job/Pod state
and exact bucket prefixes before any real operation. Do not reconstruct
acceptance JSON from this prose or mark an old run complete from its name.

## Declared scientific scope

The frozen plan contains 18 model configurations:

- 12 core configurations: Ouro R1/R2/R4 and SmolLM2, each with seeds17/29/43.
- Six R4 seed17 ablations: F4, F16, G2, G8, frozen vision during SFT, and
  no alignment. The frozen-vision SFT shares the R4 seed17 alignment
  initializer; the no-alignment model starts a fresh connector.

This expands to 34 training stages, 36 model/benchmark evaluations, 15 latency
runs, and 12 latency comparisons. Every model is evaluated on the accepted
Video-MME899 no-audio/no-subtitle subset (2,697 QA) and TempCompass multiple
choice (1,580 QA/410 official processed clips).

Latency uses the seed17 Ouro R1/R2/R4 models, each with reference cache-on,
reference no-cache, last-token logits, Triton pooling, and both optimizations.
Each nonreference run is compared with its model's reference. Measurements use
the pinned development selection; held-out quality scores do not choose a
checkpoint, prompt, timing sample or implementation. Separate CUDA profiles
are diagnostic and are excluded from timed measurements.

Including admission, per-stage checkpoints, model results, cleanup and final
aggregation, the DAG has 185 tasks: one gate, 34 train, 52 archive, 36 evaluate,
15 latency, 12 compare-latency, 34 cleanup, and one aggregate. Those are planned
task counts; historical reuse can reduce actual computation only after its
specific acceptance proof is bound.

The default order starts with R4 seed17, its frozen-vision consumer, then
SmolLM2 seed29 recovery. Remaining core models and ablations follow. Ready
cleanup tasks are placed as early as their dependencies allow. R4 alignment
weights stay until the shared frozen-vision consumer also completes; R4 SFT
weights can be released earlier after their own measurements and archival.
The separate November 17 Smol recovery does not reorder this default DAG.

MVBench licensed media/scope, a matched H20 hardware comparison, alignment and
recurrence controls with matched total compute, and the proposed selective
refresh/budget/kernel co-design remain **conditional work**. They are neither
implemented by this matrix nor counted as completed experiments. Failing a
declared test does not silently remove it from the scope.

## Allocation and storage contract

Human authorization permits both `haopengjin2ali` and `haopengjin3ali`.
The November 20 observations use the authorized `haopengjin3ali` alias;
fresh read-only checks matched the shared runtime versions and source pins.
The November 13 connection observations are historical and must not override
this newer evidence. SSH configuration is unchanged; host addresses remain
private.

Historical private packages retain their original host guards and frozen
bytes. The new reviewed Smol recovery package explicitly binds the authorized
third devbox and freshly verifies shared paths, installed tools and admission
inputs. Moving execution to another authorized host requires that same review
and verification, without editing or bypassing an old frozen package. The
public suite does not impose a blanket alias-name guard.

The current authorization permits **one to four B300 worker nodes** in
`ali-gpu` / `stream`, with **`normal` or `low` priority**. Each exact worker Job
has one Master node, eight GPUs and `restartPolicy: Never`; this is a
coordinated fleet of single-node workers, not one distributed multi-node
training run. The separate a6 recovery preparation targets one eight-GPU
node. Its actual submission and runtime state require their own evidence.
See [the current fleet workflow](operations/REMAINING_SUITE_WORKFLOW.md) for the
execution profile, admission rules, complete project Job/Pod checks and shared
task claims. A standalone
recovery and suite workers must not perform the same stage concurrently.
No launcher may infer a priority/pool change from spare capacity, preempt
another user, start Baidu parallel training or create a monitoring automation.

The earlier a3–a5 attempts and the legacy single-worker launcher example below
retain their original **low / one-node / eight-GPU** contracts. Their earlier
two-root **99 GB** admission observations are historical evidence; they do
not impose the current fleet/storage policy or authorize editing frozen
attempts in place.

Each worker's suite controller is one foreground process. Training and sharded
evaluation launch their own eight-rank process groups sequentially; the
controller itself must not be launched under `torchrun`. Startup verifies the
actual Job/Pod UID, allocation fields, eight distinct GPU UUIDs and distributed
all-reduce. A low-priority node can still be preempted externally; "one node"
is not a promise that an allocation will survive to the last task.

The current storage schema is v2. Four declared, non-overlapping roots are
accounted together: `~/.cache/Paper/DeepProbe-VLM`,
`~/.cache/Paper/DeepProbe-VLM-p1`, `~/Paper/DeepProbe-VLM`, and
`~/.config/deepprobe-vlm`. The exact remote paths are bound in the environment.
All workers share one reservation ledger. Combined projected home use,
including live reservations and the **2,000,000,000-byte safety reserve**, must
stay **below 150,000,000,000 bytes**, with a **1,000,000,000-byte filesystem
free-space guard**. Charge restored inputs, benchmark PNGs, weights, FP32 Adam
state, checkpoint rotation, logs and the 64 MiB OSS multipart scratch. New
training stages estimate their peak using real safetensor headers and recheck
capacity. Do not credit a future deletion before its cleanup receipt exists.

The user removed the OSS quota: `oss_limit_bytes: null`. Retain verified OSS
checkpoint archives; do not delete them to satisfy an obsolete bucket cap.
Eligible local weights can be removed after verified OSS archival and all
declared consumers complete. A ModelScope backup is not a prerequisite for
that local cleanup when the accepted OSS archive is retained.

Checkpoint cleanup releases local disk space. Separate process exit/model
unloading releases host RAM and GPU memory. Removing a checkpoint while its
model process is still running would not solve those other memory limits.

## 1. Prepare original-video caches in the bucket

Follow [SUITE_BENCHMARK_CACHE.md](SUITE_BENCHMARK_CACHE.md). The executable
entry points are:

- `scripts/operations/build_suite_media_binding.py`: pin accepted original
  decode evidence and exact archive catalog; fill member CRC/compressed size
  from ZIP metadata without extracting videos.
- `scripts/operations/build_suite_frame_cache.py`: two complete original-video
  decode passes, actual `uniform_indices`, direct protected upload to a new
  stream prefix, and full PNG GET/SHA readback.
- `scripts/operations/hydrate_suite_frame_cache.py`: restore only bound PNGs
  into one owned Linux-node media directory and issue data acceptance after
  complete verification.

Build F4 and F16 independently from the official original video members. F8
can be adapted from an accepted legacy cache only with its original decode
receipt, member SHA and exact frame-object mapping. Do not resample cached F8
images for the frame-count ablations. TempCompass reverse/speed/concat clips
must use the official processed members, preserving their source components.

The cache-building step writes only small manifests/receipts locally; videos
and generated PNG payloads remain in the bucket. Hydration runs on the remote
Linux host/node, never as a development-machine media download. Once all
required frame variants are present and byte verified, retain the cache
completion receipts and their exact SHA values.

## 2. Freeze source and bind real machine assets

Use a dedicated source release, then stop editing it. Pin all files under
`src`, `scripts`, `configs`, and `requirements`, plus `pyproject.toml` and
`README.md`, using `formal_integrity.source_inventory`. The separate training
source manifest must have `schema_version: 1`, purpose
`DEEPPROBE_FROZEN_P1_SOURCE`, and the returned relative-path/SHA map. Save that
manifest outside the executable source tree; verify it with
`verify_frozen_source_manifest`. Existing historical exact resumes retain
their own independently accepted source release and source manifest.

The suite's `source_files` is a list of absolute `path/bytes/sha256` pins.
Cover the complete imported package, CLI scripts and dynamic operation
imports, including `scripts/preflight.py`, `score_tempcompass_mc.py` and
`run_p1_stage_v9.py`. Do not pin only a wrapper while leaving imported code
mutable. All 34 training recipes are additionally byte-bound by the plan and
`configs/train/suite/RECIPES.json`.

Use an already verified interpreter/runtime with its required Torch,
Transformers, CUDA/Triton, PyAV, Pillow, PyYAML and OSS dependencies. The suite
enforces the frozen `experiment/src` import root and offline model/data
settings. Do not append an unpinned library directory, install a package
during a formal task, or download model weights implicitly at startup.

Prepare a private machine environment JSON. Its top-level fields are defined
by `suite_bindings.validate_environment`:

- `schema_version`, `suite_id`, `experiment_root`, `python`, `run_root` and
  `archive_root`: exact execution identity and paths. The archive root is
  the fresh stream namespace before the per-run `/runs/<run_id>/` suffix.
- `storage`: v2's four exact roots, shared `ledger_root`,
  `home_limit_bytes: 150000000000`, `home_safety_bytes: 2000000000`,
  `free_guard_bytes: 1000000000`, and `oss_limit_bytes: null`.
- `source_files` and `runtime_environment`: immutable sources and the allowed
  runtime variables. Credentials stay in the installed config files.
- `task_reserve_bytes`: explicit reserves for startup, gate, alignment, SFT,
  evaluate, latency, compare-latency, archive, cleanup and aggregate. Optional
  `task_reserve_by_id` refines a specific task's reserve.
- `tools`: exact installed `kubectl`, `kubeconfig`, `ossutil`, `oss_config`
  paths; `allocation_identity`: exact `job_name`, `namespace: default`, and
  project prefixes covering all DeepProbe workloads.
- `assets`: training, benchmarks, offline backbones and development inputs;
  `history`: independently accepted historical inputs keyed by declared task
  ID. An empty or guessed historical acceptance does not pass admission.

Training assets bind the real 91,826 train / 4,472 validation QA, formal-ready
artifact, source manifest, media root and data directory. Backbones bind local
language/vision roots and full file inventories. Development assets bind the
manifest, media root and source metadata declaring a development-only,
non-quality-selected latency sample.

Benchmark keys are `videomme899` and `tempcompass_mc`, each with string keys
`"4"`, `"8"` and `"16"`. Each value is either a verified ready data fragment or
the pinned lazy-hydration descriptor from `cache_asset_binding`. The evaluator
verifies the actual frame count and exact coverage before inference. For
hydration details and the accepted legacy descriptor, use the cache runbook.

Historical entries explicitly select `accepted` reuse or `resume`. They bind
configuration, frozen source, checkpoint file inventory, exact local path or
stream prefix, and `SUITE_HISTORY_ACCEPTED` evidence. A resume requires
`exact_resume_accepted`, matching optimizer step, complete optimizer/scheduler
state and per-rank RNG. SFT reuse must bind its actual matching alignment
initializer. Restoration is into the new suite-owned run; it does not adopt
or delete the historical run. No held-out metric authorizes historical reuse.

## 3. Build the complete command binding

The following examples use explicitly supplied `SUITE_*` variables containing
the real absolute paths and previously computed file SHA256 values. They are
invocation patterns, not a claim that a machine environment or receipt already
exists. `SUITE_EXPERIMENT_ROOT` is the frozen **experiment root**, not the
`configs` directory.

```bash
"$SUITE_PYTHON" -B "$SUITE_EXPERIMENT_ROOT/scripts/operations/build_experiment_suite_binding.py" \
  --plan "$SUITE_PLAN" --plan-sha256 "$SUITE_PLAN_SHA" \
  --environment "$SUITE_ENVIRONMENT" --environment-sha256 "$SUITE_ENVIRONMENT_SHA" \
  --config-root "$SUITE_EXPERIMENT_ROOT" --output "$SUITE_BINDING"
```

This expands every declared task into exact argv, inputs, reserves and
completion checks. `SUITE_BINDING_WRITTEN_NOT_ADMITTED` means command binding
only; it does not attest real media, historical checkpoints or GPU admission.
Compute and retain the resulting binding file's SHA before the next command.

```bash
"$SUITE_PYTHON" -B "$SUITE_EXPERIMENT_ROOT/scripts/operations/run_experiment_suite.py" \
  --plan "$SUITE_PLAN" --plan-sha256 "$SUITE_PLAN_SHA" \
  --binding "$SUITE_BINDING" --binding-sha256 "$SUITE_BINDING_SHA" \
  --state-root "$SUITE_STATE_ROOT" --config-root "$SUITE_EXPERIMENT_ROOT"
```

Without `--execute`, this validates the full plan/binding/source pins and
returns `SUITE_BINDING_VALID_NO_EXECUTION`. It neither runs experiments nor
creates a scheduler job. The initially chosen state root must be fresh; keep
preflight and launch evidence in separate paths within the accounted storage.

## 4. Complete CPU admission

```bash
PYTHONPATH="$SUITE_EXPERIMENT_ROOT/src" "$SUITE_PYTHON" -B -m deepprobe_vlm.suite_preflight \
  --plan "$SUITE_PLAN" --plan-sha256 "$SUITE_PLAN_SHA" \
  --binding "$SUITE_BINDING" --binding-sha256 "$SUITE_BINDING_SHA" \
  --environment "$SUITE_ENVIRONMENT" --environment-sha256 "$SUITE_ENVIRONMENT_SHA" \
  --state-root "$SUITE_STATE_ROOT" --output "$SUITE_CPU_PREFLIGHT"
```

Run this on the remote host with the actual bound assets. Use the implemented
`--prepare-media` option only when intentionally hydrating the already accepted
bucket frame caches into their owned node directories. A plain admission run
must not be represented as permission to download missing original datasets.

Review every failure. The required result is `SUITE_CPU_ADMITTED`, with an
empty failure list, exact plan/binding/environment/source SHA values, retained
outputs, and fresh capacity observations. A `SUITE_CPU_GATED` result keeps the
whole long suite gated. Do not bypass it by excluding a difficult ablation,
inserting an assumed receipt, or repeatedly launching the failed preflight
without correcting its cause.

The launcher requires a CPU admission generated within the last 30 minutes.
After changing inputs or completing missing prerequisites, create a new
admission output and recompute its SHA. Capacity and source identity are
checked again during launch and before tasks; a CPU estimate cannot reserve
future free space or prove a CUDA implementation is correct.

## 5. Review exact YAML and launch once

For the current one-to-four-worker profile, use
[REMAINING_SUITE_WORKFLOW.md](operations/REMAINING_SUITE_WORKFLOW.md) and the fleet
launcher. The commands below document the retained **legacy v1 single-worker
low-priority launcher**; they do not change its fixed allocation checks to
accept a normal-priority fleet. Never reuse a legacy launch intent as a new
fleet submission.

Prepare one candidate PyTorchJob YAML with the installed scheduler's
documented dry-run tooling. This repository's launcher consumes that reviewed
YAML; it does not guess scheduler flags or select a node. Bind its actual
image, volumes, user/runtime context and exact Job name. Its one container's
`command + args` must equal `suite_launch.runner_argv(...)`, using the same
plan/binding/state root and `--execute`. It must have the frozen working
directory and explicit offline/import environment, no auxiliary containers,
and the single-node stream/low/eight-GPU/Never contract.

Old scheduler output containing an interactive shell/user-switch wrapper or
duplicate environment entries is not automatically a valid suite YAML.
Review and validate the final concrete document; do not broaden the validator
to accept an unexamined alternate execution path.

```bash
"$SUITE_PYTHON" -B "$SUITE_EXPERIMENT_ROOT/scripts/operations/launch_experiment_suite.py" \
  --plan "$SUITE_PLAN" --plan-sha256 "$SUITE_PLAN_SHA" \
  --binding "$SUITE_BINDING" --binding-sha256 "$SUITE_BINDING_SHA" \
  --environment "$SUITE_ENVIRONMENT" --environment-sha256 "$SUITE_ENVIRONMENT_SHA" \
  --job "$SUITE_JOB_YAML" --job-sha256 "$SUITE_JOB_YAML_SHA" \
  --preflight "$SUITE_CPU_PREFLIGHT" --preflight-sha256 "$SUITE_CPU_PREFLIGHT_SHA" \
  --state-root "$SUITE_STATE_ROOT" --output "$SUITE_LAUNCH_EVIDENCE" --dry-run
```

The launcher checks full source inventory, fresh admission, capacity, no active
project workload, all exact per-run archive prefixes and the server-side
dry-run result. `--dry-run` creates no job. Running the same invocation with
`--submit` in place of `--dry-run` is the explicit one-shot submission. With
neither flag, it performs preparation checks only.

Keep `SUBMISSION_INTENT.json`, `SUBMISSION_CLIENT_RESULT.json` and the exact
Job UID reconciliation receipt. An ambiguous client return is not permission
to resubmit. Re-run with `--reconcile` and the same launch evidence directory;
the durable intent causes an exact Job lookup and prevents another create.
Even an empty lookup after an intent requires investigation, not blind retry.

## 6. Execution, failure and resume

On the allocated node, startup verifies live allocation/hardware identity. The
CUDA pooling gate runs the actual BF16 and FP32 checks before the matrix.
Subsequent tasks execute one process group at a time and require their exact
dependencies' validated completion receipts. A subprocess exit code alone does
not satisfy training, evaluation, latency or archival completion.

The runner keeps `SUITE_IDENTITY.json`, invocation evidence, an append-only
journal, per-task attempts and `tasks/<task_id>/COMPLETED.json`. A failed or
interrupted task preserves logs and stops downstream work. It does not raise
priority, restart from zero, silently skip incomplete outputs, or discard a
failed numerical comparison.

For an explicitly recovered invocation with the same admitted identity, the
runner command adds both `--execute --resume`. It verifies prior completion
receipts and source/binding identity and invokes the task's declared recovery
command. Training resume requires a complete compatible checkpoint including
optimizer, scheduler and all rank RNG state. A partial checkpoint with only
model weights is insufficient.

External preemption requires fresh exact Job/Pod checks, no duplicate
allocation, CPU admission and reviewed YAML again. Changing the Job name
changes the environment and its generated binding; do not edit
`SUITE_IDENTITY.json` or old completion journals to force a match. Use only the
explicit relocation procedure, preserving its old/new binding evidence chain:

1. Preserve the previous environment, generated binding and all state files.
   Create a new environment changing only `allocation_identity.job_name`,
   then use the normal binding builder to generate its new complete binding.
   All other sources, assets, history, paths, runtime, storage and scientific
   configuration must be identical.
2. Prepare an immutable relocation receipt with the actual old/new SHA values:

   ```bash
   "$SUITE_PYTHON" -B "$SUITE_EXPERIMENT_ROOT/scripts/operations/prepare_suite_relocation.py" \
     --plan "$SUITE_PLAN" --plan-sha256 "$SUITE_PLAN_SHA" \
     --previous-binding "$SUITE_PREVIOUS_BINDING" --previous-binding-sha256 "$SUITE_PREVIOUS_BINDING_SHA" \
     --previous-environment "$SUITE_PREVIOUS_ENVIRONMENT" --previous-environment-sha256 "$SUITE_PREVIOUS_ENVIRONMENT_SHA" \
     --binding "$SUITE_BINDING" --binding-sha256 "$SUITE_BINDING_SHA" \
     --environment "$SUITE_ENVIRONMENT" --environment-sha256 "$SUITE_ENVIRONMENT_SHA" \
     --state-root "$SUITE_STATE_ROOT" --output "$SUITE_RELOCATION"
   ```

3. Require `SUITE_ALLOCATION_RELOCATION_VERIFIED`. This preparation does not
   submit or alter the state journal. Compute the receipt's SHA, run fresh CPU
   admission against the new environment/binding, and review the new exact Job
   YAML. Its runner argv adds `--resume --relocation "$SUITE_RELOCATION"
   --relocation-sha256 "$SUITE_RELOCATION_SHA"` as well as `--execute`.
4. Use the launch command with the same state root, those same three relocation
   arguments, and a **new launch-evidence directory**. Server dry-run comes
   before explicit one-shot submit. The old launch-evidence directory remains
   exclusively available for reconciliation of its original submission intent.

At startup the runner appends the verified transition under
`state/relocations/`. Old `COMPLETED.json` bytes/hashes remain unchanged;
their binding identity is accepted only through the audited transition chain.
Runtime or numerical changes require a new scientific review, not an
allocation-only relocation. This mechanism does not waive the outstanding
independent audit of an older R4/SmolLM2 training checkpoint.

Latency reference/candidate comparisons require compatible hardware/runtime
and equal generated tokens. If a new allocation changes the measurement
environment, take a new declared reference/candidate pair on that allocation
and preserve earlier attempts. Do not combine measurements across nodes or
reinterpret a token mismatch as an infrastructure failure.

## 7. Archive completely, then release eligible checkpoint weights

See [CHECKPOINT_RETENTION.md](CHECKPOINT_RETENTION.md) for the exact ownership,
manifest, consumer receipt and deletion-journal contracts. New runs are claimed
before training. The full run archive includes final/recovery checkpoint state
and evidence and is immutable after its manifest is frozen. Every archived
object must pass complete OSS GET/SHA, with exact listings before and after.
Model results use a separate archive prefix so later evaluation cannot mutate
a frozen checkpoint archive.

Each cleanup task requires successful training, its full verified checkpoint
archive, the model's verified result archive and **every** declared checkpoint
consumer. Shared alignment initializers remain until all SFT consumers finish.
SFT final weights remain through both benchmarks, required latency work and
the plan's comparison/result-archive dependencies. A missing score shard or
failed comparison leaves the model weights in place.

Immediately before unlink, the retention adapter performs a fresh full archive
audit, checks local bytes and consumer output pins, and checks for live writers
under the suite/run locks. It deletes only eligible weight/optimizer payloads
inside the exact new suite-owned checkpoint trees. Configs, RNG/scheduler
state, tokenizer/code provenance, small metrics, logs, receipts and deletion
journals remain. An unowned historical run or pretrained cache never becomes
a cleanup target by matching a filename.

Completed evaluation scores are derived from actual full prediction coverage.
Final aggregation requires all 36 declared model/benchmark cells and retains
their provenance; it does not fill missing cells from another seed, an earlier
protocol, an incomplete recovery, or CPU tests. The final deliverable consists
of those scientific results, CUDA/latency evidence, exact bucket manifests,
source/config pins and the durable completion/cleanup journals.

## Historical inputs and offline runtime details

Every historical training binding also supplies its exact `source_root`. The
pinned source manifest is verified against that tree; recovery keeps the
original runtime configuration bytes and the original data/batch/schedule
contract. Recovery copies or restores into a fresh owned suite run and leaves
the original interrupted run intact. On a later interruption, the newest
structurally complete checkpoint in that owned run takes precedence over the
currently bound step350/step600 recovery input. The older Smol a3–a5 packages
retain their historical step300 initializer. Torn newer saves are preserved.

An OSS-only historical checkpoint additionally supplies an
`archive_acceptance` file pin present in the independent audit's `evidence`.
That acceptance binds `checkpoint_oss_prefix`,
`checkpoint_inventory_sha256`, and
`all_objects_full_get_sha256_verified: true`. These fields require the actual
archival audit; they must not be manufactured from a planned upload. Optional
`checkpoint_metadata` maps names such as `deepprobe_config.json`,
`training_provenance.json`, and `trainer_state.json` to small readback pins
matching the same checkpoint inventory. CPU admission checks these identities
without staging all historical weights. Fresh full weight GET/SHA verification
is deferred explicitly to that model's bounded restore, before model loading.

Set `HF_HUB_CACHE` or `HF_HOME` explicitly. Any `TRANSFORMERS_CACHE` must resolve
to the same reviewed cache. Backbone pins cover the complete pinned language
and vision snapshots, including configuration, tokenizer/processor and shard
indexes. Startup never installs or upgrades dependencies.

If a relocation splits a latency pair across GPUs or runtimes, the comparison
action remeasures the exact reference and candidate pair on its current
allocation. Earlier measurements remain immutable and both replacement runs
are included in the comparison's result archive. A token-parity failure is
preserved as a failure and does not trigger repeated trials seeking a pass.

## Preparation verification

The historical full public CPU suite passed **624 tests**, with **80 skips**, on
2026-11-12. The captured pytest output SHA-256 is
`cda7bdbf62b4f80faa5ea07385227cc3d5ce3118c04b85844e7cc4343a5c7820`.
Skipped CUDA/private-evidence checks do not count as real GPU acceptance.
This preparation did not submit the long suite, create actual F4/F16 caches,
or delete any real checkpoint.
