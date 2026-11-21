# Private disk execution for the remaining experiment suite

The current deployment uses one private-disk worker to drain the complete
185-task DAG. On November 21, real two-node CPU probes showed that the shared
home mount did not provide cross-node `flock` exclusion and could return an old
generation after another node replaced a file. The multiworker shared-journal
profile is therefore not admitted on that mount.

The optional `isolated_storage` contract reserves the worker's entire home
allowance once in the existing coordinator ledger. Existing CPU cache producers
continue using that ledger on the development machine. The GPU keeps its own
mutable budget ledger on private disk and counts all retained output in a new
home namespace against the delegated allowance. The parent hold never expires
or releases automatically. Immutable observations preserve the worker's budget
history for recovery without a shared mutable pointer.

In the isolated Pod, the old home and both vePFS aliases are read-only. A single
writable host-path overlay exposes the new home namespace; checkpoints and
CUDA caches use private disk. The accepted offline Hub remains a read-only
model input. Python module and Kubernetes discovery caches have separate writable
locations. Prepared Pod validation and live kernel mount checks enforce this
layout before any worker state or lease is written. Benchmark consumers hydrate
and retire PNGs in the new namespace while preserving the original cache READY
receipts and scientific protocol.

The current allowance is 48 GB, including a 1 GB namespace safety reserve,
under the independent 150 GB project-home limit. This is storage admission,
not a filesystem quota. Future allowance release requires exact worker
quiescence and a final namespace inventory. Scientific completions still require
their training/evaluation receipts; a passed isolation test is not such a result.

The scientific plan remains the complete 185-task DAG. Execution profile v2
keeps the existing shared-home, one-resident-chain scheduler. Opt-in profile v3
binds one to four independent eight-GPU workers, each with a 100 GiB disk
`emptyDir`. Priority remains explicitly `low` or `normal`; the profile does not
change recipes, samples, seeds, update counts, decoding or evaluation coverage.

Every worker must pass its own exact Job/Pod identity, disk mount, free-space,
GPU and eight-rank Gloo startup checks. The current node's capacity says nothing
about a future worker. All declared workers rendezvous before dispatch; there
is no hot-add or late-join behavior.

## Storage and scheduling

The global journal, source/data inputs, result files and reference ledger stay
inside the declared shared roots, with their existing 150 GB admission ceiling.
Model and native optimizer/RNG saves go under the worker's private run root.
TMPDIR, CUDA, Triton and Torch extension caches are also bound to that worker's
private mount. `HOME` remains unchanged.

Private reservations bind the exact Job UID, Pod UID, node and mount/run-root
device/inode. Actual bytes, pending reservations and safety are checked under
the worker's durable ledger lock. Consumer temporary peaks are conservatively
reserved in both storage domains. The Kubernetes `emptyDir`/ephemeral-storage
limit is an eviction accounting limit, not a hard filesystem ENOSPC quota.
Shared metadata accumulates in measured home usage and is re-admitted before
subsequent tasks; a historical capacity estimate is never a current admission.
Within each training claim, every generated shared JSON and retained metadata
copy is checked against that claim's existing shared token under the common
ledger lock. Accounting includes all its task journals, action receipts,
savepoint generations and retained metadata, plus 1 MiB of control headroom.
The copy check includes temporary publication bytes. Native progress logs stay
in the owned private run and are archived with it. A byte overflow or mismatched
token stops acceptance and preserves artifacts; this is not a lifetime quota.

Each connected initializer chain has one durable worker assignment. This
includes frozen-vision branches, both benchmarks, latency measurements and
comparisons, archives, and cleanup. A worker cannot open another resident chain
before all existing chain cleanup barriers are accepted. Lost claims and
assignments never expire or transfer automatically.

The shared reference ledger records the verified OSS copies. Private paths are
bound by run ownership and the Pod ledger rather than added to shared roots.
Small manifest/completion/deletion receipts are copied byte-for-byte to shared
state before acceptance; original owner/source records are preserved.

## Checkpoint durability

Fresh or explicitly accepted same-source recovery runs enable the savepoint
export hook. All ranks finish a native save before rank zero queues its export.
There is at most one pending export. Before the next native save can rotate any
checkpoint, every rank waits for the previous export to succeed. Formal final
completion also flushes the last export.

An export verifies complete model components, safetensor headers/shard indexes,
offline tokenizer/image processor loading, Trainer state and all eight RNG
files. The immutable generation prefix is:

```
<archive_root>/savepoints/runs/<run_id>/checkpoint-<step>/<inventory_sha256>/
```

Payloads and the manifest are fully read back; complete object identities must
remain stable across verification. `READY.json` is published last. Retained
shared metadata supports later CPU admission even after the old private disk
has disappeared. Up to four independent 64 MiB upload parts reserve 256 MiB.
Long export waits use a separate Gloo CPU control group with an explicit six
hour timeout, without changing the training DDP timeout or schedule. Rank-local
precheck and export errors converge before any later rotation or acceptance.

Final weights remain until every declared training/evaluation/latency consumer
has a matching receipt and fresh archive verification permits cleanup. Failed
or uncertain work remains for explicit recovery. Unlimited OSS retention does
not trigger remote weight deletion.

## Derive and admit a new candidate

`deepprobe_vlm.suite_private_candidate` consumes byte-pinned base/plan/change
documents and writes a new candidate. The change document explicitly provides
the new source catalog and manifest, complete reviewed history, state path,
archive namespace and v3 profile. It can also pin `f16_declarations`: exactly
the two F16 contracts and producer requests are replaced; F4/F8 declarations
remain unchanged, and no READY file is created or assumed.

```sh
python -B -m deepprobe_vlm.suite_private_candidate \
  --base "$DP_BASE" --base-sha256 "$DP_BASE_SHA" \
  --plan "$DP_PLAN" --plan-sha256 "$DP_PLAN_SHA" \
  --changes "$DP_CHANGES" --changes-sha256 "$DP_CHANGES_SHA" \
  --output "$DP_NEW_CANDIDATE_DIRECTORY"
```

The existing `deepprobe_vlm.suite_workflow` entry performs asset preparation,
CPU admission, exact empty-prefix checks and server dry-run before submission.
Use a new control directory and fresh evidence. Candidate derivation neither
retires the old fleet nor copies its COMPLETED journals. A separately reviewed
terminal/quiescent handoff must precede actual new fleet admission.

Completed work may be imported only as a fully closed checkpoint chain with
all consumers, archives and cleanups accepted. The explicit typed importer
retains the original evidence and writes `task-imports/<task-id>.json`, never a
new `COMPLETED.json`. New allocation GPU gates and result aggregation still
run. An unfinished chain uses normal reviewed history restore; an old archive
or cleanup receipt never grants deletion authority over a new private run.
The importer currently rejects groups with latency comparisons and historical
cleanup that relied on already retired remote intermediate weights. Those
histories need separate verification; neither is silently marked complete.

The current migration retains all six existing F4/F8/F16 declarations. The
optional F16 replacement interface is available for separately reviewed future
use, but no companion is enabled here without continuous placement and
latency guards.

## Explicit recovery after private disk loss

`deepprobe_vlm.suite_checkpoint_resume` reads a pinned exported generation and
produces a CPU-only proposal. It verifies the source/config/data/step contract,
model inventory, eight-rank native state and the retained metadata. The old Pod
directory need not exist. New and original source paths may differ only when
both trees verify against the same complete frozen-source manifest.

```sh
python -B -m deepprobe_vlm.suite_checkpoint_resume \
  --plan "$DP_PLAN" --plan-sha256 "$DP_PLAN_SHA" \
  --environment "$DP_ENVIRONMENT" --environment-sha256 "$DP_ENVIRONMENT_SHA" \
  --export "$DP_EXPORT_RECEIPT" --export-sha256 "$DP_EXPORT_RECEIPT_SHA" \
  --task "$DP_TASK_ID" --output "$DP_RECOVERY_PROPOSAL"
```

The default proposal contains no acceptance. An independent review must bind
the exact entry and retained evidence. Supplying its `--review` and
`--review-sha256` to a new proposal output uses the existing history importer.
Use that reviewed entry in a new state/profile/namespace candidate, with a new
Pod ledger and fresh admission. Runtime restore hashes every payload, then
uses native `--resume` to preserve optimizer, scheduler, RNG and global step.
It does not claim native state was deserialized during CPU metadata admission.
An old source without this exporter cannot use this recovery route.

One explicit exception is the independently reviewed original R4 seed-17 SFT
checkpoint 600. `legacy_private_resume` pins its separate bridge review. The
verifier hard-binds this one task, original source/config, the complete
31-object OSS generation, training data and 600-to-718 update budget. The old
scientific source stays unchanged and has no in-training export hook. Until
successful exit, checkpoint 600 remains the known durable recovery point.
The private executor uses `run_group` and retains both the exact training
command and its successful process-tree quiescence evidence. An existing
formal-completion marker alone cannot authorize an export. Only after complete
native step-718 validation does the outer executor export that native save;
its READY records `single_process_after_training_exit0` and pins the process
exit and formal-completion evidence. Ordinary hook exports retain their Gloo
protocol. Byte-identical relocation of the outer source manifest by the
workflow is supported; legacy source/config/data pins stay exact.

The action's storage estimator shares the CPU preflight's complete offline
backbone gate. Pinned Hugging Face file aliases must resolve inside the
existing personal-home boundary and retain every expected byte and revision.
Ordinary source and receipt pins still reject symlinks.

## Verification scope

Tests exercise the complete DAG with four competing workers, maximum four
simultaneous SFT claims, chain affinity, private/shared budget failures, stable
remote readback, missing shards/components/vocabulary, single-pending export
and rotation barriers, native same-source recovery admission, and receipt
survival after synthetic private-disk loss. CPU Gloo tests also use eight real
processes to verify rank-error convergence. These checks are not production
GPU/Pod/OSS acceptance or new scientific results.
