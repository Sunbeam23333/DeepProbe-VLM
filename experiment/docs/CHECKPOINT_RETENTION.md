# Serial-suite checkpoint retention

Checkpoint cleanup releases local disk space. It does not substitute for process
exit or model unloading when reclaiming host RAM or GPU memory.

`deepprobe_vlm.checkpoint_retention` is a library called by the serial runner;
it never submits jobs. Its destructive `cleanup` and `restore_checkpoint` calls
default to planning. The runner holds its suite lock and supplies a **live**
guard reporting the current run ID, source SHA and an empty active-writer list.
Each operation also holds a separate run lock. A saved JSON assertion alone is
not a live no-writer check.

## Ownership and archive

Claim a new, empty output run before training:

```python
from deepprobe_vlm import checkpoint_retention as retain
from deepprobe_vlm.checkpoint_store import OssutilStore

owner = retain.claim_owned_run(runs_root, run_id, suite_id, source_sha256)

def no_writer():
    # The caller checks its actual supervised process while holding the suite lock.
    return {"run_id": run_id, "source_sha256": source_sha256,
            "active_writers": active_process_ids_for_this_run()}
```

An existing nonempty unowned directory cannot be claimed. Historical failed runs
and pretrained caches therefore do not become deletion targets. Existing
ownership must match the exact suite, source and physical run root. Files with
symlink components or multiple hard links are rejected.

After the real training-completion validator succeeds, obtain
`checkpoint_inventory(run_root, "training/final")["inventory_sha256"]` and
write a source-bound completion receipt with these fields:

- `schema_version: 1`, `status: "SUITE_TRAINING_COMPLETE"`, `run_id`, `source_sha256`.
- `final_checkpoint_inventory_sha256`, `optimizer_steps`, `target_optimizer_steps`,
  and `exit_code: 0`. The actual and target update counts must be equal and positive.
- `outputs`: nonempty `{path, bytes, sha256}` pins for the retained native training
  completion, Trainer state and other small validation outputs.
- `output_inventory_sha256`: `retain.digest(outputs)`.

This wrapper records acceptance of real outputs; the runner must not fabricate
its step count from the target configuration. Its output pins are read and
SHA-checked again by retention. Keep the wrapper and its small proofs permanently.

Freeze a manifest after closing the run's payload logs:

```python
manifest = retain.prepare_manifest(
    owner, archive_prefix=exact_fresh_stream_prefix,
    required_consumers=declared_consumers,
    training_completion=retain.file_pin(training_receipt), guard=no_writer)
store = OssutilStore(
    ossutil_executable, ossutil_config,
    allowed_prefix=exact_fresh_stream_prefix,
    state_dir=run_root / ".retention" / "oss")
archive_receipt = retain.archive_and_verify(manifest, store, guard=no_writer)
```

The prefix must be in `ss-devbox-stream`, isolated to a specific `runs/<run_id>/`.
Use a fresh suite namespace for it. The manifest freezes every regular run file
except `.retention/`, `tmp/` and `cache/`; it includes all checkpoint state and
small training evidence. Later evaluation output and suite journals use their
own directories/prefixes. Adding objects to this frozen archive prefix makes its
exact-inventory check fail.

Archive verification lists the complete prefix, streams **every** object through
SHA256, and requires an unchanged complete listing afterward. A status field,
CRC-only upload, file size or model-header check is insufficient. Matching
partial uploads may resume; extra or different objects stop the operation.
No existing object is overwritten and no object is deleted.

The concrete store uses the installed `ossutil` config without reading credentials
into Python or logging them. `cat -q` provides the raw GET stream. Large files
use explicit multipart initiation, part upload and completion, with
`--forbid-overwrite true` on initiation and completion. One 64 MiB temporary part
is staged; full checkpoints are never duplicated just for readback. The 64 MiB
scratch and small journals must be charged to the caller's disk reserve.
Unknown initiation/completion outcomes require exact-object/session reconciliation;
they do not trigger a new upload automatically. Interrupted individual parts may
resume under the same upload ID. Credentials and raw service bodies are not
written to public results.

## Dependency-aware cleanup

Each declared consumer has:

```text
{id, kind, checkpoint, protocol_sha256, expected_units}
```

`kind` is `training`, `evaluation`, `latency` or `comparison`; `checkpoint` is
`training/final` or an explicitly consumed `training/checkpoint-N`. IDs must cover
the full dependency graph, including SFT initialization and all requested
benchmarks, latency or model comparisons. An alignment final is still needed
until its matching SFT consumer completes. A final SFT checkpoint remains until
all of its declared evaluations and measurements complete.

After independently validating an actual consumer's output and full coverage,
the runner writes `SUITE_CONSUMER_COMPLETE` with `schema_version: 1`,
`consumer_id`, `kind`, `source_sha256`, `checkpoint_inventory_sha256`,
`protocol_sha256`, `completed_units`, `failed_units: 0`, and the same nonempty
`outputs`/`output_inventory_sha256` structure. `completed_units` must equal the
declaration. Pins must refer to retained small coverage/score/acceptance artifacts,
not weights that a later cleanup legitimately removes. The retention module
checks the receipt hash, all fields and every bound output file.

```python
# Optional early release: training finished, full archive verified, and these
# intermediate recovery checkpoints have no remaining declared consumers.
plan = retain.plan_cleanup(manifest, {}, checkpoints=["training/checkpoint-718"])
retain.cleanup(plan, store, guard=no_writer, execute=True)

# Final model: requires every final-checkpoint consumer receipt, keyed by ID.
plan = retain.plan_cleanup(manifest, final_consumer_receipt_pins,
                          checkpoints=["training/final"])
deletion = retain.cleanup(plan, store, guard=no_writer, execute=True)
```

Omitting `checkpoints` selects all checkpoints conservatively. Selection never
bypasses the global successful-training receipt. Cleanup performs a fresh full
archive GET/SHA audit and rechecks local bytes and consumer outputs. Only model
`.safetensors`, native `pytorch_model*.bin`, and `optimizer.pt` under the named
checkpoint trees are removed. Configs, tokenizer/code/provenance, RNG/scheduler,
logs, scores and receipts remain.

Every file gets a durable intent before unlink and a completion record after it.
Interruption between those steps can resume without treating an unexplained
missing file as a successful deletion. Earlier legitimate intermediate cleanup
is recognized when later clearing final weights. The final `DELETION.json` has
status `OWNED_CHECKPOINT_WEIGHTS_REMOVED`, a plan hash, archive-verification receipt,
exact file list and consumer pins. This is the cleanup task's durable output.

## Restoration and limits

`restore_checkpoint(manifest, checkpoint, new_directory, store, execute=True)`
streams the chosen checkpoint directly into a new marked restore directory.
Each file is published only after its complete size/SHA matches. Matching files
survive restart; differing existing files are refused. `store.iter_object(uri)`
also supports restoring historical manifests through an explicitly validated
adapter, without adopting or deleting their original run.

The module does not infer scientific validity from a scalar status, discover
hidden downstream consumers, or police unrelated processes that ignore the
caller's suite lock. The runtime must supply the complete DAG, real task-specific
acceptance/coverage validators, a live process guard and fresh dual-root capacity
checks. CPU/fake-store tests verify these contracts; no real multipart upload or
destructive operation on an experiment checkpoint was performed during development.
