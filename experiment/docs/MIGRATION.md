# Portable migration — November 21, 2026

Code repository: <https://github.com/Sunbeam23333/DeepProbe-VLM>.
Data, checkpoint and log destination:
<https://modelscope.ai/models/JasonJin/DeepProbe-VLM/files>.
All migration files are below `migration/2026-11-21/` in that **model** repository.
The older result-only dataset repository is a separate publication destination.

## Saved code and logs

`source-and-local-evidence.tar.gz` contains 17,090 source, configuration,
operation and log files from the local project. Its 241,742,073 bytes were
uploaded and fully downloaded again; SHA256 is
`771de6a8cccd17156764cf2350f1a370d893da10ef33e44901bc0c77fb634491`.
Recognized credentials were redacted. `SNAPSHOT_MANIFEST.json` inside records
every included file and any omission. Original private evidence is retained.

`remote-code-config-logs.tar.gz` is the corresponding snapshot from both project
directories on the old development machine. `current-code.tar.gz` preserves the
current tracked repository including the migration tools. Consult the publication
receipts and actual repository inventory for transfer completion; a planned name
does not mean that its transfer has completed.

Archived control scripts describe old Job/Pod identities and absolute paths.
Keep them as provenance. For a new machine, prepare a new environment and new job
identity using the current code and restored assets. Do not replay old submission
scripts or treat an old admission/lock receipt as approval for a new allocation.

## Data transfer scope

The pinned first transfer plan contains **42,652 OSS objects / 226,675,594,709
bytes**, divided into 877 tar shards with up to 256 MiB of source payload each:

- `00-evaluation-results`: existing evaluation outputs (52.86 MB).
- `01-training-manifests`: training annotations, splits and coverage (1.63 GB).
- `02-training-videos`: the actual archived training video source (5.30 GB).
- `03-benchmark-frame-caches`: F4/F8/F16 evaluation caches and metadata (14.87 GB).
- `04-base-models-and-assets`: pinned base models and source metadata (10.28 GB).
- `05-accepted-models`: final weights for all seven accepted training stages
  (53.49 GB), including the three complete R1/R2/R4 seed17 models.
- `06-r4-native718`: R4's durable native step-718 checkpoint (22.32 GB).
- `07-original-benchmark-archives`: original benchmark archives (118.74 GB).

`TRANSFER_PLAN.json` pins source keys, ETags, ranges and known checkpoint hashes.
Each tar contains `SHARD_MANIFEST.json` with per-piece SHA256 values. A shard is
accepted only after full download and SHA256 verification from ModelScope.
`TRANSFER_COMPLETE.json` is written only after all 877 planned shards pass.
Until that receipt exists, the transfer is incomplete.

The full historical OSS inventory was approximately 1.17 TB. Redundant old
optimizer checkpoints stay in OSS; this transfer prioritizes the data and
accepted states needed on a new machine. No source checkpoint or live job is
deleted by these tools. The worker reads OSS directly and uses one temporary
readback shard outside `/home`, so it does not duplicate the dataset in the
old machine's project storage.

## Download and verify on a new machine

Install the lightweight download client and select a destination with enough
free space for the categories you need:

```bash
python -m pip install modelscope-hub
python experiment/scripts/operations/restore_migration.py \
  --output /your/storage/deepprobe-migration
```

For a smaller initial restore, repeat `--category`:

```bash
python experiment/scripts/operations/restore_migration.py \
  --output /your/storage/deepprobe-migration \
  --category 01-training-manifests \
  --category 02-training-videos \
  --category 04-base-models-and-assets \
  --category 05-accepted-models
```

If access requires authentication, set `MODELSCOPE_TOKEN` in your environment;
never add it to a script or repository. The restore tool reassembles original
objects under `OUTPUT/oss/jinhaopeng/Paper/DeepProbe-VLM/`, verifies each piece,
and verifies the full accepted-checkpoint hash where provided. Missing shards
cause a clear stop; a later run reconciles existing output before continuing.

Training manifests are under `datasets/formal-p1-20261009-v1/`. Extract the
training video archive into a media directory containing
`liwei_youtube_videos/videos/youtube_video_2024/`; manifest `video_path` values
are relative to that directory. Final checkpoint source prefixes are listed in
`TRANSFER_PLAN.json` under `accepted_checkpoint_prefixes`. Benchmark cache
receipts retain the original frame names and source provenance.

## Scientific state at migration start

At 22:43 Beijing time, a12 was still Running with its original Job/Pod identity.
R4's two new benchmark evaluations had been accepted, bringing totals to
**7/34 training stages, 3/18 complete models and 6/36 evaluations**. Its
reference latency task was running. The full suite remains incomplete.

This transfer plan is a fixed snapshot. Newly completed a12 results are retained
in its live state and in the follow-up code/log snapshots; they require a final
delta backup before the old machine can be retired. Transfer completion and
scientific experiment completion are separate conditions.
