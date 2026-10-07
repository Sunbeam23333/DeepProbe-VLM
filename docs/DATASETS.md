# Dataset acquisition and reproducible manifests

This repository distributes **code, not datasets, videos, weights, or credentials**.
No benchmark test answers may be used for SFT, prompt selection, early stopping,
or hyperparameter selection. Evaluation answers remain in a separate manifest
field and must never enter the model input. The scripts below do not download
anything unless `--download` is explicit. Run from the repository root.

## Sources and immutable revisions

Official source metadata and small annotation samples were checked on
**2026-10-07**. `configs/datasets.json` pins the following commits. Repository
contents may change upstream; do not silently replace these pins with `main`.

| Role | Official dataset | Pinned commit |
|---|---|---|
| Video SFT | [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K) | `6d8c562dc26d70042a0d9704d1cae58c94b89098` |
| Evaluation | [Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME) | `ead1408f75b618502df9a1d8e0950166bf0a2a0b` |
| Temporal evaluation | [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench) | `230a2d4fac8900333c61754641c7a13e069ac9c6` |
| Optional multi-evidence evaluation | [HERBench](https://huggingface.co/datasets/DanBenAmi/HERBench) | `30b42e97180f2c7c0fdc3f2aed412ae04f0366ca` |

LLaVA's card calls its annotation license Apache 2.0 but also restricts usage to
academic research/education and points to generated-data terms. Original video
rights remain separate. MVBench's MIT metadata does **not** license all underlying
videos; its [official card](https://huggingface.co/datasets/OpenGVLab/MVBench/blob/230a2d4fac8900333c61754641c7a13e069ac9c6/README.md)
specifically requires independent authorized access to **320 NTU RGB+D videos**.
HERBench uses CC BY-NC-SA 4.0 and source-dataset terms. Video-MME's pinned Hub
card does not provide a blanket video license: consult the
[official project](https://github.com/MME-Benchmarks/Video-MME) and original
sources. Do not upload any of these assets to this GitHub repository.

## Download plans, budgets and verification

Dependencies for metadata/downloading: `huggingface_hub`; for parquet conversion:
`pyarrow`. Manifest validation and archive handling use only Python's standard
library. Use the project's environment; an existing `HF_TOKEN`/normal Hub login
is used when required. Never put tokens in command lines or committed files.

```bash
# Default = metadata-only dry run; default selection = annotations, not videos.
python scripts/data/download_hf.py videomme
python scripts/data/download_hf.py llava_video_178k --list

# Metadata selection, bounded to 10 MB; plan records SHA, size, ETag/hash.
python scripts/data/download_hf.py videomme --download --max-gb 0.01 \
  --local-dir data/raw/videomme --manifest data/plans/videomme-annotations.json
```

The downloader requires a finite positive byte budget for actual downloads;
unknown file sizes, unmatched patterns, or changed revisions are hard errors.
`--include` may be repeated and replaces the default annotation selection.
It calls `snapshot_download` with the **exact selected file names and pinned
revision**, then verifies every downloaded file's size and LFS SHA256 or Git
blob SHA1. Local SHA256 values are included in the resulting plan. Existing plan
files are never overwritten. Files already in the Hub cache still count toward
the selected-byte cap. Allow extra disk capacity for extraction and checkpoints.

Archive extraction is a separate, explicit operation. Default = preflight/list:

```bash
python scripts/data/safe_extract.py data/raw/videomme/videos_chunked_01.zip \
  --output data/media/videomme --max-gb 12
# Re-run with --extract only after inspecting the archive member paths.
```

The extractor rejects absolute paths, parent traversal, Windows paths, symlinks,
hardlinks, devices, duplicate members, uncompressed-size overflows, and file
overwrites. It does not execute dataset loading scripts. It does not remove
archives after extraction. Inspect partial outputs after any interrupted run;
use a fresh output directory if needed. The cap is **uncompressed decimal GB**.

## 1. LLaVA-Video-178K: start with one controlled training pilot

Do not pull the complete approximately terabyte-scale repository to start.
The following is an explicit short-YouTube pilot: one annotation file and one
video archive. The selected archive is about 5.3 GB; inspect the live plan first.

```bash
python scripts/data/download_hf.py llava_video_178k --dry-run --max-gb 6 \
  --include '0_30_s_youtube_v0_1/0_30_s_youtube_oe_v0_1_qa_processed.json' \
  --include '0_30_s_youtube_v0_1/0_30_s_youtube_v0_1_videos_1.tar.gz'

python scripts/data/download_hf.py llava_video_178k --download --max-gb 6 \
  --include '0_30_s_youtube_v0_1/0_30_s_youtube_oe_v0_1_qa_processed.json' \
  --include '0_30_s_youtube_v0_1/0_30_s_youtube_v0_1_videos_1.tar.gz' \
  --local-dir data/raw/llava --manifest data/plans/llava-youtube-pilot.json

python scripts/data/safe_extract.py \
  data/raw/llava/0_30_s_youtube_v0_1/0_30_s_youtube_v0_1_videos_1.tar.gz \
  --output data/media/llava --max-gb 12 --extract

python scripts/data/convert.py llava \
  --input data/raw/llava/0_30_s_youtube_v0_1/0_30_s_youtube_oe_v0_1_qa_processed.json \
  --media-root data/media/llava --output data/manifests/llava-pilot.jsonl \
  --available-only
```

Verify the archive's top-level directory matches the annotation's `video` field.
If it adds an extra enclosing directory, set `--media-root` to that directory;
the converter never searches heuristically or silently rewrites media names.
One video archive does not contain all videos in its annotation file. Only the
explicit **training-only** `--available-only` flag permits omission and records
every omitted sample in a `.coverage.json` sidecar. This selection is a pilot,
not a random representative subset and not a publishable full-dataset result.
Without this flag, one missing file aborts conversion. It is forbidden on
benchmark converters.

Input schema verified from official samples: `id`, `video`, `data_source`,
`conversations` with alternating `{from: human/gpt, value: ...}` turns.
Each QA pair becomes one JSONL row. `<image>`/`<video>` placeholders are removed;
the model adapter injects visual tokens. Conversation context beyond the current
QA pair is not preserved by this simple SFT conversion, so use independent video
QA/caption subsets rather than context-dependent dialogues.

### Source grouping and contamination checks

All questions about one original source video stay in one split. Source IDs are
canonical across supported corpora: `youtube:<11-character-id>`,
`charades:<original-id>`, etc. Unknown naming fails and requires `--source-map`:
a local JSON object mapping annotation media path to an audited original source
ID. **Do not map cropped clips or questions to separate sources.** An example:

```json
{
  "custom/session7_clip_1.mp4": "my_source:session7",
  "custom/session7_clip_2.mp4": "my_source:session7"
}
```

LLaVA's academic collections and MVBench share source datasets. Exclude evaluation
sources from the training/validation pool before any SFT. First prepare the
benchmark manifests below, then:

```bash
python scripts/data/split.py --input data/manifests/llava-pilot.jsonl \
  --media-root data/media/llava --output-dir data/manifests/llava-split \
  --validation-fraction 0.05 --seed 42 \
  --exclude-manifest data/manifests/videomme-test.jsonl

# Add --exclude-manifest once for every MVBench/HERBench manifest used.
python scripts/data/validate.py data/manifests/llava-split/train.jsonl \
  data/manifests/llava-split/validation.jsonl --media-root data/media/llava
```

Duplicate sample IDs and cross-split source/media overlap are hard errors. Source
IDs cannot discover renamed, re-encoded, or otherwise unidentified duplicate
videos: provenance review and, where feasible, perceptual duplicate checks remain
necessary. These scripts do **not** certify absence of all benchmark contamination.

## 2. Video-MME: evaluation only, no subtitles baseline

The pinned repository has **20** `videos_chunked_*.zip` archives, not the 10
archives found in older documentation; total download is roughly 100 GB. A
complete official test manifest requires all videos. Start by listing the plan.

```bash
python scripts/data/download_hf.py videomme --dry-run --max-gb 110 \
  --include 'videomme/*.parquet' --include 'videos_chunked_*.zip'
python scripts/data/download_hf.py videomme --download --max-gb 110 \
  --include 'videomme/*.parquet' --include 'videos_chunked_*.zip' \
  --local-dir data/raw/videomme --manifest data/plans/videomme-full.json

for archive in data/raw/videomme/videos_chunked_*.zip; do
  python scripts/data/safe_extract.py "$archive" \
    --output data/media/videomme --max-gb 12 --extract
done

python scripts/data/convert.py videomme \
  --input data/raw/videomme/videomme/test-00000-of-00001.parquet \
  --media-root data/media/videomme --video-prefix videos \
  --output data/manifests/videomme-test.jsonl
```

Use the extractor's listing to confirm the actual extracted relative video
directory; change `--video-prefix` if needed. The converter uses `videoID`
(YouTube ID), **not** the numeric `video_id` benchmark index. `question_id`,
`question`, `options`, `answer`, `duration`, `domain`, and `task_type` were checked
against a small pinned parquet sample. Options become plain strings and gold
answers canonical letters. This baseline does not load subtitles/audio; disclose
that protocol and frame count in every result table.

## 3. MVBench: task-specific data and important completeness gates

```bash
python scripts/data/download_hf.py mvbench --download --max-gb 0.01 \
  --local-dir data/raw/mvbench --manifest data/plans/mvbench-annotations.json
python scripts/data/download_hf.py mvbench --dry-run --max-gb 20 --include 'video/*.zip'

# Example: one unbounded task family, not the aggregate 20-task benchmark.
python scripts/data/download_hf.py mvbench --download --max-gb 2 \
  --include 'video/clevrer.zip' --local-dir data/raw/mvbench \
  --manifest data/plans/mvbench-clevrer.json
python scripts/data/safe_extract.py data/raw/mvbench/video/clevrer.zip \
  --output data/media/mvbench/video --max-gb 4 --extract
python scripts/data/convert.py mvbench --task moving_count \
  --input data/raw/mvbench/json/moving_count.json \
  --media-root data/media/mvbench --video-prefix video \
  --output data/manifests/mvbench-moving-count-test.jsonl
```

The 20-task map is in `scripts/data/convert.py`. It follows the original
[MVBench notebook](https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/mvbench.ipynb)
for temporal bounds. Several Hub archives contain **pretrimmed** `*_segment`
media, while the original benchmark protocol uses raw videos plus start/end
times. These are not interchangeable. Bounded Charades tasks intentionally
require original `star/Charades_v1_480` or `sta/sta_video` paths; they do not apply
original timestamps to already-trimmed clips. Supply authorized raw originals
for these tasks. NTU requires separate authorized acquisition. There is no bypass.

TVQA episodic reasoning expects the original `tvqa/frames_fps3_hq` frame folders
and supports original 3-fps midpoint-segment sampling into `image_paths`, set
using `--num-frames`. Missing frames abort conversion. Do not relabel a 15/19-task
or missing-media subset as the official full MVBench score. Our reference model
sampling/prompt/scoring protocol is not a substitute for official leaderboard
evaluation: use the official pipeline for a final reported comparison.

## 4. HERBench Lite-v2: optional, explicitly versioned

```bash
python scripts/data/download_hf.py herbench --download --max-gb 0.01 \
  --local-dir data/raw/herbench --manifest data/plans/herbench-lite-v2-annotations.json
python scripts/data/download_hf.py herbench --list
```

The official repository distributes multipart tar media and describes Lite in
the first parts. **A partial multipart tar is not a complete tar archive.**
This release does not blindly concatenate/extract selected fragments or suppress
EOF/checksum errors. Follow the pinned official media acquisition/checksum
instructions, obtain a complete valid archive or authorized media directory, and
use the safe extractor on a complete archive. Check free disk space first.

```bash
python scripts/data/convert.py herbench \
  --input data/raw/herbench/data/herbench_lite_v2.parquet \
  --media-root data/media/herbench --output data/manifests/herbench-lite-v2-test.jsonl
```

The converter deliberately omits `metadata_json`, gold timestamps, and evidence
descriptions from model input records. If future evidence-recall evaluation uses
those labels, load them in a separate scorer/oracle-only path. Full/Lite/Lite-v2
are different protocols, not interchangeable results.

## Exact tokenizer-budget preflight before any GPU job

Long captions or questions can exceed the recipe's `max_text_tokens=512`.
Check **every training and validation row before allocating a long GPU run**:

```bash
python scripts/data/check_token_budget.py \
  data/manifests/llava-split/train.jsonl data/manifests/llava-split/validation.jsonl \
  --config configs/train/ouro_pilot.json \
  --report data/reports/ouro-pilot-token-budget.json
```

This downloads/loads only the tokenizer and its small configuration files at the
recipe's immutable language-model revision (`trust_remote_code=False`), never
model/vision weights. `--local-files-only` prevents even tokenizer downloads.
The check uses `MultimodalCollator.tokenize_record`, the **same code path as the
training collator**: prefix, BOS when present, question, choices and letter-only
instruction, answer, and EOS. It does not decode media or create a CUDA model.
Both train and validation loss include gold-answer tokens. The report contains
token-length quantiles, visual-plus-text maxima, **all** over-budget sample IDs,
and manifest/config/prompt-source hashes. It exits with status 2 on any overflow.
No rows are modified, silently truncated, or dropped; an existing report is not
overwritten. If it fails, explicitly increase `model.max_text_tokens` in a new
run config after checking model context and GPU memory, or separately implement
and document a curation policy. Then re-run the same preflight.

For alignment-to-SFT or resume, use the actual saved tokenizer as training does:

```bash
python scripts/data/check_token_budget.py \
  data/manifests/llava-split/train.jsonl data/manifests/llava-split/validation.jsonl \
  --config configs/train/ouro_sft.json --checkpoint runs/ouro-alignment/final \
  --report data/reports/ouro-sft-token-budget.json
```

Set `--checkpoint` to the same directory passed as training `--init-checkpoint`
or `--resume`; it must contain `tokenizer/` and `deepprobe_config.json`. The tool
checks checkpoint/config language identity and fingerprints local tokenizer
files. For held-out evaluation, add `--mode inference`: gold-answer tokens are
excluded and test rows are permitted. The default training mode rejects test
data. This text-only check does not certify media decoding, generation reserve,
backbone context capacity, or GPU memory; those require separate smoke tests.

## Offline frame caching before expensive GPU runs

The reference video decoder performs two sequential passes per clip. Cache once
on CPU/local storage before long H20/B300 training runs; otherwise decoding can
hide model throughput. This is deterministic preprocessing, not a speedup result.

```bash
python scripts/data/cache_frames.py --input data/manifests/llava-split/train.jsonl \
  --media-root data/media/llava --output-dir data/frame_cache/llava-train-f16 \
  --num-frames 16 --max-gb 100

python scripts/data/validate.py data/frame_cache/llava-train-f16/manifest.jsonl \
  --media-root data/frame_cache/llava-train-f16
```

Train/evaluate using the resulting `manifest.jsonl` and the cache directory as
`--media-root`, with the **same frame count**. Images are lossless RGB PNGs at
original selected resolution; no hidden resize or JPEG recompression. Multiple
questions sharing the same media and bounds reuse the same decode. Cache keys
include source-content SHA256, temporal bounds, frame count, decoder source hash,
and PyAV/Pillow versions; `cache_index.json` records per-frame hashes. Original
`source_video_id`, split, and clip-bound provenance are retained. Bounds are
already applied to cached frames and must not be applied again. A fresh output
directory is required; the script refuses overwrite. A budget/decode failure may
leave partial PNGs but no completed output manifest. No frame assets are shipped
or generated by repository CI.

## Shared JSONL schema and final readiness check

Required fields: `sample_id`, `source_video_id`, `question`, `answer`, `split`,
and **exactly one** of `video_path` or `image_paths`. Paths are relative to the
explicit `--media-root`; absolute paths and escaping symlinks are rejected.
`split` is one of `train`, `validation`, `test`. Optional `choices` is a list of
2–26 plain strings, with `answer` a matching uppercase letter. Video samples may
have paired `start_seconds`/`end_seconds`; consumers must honor or reject them.

```bash
# Dependency-free schema fixture only; no media or measured results are supplied.
python scripts/data/validate.py scripts/data/synthetic_schema_example.jsonl --schema-only

# Before a real GPU run: existence/root/split checks, not only --schema-only.
python scripts/data/validate.py data/manifests/videomme-test.jsonl \
  --media-root data/media/videomme

# Cross-dataset provenance check when each dataset has its own root.
python scripts/data/validate.py data/manifests/llava-split/train.jsonl \
  data/manifests/llava-split/validation.jsonl data/manifests/videomme-test.jsonl \
  --schema-only
```

Existence checks do not prove a video can decode. Run the GPU pipeline's small
data smoke test before a long H20/B300 job. Archive/source downloads are not run
by this repository's CI; unit tests use only synthetic temporary files.
