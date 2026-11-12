# Original-video caches for the serial experiment suite

F4 and F16 are sampled from the official original video member. They must not
be sampled from, or produced by repeating, the accepted F8 PNG cache. G2/G8
change spatial pooling and can use the accepted F8 cache. TempCompass sources
are its official processed ZIP members, including reverse/speed/concat clips;
the underlying Shutterstock/YouTube source alone is not an equivalent input.

The preparation and hydration CLIs default to validation/planning. Add
`--execute` on the Linux remote host to run them. `OSSStore` rejects real
execution on macOS. Credentials come from the installed OSS config, never
arguments or repository files. All objects remain in `ss-devbox-stream`.

## 1. Prepare a source binding

`scripts/operations/build_suite_media_binding.py` takes these pinned inputs:

- `--source-manifest` and `--source-manifest-sha256`: the accepted test JSONL.
- `--frame-cache-receipt` and `--frame-cache-receipt-sha256`: the accepted old
  full-decode receipt containing each original member SHA and identity.
- `--archive-catalog` and `--archive-catalog-sha256`: a JSON list of original
  archive objects. Every element has `key`, `size_bytes`, `sha256`, `etag`, and
  `receipt: {key, size_bytes, sha256}`. Receipt keys are the exact archive key
  plus `.CACHE_VERIFIED.json`.
- `--output`: a new source-binding JSON file.
- `--oss-config`: the existing stream OSS config file if not at the default.

This helper verifies the archived full-GET receipt and fresh ETag, then reads
ZIP central-directory metadata to bind member size, compressed size and CRC.
It does not extract or decode videos, copy cached PNGs, or submit GPU work.
It preserves the SHA of the accepted decode receipt in the resulting binding.
The archive catalog must contain evidence-derived values; it is not generated
by assuming that an object name or an earlier success flag proves its bytes.

The source-binding schema has `schema_version: 1`, `bucket`, `dataset`,
`manifest_sha256`, `manifest_bytes`, `expected_samples`, `expected_clips`,
`archives`, and `sources`. Every source has `media_identity`, `archive_key`,
`underlying_source_video_ids`, and a `member` with `path`, `size_bytes`,
`compressed_bytes`, `crc32`, and `sha256`. Optional `start_seconds` and
`end_seconds` must agree with every manifest question using that clip.

## 2. Generate each requested frame count directly into the bucket

`scripts/operations/build_suite_frame_cache.py` requires:

- `--source-manifest`, `--source-binding`, `--source-binding-sha256`.
- `--num-frames 4`, `8`, or `16`.
- `--output-dir`: a new directory for small metadata only.
- `--oss-prefix`: a new exact object-key prefix under
  `jinhaopeng/Paper/DeepProbe-VLM/benchmark-cache/`, containing
  `/shape-ablation/` and `/uniform4/`, `/uniform8/`, or `/uniform16/`.
- `--max-png-bytes`: an explicit cap for the total output PNG bytes.

Each ZIP range is bound to the accepted ETag using `If-Match`. One member at a
time is CRC/length/SHA checked in bounded memory. Two complete PyAV passes
count and select actual eligible frames with `data.uniform_indices`; declared
container frame counts are not used. The timeline and both eligible/original
stream indices are recorded. Very short original clips may repeat original
indices according to the existing sampling rule. No resizing is performed.

PNGs are uploaded directly with overwrite protection and verified by complete
OSS GET SHA. Neither ZIP/video nor PNG payloads are written to the development
machine or to the preparation host's local disk. Local outputs are only:

- `BUILD_PLAN.json`
- `manifest.jsonl`, preserving question order, IDs, answers and provenance
- `FRAME_CACHE_RECEIPT.json`, binding source, original indices, PNG and RGB SHA
- `CACHE_COMPLETE.json`, binding the complete exact bucket inventory

Each clip also has a bucket entry receipt. `--resume` explicitly resumes the
same local plan and exact owned bucket prefix. Complete entries and every PNG
are verified again; partial clips are re-decoded and must match any already
uploaded bytes. Extra objects, changed inputs, changed bounds or a changed
frozen builder stop the run. Resume never deletes or overwrites a differing
object. Preserve the original frozen script release for resumed builds.

## 3. Hydrate only on the evaluation node

The Python helper creates a small lazy-hydration descriptor:

```python
from deepprobe_vlm.suite_media import cache_asset_binding

asset = cache_asset_binding(
    cache_metadata_directory,
    source_manifest=original_manifest_path,
    source_binding=source_binding_path,
    media_root=new_owned_node_media_directory,
    max_bytes=explicit_node_cache_cap,
)
```

Its shape is `{cache, media_root, max_bytes}`. `cache` contains byte pins for
the original manifest and source binding, transformed manifest, frame receipt,
build plan, and completion receipt, plus the exact prefix. It is not a READY
receipt. Save it as JSON and pass its SHA to
`scripts/operations/hydrate_suite_frame_cache.py --asset-binding ...
--asset-binding-sha256 ...`; add `--execute` on the node.

The runtime API is:

```python
from deepprobe_vlm.suite_media import hydrate_frame_cache

data = hydrate_frame_cache(
    asset["cache"], media_root=asset["media_root"],
    max_bytes=asset["max_bytes"], store=stream_store, resume=False,
)
```

It first checks complete benchmark scope (Video-MME 2,697 QA/899 clips or
TempCompass 1,580 MC QA/410 official clips), original question/source identity,
frame count, original sampling indices, exact object inventory, capacity cap
and physical free space. Each PNG is fetched from its exact key and validated
by byte SHA, decoded RGB SHA, dimensions and color mode before local writing.
The local directory is owned by `.suite-media-owner.json`. Existing directories
require explicit `resume=True`/`--resume` and the identical descriptor; extra
or changed files are refused.

Only after complete validation does it write `SUITE_BENCHMARK_READY.json` and
`ENVIRONMENT_FRAGMENT.json` beside the media directory. The returned `data`
matches `suite_actions.validate_benchmark`: `manifest`, `media_root`,
`media_files` (relative `path/bytes/sha256` pins), and `acceptance`. The READY
receipt binds `media_inventory_sha256`, original-video sampling, QA count, frame
count and all small source-evidence pins. It does not claim evaluation results.
The suite's separate dual-root capacity admission still applies; this helper's
byte cap does not replace it.

## Existing accepted F8 caches

A separately pinned legacy descriptor is supported; no accepted protocol is
loosened. Its `cache` has `kind: "legacy"`, `source_manifest`, `manifest`,
`source_binding`, `receipt`, and `frame_objects`. All four files are absolute
`path/bytes/sha256` pins. `frame_objects` is the exact list of
`{relative_path, key, size_bytes, sha256}` mappings from accepted bucket
inventory evidence. The source binding must contain
`accepted_original_decode_receipt_sha256` equal to this exact old frame
receipt. Original member SHA/size, sampling indices, ordered PNG paths and
complete manifest/source coverage are checked before hydration.

Legacy adaptation accepts only unbounded F8 original clips. Missing original
member identity, missing exact bucket keys, or an unbound READY boolean is
insufficient. Build a new cache when the required evidence is unavailable.

## Verification

`tests/test_suite_media.py` exercises actual synthetic MP4/PyAV sampling,
temporal bounds, ETag/member SHA failures, byte caps, interruption/resume,
immutable conflicts, the legacy source binding, and end-to-end 1,580 QA/410
synthetic-clip hydration accepted by `suite_actions.validate_benchmark`.
These CPU tests are not benchmark scores or evidence that real media was
downloaded during development.
