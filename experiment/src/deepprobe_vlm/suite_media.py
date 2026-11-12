"""Build shape-ablation frame caches directly from pinned original OSS ZIPs.

No original video or PNG is written to local storage. The only local outputs
are the transformed manifest and small, independently verifiable receipts.
TempCompass sources are its official *processed* clips (including reverse,
speed and concat variants), never their underlying unprocessed source videos.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import configparser
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import zipfile

from .data import uniform_indices
from .manifest import safe_relative_path, validate_records

BUCKET = "ss-devbox-stream"
ENDPOINT = "https://oss-cn-wulanchabu.aliyuncs.com"
PROJECT_PREFIX = "jinhaopeng/Paper/DeepProbe-VLM/benchmark-cache/"
BLOCK = 8 * 1024 * 1024
MAX_MEMBER_BYTES = 1_000_000_000
MAX_FRAME_PIXELS = 3840 * 2160
MAX_PNG_BYTES = 32 * 1024 * 1024
MAX_METADATA_BYTES = 64 * 1024 * 1024
SAMPLING = "deepprobe_vlm.data.uniform_indices"


class CacheError(ValueError):
    """A source binding, byte budget or immutable object contract failed."""


def require(condition, reason):
    if not condition:
        raise CacheError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def _sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _positive(value):
    return type(value) is int and value > 0


def safe(path):
    path = Path(path).absolute()
    require(".." not in path.parts and not any(p.is_symlink() for p in [path, *path.parents]),
            "local path contains traversal or a symlink")
    return path


def _key(value):
    require(isinstance(value, str) and value.startswith(PROJECT_PREFIX)
            and ".." not in value.split("/") and "\\" not in value
            and not any(c.isspace() for c in value), "unsafe or foreign OSS key")
    return value


def _identity(row):
    return row.get("processed_media_identity") or row.get("source_video_id")


def _bounds(row):
    start, end = row.get("start_seconds"), row.get("end_seconds")
    if start is not None or end is not None:
        require(all(type(v) in (float, int) and math.isfinite(v) for v in (start, end))
                and 0 <= start < end, "invalid source temporal bounds")
    return start, end


def validate_binding(binding, manifest_raw):
    """Validate a reviewed schema-1 source map; this never opens media/network.

    An archive entry contains key, size_bytes, sha256, etag and receipt with
    key/size_bytes/sha256. A source contains media_identity, archive_key,
    underlying_source_video_ids, and member with path, size_bytes,
    compressed_bytes, crc32 and sha256. Optional source temporal bounds must
    match every question using that media identity. Pin the whole binding in
    the CLI; passing a mapping directly is intended for a reviewed caller.
    """
    require(isinstance(binding, dict) and binding.get("schema_version") == 1,
            "unsupported source binding schema")
    require(binding.get("bucket") == BUCKET and binding.get("dataset") in {"videomme", "tempcompass"},
            "source dataset or bucket differs")
    require(binding.get("manifest_sha256") == digest(manifest_raw)
            and binding.get("manifest_bytes") == len(manifest_raw), "source manifest pin differs")
    require(len(manifest_raw) <= MAX_METADATA_BYTES, "manifest exceeds metadata bound")
    rows = validate_records([json.loads(line) for line in manifest_raw.decode().splitlines() if line.strip()],
                            check_media=False)
    require(all(row.get("dataset") == binding["dataset"] and row["split"] == "test" for row in rows),
            "cache builder only accepts the pinned benchmark test manifest")
    require(binding.get("expected_samples") == len(rows), "source question coverage differs")
    archives, sources = binding.get("archives"), binding.get("sources")
    require(isinstance(archives, list) and archives and isinstance(sources, list) and sources,
            "missing archive or original member map")
    by_archive = {}
    for archive in archives:
        key = _key(archive.get("key"))
        require(key.endswith(".zip") and key not in by_archive
                and _positive(archive.get("size_bytes")) and _sha(archive.get("sha256"))
                and isinstance(archive.get("etag"), str) and archive["etag"]
                and not any(c in archive["etag"] for c in "\r\n"), "invalid original archive identity")
        receipt = archive.get("receipt", {})
        require(_key(receipt.get("key")) == key + ".CACHE_VERIFIED.json"
                and _positive(receipt.get("size_bytes")) and receipt["size_bytes"] <= MAX_METADATA_BYTES
                and _sha(receipt.get("sha256")), "missing pinned archive verification receipt")
        by_archive[key] = archive
    by_source = {}
    for source in sources:
        identity, member = source.get("media_identity"), source.get("member", {})
        require(isinstance(identity, str) and identity and identity not in by_source
                and source.get("archive_key") in by_archive, "missing or duplicate original source identity")
        path = safe_relative_path(member.get("path"))
        require(Path(path).suffix.lower() in {".mp4", ".webm", ".mkv", ".avi", ".mov"},
                "source member must be an original video, never cached image frames")
        require(_positive(member.get("size_bytes")) and member["size_bytes"] <= MAX_MEMBER_BYTES
                and _positive(member.get("compressed_bytes"))
                and type(member.get("crc32")) is int and 0 <= member["crc32"] <= 0xFFFFFFFF
                and _sha(member.get("sha256")), "invalid original video member metadata")
        underlying = source.get("underlying_source_video_ids")
        require(isinstance(underlying, list) and underlying
                and all(isinstance(value, str) and value for value in underlying)
                and len(set(underlying)) == len(underlying), "missing underlying source provenance")
        _bounds(source)
        if binding["dataset"] == "tempcompass":
            require(identity.startswith("tempcompass:") and Path(path).stem == identity.split(":", 1)[1],
                    "TempCompass must use its official processed clip member")
        by_source[identity] = source
    require(set(by_source) == {_identity(row) for row in rows}
            and binding.get("expected_clips") == len(by_source), "original clip coverage differs")
    require(set(by_archive) == {row["archive_key"] for row in sources}, "unused archive in source map")
    for row in rows:
        source = by_source[_identity(row)]
        require(_bounds(row) == _bounds(source), "question/source temporal bounds differ")
        if binding["dataset"] == "tempcompass":
            require(_identity(row) == "tempcompass:" + str(row.get("video_id")),
                    "TempCompass question and processed member identity differ")
        declared_sources = row.get("underlying_source_video_ids", [row["source_video_id"]])
        require(declared_sources == source["underlying_source_video_ids"], "source provenance differs")
    return rows, by_archive, by_source


class OSSStore:
    """Installed OSS SDK adapter; neither credentials nor video bytes enter argv."""
    def __init__(self, config_path, *, allowed_prefix):
        require(platform.system() == "Linux", "real media execution belongs on the remote Linux host")
        self.allowed_prefix = _key(allowed_prefix)
        require(self.allowed_prefix.endswith("/"), "destination must be an exact prefix ending in slash")
        import oss2
        self.oss2 = oss2
        config = configparser.ConfigParser()
        config.read(safe(config_path))
        require(config.has_section("default") and config["default"].get("region") == "cn-wulanchabu",
                "installed stream OSS config unavailable")
        credentials = config["default"]
        self.bucket = oss2.Bucket(oss2.Auth(credentials["accesskeyid"], credentials["accesskeysecret"]),
                                 ENDPOINT, BUCKET, connect_timeout=60, enable_crc=True)

    def head(self, key):
        result = self.bucket.head_object(_key(key))
        return {"size_bytes": result.content_length, "etag": result.etag.strip('"')}

    def get(self, key, maximum=MAX_METADATA_BYTES):
        try:
            stream = self.bucket.get_object(_key(key))
        except self.oss2.exceptions.NoSuchKey:
            return None
        try:
            data = stream.read(maximum + 1)
            require(len(data) <= maximum, "OSS object exceeds byte bound")
            return data
        finally:
            stream.close()

    def range(self, key, start, end, etag, total):
        stream = self.bucket.get_object(_key(key), byte_range=(start, end), headers={"If-Match": etag})
        try:
            content_range = stream.headers.get("Content-Range", stream.headers.get("content-range"))
            require(stream.status == 206 and content_range == f"bytes {start}-{end}/{total}",
                    "range response is not bound to the requested archive bytes")
            raw = stream.read(end - start + 2)
            require(len(raw) == end - start + 1, "range response byte count differs")
            return raw
        finally:
            stream.close()

    def put_verified(self, key, raw):
        require(_key(key).startswith(self.allowed_prefix), "write escapes new frame cache prefix")
        old = self.get(key, len(raw))
        if old is not None:
            require(old == raw, "immutable object differs; existing bytes were not overwritten")
            return
        self.bucket.put_object(key, raw, headers={"x-oss-forbid-overwrite": "true",
                                                 "x-oss-meta-sha256": digest(raw)})
        require(self.get(key, len(raw)) == raw, "full OSS GET readback differs")

    def keys(self, prefix):
        marker, seen = "", set()
        while True:
            page = self.bucket.list_objects(prefix=_key(prefix), marker=marker, max_keys=1000)
            for row in page.object_list:
                yield row.key
            if not page.is_truncated:
                break
            marker = page.next_marker
            require(marker and marker not in seen, "OSS listing pagination stalled")
            seen.add(marker)


class RangeReader(io.RawIOBase):
    """A ZIP-compatible seekable stream with one bounded, ETag-bound block."""
    def __init__(self, store, archive):
        self.store, self.archive = store, archive
        self.position, self.start, self.cached = 0, 0, b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        require(whence in (0, 1, 2), "invalid archive seek")
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.archive["size_bytes"])
        require(0 <= position <= self.archive["size_bytes"], "archive seek outside pinned bounds")
        self.position = position
        return position

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.archive["size_bytes"] - self.position
        require(size <= 2 * BLOCK, "ZIP requested an unbounded read")
        end, result = min(self.archive["size_bytes"], self.position + size), bytearray()
        while self.position < end:
            if not self.start <= self.position < self.start + len(self.cached):
                self.start = self.position
                last = min(self.archive["size_bytes"], self.start + BLOCK) - 1
                self.cached = self.store.range(self.archive["key"], self.start, last,
                                               self.archive["etag"], self.archive["size_bytes"])
                require(len(self.cached) == last - self.start + 1, "short archive range")
            amount = min(end - self.position, self.start + len(self.cached) - self.position)
            require(amount > 0, "empty archive range")
            offset = self.position - self.start
            result.extend(self.cached[offset:offset + amount])
            self.position += amount
        return bytes(result)


def verify_archive(store, archive):
    receipt = archive["receipt"]
    raw = store.get(receipt["key"], receipt["size_bytes"])
    require(raw is not None and len(raw) == receipt["size_bytes"] and digest(raw) == receipt["sha256"],
            "pinned archive receipt changed")
    document = json.loads(raw)
    binding = document.get("binding", {})
    require(binding.get("bucket") == BUCKET and binding.get("key") == archive["key"]
            and binding.get("sha256") == archive["sha256"] and binding.get("size_bytes") == archive["size_bytes"]
            and document.get("source_stream_sha256") == archive["sha256"]
            and document.get("source_bytes") == archive["size_bytes"]
            and document.get("oss_readback") == {"bytes": archive["size_bytes"], "sha256": archive["sha256"],
                "verification": "complete_OSS_GET_SHA256_and_size"}, "archive full-readback acceptance missing")
    head = store.head(archive["key"])
    require(head["size_bytes"] == archive["size_bytes"] and head["etag"].strip('"') == archive["etag"].strip('"'),
            "archive size or audited ETag changed")


def extract_member(store, archive, member):
    """CRC, length and SHA-check one original member before decoding it."""
    target, hasher, length = io.BytesIO(), hashlib.sha256(), 0
    try:
        with zipfile.ZipFile(RangeReader(store, archive)) as bundle:
            matches = [row for row in bundle.infolist() if row.filename == member["path"]]
            require(len(matches) == 1, "missing or duplicate ZIP member")
            info = matches[0]
            require(info.file_size == member["size_bytes"] and info.compress_size == member["compressed_bytes"]
                    and info.CRC == member["crc32"] and not info.flag_bits & 1,
                    "original ZIP member metadata changed")
            with bundle.open(info) as stream:
                for raw in iter(lambda: stream.read(BLOCK), b""):
                    length += len(raw)
                    require(length <= member["size_bytes"] <= MAX_MEMBER_BYTES, "original member byte limit exceeded")
                    target.write(raw)
                    hasher.update(raw)
        require(length == member["size_bytes"] and hasher.hexdigest() == member["sha256"],
                "original video member SHA or length differs")
        target.seek(0)
        return target
    except BaseException:
        target.close()
        raise


def decode_selected(source, frames, *, start_seconds=None, end_seconds=None):
    """Always count actual eligible frames, then decode the same stream again.

    The second pass is complete too; a truncated tail cannot be hidden by an
    early selected frame. The returned ordered entries include both original
    stream indices and within-clip indices. Repeated indices only occur when
    the original clip has fewer decoded frames than requested.
    """
    import av
    start_seconds, end_seconds = _bounds({"start_seconds": start_seconds, "end_seconds": end_seconds})

    def scan(wanted=None):
        source.seek(0)
        eligible, total, selected, timeline = 0, 0, {}, hashlib.sha256()
        with av.open(source, options={"err_detect": "explode"}) as container:
            require(len(container.streams.video) > 0, "original member has no video stream")
            container.streams.video[0].thread_count = 1
            for original_index, frame in enumerate(container.decode(video=0)):
                total += 1
                require(0 < frame.width * frame.height <= MAX_FRAME_PIXELS, "decoded frame exceeds pixel memory bound")
                timestamp = frame.time
                timeline.update(encode([original_index, frame.pts, str(frame.time_base), frame.width, frame.height]))
                if start_seconds is not None:
                    require(timestamp is not None, "bounded clip has missing frame timestamps")
                    if timestamp < start_seconds or timestamp >= end_seconds:
                        continue
                if wanted is not None and eligible in wanted:
                    selected[eligible] = {"image": frame.to_image().convert("RGB"),
                                          "source_frame_index": original_index,
                                          "timestamp_seconds": timestamp}
                eligible += 1
        return eligible, total, timeline.hexdigest(), selected

    count, total, timeline, _ = scan()
    indices = uniform_indices(count, frames)
    count2, total2, timeline2, selected = scan(set(indices))
    require((count2, total2, timeline2) == (count, total, timeline) and set(selected) == set(indices),
            "original video does not decode consistently in two full passes")
    return {"decoded_frame_count": count, "original_stream_frame_count": total,
            "selected_indices": indices, "full_decode_passes": 2,
            "timeline_sha256": timeline, "selected": selected}


def _write_once(path, raw):
    path = safe(path)
    require(len(raw) <= MAX_METADATA_BYTES, "local metadata byte bound exceeded")
    if path.exists():
        require(path.is_file() and path.read_bytes() == raw, "local immutable metadata differs")
        return
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def cache_plan(binding, manifest_raw, *, frames, oss_prefix, max_bytes):
    rows, archives, sources = validate_binding(binding, manifest_raw)
    require(type(frames) is int and frames in (4, 8, 16), "suite cache supports exactly F4/F8/F16")
    require(_positive(max_bytes), "positive frame cache byte cap required")
    prefix = _key(oss_prefix)
    require(prefix.endswith("/") and "/shape-ablation/" in prefix and f"/uniform{frames}/" in prefix,
            "destination must be a new, explicit shape-ablation/uniformF prefix")
    require(all(not key.startswith(prefix) and not prefix.startswith(key) for key in archives),
            "destination overlaps an original archive")
    return {"schema_version": 1, "kind": "suite_original_video_frame_cache", "bucket": BUCKET,
            "prefix": prefix, "num_frames": frames, "max_png_bytes": max_bytes,
            "binding_sha256": digest(encode(binding)), "manifest_sha256": digest(manifest_raw),
            "dataset": binding["dataset"], "sample_count": len(rows), "source_count": len(sources),
            "sampling": SAMPLING, "resample_existing_frame_cache": False,
            "local_video_or_png_payloads": False, "builder_sha256": digest(Path(__file__).read_bytes())}


def create_source_binding(manifest_raw, accepted_frame_receipt_raw, archive_catalog, *, store):
    """Normalize archived original-video identities without reading old PNGs.

    This is a read-only preparation step. Caller/CLI pins the accepted old
    receipt and catalog. ZIP central directories supply missing compressed
    lengths/CRCs; original member SHA comes from the accepted decode evidence.
    It does not certify new frames or open/decode any ZIP member.
    """
    rows = validate_records([json.loads(line) for line in manifest_raw.decode().splitlines() if line.strip()],
                            check_media=False)
    datasets = {row.get("dataset") for row in rows}
    require(len(datasets) == 1 and next(iter(datasets)) in {"tempcompass", "videomme"},
            "source manifest benchmark differs")
    accepted = json.loads(accepted_frame_receipt_raw)
    require(accepted.get("complete") is True and accepted.get("failures", []) == []
            and isinstance(accepted.get("entries"), list), "accepted source decode receipt is incomplete")
    accepted_entries = {_identity(row): row for row in accepted["entries"]}
    require(len(accepted_entries) == len(accepted["entries"]), "duplicate original source receipt identity")
    wanted = {_identity(row) for row in rows}
    require(wanted <= set(accepted_entries), "source decode evidence does not cover the manifest")
    require(isinstance(archive_catalog, list) and archive_catalog, "original archive catalog missing")
    directories = {}
    for archive in archive_catalog:
        verify_archive(store, archive)
        with zipfile.ZipFile(RangeReader(store, archive)) as bundle:
            directory = {}
            for item in bundle.infolist():
                require(item.filename not in directory, "duplicate ZIP member path")
                directory[item.filename] = item
            directories[archive["key"]] = directory
    sources, used = [], set()
    for identity in sorted(wanted):
        entry = accepted_entries[identity]
        require(entry.get("decode_complete") is True and _sha(entry.get("original_sha256")),
                "accepted original video SHA or complete decode missing")
        matches = [archive for archive in archive_catalog if archive["sha256"] == entry.get("archive_sha256")
                   and (entry.get("archive_key") == archive["key"] if entry.get("archive_key")
                        else archive["key"].endswith("/" + entry.get("archive_path", "")))]
        require(len(matches) == 1, "source archive identity is missing or ambiguous")
        archive = matches[0]
        accepted_etag = entry.get("archive_etag", entry.get("oss_archive_etag"))
        require(isinstance(accepted_etag, str) and accepted_etag.strip('"') == archive["etag"].strip('"'),
                "accepted source archive ETag differs")
        member = directories[archive["key"]].get(entry.get("member_path"))
        require(member is not None and not member.flag_bits & 1
                and member.file_size == entry.get("original_size_bytes"), "accepted member identity differs")
        source_rows = [row for row in rows if _identity(row) == identity]
        bounds = {_bounds(row) for row in source_rows}
        require(len(bounds) == 1, "same processed source has inconsistent temporal bounds")
        underlying = source_rows[0].get("underlying_source_video_ids", [source_rows[0]["source_video_id"]])
        require(entry.get("underlying_source_video_ids", [entry.get("source_video_id")]) == underlying,
                "accepted original source provenance differs")
        source = {"media_identity": identity, "archive_key": archive["key"],
                  "underlying_source_video_ids": underlying,
                  "member": {"path": member.filename, "size_bytes": member.file_size,
                             "compressed_bytes": member.compress_size, "crc32": member.CRC,
                             "sha256": entry["original_sha256"]}}
        start, end = next(iter(bounds))
        if start is not None:
            source.update(start_seconds=start, end_seconds=end)
        sources.append(source)
        used.add(archive["key"])
    binding = {"schema_version": 1, "bucket": BUCKET, "dataset": next(iter(datasets)),
               "manifest_sha256": digest(manifest_raw), "manifest_bytes": len(manifest_raw),
               "expected_samples": len(rows), "expected_clips": len(sources),
               "accepted_original_decode_receipt_sha256": digest(accepted_frame_receipt_raw),
               "archive_catalog_sha256": digest(encode(archive_catalog)),
               "archives": [archive for archive in archive_catalog if archive["key"] in used], "sources": sources}
    validate_binding(binding, manifest_raw)
    return binding


def build_frame_cache(source_manifest, source_binding, *, frames, output_dir_new,
                      oss_prefix_new, max_bytes, store, resume=False, decoder=decode_selected):
    """Upload a genuine cache and emit local metadata; exact partial resume is explicit.

    Resume never trusts a prior completion flag: every source receipt and every
    frame is read and SHA-checked again. Unknown extra objects or changed plans
    stop execution. A partial source is safely re-decoded against its original
    SHA and existing immutable bytes; no deletes or blind overwrites occur.
    """
    manifest_raw = safe(source_manifest).read_bytes()
    binding = deepcopy(source_binding)
    rows, archives, sources = validate_binding(binding, manifest_raw)
    plan = cache_plan(binding, manifest_raw, frames=frames, oss_prefix=oss_prefix_new, max_bytes=max_bytes)
    prefix, output = plan["prefix"], safe(output_dir_new)
    plan_raw = encode(plan)
    expected_keys = {prefix + name for name in ("BUILD_PLAN.json", "manifest.jsonl", "FRAME_CACHE_RECEIPT.json",
                                                "CACHE_COMPLETE.json")}
    paths = {}
    for identity in sources:
        token = digest(identity.encode())
        paths[identity] = [f"frames/{token}/{slot:02d}.png" for slot in range(frames)]
        expected_keys.add(prefix + f"entries/{token}.json")
        expected_keys.update(prefix + path for path in paths[identity])
    existing_keys = set(store.keys(prefix))
    if resume:
        require(output.is_dir() and (output / "BUILD_PLAN.json").read_bytes() == plan_raw,
                "resume requires the exact existing local build plan")
        require(store.get(prefix + "BUILD_PLAN.json") == plan_raw and existing_keys <= expected_keys,
                "resume prefix ownership or inventory differs")
    else:
        require(not output.exists() and not existing_keys, "cache needs a new local directory and empty exact OSS prefix")
    for archive in archives.values():
        verify_archive(store, archive)
    if not resume:
        output.mkdir(parents=True)
        _write_once(output / "BUILD_PLAN.json", plan_raw)
        store.put_verified(prefix + "BUILD_PLAN.json", plan_raw)
    entries, total_bytes = [], 0
    for identity, source in sources.items():
        token, archive, member = digest(identity.encode()), archives[source["archive_key"]], source["member"]
        entry_key = prefix + f"entries/{token}.json"
        prior_raw = store.get(entry_key) if resume else None
        if prior_raw is not None:
            entry = json.loads(prior_raw)
            require(entry.get("source_binding") == source and entry.get("plan_sha256") == digest(plan_raw)
                    and entry.get("decode_complete") is True and entry.get("full_decode_passes") == 2
                    and entry.get("image_paths") == paths[identity], "resumed source binding differs")
            count = entry.get("decoded_frame_count")
            require(_positive(count) and entry.get("selected_indices") == uniform_indices(count, frames)
                    and len(entry.get("frames", [])) == frames, "resumed sampling differs")
            for slot, item in enumerate(entry["frames"]):
                require(item.get("relative_path") == paths[identity][slot]
                        and item.get("eligible_frame_index") == entry["selected_indices"][slot]
                        and _positive(item.get("size_bytes")) and item["size_bytes"] <= MAX_PNG_BYTES
                        and _sha(item.get("sha256")), "resumed frame metadata differs")
                raw = store.get(prefix + item["relative_path"], item["size_bytes"])
                require(raw is not None and len(raw) == item["size_bytes"] and digest(raw) == item["sha256"],
                        "resumed frame full GET SHA differs")
                total_bytes += len(raw)
                require(total_bytes <= max_bytes, "total PNG cache byte cap exceeded")
            entries.append(entry)
            continue
        with extract_member(store, archive, member) as original:
            decoded = decoder(original, frames, start_seconds=source.get("start_seconds"),
                              end_seconds=source.get("end_seconds"))
        count, indices = decoded["decoded_frame_count"], decoded["selected_indices"]
        require(_positive(count) and indices == uniform_indices(count, frames)
                and decoded["full_decode_passes"] == 2, "decoder sampling or full-pass contract differs")
        frame_rows = []
        for slot, index in enumerate(indices):
            selected = decoded["selected"][index]
            image = selected["image"]
            require(image.mode == "RGB" and 0 < image.width * image.height <= MAX_FRAME_PIXELS,
                    "selected image shape or color mode differs")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", optimize=False)
            raw = buffer.getvalue()
            require(0 < len(raw) <= MAX_PNG_BYTES and total_bytes + len(raw) <= max_bytes,
                    "per-frame or total PNG cache byte cap exceeded")
            frame_row = {"relative_path": paths[identity][slot], "size_bytes": len(raw), "sha256": digest(raw),
                         "rgb_sha256": digest(image.tobytes()), "width": image.width, "height": image.height,
                         "eligible_frame_index": index, "source_frame_index": selected["source_frame_index"],
                         "timestamp_seconds": selected["timestamp_seconds"]}
            store.put_verified(prefix + frame_row["relative_path"], raw)
            total_bytes += len(raw)
            frame_rows.append(frame_row)
        entry = {"media_identity": identity, "source_binding": source, "plan_sha256": digest(plan_raw),
                 "decode_complete": True, "full_decode_passes": 2, "decoded_frame_count": count,
                 "original_stream_frame_count": decoded["original_stream_frame_count"],
                 "timeline_sha256": decoded["timeline_sha256"], "selected_indices": indices,
                 "image_paths": paths[identity], "frames": frame_rows}
        store.put_verified(entry_key, encode(entry))
        entries.append(entry)
        del decoded
    for archive in archives.values():
        verify_archive(store, archive)
    transformed = deepcopy(rows)
    for row in transformed:
        row.pop("video_path", None)
        row["image_paths"] = paths[_identity(row)]
    validate_records(transformed, check_media=False)
    manifest = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
                        for row in transformed)
    receipt = {"schema_version": 1, "kind": "suite_original_video_frame_cache", "complete": True,
               "dataset": binding["dataset"], "num_frames": frames, "sampling": SAMPLING,
               "resample_existing_frame_cache": False, "frames_resized": False,
               "source_count": len(sources), "sample_count": len(rows), "frame_count": len(sources) * frames,
               "png_bytes": total_bytes, "all_png_oss_get_sha256_verified": True,
               "plan_sha256": digest(plan_raw), "manifest_sha256": digest(manifest),
               "original_archive_acceptance": "pinned_full_GET_receipt_plus_fresh_ETag_and_member_SHA",
               "entries": entries}
    receipt_raw = encode(receipt)
    for name, raw in (("manifest.jsonl", manifest), ("FRAME_CACHE_RECEIPT.json", receipt_raw)):
        require(len(raw) <= MAX_METADATA_BYTES, "generated metadata exceeds byte bound")
        store.put_verified(prefix + name, raw)
        _write_once(output / name, raw)
    require(set(store.keys(prefix)) == expected_keys - {prefix + "CACHE_COMPLETE.json"}
            or (resume and set(store.keys(prefix)) == expected_keys), "exact output OSS inventory differs")
    complete = {"schema_version": 1, "status": "FRAME_CACHE_COMPLETE", "bucket": BUCKET,
                "prefix": prefix, "plan_sha256": digest(plan_raw), "source_binding_sha256": digest(encode(binding)),
                "manifest_sha256": digest(manifest), "frame_cache_receipt_sha256": digest(receipt_raw),
                "source_count": len(sources), "sample_count": len(rows), "frame_count": len(sources) * frames,
                "png_bytes": total_bytes, "exact_prefix_object_count": len(expected_keys),
                "original_video_sha_verified": True, "all_png_full_oss_get_verified": True,
                "local_video_or_png_payloads": False, "formal_evaluation_completed": False}
    store.put_verified(prefix + "CACHE_COMPLETE.json", encode(complete))
    require(set(store.keys(prefix)) == expected_keys, "final exact output OSS inventory differs")
    _write_once(output / "CACHE_COMPLETE.json", encode(complete))
    return complete


def cache_asset_binding(metadata_root, *, source_manifest, source_binding, media_root, max_bytes):
    """Export a small, pinned lazy-hydration descriptor; no READY is fabricated."""
    from .checkpoint_retention import file_pin
    metadata_root = safe(metadata_root)
    cache = {"kind": "suite", "source_manifest": file_pin(source_manifest), "source_binding": file_pin(source_binding)}
    for key, name in (("manifest", "manifest.jsonl"), ("receipt", "FRAME_CACHE_RECEIPT.json"),
                       ("complete", "CACHE_COMPLETE.json"), ("plan", "BUILD_PLAN.json")):
        cache[key] = file_pin(metadata_root / name)
    cache["prefix"] = json.loads((metadata_root / "CACHE_COMPLETE.json").read_bytes())["prefix"]
    return {"cache": cache, "media_root": str(safe(media_root)), "max_bytes": max_bytes}


def _read_pin(pin):
    from .checkpoint_retention import check_pin
    check_pin(pin)
    require(pin["bytes"] <= MAX_METADATA_BYTES, "metadata pin exceeds byte bound")
    raw = safe(pin["path"]).read_bytes()
    require(len(raw) == pin["bytes"] and digest(raw) == pin["sha256"], "metadata changed while reading")
    return raw


def _benchmark_scope(rows, dataset, frames):
    count = 2697 if dataset == "videomme" else 1580
    require(len(rows) == count and len({row["sample_id"] for row in rows}) == count,
            "benchmark QA coverage is incomplete")
    require(all(len(row.get("image_paths", [])) == frames and row["split"] == "test"
                and row.get("dataset") == dataset for row in rows), "benchmark frame count or split differs")
    if dataset == "videomme":
        from .evaluate import validate_videomme_subset
        validate_videomme_subset(rows)
    else:
        require(len({row.get("video_id") for row in rows}) == 410
                and all(row.get("original_answer") and row.get("underlying_source_video_ids") for row in rows),
                "TempCompass official clip/answer provenance is incomplete")


def _bound_cache_inventory(cache):
    """Validate source→sample→original index→PNG mappings before materialization."""
    kind = cache.get("kind")
    require(kind in ("suite", "legacy"), "unknown frame cache evidence kind")
    keys = ["manifest", "receipt", "source_binding", "source_manifest"]
    if kind == "suite":
        keys += ["plan", "complete"]
    raw = {key: _read_pin(cache[key]) for key in keys}
    binding = json.loads(raw["source_binding"])
    original_rows, archives, sources = validate_binding(binding, raw["source_manifest"])
    receipt = json.loads(raw["receipt"])
    require(receipt.get("complete") is True and receipt.get("sampling") == SAMPLING,
            "complete original uniform-sampling receipt missing")
    frames = receipt.get("num_frames")
    require(type(frames) is int and frames in (4, 8, 16), "cache frame count is unsupported")
    rows = validate_records([json.loads(line) for line in raw["manifest"].decode().splitlines() if line.strip()],
                            check_media=False)
    _benchmark_scope(rows, binding["dataset"], frames)
    strip_media = lambda row: {key: value for key, value in row.items() if key not in ("video_path", "image_paths")}
    require([strip_media(row) for row in rows] == [strip_media(row) for row in original_rows],
            "cache manifest questions, answers, source identity or order changed")
    entries = receipt.get("entries", [])
    identity_for = (lambda entry: entry.get("media_identity")) if kind == "suite" else _identity
    by_entry = {identity_for(entry): entry for entry in entries}
    require(len(by_entry) == len(entries) and set(by_entry) == set(sources), "frame receipt source coverage differs")
    if kind == "suite":
        plan, complete = json.loads(raw["plan"]), json.loads(raw["complete"])
        prefix = _key(cache["prefix"])
        require(plan.get("binding_sha256") == digest(encode(binding))
                and complete.get("source_binding_sha256") == digest(encode(binding))
                and complete.get("status") == "FRAME_CACHE_COMPLETE"
                and complete.get("plan_sha256") == digest(raw["plan"])
                and complete.get("manifest_sha256") == digest(raw["manifest"])
                and complete.get("frame_cache_receipt_sha256") == digest(raw["receipt"])
                and plan.get("prefix") == complete.get("prefix") == prefix
                and receipt.get("plan_sha256") == digest(raw["plan"])
                and receipt.get("manifest_sha256") == digest(raw["manifest"])
                and plan.get("num_frames") == frames, "cache completion/source/manifest pins differ")
        objects = None
    else:
        require(frames == 8 and binding.get("accepted_original_decode_receipt_sha256") == digest(raw["receipt"]),
                "legacy F8 needs the exact accepted original decode receipt")
        objects = cache.get("frame_objects")
        require(isinstance(objects, list) and objects, "legacy cache needs exact frame-to-OSS mapping")
        require(len({item["relative_path"] for item in objects}) == len(objects), "duplicate legacy object mapping")
        objects = {item["relative_path"]: item for item in objects}
    inventory, claimed = [], {}
    for identity, source in sources.items():
        entry = by_entry[identity]
        count = entry.get("decoded_frame_count")
        require(entry.get("decode_complete") is True and _positive(count)
                and entry.get("selected_indices") == uniform_indices(count, frames)
                and len(entry.get("frames", [])) == frames, "original frame sampling evidence differs")
        if kind == "suite":
            require(entry.get("source_binding") == source and entry.get("plan_sha256") == digest(raw["plan"])
                    and entry.get("full_decode_passes") == 2, "new cache original member binding differs")
        else:
            require(_bounds(source) == (None, None), "bounded legacy cache requires fresh original-video decoding")
            archive = archives[source["archive_key"]]
            require(entry.get("member_path") == source["member"]["path"]
                    and entry.get("original_sha256") == source["member"]["sha256"]
                    and entry.get("original_size_bytes") == source["member"]["size_bytes"]
                    and entry.get("archive_sha256") == archive["sha256"], "legacy original member evidence differs")
        ordered_paths = []
        for slot, item in enumerate(entry["frames"]):
            relative = safe_relative_path(item.get("relative_path"))
            require(_positive(item.get("size_bytes")) and item["size_bytes"] <= MAX_PNG_BYTES
                    and _sha(item.get("sha256")) and _sha(item.get("rgb_sha256"))
                    and _positive(item.get("width")) and _positive(item.get("height"))
                    and item["width"] * item["height"] <= MAX_FRAME_PIXELS,
                    "invalid frame receipt shape or SHA")
            index = item.get("eligible_frame_index") if kind == "suite" else item.get("source_frame_index")
            require(index == entry["selected_indices"][slot], "frame receipt selected index differs")
            if kind == "suite":
                key = prefix + relative
            else:
                mapping = objects.get(relative, {})
                require(mapping.get("sha256") == item["sha256"] and mapping.get("size_bytes") == item["size_bytes"],
                        "legacy exact OSS object mapping differs")
                key = mapping.get("key")
            _key(key)
            require(relative not in claimed, "two clips claim the same frame path")
            claimed[relative] = identity
            inventory.append({"path": relative, "bytes": item["size_bytes"], "sha256": item["sha256"],
                              "key": key, "rgb_sha256": item["rgb_sha256"], "width": item["width"], "height": item["height"]})
            ordered_paths.append(relative)
        require(entry.get("image_paths") == ordered_paths, "ordered frame paths differ")
        require(all(row["image_paths"] == ordered_paths for row in rows if _identity(row) == identity),
                "question frame paths differ from original source receipt")
    require(set(claimed) == {path for row in rows for path in row["image_paths"]}, "manifest frame inventory differs")
    if objects is not None:
        require(set(objects) == set(claimed), "unused legacy frame object mappings")
    return rows, binding, sorted(inventory, key=lambda item: item["path"]), frames, raw, keys


def hydrate_frame_cache(cache, *, media_root, max_bytes, store, resume=False):
    """Download bound PNGs to one owned Linux-node directory and export READY.

    The operation verifies every source archive receipt and every PNG byte/RGB
    hash. Existing owned frames are read and checked before reuse. A manifest,
    source binding or file discrepancy stops the operation; there is no delete.
    """
    require(platform.system() == "Linux", "media hydration belongs on the remote Linux node")
    from .checkpoint_retention import file_pin, digest as structured_digest
    from PIL import Image
    rows, binding, inventory, frames, raw, evidence_keys = _bound_cache_inventory(cache)
    require(_positive(max_bytes), "positive node media capacity cap required")
    needed = sum(item["bytes"] for item in inventory)
    require(needed <= max_bytes, "bound frame inventory exceeds node media capacity cap")
    root = safe(media_root)
    owner = {"schema_version": 1, "kind": "suite_benchmark_media", "cache_binding_sha256": structured_digest(cache),
             "media_root": str(root), "max_bytes": max_bytes, "png_bytes": needed}
    owner_raw = encode(owner)
    if root.exists():
        require(resume and root.is_dir() and (root / ".suite-media-owner.json").read_bytes() == owner_raw,
                "existing media directory is not this exact owned cache")
    else:
        require(root.parent.is_dir(), "owned media directory parent is missing")
    for archive in binding["archives"]:
        verify_archive(store, archive)
    if cache["kind"] == "suite":
        prefix = cache["prefix"]
        for key, name in (("plan", "BUILD_PLAN.json"), ("manifest", "manifest.jsonl"),
                          ("receipt", "FRAME_CACHE_RECEIPT.json"), ("complete", "CACHE_COMPLETE.json")):
            require(store.get(prefix + name, len(raw[key])) == raw[key], "bucket cache metadata changed")
        expected_keys = {prefix + name for name in ("BUILD_PLAN.json", "manifest.jsonl", "FRAME_CACHE_RECEIPT.json",
                                                     "CACHE_COMPLETE.json")}
        expected_keys.update(item["key"] for item in inventory)
        expected_keys.update(prefix + "entries/" + digest(source["media_identity"].encode()) + ".json"
                             for source in binding["sources"])
        require(set(store.keys(prefix)) == expected_keys, "cache exact bucket inventory changed")
    allowed = {item["path"] for item in inventory} | {".suite-media-owner.json"}
    local_size = 0
    if root.exists():
        for path in root.rglob("*"):
            safe(path)
            if path.is_file():
                require(path.relative_to(root).as_posix() in allowed and path.stat().st_nlink == 1,
                        "unowned or linked file in media directory")
                if path.name != ".suite-media-owner.json":
                    local_size += path.stat().st_size
    require(shutil.disk_usage(root if root.exists() else root.parent).free >= max(0, needed - local_size),
            "insufficient physical free space for bounded frame cache")
    root.mkdir(exist_ok=True)
    _write_once(root / ".suite-media-owner.json", owner_raw)
    for item in inventory:
        # Re-read the exact bucket object even when a local frame exists. A
        # READY marker can therefore never hide later archive/object drift.
        payload = store.get(item["key"], item["bytes"])
        require(payload is not None and len(payload) == item["bytes"] and digest(payload) == item["sha256"],
                "frame full OSS GET SHA differs during hydration")
        with Image.open(io.BytesIO(payload)) as image:
            require(image.format == "PNG" and image.mode == "RGB" and image.size == (item["width"], item["height"]),
                    "decoded PNG shape/color mode differs")
            image.load()
            require(digest(image.tobytes()) == item["rgb_sha256"], "decoded PNG RGB SHA differs")
        path = safe(root / item["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_once(path, payload)
    files = [{"path": item["path"], "bytes": item["bytes"], "sha256": item["sha256"]} for item in inventory]
    for item in files:
        require(file_pin(root / item["path"]) == {**item, "path": str(root / item["path"])},
                "final local cache byte/SHA identity differs")
    for key in evidence_keys:
        require(_read_pin(cache[key]) == raw[key], "source evidence changed during hydration")
    # Small evidence remains separate from the owned frame payload directory.
    metadata = safe(root.parent / (root.name + ".acceptance"))
    metadata.mkdir(exist_ok=True)
    acceptance = {"schema_version": 1, "status": "SUITE_BENCHMARK_READY", "manifest_sha256": cache["manifest"]["sha256"],
                  "num_frames": frames, "media_inventory_sha256": structured_digest(files),
                  "sampling": "uniform_indices_from_original_video", "qa": len(rows), "source_count": len(binding["sources"]),
                  "evidence": [cache[key] for key in evidence_keys], "cache_binding_sha256": structured_digest(cache),
                  "png_bytes": needed, "all_png_full_oss_get_and_rgb_verified": True,
                  "formal_evaluation_completed": False}
    _write_once(metadata / "SUITE_BENCHMARK_READY.json", encode(acceptance))
    result = {"manifest": cache["manifest"], "media_root": str(root), "media_files": files,
              "acceptance": file_pin(metadata / "SUITE_BENCHMARK_READY.json")}
    _write_once(metadata / "ENVIRONMENT_FRAGMENT.json", encode(result))
    return result


def hydrate_main(argv=None):
    parser = argparse.ArgumentParser(description=hydrate_frame_cache.__doc__)
    parser.add_argument("--asset-binding", required=True, type=Path)
    parser.add_argument("--asset-binding-sha256", required=True)
    parser.add_argument("--oss-config", type=Path, default=Path("/home/jinhaopeng/.ossutilconfig"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        raw = safe(args.asset_binding).read_bytes()
        require(_sha(args.asset_binding_sha256) and digest(raw) == args.asset_binding_sha256,
                "asset binding SHA differs")
        asset = json.loads(raw)
        rows, binding, inventory, frames, _, _ = _bound_cache_inventory(asset["cache"])
        require(sum(item["bytes"] for item in inventory) <= asset["max_bytes"], "node frame cache capacity cap exceeded")
        if not args.execute:
            print(json.dumps({"status": "PREPARED_NOT_EXECUTED", "qa": len(rows), "num_frames": frames,
                              "png_bytes": sum(item["bytes"] for item in inventory), "dataset": binding["dataset"]}))
            return 0
        store = OSSStore(args.oss_config, allowed_prefix=PROJECT_PREFIX + "suite-media-readonly/")
        fragment = hydrate_frame_cache(asset["cache"], media_root=asset["media_root"], max_bytes=asset["max_bytes"],
                                       store=store, resume=args.resume)
        print(json.dumps({"status": "SUITE_BENCHMARK_READY", "acceptance": fragment["acceptance"],
                          "media_root": fragment["media_root"], "frame_count": len(fragment["media_files"])}))
        return 0
    except Exception as error:
        print(json.dumps({"status": "FAILED", "reason": str(error) if isinstance(error, CacheError)
                          else type(error).__name__, "automatic_retry": False}), file=sys.stderr)
        return 2


def binding_main(argv=None):
    parser = argparse.ArgumentParser(description=create_source_binding.__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--frame-cache-receipt", required=True, type=Path)
    parser.add_argument("--frame-cache-receipt-sha256", required=True)
    parser.add_argument("--archive-catalog", required=True, type=Path)
    parser.add_argument("--archive-catalog-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--oss-config", type=Path, default=Path("/home/jinhaopeng/.ossutilconfig"))
    parser.add_argument("--execute", action="store_true", help="read original archive metadata and write binding")
    args = parser.parse_args(argv)
    try:
        inputs = []
        for path, sha256 in ((args.source_manifest, args.source_manifest_sha256),
                             (args.frame_cache_receipt, args.frame_cache_receipt_sha256),
                             (args.archive_catalog, args.archive_catalog_sha256)):
            raw = safe(path).read_bytes()
            require(len(raw) <= MAX_METADATA_BYTES and _sha(sha256) and digest(raw) == sha256,
                    "source binding preparation input SHA differs")
            inputs.append(raw)
        if not args.execute:
            print(json.dumps({"status": "PREPARED_NOT_EXECUTED", "input_sha256": [digest(raw) for raw in inputs],
                              "original_media_decode": False, "original_media_written": False}))
            return 0
        store = OSSStore(args.oss_config, allowed_prefix=PROJECT_PREFIX + "suite-media-readonly/")
        binding = create_source_binding(inputs[0], inputs[1], json.loads(inputs[2]), store=store)
        raw = encode(binding)
        _write_once(safe(args.output), raw)
        print(json.dumps({"status": "SOURCE_BINDING_PREPARED", "sha256": digest(raw),
                          "source_count": binding["expected_clips"], "sample_count": binding["expected_samples"],
                          "new_frame_cache_completed": False}))
        return 0
    except Exception as error:
        print(json.dumps({"status": "FAILED", "reason": str(error) if isinstance(error, CacheError)
                          else type(error).__name__}), file=sys.stderr)
        return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--source-binding", required=True, type=Path)
    parser.add_argument("--source-binding-sha256", required=True)
    parser.add_argument("--num-frames", required=True, type=int, choices=(4, 8, 16))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--oss-prefix", required=True, help="key prefix within ss-devbox-stream")
    parser.add_argument("--max-png-bytes", required=True, type=int)
    parser.add_argument("--oss-config", type=Path, default=Path("/home/jinhaopeng/.ossutilconfig"))
    parser.add_argument("--resume", action="store_true", help="reverify one exact owned partial cache")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        raw = safe(args.source_binding).read_bytes()
        require(len(raw) <= MAX_METADATA_BYTES and _sha(args.source_binding_sha256)
                and digest(raw) == args.source_binding_sha256, "source binding SHA differs")
        binding = json.loads(raw)
        plan = cache_plan(binding, safe(args.source_manifest).read_bytes(), frames=args.num_frames,
                          oss_prefix=args.oss_prefix, max_bytes=args.max_png_bytes)
        if not args.execute:
            print(json.dumps({"status": "PREPARED_NOT_EXECUTED", "plan": plan}, sort_keys=True))
            return 0
        store = OSSStore(args.oss_config, allowed_prefix=args.oss_prefix)
        result = build_frame_cache(args.source_manifest, binding, frames=args.num_frames,
                    output_dir_new=args.output_dir, oss_prefix_new=args.oss_prefix, max_bytes=args.max_png_bytes,
                    store=store, resume=args.resume)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as error:
        # SDK exceptions can contain signed request details; report only type.
        print(json.dumps({"status": "FAILED", "reason": str(error) if isinstance(error, CacheError)
                          else type(error).__name__, "automatic_retry": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
