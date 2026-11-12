# Declared-only variant of train.py; existing native runner remains unchanged.
"""Full-parameter native-looped video SFT via HF Trainer / Accelerate DDP."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import importlib.metadata
import re
import subprocess

import torch
import torch.distributed as dist
from transformers import AutoImageProcessor, AutoTokenizer, Trainer, TrainingArguments, set_seed

from .data import VideoManifestDataset, MultimodalCollator
from .formal_integrity import (manifest_media, verify_attested_media_metadata,
                               verify_attested_training_media,
                               verify_frozen_source_manifest)
from .manifest import load_manifest, validate_disjoint
from .modeling import DeepProbeConfig, DeepProbeVLM
from .resume_checkpoint import verify_resume_checkpoint


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


FORMAL_POLICY_ID = "deepprobe-formal-benchmark-source-v1"
FORMAL_TRAINING_GATES = {
    "pinned_annotations", "videomme_fixed_subset", "benchmark_source_identity",
    "training_manifests", "training_candidate_coverage", "training_data_attestation",
    "training_media_integrity", "declared_source_leakage", "artifact_integrity",
}


def _read_json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _canonical_sha256(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _passed_formal_gates(value) -> bool:
    return (isinstance(value, dict) and set(value) == FORMAL_TRAINING_GATES
            and all(isinstance(item, dict) and item.get("status") == "passed"
                    for item in value.values()))


def verify_formal_ready(path: str | Path, train_hash: str, eval_hash: str,
                        training_data_dir: str | Path, media_root: str | Path) -> dict:
    """Authenticate formal receipts and the inventory bound to the manifests.

    Current media bytes are checked separately by rank zero in ``main`` before
    and after training.  A receipt alone never authorizes formal training.
    """
    receipt_path = Path(path)
    if not receipt_path.is_file():
        raise ValueError("Formal training requires an existing FORMAL_READY.json")
    data_dir = Path(training_data_dir)
    ready = _read_json_object(receipt_path)
    if (ready.get("schema_version") != 1 or ready.get("policy_id") != FORMAL_POLICY_ID
            or not isinstance(ready.get("formal_evaluation_ready"), bool)
            or ready.get("formal_training_ready") is not True or ready.get("status") not in {
            "ready_for_training", "ready_for_evaluation"}):
        raise ValueError("Formal source/data gates are not ready for training")
    if ready.get("benchmark_source_identity_gate") != "passed_on_declared_subset":
        raise ValueError("Formal benchmark source identity gate did not pass")
    gates = ready.get("training_readiness_gates")
    if not _passed_formal_gates(gates):
        raise ValueError("FORMAL_READY requires the complete passed training gate set")
    immutable = ready.get("immutable_input_sha256")
    if (not isinstance(immutable, dict) or immutable.get("training_train") != train_hash
            or immutable.get("training_validation") != eval_hash):
        raise ValueError("FORMAL_READY.json is not bound to these train/validation manifests")
    artifacts = ready.get("artifact_sha256")
    if (not isinstance(artifacts, dict) or not artifacts
            or not {"source_policy.json", "source_gate_report.json",
                    "training_media_inventory.json"}.issubset(artifacts)):
        raise ValueError("FORMAL_READY.json lacks its generated artifact hash set")
    for name, expected in artifacts.items():
        if (not isinstance(name, str) or Path(name).name != name
                or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)):
            raise ValueError("FORMAL_READY.json contains an invalid artifact entry")
        artifact = receipt_path.parent / name
        if artifact.is_symlink() or not artifact.is_file() or _sha256_file(artifact) != expected:
            raise ValueError(f"Formal gate artifact missing or changed: {name}")

    policy = _read_json_object(receipt_path.parent / "source_policy.json")
    if policy.get("schema_version") != 1 or policy.get("policy_id") != FORMAL_POLICY_ID:
        raise ValueError("Formal source policy schema/identity differs")
    report = _read_json_object(receipt_path.parent / "source_gate_report.json")
    if (report.get("schema_version") != 1 or report.get("status") != ready.get("status")
            or not isinstance(report.get("formal_evaluation_ready"), bool)
            or report.get("formal_training_ready") is not True
            or report.get("formal_evaluation_ready") != ready.get("formal_evaluation_ready")
            or report.get("benchmark_source_identity_gate") != "passed_on_declared_subset"
            or report.get("training_readiness_gates") != gates
            or not _passed_formal_gates(report.get("training_readiness_gates"))):
        raise ValueError("source_gate_report status/training gates differ from FORMAL_READY")

    required_data = {
        name: data_dir / name for name in (
            "train.jsonl", "validation.jsonl", "coverage.jsonl", "READY.json",
            "preparation_report.json", "training_media_inventory.json")
    }
    for name, item in required_data.items():
        if item.is_symlink() or not item.is_file():
            raise ValueError(f"Formal training-data artifact missing or unsafe: {name}")
    data_hashes = {name: _sha256_file(item) for name, item in required_data.items()}
    if data_hashes["train.jsonl"] != train_hash or data_hashes["validation.jsonl"] != eval_hash:
        raise ValueError("Formal training data directory differs from CLI manifests")
    if (immutable.get("training_coverage") != data_hashes["coverage.jsonl"]
            or immutable.get("training_media_inventory") != data_hashes["training_media_inventory.json"]):
        raise ValueError("FORMAL_READY coverage/media inventory hashes differ")

    data_ready = _read_json_object(required_data["READY.json"])
    preparation = _read_json_object(required_data["preparation_report.json"])
    media_inventory = _read_json_object(required_data["training_media_inventory.json"])
    if media_inventory.get("media_root") != str(Path(media_root).resolve()):
        raise ValueError("Formal training media inventory used a different MEDIA_ROOT")
    if (data_ready.get("schema_version") != 1
            or data_ready.get("purpose") != "FORMAL_TRAINING_DATA"
            or data_ready.get("formal_data_ready") is not True
            or data_ready.get("report_sha256") != data_hashes["preparation_report.json"]):
        raise ValueError("Training-data READY schema/status/report binding differs")
    if (preparation.get("schema_version") != 1
            or preparation.get("status") != "formal_training_data_ready"
            or preparation.get("formal_data_ready") is not True):
        raise ValueError("Training preparation report is not formal-ready")
    expected_outputs = {
        "train.jsonl": train_hash, "validation.jsonl": eval_hash,
        "coverage.jsonl": data_hashes["coverage.jsonl"],
        "training_media_inventory.json": data_hashes["training_media_inventory.json"],
    }
    if any(preparation.get("output_sha256", {}).get(name) != digest
           for name, digest in expected_outputs.items()):
        raise ValueError("Training preparation report output hashes differ")
    if (data_ready.get("manifest_sha256", {}).get("train.jsonl") != train_hash
            or data_ready.get("manifest_sha256", {}).get("validation.jsonl") != eval_hash
            or data_ready.get("coverage_sha256") != data_hashes["coverage.jsonl"]
            or data_ready.get("training_media_inventory_sha256") != data_hashes["training_media_inventory.json"]):
        raise ValueError("Training-data READY manifest/coverage/media hashes differ")

    entries = media_inventory.get("entries")
    if (media_inventory.get("schema_version") != 1
            or media_inventory.get("kind") != "formal_training_media_sha256_inventory"
            or not isinstance(entries, list) or not entries):
        raise ValueError("Training media inventory schema/entries differ")
    seen_media = set()
    for entry in entries:
        if (not isinstance(entry, dict) or not isinstance(entry.get("relative_path"), str)
                or Path(entry["relative_path"]).is_absolute() or ".." in Path(entry["relative_path"]).parts
                or "\\" in entry["relative_path"] or "://" in entry["relative_path"]
                or not isinstance(entry.get("source_video_id"), str)
                or isinstance(entry.get("size_bytes"), bool) or not isinstance(entry.get("size_bytes"), int)
                or entry["size_bytes"] <= 0 or not isinstance(entry.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])):
            raise ValueError("Training media inventory contains an invalid entry")
        if entry["relative_path"] in seen_media:
            raise ValueError("Training media inventory contains a duplicate path")
        seen_media.add(entry["relative_path"])
    entries_sha = _canonical_sha256(entries)
    total_bytes = sum(entry["size_bytes"] for entry in entries)
    if (media_inventory.get("file_count") != len(entries)
            or media_inventory.get("total_bytes") != total_bytes
            or media_inventory.get("entries_sha256") != entries_sha
            or data_ready.get("training_media_entries_sha256") != entries_sha
            or data_ready.get("training_media_file_count") != len(entries)
            or data_ready.get("training_media_total_bytes") != total_bytes):
        raise ValueError("Training media inventory count/content binding differs")
    preparation_media = preparation.get("training_media_inventory", {})
    if (preparation_media.get("relative_path") != "training_media_inventory.json"
            or preparation_media.get("sha256") != data_hashes["training_media_inventory.json"]
            or preparation_media.get("entries_sha256") != entries_sha
            or preparation_media.get("file_count") != len(entries)
            or preparation_media.get("total_bytes") != total_bytes):
        raise ValueError("Training preparation report media binding differs")

    manifests = report.get("training_manifests", {})
    coverage = report.get("training_candidate_coverage", {})
    attestation = report.get("training_data_attestation", {})
    media_attestation = report.get("training_media_attestation", {})
    if manifests.get("valid") is not True or manifests.get("sha256") != {
            "train": train_hash, "validation": eval_hash}:
        raise ValueError("source_gate_report manifest proof differs")
    if coverage.get("valid") is not True or coverage.get("sha256") != data_hashes["coverage.jsonl"]:
        raise ValueError("source_gate_report coverage proof differs")
    attested_media = attestation.get("media_inventory", {})
    if (attestation.get("provided") is not True or attestation.get("valid") is not True
            or attestation.get("report_sha256") != data_hashes["preparation_report.json"]
            or attestation.get("ready_sha256") != data_hashes["READY.json"]
            or attested_media.get("sha256") != data_hashes["training_media_inventory.json"]
            or attested_media.get("entries_sha256") != entries_sha
            or attested_media.get("file_count") != len(entries)
            or attested_media.get("total_bytes") != total_bytes):
        raise ValueError("source_gate_report training-data attestation differs")
    if (media_attestation.get("checked") is not True or media_attestation.get("passed") is not True
            or media_attestation.get("entries_sha256") != entries_sha
            or media_attestation.get("file_count") != len(entries)
            or media_attestation.get("total_bytes") != total_bytes):
        raise ValueError("source_gate_report training media attestation differs")
    current_inventory = _read_json_object(receipt_path.parent / "training_media_inventory.json")
    if (current_inventory.get("complete") is not True
            or current_inventory.get("root") != str(Path(media_root).resolve())
            or current_inventory.get("entries") != entries):
        raise ValueError("Formal-gate current media inventory differs from attested media")
    return {
        "path": str(receipt_path.resolve()),
        "sha256": _sha256_file(receipt_path),
        "policy_id": ready.get("policy_id"),
        "benchmark_source_identity_gate": ready["benchmark_source_identity_gate"],
        "formal_evaluation_ready": ready.get("formal_evaluation_ready") is True,
        "immutable_input_sha256": immutable,
        "training_media_entries_sha256": entries_sha,
    }


FORMAL_TRAIN_COMPLETE = "FORMAL_TRAIN_COMPLETE.json"


def _formal_source_root() -> Path:
    """The code tree actually imported for this train.py process."""
    return Path(__file__).resolve().parents[2]


def rank_zero_integrity_check(phase: str, verify, *, expected_rank: int,
                              expected_world_size: int) -> dict:
    """Run one live integrity check and deliver the same result to all ranks.

    A rank-zero exception is serialized before any rank raises, so peers do not
    silently proceed into Trainer when the verifier rejects current bytes.
    """
    initialized = dist.is_available() and dist.is_initialized()
    if expected_world_size < 1 or expected_rank < 0 or expected_rank >= expected_world_size:
        raise ValueError(f"{phase}: invalid distributed rank/world size")
    if expected_world_size > 1 and not initialized:
        raise ValueError(f"{phase}: distributed process group is not initialized")
    if initialized and (dist.get_world_size() != expected_world_size
                        or dist.get_rank() != expected_rank):
        raise ValueError(f"{phase}: distributed rank/world size differs from TrainingArguments")
    if expected_world_size == 1:
        return verify()
    packet = [None]
    if expected_rank == 0:
        try:
            packet[0] = {"ok": True, "result": verify()}
        except Exception as exc:
            packet[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    dist.broadcast_object_list(packet, src=0)
    value = packet[0]
    if not isinstance(value, dict) or value.get("ok") is not True:
        detail = value.get("error") if isinstance(value, dict) else "invalid rank-zero result"
        raise ValueError(f"{phase}: rank-zero formal integrity check failed: {detail}")
    result = value.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"{phase}: rank-zero formal integrity result is invalid")
    return result


def _verify_live_formal_inputs(*, source_manifest: Path, source_manifest_sha256: str,
                               formal_ready: Path, training_data_dir: Path,
                               train_hash: str, eval_hash: str, media_root: Path,
                               expected_media: dict[str, str],
                               full_media_hash: bool) -> dict:
    receipt = verify_formal_ready(formal_ready, train_hash, eval_hash,
                                  training_data_dir, media_root)
    source = verify_frozen_source_manifest(
        _formal_source_root(), source_manifest, source_manifest_sha256)
    media_verifier = (verify_attested_training_media if full_media_hash
                      else verify_attested_media_metadata)
    media = media_verifier(
        training_data_dir / "training_media_inventory.json", media_root,
        receipt["immutable_input_sha256"]["training_media_inventory"], expected_media)
    return {"receipt": receipt, "source": source, "media": media}


def _require_unchanged_formal_inputs(before: dict, after: dict) -> dict:
    if before != after:
        raise ValueError("Formal source/media changed since the initial full SHA-256 check")
    return after


def _write_formal_completion(output: Path, *, stage: str, world_size: int,
                             train_hash: str, eval_hash: str, before: dict,
                             after: dict) -> dict:
    """Create the formal success marker only after the final live byte check."""
    for name in ("run_manifest.json", "trainer_state.json", "train_results.json",
                 "eval_results.json"):
        path = output / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Formal completion requires {name}")
    if not (output / "final" / "deepprobe_config.json").is_file():
        raise ValueError("Formal completion requires the final model")
    state = _read_json_object(output / "trainer_state.json")
    if type(state.get("global_step")) is not int or state["global_step"] <= 0:
        raise ValueError("Formal completion requires a positive optimizer step")
    _require_unchanged_formal_inputs(before, after)
    from . import declared_training as declared
    manifest = _read_json_object(output / "run_manifest.json")
    identity = declared.completion_identity(manifest["training_contract"], state)
    receipt = {
        **identity,
        "schema_version": 1, "status": "formal_training_complete",
        "stage": stage, "world_size": world_size,
        "optimizer_step": state["global_step"],
        "train_manifest_sha256": train_hash,
        "eval_manifest_sha256": eval_hash,
        "formal_ready_sha256": before["receipt"]["sha256"],
        "frozen_source_manifest_sha256": before["source"]["manifest_sha256"],
        "media_before": before["media"], "media_after": after["media"],
        "run_manifest_sha256": _sha256_file(output / "run_manifest.json"),
        "trainer_state_sha256": _sha256_file(output / "trainer_state.json"),
    }
    path = output / FORMAL_TRAIN_COMPLETE
    with path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


class DeepProbeTrainer(Trainer):
    """Keep component checkpoints plus native Trainer optimizer/RNG state.

    This implementation targets single GPU and replicated DDP, not sharded
    FSDP/ZeRO saving; refusing unsupported modes is preferable to partial saves.
    """
    def __init__(self, *args, image_processor=None, training_sources=None,
                 tuning_sources=None, training_contract=None, **kwargs):
        self.image_processor = image_processor
        self.training_sources = sorted(training_sources or [])
        self.tuning_sources = sorted(tuning_sources or [])
        self.training_contract = training_contract
        super().__init__(*args, **kwargs)
        if self.is_deepspeed_enabled or self.is_fsdp_enabled:
            raise ValueError("Only ordinary DDP is supported by this checkpoint format")
        self.model_accepts_loss_kwargs = False

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        self.model.save_pretrained(output_dir)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(Path(output_dir) / "tokenizer")
        if self.image_processor is not None:
            self.image_processor.save_pretrained(Path(output_dir) / "image_processor")
        (Path(output_dir) / "training_provenance.json").write_text(json.dumps(
            {"training_sources": self.training_sources, "tuning_sources": self.tuning_sources,
             "training_contract": self.training_contract}, indent=2) + "\n")
        torch.save(self.args, Path(output_dir) / "training_args.bin")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        (model or self.model).load_checkpoint_weights(resume_from_checkpoint)

    def save_final_resume_checkpoint(self):
        """Persist a native Trainer checkpoint at the exact final step.

        ``save_steps`` need not divide the final step.  The pinned Transformers
        runtime's checkpoint path is deliberately used here because it saves
        optimizer, scheduler, scaler, Trainer state and every DDP rank's RNG in
        addition to this class's portable component weights.
        """
        if self.args.save_only_model:
            raise ValueError("Formal final checkpoint must include optimizer/RNG state")
        if self.state.global_step <= 0:
            raise ValueError("Cannot save a final resume checkpoint before an optimizer step")
        self._save_checkpoint(self.model, trial=None)
        self.accelerator.wait_for_everyone()
        return Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"


class DiagnosticDeepProbeTrainer(DeepProbeTrainer):
    """Use the ordinary Trainer update with one opt-in read-only step trace."""

    def __init__(self, *args, step_trace, **kwargs):
        from .step_diagnostic import FirstStepTraceCallback

        self.step_trace = step_trace
        self._inside_traced_training_step = False
        super().__init__(*args, **kwargs)
        self.add_callback(FirstStepTraceCallback(step_trace))

    def training_step(self, model, inputs, num_items_in_batch=None):
        from .step_diagnostic import EmbeddingGradientTap, SAMPLE_ID_KEY, ddp_bucket_snapshot

        sample_ids = inputs.pop(SAMPLE_ID_KEY, None)
        tracing = self.step_trace.should_trace(self.state.global_step)
        gradient_tap = None
        if tracing:
            if sample_ids is None:
                raise RuntimeError("Diagnostic training batch has no sample IDs")
            ddp_model = (model if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                         else self.model_wrapped)
            self.step_trace.batch_before_forward(
                list(sample_ids), inputs,
                ddp_bucket_before_forward=ddp_bucket_snapshot(ddp_model))
            supports_embedding_tap = EmbeddingGradientTap.supports(self.model)
            stage = getattr(self.model, "stage", None)
            if stage == "sft" and not supports_embedding_tap:
                raise RuntimeError("Diagnostic SFT requires an embedding gradient boundary")
            if supports_embedding_tap:
                embedding = self.model.language_model.get_input_embeddings()
                weight = getattr(embedding, "weight", None)
                trainable = isinstance(weight, torch.nn.Parameter) and weight.requires_grad
                if stage == "alignment":
                    # Alignment trains the projector only; no embedding weight
                    # gradient exists to capture at this boundary.
                    if trainable:
                        raise RuntimeError("Diagnostic alignment expected a frozen input embedding")
                elif not trainable:
                    raise RuntimeError("Diagnostic SFT requires a trainable input embedding")
                else:
                    gradient_tap = EmbeddingGradientTap(self.model, inputs["input_ids"])
                    gradient_tap.install()
        self._inside_traced_training_step = tracing
        try:
            loss = super().training_step(model, inputs, num_items_in_batch)
            embedding_gradients = (gradient_tap.summary(sync_gradients=self.accelerator.sync_gradients)
                                   if gradient_tap is not None else None)
        finally:
            self._inside_traced_training_step = False
            if gradient_tap is not None:
                gradient_tap.close()
        if tracing:
            self.step_trace.after_backward(
                model=self.model, scaled_loss=loss,
                sync_gradients=self.accelerator.sync_gradients,
                embedding_gradients=embedding_gradients,
                ddp_bucket_after_backward=ddp_bucket_snapshot(ddp_model))
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        result = super().compute_loss(model, inputs, return_outputs=return_outputs,
                                      num_items_in_batch=num_items_in_batch)
        if self._inside_traced_training_step:
            self.step_trace.retain_raw_loss(result[0] if return_outputs else result)
        return result

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        from .step_diagnostic import SAMPLE_ID_KEY

        inputs.pop(SAMPLE_ID_KEY, None)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


def verify_checkpoint_config(saved: DeepProbeConfig, current: DeepProbeConfig,
                             *, exact_resume: bool) -> None:
    """Require alignment/SFT architecture parity and exact resume identity."""
    for name in ("language_model", "language_revision", "language_backend",
                 "vision_model", "vision_revision", "num_loops", "num_frames",
                 "pool_grid", "max_text_tokens", "attn_implementation"):
        if getattr(saved, name) != getattr(current, name):
            raise ValueError(f"Checkpoint/config mismatch: {name}")
    if exact_resume and saved != current:
        raise ValueError("Exact resume requires an unchanged model configuration")


def validate_p0_resumed_final_export_option(cli, training: dict, *, formal_training: bool) -> None:
    """Allow the P0 resume branch to rely on its complete final-step checkpoint.

    This only skips the redundant post-training ``final/`` model export.  It
    must not change a formal run, an initializer run, or the checkpoint save
    schedule used for direct model/optimizer/scheduler/RNG comparison.
    """
    if not cli.p0_skip_resumed_final_export:
        return
    if formal_training:
        raise ValueError("Formal training cannot skip its final model export")
    if not cli.resume or cli.init_checkpoint:
        raise ValueError("P0 final-export omission requires an exact --resume branch")
    if training.get("save_strategy", "steps") != "steps":
        raise ValueError("P0 final-export omission requires step checkpoints")
    final_step, save_steps = training.get("max_steps"), training.get("save_steps", 100)
    if (type(final_step) is not int or type(save_steps) is not int
            or not 0 < save_steps <= final_step or final_step % save_steps):
        raise ValueError("P0 final-export omission requires a checkpoint at the final step")
    save_total_limit = training.get("save_total_limit", 2)
    if type(save_total_limit) is not int or save_total_limit < 2:
        raise ValueError("P0 final-export omission must retain the fork and final checkpoints")
    if training.get("save_only_model", False):
        raise ValueError("P0 final-export omission requires optimizer/RNG checkpoint state")
    if (Path(cli.output_dir) / "final").exists() or (Path(cli.output_dir) / "final").is_symlink():
        raise ValueError("P0 final-export omission requires no preexisting final directory")


def connector_state_sha256(model):
    rows = {}
    for name, value in sorted(model.projector.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        if not torch.isfinite(tensor).all():
            raise ValueError("Nonfinite initial connector tensor")
        rows[name] = {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
                      "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest()}
    if not rows:
        raise ValueError("Missing initial connector state")
    return _canonical_sha256(rows)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--eval-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--media-root")
    p.add_argument("--resume", help="Resume weights, optimizer, scheduler, RNG and progress")
    p.add_argument("--init-checkpoint", help="Initialize weights only, e.g. alignment -> SFT")
    p.add_argument("--no-alignment", action="store_true", help="Explicit declared random-connector SFT")
    p.add_argument("--formal-ready", help="Verified FORMAL_READY.json required by formal-training configs")
    p.add_argument("--formal-training-data-dir",
                   help="Directory containing hash-bound formal train/validation/coverage receipts")
    p.add_argument("--frozen-source-manifest",
                   help="Actual SOURCE_MANIFEST.json for the imported frozen experiment tree")
    p.add_argument("--frozen-source-manifest-sha256",
                   help="Exact frozen experiment manifest identity; required for formal training")
    p.add_argument("--diagnostic-step", type=int,
                   help="Opt-in per-rank evidence for one optimizer step; does not change the schedule")
    p.add_argument("--p0-skip-resumed-final-export", action="store_true",
                   help="P0 exact-resume branch only: retain the complete final-step checkpoint without a duplicate final/ model export")
    return p


def main(argv=None):
    cli = parser().parse_args(argv)
    if cli.resume and cli.init_checkpoint:
        raise ValueError("Choose --resume OR --init-checkpoint")
    output = Path(cli.output_dir)
    if output.exists() and any(output.iterdir()) and not cli.resume:
        raise FileExistsError("Output directory is nonempty; choose a fresh run directory or explicit --resume")
    raw = json.loads(Path(cli.config).read_text())
    from . import declared_training as declared
    mode = declared.initialization_mode(raw, init_checkpoint=cli.init_checkpoint,
                                        resume=cli.resume, no_alignment=cli.no_alignment)
    config = DeepProbeConfig.from_dict(raw)
    settings = dict(raw.get("training", {}))
    stage = settings.pop("stage", "sft")
    if cli.diagnostic_step is not None:
        max_steps = settings.get("max_steps")
        if (cli.diagnostic_step <= 0 or isinstance(max_steps, bool)
                or not isinstance(max_steps, int) or cli.diagnostic_step > max_steps):
            raise ValueError("--diagnostic-step requires 1 <= step <= explicit training.max_steps")
    formal_training = settings.pop("formal_training", False)
    if not isinstance(formal_training, bool):
        raise ValueError("training.formal_training must be boolean")
    require_final_resume_checkpoint = settings.pop(
        "require_final_resume_checkpoint", formal_training
    )
    if not isinstance(require_final_resume_checkpoint, bool):
        raise ValueError("training.require_final_resume_checkpoint must be boolean")
    if formal_training and not require_final_resume_checkpoint:
        raise ValueError("Formal training requires a final resumable checkpoint")
    validate_p0_resumed_final_export_option(cli, settings, formal_training=formal_training)
    source_manifest_sha256 = cli.frozen_source_manifest_sha256
    if source_manifest_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", source_manifest_sha256):
        raise ValueError("--frozen-source-manifest-sha256 must be a lowercase SHA-256")
    if formal_training and (source_manifest_sha256 is None or not cli.frozen_source_manifest):
        raise ValueError("Formal training requires --frozen-source-manifest and its SHA-256")
    if formal_training and (output / FORMAL_TRAIN_COMPLETE).exists():
        raise FileExistsError("A completed formal output cannot be resumed or overwritten")
    set_seed(settings.get("seed", 17))
    train_root = Path(cli.media_root) if cli.media_root else Path(cli.train_manifest).resolve().parent
    eval_root = Path(cli.media_root) if cli.media_root else Path(cli.eval_manifest).resolve().parent
    train_records = load_manifest(cli.train_manifest, media_root=train_root)
    eval_records = load_manifest(cli.eval_manifest, media_root=eval_root)
    validate_disjoint(train_records, eval_records)
    if len(train_records) != declared.TRAIN_EXAMPLES:
        raise ValueError("Declared stage requires the frozen 91,826 training examples")
    train_hash = _sha256_file(Path(cli.train_manifest))
    eval_hash = _sha256_file(Path(cli.eval_manifest))
    if formal_training and (not cli.formal_ready or not cli.formal_training_data_dir):
        raise ValueError("Formal-training config requires --formal-ready and --formal-training-data-dir")
    if cli.formal_ready and not formal_training:
        raise ValueError("--formal-ready requires training.formal_training=true")
    if any(x["split"] not in {"train", "training"} for x in train_records):
        raise ValueError("Training manifest must use train/training split; benchmark test is not training data")
    if any(x["split"] not in {"val", "valid", "validation", "dev"} for x in eval_records):
        raise ValueError("Training-time evaluation must use validation/dev, never held-out test")
    if settings.get("bf16", True) and not torch.cuda.is_available():
        raise ValueError("bf16 training requires CUDA in this recipe; disable explicitly for CPU tests")
    if settings.get("bf16", True) and not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA device does not support bf16")
    training_defaults = dict(per_device_train_batch_size=1, per_device_eval_batch_size=1,
                             gradient_accumulation_steps=8, learning_rate=2e-5,
                             num_train_epochs=1, max_steps=-1, bf16=True,
                             gradient_checkpointing=True, seed=17,
                             save_steps=100, logging_steps=10, save_total_limit=2,
                             dataloader_num_workers=0, optim="adamw_torch", report_to="none",
                             eval_strategy="epoch", prediction_loss_only=True,
                             remove_unused_columns=False, ddp_find_unused_parameters=True,
                             gradient_checkpointing_kwargs={"use_reentrant": False})
    training_defaults.update(settings)
    # These are correctness invariants, not user-tunable performance flags.
    # With find_unused_parameters=False, DDP starts with one large bucket and
    # rebuilds it after the first backward pass. A resumed process would then
    # reduce its first gradient with a different bucket layout from the
    # uninterrupted process. True keeps the bucket assignment fixed across a
    # restart (and is supported with the non-reentrant checkpoint below).
    training_defaults.update(remove_unused_columns=False, prediction_loss_only=True,
                             ddp_find_unused_parameters=True,
                             gradient_checkpointing_kwargs={"use_reentrant": False})
    args = TrainingArguments(output_dir=cli.output_dir, **training_defaults)
    formal_inputs = None
    formal_receipt = None
    if formal_training:
        # TrainingArguments initializes Accelerate's process group before any
        # collective.  The actual source tree and media bytes are checked by
        # only global rank zero and its verdict is broadcast to all ranks.
        device = args.device
        if device.type == "cuda":
            # NCCL object collectives use the current CUDA device.  Make the
            # local-rank mapping explicit before peers wait on rank zero.
            torch.cuda.set_device(device)
        declared_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if declared_world_size > 1 and declared_world_size != args.world_size:
            raise ValueError("Formal torchrun WORLD_SIZE differs from TrainingArguments")
        expected_media = manifest_media(train_records + eval_records)
        formal_gate_args = dict(
            source_manifest=Path(cli.frozen_source_manifest),
            source_manifest_sha256=source_manifest_sha256,
            formal_ready=Path(cli.formal_ready),
            training_data_dir=Path(cli.formal_training_data_dir),
            train_hash=train_hash, eval_hash=eval_hash,
            media_root=train_root, expected_media=expected_media)
        formal_inputs = rank_zero_integrity_check(
            "formal admission", lambda: _verify_live_formal_inputs(
                **formal_gate_args, full_media_hash=True),
            expected_rank=args.process_index, expected_world_size=args.world_size)
        formal_receipt = formal_inputs["receipt"]
    checkpoint = cli.resume or cli.init_checkpoint
    if cli.resume:
        verify_resume_checkpoint(
            cli.resume, world_size=args.world_size, require_scaler=args.fp16,
            expected_max_steps=args.max_steps if args.max_steps > 0 else None)
    training_sources = {x["source_video_id"] for x in train_records}
    tuning_sources = {x["source_video_id"] for x in eval_records}
    previous_provenance = None
    if checkpoint:
        provenance = Path(checkpoint) / "training_provenance.json"
        if not provenance.is_file():
            raise ValueError("Training checkpoint is missing portable training_provenance.json")
        previous_provenance = json.loads(provenance.read_text())
        declared.verify_previous_contract(previous_provenance.get("training_contract"), raw, resume=bool(cli.resume))
        if cli.init_checkpoint:
            declared.verify_alignment_completion(checkpoint, previous_provenance["training_contract"])
        training_sources.update(previous_provenance["training_sources"])
        tuning_sources.update(previous_provenance.get("tuning_sources", []))
        if training_sources & {x["source_video_id"] for x in eval_records}:
            raise ValueError("Evaluation sources occurred in earlier-stage training")
        saved = DeepProbeConfig.from_dict(json.loads((Path(checkpoint) / "deepprobe_config.json").read_text()))
        # An alignment checkpoint is only a valid initializer for the same
        # architecture and input contract. In particular, every loop-depth
        # baseline must receive its own matching alignment stage; silently
        # attaching an R4 checkpoint to an R1 SFT run would invalidate the
        # experiment even though the recurrent block shares parameters.
        verify_checkpoint_config(saved, config, exact_resume=bool(cli.resume))
    backbone_identity = None
    if mode == "pretrained_backbones_random_connector":
        backbone_identity = rank_zero_integrity_check(
            "declared pretrained initialization", lambda: declared.pretrained_identity(raw),
            expected_rank=args.process_index, expected_world_size=args.world_size)
    # FP32 trainable parameters + BF16 autocast preserve FP32 Adam moments.
    model = DeepProbeVLM.from_backbones(config, checkpoint=checkpoint, torch_dtype=torch.float32)
    model.set_stage(stage)
    if backbone_identity is not None:
        after_loading = rank_zero_integrity_check(
            "declared initialization after model load", lambda: declared.pretrained_identity(raw),
            expected_rank=args.process_index, expected_world_size=args.world_size)
        if after_loading != backbone_identity:
            raise ValueError("Pretrained model bytes changed while being loaded")
    initialization = declared.make_initialization(
        raw, mode, backbones=backbone_identity, projector_sha256=connector_state_sha256(model),
        previous=previous_provenance.get("training_contract") if previous_provenance else None,
        checkpoint_provenance_sha256=_sha256_file(Path(checkpoint) / "training_provenance.json") if checkpoint else None)
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(checkpoint) / "tokenizer") if checkpoint else config.language_model,
        revision=None if checkpoint else config.language_revision, trust_remote_code=False)
    processor = AutoImageProcessor.from_pretrained(
        str(Path(checkpoint) / "image_processor") if checkpoint else config.vision_model,
        revision=None if checkpoint else config.vision_revision, use_fast=False)
    collator = MultimodalCollator(tokenizer, num_visual_tokens=config.num_visual_tokens,
                                 max_text_tokens=config.max_text_tokens)
    if cli.diagnostic_step is not None:
        from .step_diagnostic import DiagnosticCollator

        collator = DiagnosticCollator(collator)
    contract = {"stage": stage, "declared_protocol": declared.validate_declaration(raw),
                "initialization": initialization, "config_sha256": _sha256_file(Path(cli.config)),
                "frozen_source_manifest_sha256": source_manifest_sha256,
                "train_manifest_sha256": train_hash, "eval_manifest_sha256": eval_hash,
                "formal_training": formal_training,
                "require_final_resume_checkpoint": require_final_resume_checkpoint,
                "formal_ready_sha256": formal_receipt["sha256"] if formal_receipt else None,
                "global_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size,
                "world_size": args.world_size,
                "training": {k: str(getattr(args, k)) for k in (
                    "per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate", "weight_decay",
                    "optim", "lr_scheduler_type", "warmup_steps", "warmup_ratio", "seed", "bf16",
                    "gradient_checkpointing", "ddp_find_unused_parameters",
                    "ddp_bucket_cap_mb", "ddp_broadcast_buffers",
                    "max_steps", "num_train_epochs")}}
    if cli.resume and previous_provenance.get("training_contract") != contract:
        raise ValueError("Exact resume training contract changed (stage/data/batch/schedule); use --init-checkpoint for a new run")
    trainer_class = DeepProbeTrainer
    diagnostic_kwargs = {}
    if cli.diagnostic_step is not None:
        from .step_diagnostic import FirstStepTrace

        trainer_class = DiagnosticDeepProbeTrainer
        diagnostic_kwargs["step_trace"] = FirstStepTrace(
            output, target_step=cli.diagnostic_step, rank=args.process_index,
            world_size=args.world_size, resume_checkpoint=cli.resume)
    trainer = trainer_class(
        model=model, args=args, processing_class=tokenizer, image_processor=processor,
        training_sources=training_sources,
        tuning_sources=tuning_sources, training_contract=contract,
        train_dataset=VideoManifestDataset(train_records, processor, config.num_frames, train_root),
        eval_dataset=VideoManifestDataset(eval_records, processor, config.num_frames, eval_root),
        data_collator=collator, **diagnostic_kwargs)
    if trainer.is_world_process_zero():
        output.mkdir(parents=True, exist_ok=True)
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        manifest = {"status": "REAL_TRAINING_CONFIGURATION_NOT_A_RESULT", "model": asdict(config),
                    "training": training_defaults, "stage": stage, "parameter_counts": model.parameter_counts(),
                    "world_size": args.world_size, "python": platform.python_version(),
                    "torch": torch.__version__, "cuda": torch.version.cuda,
                    "train_manifest_sha256": train_hash, "eval_manifest_sha256": eval_hash,
                    "training_contract": contract,
                    "packages": {k: importlib.metadata.version(k) for k in ("transformers", "accelerate", "av", "safetensors", "numpy", "Pillow")},
                    "git_revision": revision.stdout.strip() if revision.returncode == 0 else "unavailable",
                    "init_checkpoint": cli.init_checkpoint, "resume": cli.resume,
                    "p0_skip_resumed_final_export": cli.p0_skip_resumed_final_export,
                    "formal_data_receipt": formal_receipt,
                    "formal_integrity_before": formal_inputs,
                    "training_sources": sorted(training_sources), "tuning_sources": sorted(tuning_sources)}
        if cli.diagnostic_step is not None:
            manifest["diagnostic_step"] = cli.diagnostic_step
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if formal_training:
        # Model loading may take time after admission.  A cheap identity pass
        # immediately before the first optimizer step closes that common race
        # without reading the full media corpus a second time.
        rank_zero_integrity_check(
            "formal pre-train identity", lambda: _require_unchanged_formal_inputs(
                formal_inputs, _verify_live_formal_inputs(
                    **formal_gate_args, full_media_hash=False)),
            expected_rank=args.process_index, expected_world_size=args.world_size)
    result = trainer.train(resume_from_checkpoint=cli.resume)
    if require_final_resume_checkpoint:
        # All ranks must participate so DDP checkpoints contain rank0..N RNG state.
        trainer.save_final_resume_checkpoint()
    if not cli.p0_skip_resumed_final_export:
        trainer.save_model(str(Path(cli.output_dir) / "final"))
    trainer.save_state()
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.save_metrics("eval", trainer.evaluate())
    if formal_training:
        # The runner independently checks again for stage acceptance.  Direct
        # train.py callers get the same fail-closed end gate and only a passed
        # check can create the formal completion marker.
        formal_after = rank_zero_integrity_check(
            "formal post-train bytes", lambda: _require_unchanged_formal_inputs(
                formal_inputs, _verify_live_formal_inputs(
                    **formal_gate_args, full_media_hash=True)),
            expected_rank=args.process_index, expected_world_size=args.world_size)
        rank_zero_integrity_check(
            "formal completion", lambda: _write_formal_completion(
                output, stage=stage, world_size=args.world_size,
                train_hash=train_hash, eval_hash=eval_hash,
                before=formal_inputs, after=formal_after),
            expected_rank=args.process_index, expected_world_size=args.world_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
