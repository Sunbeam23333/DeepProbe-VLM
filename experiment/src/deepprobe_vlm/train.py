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
import subprocess

import torch
from transformers import AutoImageProcessor, AutoTokenizer, Trainer, TrainingArguments, set_seed

from .data import VideoManifestDataset, MultimodalCollator
from .manifest import load_manifest, validate_disjoint
from .modeling import DeepProbeConfig, DeepProbeVLM


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


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--eval-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--media-root")
    p.add_argument("--resume", help="Resume weights, optimizer, scheduler, RNG and progress")
    p.add_argument("--init-checkpoint", help="Initialize weights only, e.g. alignment -> SFT")
    return p


def main(argv=None):
    cli = parser().parse_args(argv)
    if cli.resume and cli.init_checkpoint:
        raise ValueError("Choose --resume OR --init-checkpoint")
    output = Path(cli.output_dir)
    if output.exists() and any(output.iterdir()) and not cli.resume:
        raise FileExistsError("Output directory is nonempty; choose a fresh run directory or explicit --resume")
    raw = json.loads(Path(cli.config).read_text())
    config = DeepProbeConfig.from_dict(raw)
    settings = dict(raw.get("training", {}))
    stage = settings.pop("stage", "sft")
    set_seed(settings.get("seed", 17))
    train_root = Path(cli.media_root) if cli.media_root else Path(cli.train_manifest).resolve().parent
    eval_root = Path(cli.media_root) if cli.media_root else Path(cli.eval_manifest).resolve().parent
    train_records = load_manifest(cli.train_manifest, media_root=train_root)
    eval_records = load_manifest(cli.eval_manifest, media_root=eval_root)
    validate_disjoint(train_records, eval_records)
    if any(x["split"] not in {"train", "training"} for x in train_records):
        raise ValueError("Training manifest must use train/training split; benchmark test is not training data")
    if any(x["split"] not in {"val", "valid", "validation", "dev"} for x in eval_records):
        raise ValueError("Training-time evaluation must use validation/dev, never held-out test")
    if settings.get("bf16", True) and not torch.cuda.is_available():
        raise ValueError("bf16 training requires CUDA in this recipe; disable explicitly for CPU tests")
    if settings.get("bf16", True) and not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA device does not support bf16")
    checkpoint = cli.resume or cli.init_checkpoint
    training_sources = {x["source_video_id"] for x in train_records}
    tuning_sources = {x["source_video_id"] for x in eval_records}
    previous_provenance = None
    if checkpoint:
        provenance = Path(checkpoint) / "training_provenance.json"
        if not provenance.is_file():
            raise ValueError("Training checkpoint is missing portable training_provenance.json")
        previous_provenance = json.loads(provenance.read_text())
        training_sources.update(previous_provenance["training_sources"])
        tuning_sources.update(previous_provenance.get("tuning_sources", []))
        if training_sources & {x["source_video_id"] for x in eval_records}:
            raise ValueError("Evaluation sources occurred in earlier-stage training")
        saved = DeepProbeConfig.from_dict(json.loads((Path(checkpoint) / "deepprobe_config.json").read_text()))
        for name in ("language_model", "language_revision", "language_backend", "vision_model", "vision_revision", "pool_grid"):
            if getattr(saved, name) != getattr(config, name):
                raise ValueError(f"Checkpoint/config mismatch: {name}")
        if cli.resume and saved != config:
            raise ValueError("Exact resume requires an unchanged model configuration")
    # FP32 trainable parameters + BF16 autocast preserve FP32 Adam moments.
    model = DeepProbeVLM.from_backbones(config, checkpoint=checkpoint, torch_dtype=torch.float32)
    model.set_stage(stage)
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(checkpoint) / "tokenizer") if checkpoint else config.language_model,
        revision=None if checkpoint else config.language_revision, trust_remote_code=False)
    processor = AutoImageProcessor.from_pretrained(
        str(Path(checkpoint) / "image_processor") if checkpoint else config.vision_model,
        revision=None if checkpoint else config.vision_revision, use_fast=False)
    collator = MultimodalCollator(tokenizer, num_visual_tokens=config.num_visual_tokens,
                                 max_text_tokens=config.max_text_tokens)
    training_defaults = dict(per_device_train_batch_size=1, per_device_eval_batch_size=1,
                             gradient_accumulation_steps=8, learning_rate=2e-5,
                             num_train_epochs=1, max_steps=-1, bf16=True,
                             gradient_checkpointing=True, seed=17,
                             save_steps=100, logging_steps=10, save_total_limit=2,
                             dataloader_num_workers=0, optim="adamw_torch", report_to="none",
                             eval_strategy="epoch", prediction_loss_only=True,
                             remove_unused_columns=False, ddp_find_unused_parameters=False,
                             gradient_checkpointing_kwargs={"use_reentrant": False})
    training_defaults.update(settings)
    # These are correctness invariants, not user-tunable performance flags.
    training_defaults.update(remove_unused_columns=False, prediction_loss_only=True,
                             ddp_find_unused_parameters=False,
                             gradient_checkpointing_kwargs={"use_reentrant": False})
    args = TrainingArguments(output_dir=cli.output_dir, **training_defaults)
    train_hash = hashlib.sha256(Path(cli.train_manifest).read_bytes()).hexdigest()
    eval_hash = hashlib.sha256(Path(cli.eval_manifest).read_bytes()).hexdigest()
    contract = {"stage": stage, "train_manifest_sha256": train_hash, "eval_manifest_sha256": eval_hash,
                "global_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size,
                "world_size": args.world_size,
                "training": {k: str(getattr(args, k)) for k in (
                    "per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate", "weight_decay",
                    "optim", "lr_scheduler_type", "warmup_steps", "warmup_ratio", "seed", "bf16",
                    "gradient_checkpointing", "max_steps", "num_train_epochs")}}
    if cli.resume and previous_provenance.get("training_contract") != contract:
        raise ValueError("Exact resume training contract changed (stage/data/batch/schedule); use --init-checkpoint for a new run")
    trainer = DeepProbeTrainer(
        model=model, args=args, processing_class=tokenizer, image_processor=processor,
        training_sources=training_sources,
        tuning_sources=tuning_sources, training_contract=contract,
        train_dataset=VideoManifestDataset(train_records, processor, config.num_frames, train_root),
        eval_dataset=VideoManifestDataset(eval_records, processor, config.num_frames, eval_root),
        data_collator=collator)
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
                    "training_sources": sorted(training_sources), "tuning_sources": sorted(tuning_sources)}
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    result = trainer.train(resume_from_checkpoint=cli.resume)
    trainer.save_model(str(Path(cli.output_dir) / "final"))
    trainer.save_state()
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.save_metrics("eval", trainer.evaluate())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
