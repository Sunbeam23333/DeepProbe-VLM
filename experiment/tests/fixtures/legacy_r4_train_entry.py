"""AST-only excerpts from the audited original R4 SFT training entry.

Original train.py SHA-256: cf7877633d210f7cef0651ee39278223496db1289c7ef146a3f98879abc1808a
These exact class/method and main statements are parsed, never executed.
"""

class DeepProbeTrainer(Trainer):
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


def main(argv=None):
    trainer_class = DeepProbeTrainer
    trainer = trainer_class(
        model=model, args=args, processing_class=tokenizer, image_processor=processor,
        training_sources=training_sources,
        tuning_sources=tuning_sources, training_contract=contract,
        train_dataset=VideoManifestDataset(train_records, processor, config.num_frames, train_root),
        eval_dataset=VideoManifestDataset(eval_records, processor, config.num_frames, eval_root),
        data_collator=collator, **diagnostic_kwargs)
    if require_final_resume_checkpoint:
        # All ranks must participate so DDP checkpoints contain rank0..N RNG state.
        trainer.save_final_resume_checkpoint()
