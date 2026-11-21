# DeepProbe-VLM progress snapshot — 2026-11-21, 08:38:23–08:38:25 Beijing time

Attempt a6 has one Running worker and Pod on stream, at normal priority, with one node and eight GPUs. The exact identity was verified; the container has zero restarts. STARTUP and the engineering `gpu_gate` passed. The active task is restoring accepted historical R4 seed17 alignment. This is not a new alignment completion: no new optimizer update, scientific training completion or model/benchmark evaluation result is verified by this observation.

The bound plan retains 185 operational tasks across 18 models. Remaining model work includes 27 fresh training stages, the final 118 R4 SFT updates, and 32 new model/benchmark evaluations. One completed operational task means the GPU engineering gate passed; it is not a scientific completion percentage. Public main is synchronized at `a114a4c`.

The 08:32:28 admission accepted CPU checks and the server dry-run, with 52 exact empty archive prefixes and no active project Job at that check. Its home-capacity projection was 120,429,007,416 bytes against a 150,000,000,000-byte limit. This is a dated projection, not current actual use; capacity is rechecked per operation. No OSS capacity limit is configured.

At 08:33:49, Video-MME899 F4 cache preparation had 629 cache metadata entries and 2,516 PNGs, comprising 3,146 objects / 996,822,917 bytes. It had neither a complete marker nor READY/PREPARED_ASSET receipts. TempCompass F4, Video-MME899 F16 and TempCompass F16 each had zero observed objects and were also awaiting READY. The 629 entries are not accepted READY clips or completed evaluations. Each evaluation still requires its own verified cache receipt.

Accepted [November 19 scientific results](https://github.com/Sunbeam23333/DeepProbe-VLM/tree/main/experiment/results/releases/2026-11-19) remain the baseline. No new held-out score or full-model latency comparison is claimed here. Earlier releases and private drafts are preserved.

[progress.json](progress.json) records the separate runtime and cache observation times, scope limits, and evidence byte counts/SHA256 digests. The runtime observation is bounded and non-atomic. Raw private evidence, cluster identifiers, filesystem paths, storage prefixes, credentials, media and weights are excluded.

The completed GPU engineering gate includes new single-device synthetic pooling measurements at shape [8,729,1152], G4. The reference → Triton median CUDA-event time is **0.249408 → 0.096992 ms (2.5714×)** for BF16 and **0.242176 → 0.097776 ms (2.4768×)** for FP32. Numerical checks passed, with maximum absolute errors 1.19209e-7 and 1.49012e-7. Each backend used 25 warmups and 100 measured calls with alternating order. Timing includes output allocation and excludes the gate and projector. These are operator measurements, not full-model inference or training speedups; no checkpoint was loaded.

The unchanged original [BF16 SUMMARY](cuda_pool/bf16/SUMMARY.json) and [FP32 SUMMARY](cuda_pool/fp32/SUMMARY.json) provide full summary statistics, numerical tolerances and protocol. Their completion times (08:36:56 and 08:37:08 Beijing time) precede the runtime observation above. The November 19 measurements remain a separate historical result; this package does not overwrite them.
