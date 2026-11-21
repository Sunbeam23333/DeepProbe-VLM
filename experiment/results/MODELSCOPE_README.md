---
license: Apache License 2.0
---

# DeepProbe-VLM experiment results

Verified results and reproducibility records for DeepProbe-VLM. The repository visibility is unchanged.

## Latest release

[2026-11-19 result package](https://modelscope.ai/datasets/JasonJin/DeepProbe-VLM/files?branch=master&path=releases%2F2026-11-19) · [GitHub source and experiment scripts](https://github.com/Sunbeam23333/DeepProbe-VLM)

The release contains four scored prediction sets (8,554 rows), paired source-cluster statistics, 400 raw CUDA operator timing records, numerical checks, exact SHA256 manifests, and a separate progress snapshot.

- **Video-MME 899-video subset**: R1 6.3404%, R2 18.4650%; paired difference +12.1246 percentage points, 95% CI [10.6044, 13.5716]. Invalid answers remain in the denominator. This is not the official full 900-video result.
- **TempCompass**: R1 42.5316%, R2 50.4430%; paired difference +7.9114 percentage points, 95% CI [5.1876, 10.6785]. The accepted official `disable_llm` multiple-choice rules are used.
- **Synthetic weighted pooling**: reference/Triton median CUDA-event ratios are 2.4096× for BF16 and 2.2835× for FP32 at shape [8,729,1152], G4. These measurements do not establish full-model or training speedup.

The accepted quality comparison uses training seed17 only. Bootstrap intervals represent test-source sampling uncertainty, not training-seed robustness. Overall intervals were independently recomputed; subgroup intervals were not part of that independent recomputation. The high Video-MME invalid-response rates limit the absolute results.

The `releases/2026-11-19/README.md` file explains the artifact layout and limitations. `PUBLICATION_MANIFEST.json` pins all published payloads, `SOURCE_MAP.json` maps derivatives to original evidence digests, and `VALIDATION.json` records count and metric checks. Private model filesystem names in Video-MME predictions were replaced with public experiment identifiers. No scores were changed.

R4 SFT, SmolLM2, the remaining seeds and ablations, and formal full-model latency measurements remain incomplete in this snapshot. The current small result release is not a backup of model weights. Future fully verified model backups, if present, use a separate `artifacts/` prefix.

The later [current GitHub status](https://github.com/Sunbeam23333/DeepProbe-VLM/blob/main/experiment/results/STATUS.md) and [November 21 progress-v6](https://github.com/Sunbeam23333/DeepProbe-VLM/tree/main/experiment/results/releases/2026-11-21-progress-v6) describe a separately admitted isolated a6 suite worker. At **November 21 08:38:25 Beijing time**, its Job and Pod were Running on one normal-priority stream node with eight B300 GPUs and zero restarts. STARTUP and the engineering `gpu_gate` passed. One of 185 operational tasks was complete; the active task was restoring accepted historical R4 alignment. No new optimizer update, scientific training completion or model/benchmark result was observed. Public source main is synchronized at `a114a4c`.

The new GPU gate measured synthetic weighted pooling at shape [8,729,1152], G4. Reference → Triton median CUDA-event time was **0.249408 → 0.096992 ms (2.5714×)** for BF16 and **0.242176 → 0.097776 ms (2.4768×)** for FP32; both numerical checks passed. Each backend used 25 warmups and 100 measured calls in alternating order. Timing includes output allocation, excludes the gate and projector, and uses no model checkpoint. These new operator measurements do not establish full-model or training speedup. The original November 19 measurements above are unchanged.

Accepted Smol alignment a6 reached 718 steps; its independent archive recovery and suite-history import are accepted, while the original archive-phase Job failure remains unchanged. R4 cp600 archive/metadata checks are accepted; runtime restoration and 118 remaining SFT updates are pending. Scientific totals remain 6 accepted training stages, 2 complete two-stage models, 4 evaluations and 0 full-model latency runs. Remaining work is 27 fresh training stages, R4's 118 updates, 32 new evaluations, 15 latency runs and 12 comparisons.

At 08:33:49, Video-MME899 F4 preparation had 629 cache metadata entries and 2,516 PNGs, totaling 3,146 objects / 996,822,917 bytes; READY was absent. The other three new F4/F16 shapes had zero observed objects. Both original F8 caches remain READY. The 08:32:28 admission projection was 120.429 GB under the 150 GB home budget, with private scratch and a retained parent reservation; this is not current disk use. OSS has no configured capacity limit. Each evaluation requires its own cache READY, and local checkpoints are retired only after verified archival and every dependent consumer completes. Earlier sealed releases and accepted November 19 scores remain unchanged.

Benchmark media is not redistributed here. Access and reuse of benchmark assets remain subject to their upstream licenses and terms.
