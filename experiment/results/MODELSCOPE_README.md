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

The later [current GitHub status](https://github.com/Sunbeam23333/DeepProbe-VLM/blob/main/experiment/results/STATUS.md) and [November 21 progress-v4](https://modelscope.ai/datasets/JasonJin/DeepProbe-VLM/files?branch=master&path=releases%2F2026-11-21-progress-v4) record accepted a6 alignment/archive recovery/native history and the separately repaired full185 worker. Latest runtime code is `65e562c`. V6 passed 68 CPU checks (four explicit F4/F16 cache deferrals, zero failures), exact server dry-run and independent admission. Attempt 2 was actually submitted and passed eight-device hardware/startup checks, but was **observed Failed at November 21 04:36:26 Beijing time**, exit1 with zero restarts. Its first engineering `gpu_gate` failed during Triton cache initialization because `/.triton` was unwritable, before the intended numerical kernel launched. This does not establish a CUDA numerical discrepancy. No training/evaluation task or new optimizer update occurred; explicit task-owned cache paths and fresh replacement admission are being prepared.

The first worker's controller-timeout failure remains unchanged and its 18-object small archive passed full GET/SHA. Both F8 caches remain READY; controlled F4/F16 CPU recovery is active, with four new READY receipts pending. The reviewed worst checkpoint-chain peak is 142.546 GB under the 150 GB home budget. The admitted allocation is one normal-priority stream node with eight GPUs. The remaining scientific work is 27 new training stages, R4's 118 SFT updates, 32 new evaluations, 15 full-model latency runs and 12 comparisons. Local checkpoints are retired only after verified archival and all consumers complete. These are progress facts, not new quality or full-model performance results; the sealed earlier releases and accepted November 19 scores remain unchanged.

Benchmark media is not redistributed here. Access and reuse of benchmark assets remain subject to their upstream licenses and terms.
