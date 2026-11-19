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

Benchmark media is not redistributed here. Access and reuse of benchmark assets remain subject to their upstream licenses and terms.
