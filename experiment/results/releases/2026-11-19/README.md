# DeepProbe-VLM verified results — 2026-11-19

This release contains the accepted R1/R2 seed17 benchmark results, four scored per-example prediction files (8,554 rows), source-cluster paired statistics, and 400 raw CUDA operator timing records with summaries and numerical checks. `PUBLICATION_MANIFEST.json` pins every byte.

- Video-MME **899-video subset**: R1 6.3404% (171/2,697), R2 18.4650% (498/2,697). Paired difference +12.1246 percentage points, 95% CI [10.6044, 13.5716]. Invalid answers remain wrong: R1 82.1283%, R2 52.2432%.
- TempCompass: R1 42.5316% (672/1,580), R2 50.4430% (797/1,580). Paired difference +7.9114 percentage points, 95% CI [5.1876, 10.6785]. Official disable_llm multiple-choice rules are retained.
- Synthetic weighted pooling: reference/Triton CUDA-event median ratio **2.4096× BF16** and **2.2835× FP32**, one device, shape [8,729,1152], G4. These are operator measurements, not end-to-end inference or training speedups. The runtime reports NVIDIA L20D; see the retained runtime records.

Only training seed17 has accepted quality results. The bootstrap resamples test-source clusters, not training seeds. Overall intervals were independently recomputed at acceptance; subgroup intervals were not part of that independent recomputation. Cold/mixed per-example timings in Video-MME predictions retain their original labels and must not be presented as warmed model latency.

`RESULTS.md` and `accepted_results.json` preserve the original accepted summary. `SOURCE_MAP.json` identifies exact originals and published derivatives. Video-MME prediction model_name paths were replaced by public experiment identifiers; correctness, generated answers, source/sample identities, and protocol/timing fields are unchanged. TempCompass predictions and paired-statistics files are byte-identical originals. `VALIDATION.json` records fresh count, correctness and operator timing recomputation for this publication.

`status.json` is a separate progress snapshot: R4 SFT checkpoint600 needs five missing payloads restored; SmolLM2 seed29 alignment a5 failed after step354 and has a CPU-verified checkpoint350 candidate, without accepted a5 failure archive or new optimizer/RNG restoration. No new completed model or quality result is inferred. The 18-model core matrix still has multi-seed, ablation and full-model latency work pending.

The new runner supports preparing 1–4 B300 workers at normal priority. Project /home and project OSS budgets are separately below 150,000,000,000 bytes, with shared reservations across workers. Model checkpoint retirement requires verified durable backup and completed consumers. This release itself contains results/documents, not model-weight backups or benchmark media.

Sources: [GitHub](https://github.com/Sunbeam23333/DeepProbe-VLM) · [ModelScope dataset](https://modelscope.ai/datasets/JasonJin/DeepProbe-VLM). Benchmark assets remain governed by their source licenses and access conditions; no source media is redistributed here.
