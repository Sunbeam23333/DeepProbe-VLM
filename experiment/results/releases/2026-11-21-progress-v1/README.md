# DeepProbe-VLM progress snapshot — 2026-11-21

SmolLM2 alignment completed all 718 optimizer updates. Its original Job then failed during archival. A separately reviewed archive-recovery client is running; completed upload, independent audit and archive acceptance are not yet claimed by this snapshot.

The full plan retains 18 models, 34 training stages, 36 model/benchmark evaluations, 15 full-model latency runs, 12 latency comparisons and 185 operational tasks. No full-suite GPU worker has been submitted at this snapshot. The prepared worker uses normal priority and eight GPUs; the measured 150 GB home budget currently supports one worker.

The scheduler now completes one connected checkpoint chain before opening an independent chain. It waits for intermediate archive acceptance and final consumer cleanup, preventing accumulated alignment models from blocking SFT. 190 scoped CPU tests passed, including synthetic full-DAG execution with 1–4 competing claimants. Those synthetic completions are not experiment results.

There are no new held-out scores. The [previous complete result release](https://github.com/Sunbeam23333/DeepProbe-VLM/tree/main/experiment/results/releases/2026-11-19) contains the accepted R1/R2 predictions, statistics and synthetic CUDA-operator measurements. See [current GitHub status](https://github.com/Sunbeam23333/DeepProbe-VLM/blob/main/experiment/results/STATUS.md) for later observations.

[progress.json](progress.json) records the snapshot's evidence hashes and explicit limits. Raw cluster snapshots, credentials, video media and model weights are not included. This upload does not change repository visibility.
