# DeepProbe-VLM progress — 2026-11-21, attempt 2

The repaired full185 attempt-2 worker is **observed Failed** at **04:36:26 Beijing time**,
exit1 with zero restarts and an exact identity match to the successful submission.
Its eight-device hardware and startup checks passed, including BF16 attention
forward/backward and all-reduce. The first engineering `gpu_gate` failed during
Triton cache initialization because `/.triton` was unwritable. The intended numerical
kernel had not launched; this does not establish a CUDA numerical discrepancy.
No training/evaluation task, scientific completion or new optimizer update occurred.
An explicit task-owned cache-directory repair and fresh replacement admission are
being prepared; this snapshot claims no replacement execution.

V6 CPU admission passed 68 checks, with four F4/F16 cache shapes explicitly deferred
and no failures. Exact server dry-run and independent admission passed. The worker
uses one normal-priority stream node and eight GPUs. The reviewed checkpoint-chain
peak is 142,546,098,993 bytes under a 150,000,000,000-byte home budget, rechecked per task.

The first worker remains Failed after its controller query timeout, with no
scientific task or checkpoint produced. Its 18-object / 103,783-byte failure archive
passed full GET/SHA verification. The separate repair uses bounded server metadata
pagination and exact project reads; 137 scoped CPU tests passed. Runtime source
commit: `65e562c67d0adfd1573c99dca722082b231c03a2`.

Smol a6 alignment completed 718/718 updates; archive recovery and native history
import are accepted. Its original scheduler Job remains Failed after archival failure.
R4 checkpoint600 still needs actual runtime restoration and 118 SFT updates.
The remaining plan has 27 fresh training stages, 32 new evaluations, 15 full-model
latency runs and 12 comparisons. Both F8 caches are READY; F4/F16 CPU recovery is
active and four new READY receipts remain pending. Local checkpoint deletion requires
verified archival and completion of all consumers.

There are **no new held-out scores or full-model speedup claims**. The accepted
[November 19 result package](https://github.com/Sunbeam23333/DeepProbe-VLM/tree/main/experiment/results/releases/2026-11-19)
still contains the R1/R2 seed17 results, 8,554 prediction rows and 400 synthetic CUDA
operator timing rows. Its limitations remain unchanged: Video-MME is a declared
899-video subset, invalid answers stay in the denominator, bootstrap uncertainty
does not measure training-seed robustness, and pooling speed ratios are operator-only.

[progress.json](progress.json) contains the dated sanitized facts and evidence hashes.
The manifest pins this immutable snapshot. Original media, weights, credentials,
private paths, internal Job UIDs and raw cluster responses are excluded.
