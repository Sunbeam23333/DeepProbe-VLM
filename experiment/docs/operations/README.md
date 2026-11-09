# Operational tools and evidence

The reusable preparation, source/content audits, checkpoint checks, archive
workers and guarded launch tools live in [scripts/operations](../../scripts/operations).
These tools record immutable run identities and require explicit configuration
and receipts; a successful CPU test does not authorize a GPU submission.

Published progress and accepted measurements are in
[results/STATUS.md](../../results/STATUS.md) and
[results/RESULTS.md](../../results/RESULTS.md). The accompanying JSON files expose
exact denominators, protocols, archive digests and the limits of the claims.

Deployment defaults refer to the original experiment environment. Adapt paths,
account aliases and storage configuration to an authorized environment before
use. Credentials must be supplied externally. Do not invoke launch, acquisition,
archive or cleanup commands just to check installation; use the CPU test suite.
The `benchmark_media_coordinate.py` entry point requires `--execute` before
reading its private plan or attempting remote writes. The historical
`benchmark_media_coordinate_v4.py` remains preserved and is not a general entry
point. Environment bootstrap requires an explicit `DEEPPROBE_GPU_RUNTIME`; it
does not select another project's environment by default.
For the original deployment, GPU work uses the stream queue, low priority, one
node with eight GPUs, restartNever, and at most one project job including Pending.

The private `evidence/` tree and historical operational ledgers remain outside
Git. They include frozen deployment packages, full cluster snapshots, licensed
benchmark records and run-specific receipts. Their preservation is independent
of this public code release. Public evidence indexes identify original bytes by
SHA256 without implying public access to the underlying OSS objects. The working
source release is not a substitute for the immutable package bound to an old run.

Unified evaluation workers now bind the model family, backend, recurrence depth,
training seed, checkpoint and independent acceptance explicitly through
`model_registry.py`. Output names are not parsed into model settings. Plans must
provide matching `model_sequence` and `model_bindings`; old plans without those
bindings fail closed. The prepared path covers directly initialized, matched
two-stage SFTs. Resumed-SFT provenance and no-alignment/frame/pooling ablations
need separately reviewed source-contract variants. This CPU-tested interface
does not imply that R4, Smol or extra-seed evaluations have run.

Tests that use those private artifacts are integration checks. They require the
original authorized inputs; no benchmark media, model weights, upstream scorer
copies or review images are redistributed as test fixtures.
