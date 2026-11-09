# Test suites

Run the public CPU suite from `experiment/` after installing the training and test dependencies:

```bash
python -m pytest -q
```

The default suite uses source-controlled code, synthetic fixtures, and the
project-owned training configurations in `fixtures/training_configs/`. Their
original relative paths and byte hashes are recorded in `SOURCE.json`. Tests marked
`private_evidence` are skipped before their setup runs. They validate historical
training configurations, archive receipts, benchmark annotations, visual reviews,
or a pinned upstream scorer under `docs/operations/evidence/`; those artifacts are
not distributed in Git. No benchmark download or remote job is started by these
tests.

To run those integration checks, first obtain the required immutable artifacts
separately under their existing local paths, with authorization and applicable
dataset/source permissions. Then opt in explicitly:

```bash
python -m pytest -q --run-private-evidence -m private_evidence
```

Use `--run-private-evidence` without `-m` to run both suites. The original
integrity checks remain in force: missing, modified, or incomplete artifacts
do not become synthetic substitutes or passing evidence. Some independent
hardware/platform tests also have their own documented skip conditions; a CPU
pass does not validate CUDA/DDP training or benchmark quality.
