#!/usr/bin/env bash
set -euo pipefail
# Same batch/frames/loops as H20: do not confuse hardware and workload changes.
python scripts/preflight.py --expected-gpu B300 --output runs/preflight_b300.json
bash scripts/launch/train.sh configs/train/ouro_sft.json "$@"
