#!/usr/bin/env bash
set -euo pipefail
python scripts/preflight.py --expected-gpu H20 --output runs/preflight_h20.json
bash scripts/launch/train.sh configs/train/ouro_sft.json "$@"
