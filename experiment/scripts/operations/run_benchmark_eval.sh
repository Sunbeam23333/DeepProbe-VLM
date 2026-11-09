#!/usr/bin/env bash
# Invoked only inside one allocated ali-gpu/stream eight-GPU low job.
set -euo pipefail
EXPERIMENT=$(cd "$(dirname "$0")/../.." && pwd)
BASE=/home/jinhaopeng/.cache/Paper/DeepProbe-VLM
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HOME="$BASE/hf" HF_HUB_CACHE="$BASE/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
exec "$BASE/env/bin/python" "$EXPERIMENT/scripts/operations/run_benchmark_eval.py" "$@"
