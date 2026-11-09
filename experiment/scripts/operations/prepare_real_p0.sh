#!/usr/bin/env bash
# Engineering-only pilot. Malformed upstream benchmark IDs remain unresolved.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BASE=/home/jinhaopeng/.cache/Paper/DeepProbe-VLM
PY=$BASE/env/bin/python
OUTPUT=${1:-$BASE/p0-data-v1}
CONFIG=${2:-$ROOT/configs/train/ouro_pilot.json}
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HOME=$BASE/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
revision() { "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["datasets"][sys.argv[2]]["revision"])' "$ROOT/configs/datasets.json" "$1"; }
LLAVA=$BASE/assets/llava_video_178k/$(revision llava_video_178k)
VMME=$BASE/assets/videomme/$(revision videomme)
MVB=$BASE/assets/mvbench/$(revision mvbench)
timeout --signal=TERM --kill-after=30 1800 "$PY" "$ROOT/scripts/operations/prepare_p0_data.py" \
  --llava-root "$LLAVA" --llava-plan "$LLAVA/ASSET_COMMITTED.json" \
  --videomme-root "$VMME" --videomme-plan "$VMME/ASSET_COMMITTED.json" \
  --mvbench-root "$MVB" --mvbench-plan "$MVB/ASSET_COMMITTED.json" \
  --media-root "$BASE/media/llava-p0" --output-dir "$OUTPUT" --config "$CONFIG" \
  --allow-unresolved-benchmark-sources-for-p0 --exclude-unresolved-training-sources
