#!/usr/bin/env bash
# Execute P0-05 inside one already allocated full GPU node. This never submits a job.
# The Python runner performs verified incremental/final OSS archival itself.
set -euo pipefail

if [[ ${1:-} == --help || ( $# -ne 4 && $# -ne 6 ) ]]; then
  echo 'Usage: run_p0_05.sh RUN_ID DATA_ROOT MEDIA_ROOT CONFIG_PATH [--diagnostic-step STEP]'
  [[ ${1:-} == --help ]] && exit 0
  exit 2
fi
[[ "$1" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid RUN_ID' >&2; exit 2; }
DIAGNOSTIC_ARGS=()
if [[ $# -eq 6 ]]; then
  [[ "$5" == --diagnostic-step && "$6" =~ ^[1-9][0-9]*$ ]] || {
    echo 'Expected --diagnostic-step followed by a positive integer' >&2; exit 2;
  }
  DIAGNOSTIC_ARGS=(--diagnostic-step "$6")
fi

EXPERIMENT=$(cd "$(dirname "$0")/../.." && pwd)
BASE=/home/jinhaopeng/.cache/Paper/DeepProbe-VLM
PY=$BASE/env/bin/python
RUN_ID=$1
DATA_ROOT=$(realpath "$2")
MEDIA_ROOT=$(realpath "$3")
CONFIG_PATH=$(realpath "$4")

[[ -x "$PY" ]] || { echo "Missing runtime: $PY" >&2; exit 2; }
for path in "$EXPERIMENT" "$BASE" "$DATA_ROOT" "$MEDIA_ROOT" "$CONFIG_PATH"; do
  [[ "$path" == /home/jinhaopeng/* ]] || { echo "Input must remain below /home/jinhaopeng: $path" >&2; exit 2; }
done
[[ -f "$DATA_ROOT/train.jsonl" && -f "$DATA_ROOT/validation.jsonl" ]] || {
  echo 'DATA_ROOT must contain train.jsonl and validation.jsonl' >&2; exit 2;
}

mkdir -p "$BASE/locks" "$BASE/runs"
exec 9>"$BASE/locks/p0-05.lock"
flock -n 9 || { echo 'Another P0-05 run holds the project lock.' >&2; exit 73; }

export PATH=/home/jinhaopeng/.local/bin:$PATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HOME="$BASE/hf" HF_HUB_CACHE="$BASE/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

exec "$PY" "$EXPERIMENT/scripts/operations/run_p0_05.py" \
  --run-id "$RUN_ID" \
  --config "$CONFIG_PATH" \
  --train-manifest "$DATA_ROOT/train.jsonl" \
  --eval-manifest "$DATA_ROOT/validation.jsonl" \
  --media-root "$MEDIA_ROOT" \
  --output-dir "$BASE/runs/$RUN_ID" \
  --python "$PY" \
  --world-size 8 \
  --overfit-samples 4 \
  --overfit-steps 60 \
  --overfit-learning-rate 0.00001 \
  --overfit-warmup-steps 5 \
  --overfit-max-loss-ratio 0.5 \
  --overfit-min-token-accuracy 0.95 \
  --project-storage-root "$BASE" \
  --project-storage-limit-bytes 100000000000 \
  --reserved-additional-bytes 75000000000 \
  "${DIAGNOSTIC_ARGS[@]}"
