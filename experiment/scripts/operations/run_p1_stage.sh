#!/usr/bin/env bash
# Run one finite formal P1 alignment or SFT stage on an already allocated node.
# This file never submits a job.  Submit its invocation through ali-gpu/stream
# as one full eight-GPU node with `sslaunch -p low`.
set -euo pipefail

if [[ ${1:-} == --help || ( $# != 7 && $# != 9 ) ]]; then
  cat <<'EOF'
Usage: run_p1_stage.sh RUN_ID alignment|sft DATA_ROOT MEDIA_ROOT CONFIG_PATH FORMAL_READY_JSON SOURCE_MANIFEST_JSON [--alignment-final PATH | --resume-checkpoint PATH]

Fresh alignment uses seven arguments.  Fresh SFT requires --alignment-final.
Either stage may instead use --resume-checkpoint with a complete checkpoint
from a prior attempt and a new RUN_ID.  SOURCE_MANIFEST_JSON is the exact
SHA-256 inventory of the frozen experiment tree and must live outside it.
EOF
  [[ ${1:-} == --help ]] && exit 0
  exit 2
fi

[[ "$1" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'RUN_ID contains invalid characters.' >&2; exit 2; }
[[ "$2" == alignment || "$2" == sft ]] || { echo 'Stage must be alignment or sft.' >&2; exit 2; }
if [[ $# == 9 ]]; then
  [[ "$8" == --alignment-final || "$8" == --resume-checkpoint ]] || {
    echo 'Optional mode must be --alignment-final or --resume-checkpoint.' >&2; exit 2;
  }
fi
[[ "$2" != alignment || ${8:-} != --alignment-final ]] || {
  echo 'alignment does not accept --alignment-final.' >&2; exit 2;
}
[[ "$2" != sft || $# == 9 ]] || {
  echo 'Fresh sft requires --alignment-final; resumed sft requires --resume-checkpoint.' >&2; exit 2;
}

EXPERIMENT=$(cd "$(dirname "$0")/../.." && pwd)
BASE=/home/jinhaopeng/.cache/Paper/DeepProbe-VLM
PY=$BASE/env/bin/python
[[ -x "$PY" ]] || { echo "Missing project runtime: $PY" >&2; exit 2; }

mkdir -p "$BASE/locks"
exec 9>"$BASE/locks/p1-stage.lock"
flock -n 9 || { echo 'Another formal P1 stage holds the project lock.' >&2; exit 73; }

export PATH=/home/jinhaopeng/.local/bin:$PATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HOME="$BASE/hf" HF_HUB_CACHE="$BASE/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export NCCL_ASYNC_ERROR_HANDLING=1

EXTRA=()
if [[ $# == 9 ]]; then
  EXTRA=("$8" "$9")
fi
exec "$PY" "$EXPERIMENT/scripts/operations/run_p1_stage.py" \
  --experiment "$EXPERIMENT" --base "$BASE" \
  --run-id "$1" --stage "$2" --data-root "$3" --media-root "$4" \
  --config "$5" --formal-ready "$6" --source-manifest "$7" \
  "${EXTRA[@]}"
