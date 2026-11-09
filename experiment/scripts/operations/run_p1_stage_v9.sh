#!/usr/bin/env bash
# Declared variants only; invoke on an already allocated stream/low 1x8 node.
# Scheduler submission and independent CPU admission remain external.
set -euo pipefail
if [[ ${1:-} == --help || $# -lt 7 ]]; then
  cat <<'EOF'
Usage: run_p1_stage_v9.sh RUN_ID alignment|sft DATA_ROOT MEDIA_ROOT CONFIG FORMAL_READY SOURCE_MANIFEST [--alignment-final PATH | --resume-checkpoint PATH] [--no-alignment]

The JSON must contain a frozen experiment_protocol. Ordinary SFT still needs
its same-seed alignment final. --no-alignment explicitly permits only the
R4/F8/G4/seed17 direct-SFT diagnostic, including its own exact resume attempts.
This new entry point never changes a frozen v8 release or submits a GPU job.
EOF
  [[ ${1:-} == --help ]] && exit 0
  exit 2
fi
RUN_ID=$1
STAGE=$2
DATA_ROOT=$3
MEDIA_ROOT=$4
CONFIG=$5
FORMAL_READY=$6
SOURCE_MANIFEST=$7
shift 7
EXTRA=()
INITIALIZER_SEEN=false
NO_ALIGNMENT_SEEN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --alignment-final|--resume-checkpoint)
      [[ "$INITIALIZER_SEEN" == false && $# -ge 2 && "$2" != --* ]] || exit 2
      INITIALIZER_SEEN=true
      EXTRA+=("$1" "$2")
      shift 2
      ;;
    --no-alignment)
      [[ "$NO_ALIGNMENT_SEEN" == false ]] || exit 2
      NO_ALIGNMENT_SEEN=true
      EXTRA+=("$1")
      shift
      ;;
    *) echo 'Unsupported declared-runner option.' >&2; exit 2 ;;
  esac
done
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
[[ "$STAGE" == alignment || "$STAGE" == sft ]] || exit 2
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
exec "$PY" "$EXPERIMENT/scripts/operations/run_p1_stage_v9.py" \
  --experiment "$EXPERIMENT" --base "$BASE" --run-id "$RUN_ID" --stage "$STAGE" \
  --data-root "$DATA_ROOT" --media-root "$MEDIA_ROOT" --config "$CONFIG" \
  --formal-ready "$FORMAL_READY" --source-manifest "$SOURCE_MANIFEST" "${EXTRA[@]}"
