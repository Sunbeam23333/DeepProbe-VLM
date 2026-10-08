#!/usr/bin/env bash
# Run from experiment/. Remaining arguments go to the training CLI.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/launch/train.sh CONFIG --train-manifest ... --media-root ... --output-dir ..." >&2
  exit 2
fi
config="$1"
shift
num_gpus="${DEEP_PROBE_GPUS:-8}"
torchrun --standalone --nnodes=1 --nproc-per-node="$num_gpus" \
  -m deepprobe_vlm.train --config "$config" "$@"
