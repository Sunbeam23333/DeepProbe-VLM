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
# Static rendezvous avoids the hostname lookup used by --standalone on clusters.
# Override MASTER_PORT when multiple single-node jobs share the host/network.
torchrun --nnodes=1 --node-rank=0 --nproc-per-node="$num_gpus" \
  --master-addr="${MASTER_ADDR:-127.0.0.1}" --master-port="${MASTER_PORT:-29500}" \
  -m deepprobe_vlm.train --config "$config" "$@"
