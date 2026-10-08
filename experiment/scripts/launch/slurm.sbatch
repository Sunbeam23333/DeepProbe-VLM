#!/usr/bin/env bash
#SBATCH --job-name=deepprobe-pilot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
# Edit the partition/account for your cluster; no resource purchase is automatic.
# Submit from experiment/ after activating its environment.
set -euo pipefail
: "${DEEP_PROBE_TRAIN:?Set DEEP_PROBE_TRAIN to a training manifest}"
: "${DEEP_PROBE_VAL:?Set DEEP_PROBE_VAL to a source-disjoint validation manifest}"
: "${DEEP_PROBE_MEDIA:?Set DEEP_PROBE_MEDIA to the media root}"
: "${DEEP_PROBE_OUTPUT:?Set DEEP_PROBE_OUTPUT to a new run directory}"
bash scripts/launch/train.sh configs/train/ouro_pilot.json \
  --train-manifest "$DEEP_PROBE_TRAIN" --eval-manifest "$DEEP_PROBE_VAL" \
  --media-root "$DEEP_PROBE_MEDIA" --output-dir "$DEEP_PROBE_OUTPUT"
