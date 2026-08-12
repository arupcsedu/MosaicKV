#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_PROCID:?this worker must be launched as an eight-task Slurm step}"
: "${MOSAICKV_QUALITY_INDEX:?MOSAICKV_QUALITY_INDEX is required}"

if (( SLURM_PROCID < 0 || SLURM_PROCID >= 8 )); then
  echo "invalid campaign worker rank: ${SLURM_PROCID}" >&2
  exit 2
fi

exec "${MOSAICKV_PYTHON}" scripts/run_gpu_pilot_worker.py \
  --index "${MOSAICKV_QUALITY_INDEX}" \
  --array-index "${SLURM_PROCID}"
