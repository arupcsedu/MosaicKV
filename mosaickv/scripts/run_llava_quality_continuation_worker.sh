#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_PROCID:?this worker must be launched as a six-task Slurm step}"
: "${MOSAICKV_CONTINUATION_INDEX:?MOSAICKV_CONTINUATION_INDEX is required}"

if (( SLURM_PROCID < 0 || SLURM_PROCID >= 6 )); then
  echo "invalid continuation worker rank: ${SLURM_PROCID}" >&2
  exit 2
fi

exec "${MOSAICKV_PYTHON}" scripts/run_llava_quality_continuation_worker.py \
  --index "${MOSAICKV_CONTINUATION_INDEX}" \
  --worker-index "${SLURM_PROCID}"
