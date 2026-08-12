#!/bin/bash

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 ROUND WAVE OUTPUT_ROOT" >&2
  exit 2
fi

ROUND=$1
WAVE=$2
OUTPUT_ROOT=$3
LOCAL_RANK=${SLURM_PROCID:?SLURM_PROCID is required}
if [[ ${ROUND} != 0 && ${ROUND} != 1 ]]; then
  echo "ROUND must be 0 or 1" >&2
  exit 2
fi
if [[ ${WAVE} != 0 && ${WAVE} != 1 ]]; then
  echo "WAVE must be 0 or 1" >&2
  exit 2
fi
if [[ ${LOCAL_RANK} != 0 && ${LOCAL_RANK} != 1 ]]; then
  echo "SLURM_PROCID must be 0 or 1" >&2
  exit 2
fi

"${MOSAICKV_PYTHON}" scripts/verify_choice_c_gpu_binding.py \
  --output-root "${OUTPUT_ROOT}" \
  --round "${ROUND}" \
  --wave "${WAVE}" \
  --rank "${LOCAL_RANK}"

# Each candidate is paired with an independent reference on the other GPU.
# Round 1 swaps every candidate/reference placement across the two GPUs.
case "${ROUND}:${WAVE}:${LOCAL_RANK}" in
  0:0:0) FACTOR_INDEX=0; MEASUREMENT_RANK=0 ;;
  0:0:1) FACTOR_INDEX=6; MEASUREMENT_RANK=1 ;;
  0:1:0) FACTOR_INDEX=7; MEASUREMENT_RANK=2 ;;
  0:1:1) FACTOR_INDEX=0; MEASUREMENT_RANK=3 ;;
  1:0:0) FACTOR_INDEX=6; MEASUREMENT_RANK=0 ;;
  1:0:1) FACTOR_INDEX=0; MEASUREMENT_RANK=1 ;;
  1:1:0) FACTOR_INDEX=0; MEASUREMENT_RANK=2 ;;
  1:1:1) FACTOR_INDEX=7; MEASUREMENT_RANK=3 ;;
  *)
    echo "unsupported reduced-factorial assignment" >&2
    exit 2
    ;;
esac

exec "${MOSAICKV_PYTHON}" scripts/run_choice_c_latency_factorial_worker.py \
  --factor-index "${FACTOR_INDEX}" \
  --round "${ROUND}" \
  --rank "${MEASUREMENT_RANK}" \
  --spec "${MOSAICKV_LATENCY_SPEC}" \
  --profile "${MOSAICKV_LATENCY_PROFILE}" \
  --ledger "${MOSAICKV_LATENCY_LEDGER}" \
  --output-root "${OUTPUT_ROOT}"
