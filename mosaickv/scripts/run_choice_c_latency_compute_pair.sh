#!/bin/bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 ROUND OUTPUT_ROOT" >&2
  exit 2
fi

ROUND=$1
OUTPUT_ROOT=$2
LOCAL_RANK=${SLURM_PROCID:?SLURM_PROCID is required}
if [[ ${ROUND} != 0 && ${ROUND} != 1 ]]; then
  echo "ROUND must be 0 or 1" >&2
  exit 2
fi
if [[ ${LOCAL_RANK} != 0 && ${LOCAL_RANK} != 1 ]]; then
  echo "SLURM_PROCID must be 0 or 1" >&2
  exit 2
fi

"${MOSAICKV_PYTHON}" scripts/verify_choice_c_gpu_binding.py \
  --output-root "${OUTPUT_ROOT}" \
  --round "${ROUND}" \
  --wave 0 \
  --rank "${LOCAL_RANK}"

if [[ ${ROUND} == 0 ]]; then
  if [[ ${LOCAL_RANK} == 0 ]]; then
    COMPUTE_MODE=reference
  else
    COMPUTE_MODE=optimized
  fi
elif [[ ${LOCAL_RANK} == 0 ]]; then
  COMPUTE_MODE=optimized
else
  COMPUTE_MODE=reference
fi

exec "${MOSAICKV_PYTHON}" scripts/run_choice_c_latency_compute_worker.py \
  --compute-mode "${COMPUTE_MODE}" \
  --round "${ROUND}" \
  --rank "${LOCAL_RANK}" \
  --spec "${MOSAICKV_LATENCY_SPEC}" \
  --profile "${MOSAICKV_LATENCY_PROFILE}" \
  --ledger "${MOSAICKV_LATENCY_LEDGER}" \
  --output-root "${OUTPUT_ROOT}"
