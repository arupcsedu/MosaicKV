#!/bin/bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_ROOT" >&2
  exit 2
fi

OUTPUT_ROOT=$1
LOCAL_RANK=${SLURM_PROCID:?SLURM_PROCID is required}
if [[ ${LOCAL_RANK} != 0 && ${LOCAL_RANK} != 1 ]]; then
  echo "SLURM_PROCID must be 0 or 1" >&2
  exit 2
fi

"${MOSAICKV_PYTHON}" scripts/verify_choice_c_gpu_binding.py \
  --output-root "${OUTPUT_ROOT}" \
  --round 0 \
  --wave 0 \
  --rank "${LOCAL_RANK}"

exec "${MOSAICKV_PYTHON}" scripts/run_choice_c_latency_blocked_worker.py \
  --rank "${LOCAL_RANK}" \
  --cycles "${MOSAICKV_BLOCKED_CYCLES}" \
  --spec "${MOSAICKV_LATENCY_SPEC}" \
  --profile "${MOSAICKV_LATENCY_PROFILE}" \
  --ledger "${MOSAICKV_LATENCY_LEDGER}" \
  --output-root "${OUTPUT_ROOT}"
