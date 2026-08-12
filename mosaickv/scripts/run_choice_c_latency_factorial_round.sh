#!/bin/bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 ROUND OUTPUT_ROOT" >&2
  exit 2
fi

ROUND=$1
OUTPUT_ROOT=$2
RANK=${SLURM_PROCID:?SLURM_PROCID is required}
if [[ ${ROUND} != 0 && ${ROUND} != 1 ]]; then
  echo "ROUND must be 0 or 1" >&2
  exit 2
fi
if (( RANK < 0 || RANK > 7 )); then
  echo "SLURM_PROCID must be in [0, 7]" >&2
  exit 2
fi

# Rotate the complete 2^3 factorial by three ranks in round 1. Each factor
# cell therefore runs on a different physical GPU and CPU locality domain.
FACTOR_INDEX=$(( (RANK + 3 * ROUND) % 8 ))

exec "${MOSAICKV_PYTHON}" scripts/run_choice_c_latency_factorial_worker.py \
  --factor-index "${FACTOR_INDEX}" \
  --round "${ROUND}" \
  --rank "${RANK}" \
  --spec "${MOSAICKV_LATENCY_SPEC}" \
  --profile "${MOSAICKV_LATENCY_PROFILE}" \
  --ledger "${MOSAICKV_LATENCY_LEDGER}" \
  --output-root "${OUTPUT_ROOT}"
