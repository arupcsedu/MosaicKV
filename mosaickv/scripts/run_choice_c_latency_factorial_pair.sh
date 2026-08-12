#!/bin/bash

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 ROUND WAVE OUTPUT_ROOT" >&2
  exit 2
fi

ROUND=$1
WAVE=$2
OUTPUT_ROOT=$3
RANK=${SLURM_PROCID:?SLURM_PROCID is required}
if [[ ${ROUND} != 0 && ${ROUND} != 1 ]]; then
  echo "ROUND must be 0 or 1" >&2
  exit 2
fi
if (( WAVE < 0 || WAVE > 3 )); then
  echo "WAVE must be in [0, 3]" >&2
  exit 2
fi
if (( RANK < 0 || RANK > 1 )); then
  echo "SLURM_PROCID must be 0 or 1" >&2
  exit 2
fi

# Four waves cover all eight factor cells. The odd round offset moves every
# factor cell to the other task/GPU/locality domain in round 1.
FACTOR_INDEX=$(( (2 * WAVE + RANK + 3 * ROUND) % 8 ))

exec "${MOSAICKV_PYTHON}" scripts/run_choice_c_latency_factorial_worker.py \
  --factor-index "${FACTOR_INDEX}" \
  --round "${ROUND}" \
  --rank "${RANK}" \
  --spec "${MOSAICKV_LATENCY_SPEC}" \
  --profile "${MOSAICKV_LATENCY_PROFILE}" \
  --ledger "${MOSAICKV_LATENCY_LEDGER}" \
  --output-root "${OUTPUT_ROOT}"
