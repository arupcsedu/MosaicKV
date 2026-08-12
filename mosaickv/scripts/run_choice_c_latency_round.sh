#!/bin/bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 ROUND OUTPUT_ROOT" >&2
  exit 2
fi

ROUND=$1
OUTPUT_ROOT=$2
RANK=${SLURM_PROCID:?SLURM_PROCID is required}

case "${ROUND}:${RANK}" in
  0:0|1:1)
    MODE=per_group_reference
    ;;
  0:1|1:0)
    MODE=deferred_layer
    ;;
  *)
    echo "unsupported round/rank assignment: ${ROUND}:${RANK}" >&2
    exit 2
    ;;
esac

exec "${MOSAICKV_PYTHON}" scripts/run_choice_c_latency_worker.py \
  --mode "${MODE}" \
  --replica "round${ROUND}-rank${RANK}" \
  --spec "${MOSAICKV_LATENCY_SPEC}" \
  --profile "${MOSAICKV_LATENCY_PROFILE}" \
  --ledger "${MOSAICKV_LATENCY_LEDGER}" \
  --output-root "${OUTPUT_ROOT}"
