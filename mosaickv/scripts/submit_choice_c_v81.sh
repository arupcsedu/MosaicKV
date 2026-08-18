#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
V81_ROOT=${MOSAICKV_V81_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v81-disjoint-exact-future-rerank-calibration-v1}
V81_SPEC=${PACKAGE_ROOT}/configs/experiments/choice_c_v81_disjoint_exact_future_rerank_calibration.yaml
FROZEN=${V81_ROOT}/analysis/frozen_candidate.yaml

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"

if [[ -e "${V81_ROOT}" ]]; then
  printf 'refusing to reuse existing v81 output root: %s\n' "${V81_ROOT}" >&2
  exit 2
fi

LEDGER_JOB=$(sbatch --parsable \
  --export="ALL,MOSAICKV_V81_OUTPUT_ROOT=${V81_ROOT}" \
  slurm/choice_c_v81_materialize_exact_future_calibration.sbatch)
CALIBRATION_JOB=$(sbatch --parsable \
  --dependency="afterok:${LEDGER_JOB}" \
  --export="ALL,MOSAICKV_V81_OUTPUT_ROOT=${V81_ROOT},MOSAICKV_V81_SPEC=${V81_SPEC}" \
  slurm/choice_c_v81_exact_future_calibration_a100.sbatch)
FREEZE_JOB=$(sbatch --parsable \
  --dependency="afterok:${CALIBRATION_JOB}" \
  --export="ALL,MOSAICKV_V81_OUTPUT_ROOT=${V81_ROOT},MOSAICKV_V81_SPEC=${V81_SPEC},MOSAICKV_V81_FROZEN=${FROZEN}" \
  slurm/choice_c_v81_freeze.sbatch)

printf 'v81 ledger: %s\n' "${LEDGER_JOB}"
printf 'v81 exact/future calibration: %s\n' "${CALIBRATION_JOB}"
printf 'v81 freeze: %s\n' "${FREEZE_JOB}"
printf 'No held-out campaign was submitted. It remains blocked unless both unchanged gates pass and v81 creates a frozen candidate.\n'
