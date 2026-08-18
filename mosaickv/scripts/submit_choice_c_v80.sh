#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
V80_ROOT=${MOSAICKV_V80_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v80-disjoint-risk-gated-swap-calibration-v1}
V80_SPEC=${PACKAGE_ROOT}/configs/experiments/choice_c_v80_disjoint_risk_gated_swap_calibration.yaml
FROZEN=${V80_ROOT}/analysis/frozen_candidate.yaml

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"

if [[ -e "${V80_ROOT}" ]]; then
  printf 'refusing to reuse existing v80 output root: %s\n' "${V80_ROOT}" >&2
  exit 2
fi

LEDGER_JOB=$(sbatch --parsable \
  --export="ALL,MOSAICKV_V80_OUTPUT_ROOT=${V80_ROOT}" \
  slurm/choice_c_v80_materialize_risk_calibration.sbatch)
CALIBRATION_JOB=$(sbatch --parsable \
  --dependency="afterok:${LEDGER_JOB}" \
  --export="ALL,MOSAICKV_V80_OUTPUT_ROOT=${V80_ROOT},MOSAICKV_V80_SPEC=${V80_SPEC}" \
  slurm/choice_c_v80_risk_calibration_a100.sbatch)
FREEZE_JOB=$(sbatch --parsable \
  --dependency="afterok:${CALIBRATION_JOB}" \
  --export="ALL,MOSAICKV_V80_OUTPUT_ROOT=${V80_ROOT},MOSAICKV_V80_SPEC=${V80_SPEC},MOSAICKV_V80_FROZEN=${FROZEN}" \
  slurm/choice_c_v80_freeze.sbatch)

printf 'v80 risk-gated ledger: %s\n' "${LEDGER_JOB}"
printf 'v80 risk-gated calibration: %s\n' "${CALIBRATION_JOB}"
printf 'v80 freeze: %s\n' "${FREEZE_JOB}"
printf 'No held-out campaign was submitted. A future held-out protocol requires both unchanged gates and a v80 freeze.\n'
