#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
V78_ROOT=${MOSAICKV_V78_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v78-disjoint-visual-balance-calibration-v1}
V78_SPEC=${PACKAGE_ROOT}/configs/experiments/choice_c_v78_disjoint_visual_balance_calibration.yaml
FROZEN=${V78_ROOT}/analysis/frozen_candidate.yaml

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"

if [[ -e "${V78_ROOT}" ]]; then
  printf 'refusing to reuse existing v78 output root: %s\n' "${V78_ROOT}" >&2
  exit 2
fi

LEDGER_JOB=$(sbatch --parsable \
  --export="ALL,MOSAICKV_V78_OUTPUT_ROOT=${V78_ROOT}" \
  slurm/choice_c_v78_materialize_visual_calibration.sbatch)
CALIBRATION_JOB=$(sbatch --parsable \
  --dependency="afterok:${LEDGER_JOB}" \
  --export="ALL,MOSAICKV_V78_OUTPUT_ROOT=${V78_ROOT},MOSAICKV_V78_SPEC=${V78_SPEC}" \
  slurm/choice_c_v78_visual_calibration_a100.sbatch)
FREEZE_JOB=$(sbatch --parsable \
  --dependency="afterok:${CALIBRATION_JOB}" \
  --export="ALL,MOSAICKV_V78_OUTPUT_ROOT=${V78_ROOT},MOSAICKV_V78_SPEC=${V78_SPEC},MOSAICKV_V78_FROZEN=${FROZEN}" \
  slurm/choice_c_v78_freeze.sbatch)

printf 'v78 visual-balance ledger: %s\n' "${LEDGER_JOB}"
printf 'v78 visual-balance calibration: %s\n' "${CALIBRATION_JOB}"
printf 'v78 freeze: %s\n' "${FREEZE_JOB}"
printf 'v77 remains blocked and was not submitted. A future held-out protocol requires a passing v78 freeze.\n'
