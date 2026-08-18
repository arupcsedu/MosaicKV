#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
V84_ROOT=${MOSAICKV_V84_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v84-disjoint-risk-gated-hybrid-calibration-v1}
V84_SPEC=${PACKAGE_ROOT}/configs/experiments/choice_c_v84_disjoint_risk_gated_hybrid_calibration.yaml
FROZEN=${V84_ROOT}/analysis/frozen_candidate.yaml

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"

if [[ -e "${V84_ROOT}" ]]; then
  printf 'refusing to reuse existing v84 output root: %s\n' "${V84_ROOT}" >&2
  exit 2
fi

LEDGER_JOB=$(sbatch --parsable \
  --export="ALL,MOSAICKV_V84_OUTPUT_ROOT=${V84_ROOT}" \
  slurm/choice_c_v84_materialize_risk_calibration.sbatch)
CALIBRATION_JOB=$(sbatch --parsable \
  --dependency="afterok:${LEDGER_JOB}" \
  --export="ALL,MOSAICKV_V84_OUTPUT_ROOT=${V84_ROOT},MOSAICKV_V84_SPEC=${V84_SPEC}" \
  slurm/choice_c_v84_risk_calibration_a100.sbatch)
FREEZE_JOB=$(sbatch --parsable \
  --dependency="afterok:${CALIBRATION_JOB}" \
  --export="ALL,MOSAICKV_V84_OUTPUT_ROOT=${V84_ROOT},MOSAICKV_V84_SPEC=${V84_SPEC},MOSAICKV_V84_FROZEN=${FROZEN}" \
  slurm/choice_c_v84_freeze.sbatch)

printf 'v84 ledger: %s\n' "${LEDGER_JOB}"
printf 'v84 risk-gated calibration: %s\n' "${CALIBRATION_JOB}"
printf 'v84 freeze: %s\n' "${FREEZE_JOB}"
printf 'No held-out campaign was submitted. It remains blocked unless every v84 gate passes and a frozen candidate is created.\n'
