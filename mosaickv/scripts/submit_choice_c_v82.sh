#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
V82_ROOT=${MOSAICKV_V82_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v82-disjoint-hybrid-divergence-calibration-v1}
V82_SPEC=${PACKAGE_ROOT}/configs/experiments/choice_c_v82_disjoint_hybrid_divergence_calibration.yaml
FROZEN=${V82_ROOT}/analysis/frozen_candidate.yaml

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"

if [[ -e "${V82_ROOT}" ]]; then
  printf 'refusing to reuse existing v82 output root: %s\n' "${V82_ROOT}" >&2
  exit 2
fi

LEDGER_JOB=$(sbatch --parsable \
  --export="ALL,MOSAICKV_V82_OUTPUT_ROOT=${V82_ROOT}" \
  slurm/choice_c_v82_materialize_hybrid_calibration.sbatch)
CALIBRATION_JOB=$(sbatch --parsable \
  --dependency="afterok:${LEDGER_JOB}" \
  --export="ALL,MOSAICKV_V82_OUTPUT_ROOT=${V82_ROOT},MOSAICKV_V82_SPEC=${V82_SPEC}" \
  slurm/choice_c_v82_hybrid_calibration_a100.sbatch)
FREEZE_JOB=$(sbatch --parsable \
  --dependency="afterok:${CALIBRATION_JOB}" \
  --export="ALL,MOSAICKV_V82_OUTPUT_ROOT=${V82_ROOT},MOSAICKV_V82_SPEC=${V82_SPEC},MOSAICKV_V82_FROZEN=${FROZEN}" \
  slurm/choice_c_v82_freeze.sbatch)

printf 'v82 ledger: %s\n' "${LEDGER_JOB}"
printf 'v82 hybrid calibration: %s\n' "${CALIBRATION_JOB}"
printf 'v82 freeze: %s\n' "${FREEZE_JOB}"
printf 'No held-out campaign was submitted. It remains blocked unless both unchanged gates pass and v82 creates a frozen candidate.\n'
