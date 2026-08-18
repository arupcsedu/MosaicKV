#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
V74_ROOT=${MOSAICKV_V74_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v74-label-blind-calibration-v6}
V74_SPEC=${PACKAGE_ROOT}/configs/experiments/choice_c_v74_label_blind_calibration.yaml
FROZEN=${V74_ROOT}/analysis/frozen_candidate.yaml

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"

LEDGER_JOB=$(sbatch --parsable \
  --export="ALL,MOSAICKV_V74_OUTPUT_ROOT=${V74_ROOT}" \
  slurm/choice_c_v74_materialize_calibration.sbatch)
CALIBRATION_JOB=$(sbatch --parsable \
  --dependency="afterok:${LEDGER_JOB}" \
  --export="ALL,MOSAICKV_V74_OUTPUT_ROOT=${V74_ROOT},MOSAICKV_V74_SPEC=${V74_SPEC}" \
  slurm/choice_c_v74_calibration_a100.sbatch)
FREEZE_JOB=$(sbatch --parsable \
  --dependency="afterok:${CALIBRATION_JOB}" \
  --export="ALL,MOSAICKV_V74_OUTPUT_ROOT=${V74_ROOT},MOSAICKV_V74_SPEC=${V74_SPEC},MOSAICKV_V74_FROZEN=${FROZEN}" \
  slurm/choice_c_v74_freeze.sbatch)
printf 'v74 ledger: %s\n' "${LEDGER_JOB}"
printf 'v74 calibration: %s\n' "${CALIBRATION_JOB}"
printf 'v74 freeze: %s\n' "${FREEZE_JOB}"
printf 'v75 held-out jobs were not submitted; use submit_choice_c_v75_after_freeze.sh only after the v74 gate passes.\n'
