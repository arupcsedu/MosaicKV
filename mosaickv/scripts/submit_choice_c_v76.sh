#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
V76_ROOT=${MOSAICKV_V76_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v76-disjoint-ocr-calibration-v1}
V76_SPEC=${PACKAGE_ROOT}/configs/experiments/choice_c_v76_disjoint_ocr_calibration.yaml
FROZEN=${V76_ROOT}/analysis/frozen_candidate.yaml

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"

LEDGER_JOB=$(sbatch --parsable \
  --export="ALL,MOSAICKV_V76_OUTPUT_ROOT=${V76_ROOT}" \
  slurm/choice_c_v76_materialize_ocr_calibration.sbatch)
CALIBRATION_JOB=$(sbatch --parsable \
  --dependency="afterok:${LEDGER_JOB}" \
  --export="ALL,MOSAICKV_V76_OUTPUT_ROOT=${V76_ROOT},MOSAICKV_V76_SPEC=${V76_SPEC}" \
  slurm/choice_c_v76_ocr_calibration_a100.sbatch)
FREEZE_JOB=$(sbatch --parsable \
  --dependency="afterok:${CALIBRATION_JOB}" \
  --export="ALL,MOSAICKV_V76_OUTPUT_ROOT=${V76_ROOT},MOSAICKV_V76_SPEC=${V76_SPEC},MOSAICKV_V76_FROZEN=${FROZEN}" \
  slurm/choice_c_v76_freeze.sbatch)

printf 'v76 OCR ledger: %s\n' "${LEDGER_JOB}"
printf 'v76 OCR calibration: %s\n' "${CALIBRATION_JOB}"
printf 'v76 freeze: %s\n' "${FREEZE_JOB}"
printf 'v77 was not submitted; use submit_choice_c_v77_after_freeze.sh only after both v76 gates pass.\n'
