#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
V79_ROOT=${MOSAICKV_V79_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v79-disjoint-adaptive-swap-calibration-v1}
V79_SPEC=${PACKAGE_ROOT}/configs/experiments/choice_c_v79_disjoint_adaptive_swap_calibration.yaml
FROZEN=${V79_ROOT}/analysis/frozen_candidate.yaml

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"

if [[ -e "${V79_ROOT}" ]]; then
  printf 'refusing to reuse existing v79 output root: %s\n' "${V79_ROOT}" >&2
  exit 2
fi

LEDGER_JOB=$(sbatch --parsable \
  --export="ALL,MOSAICKV_V79_OUTPUT_ROOT=${V79_ROOT}" \
  slurm/choice_c_v79_materialize_adaptive_calibration.sbatch)
CALIBRATION_JOB=$(sbatch --parsable \
  --dependency="afterok:${LEDGER_JOB}" \
  --export="ALL,MOSAICKV_V79_OUTPUT_ROOT=${V79_ROOT},MOSAICKV_V79_SPEC=${V79_SPEC}" \
  slurm/choice_c_v79_adaptive_calibration_a100.sbatch)
FREEZE_JOB=$(sbatch --parsable \
  --dependency="afterok:${CALIBRATION_JOB}" \
  --export="ALL,MOSAICKV_V79_OUTPUT_ROOT=${V79_ROOT},MOSAICKV_V79_SPEC=${V79_SPEC},MOSAICKV_V79_FROZEN=${FROZEN}" \
  slurm/choice_c_v79_freeze.sbatch)

printf 'v79 adaptive-swap ledger: %s\n' "${LEDGER_JOB}"
printf 'v79 adaptive-swap calibration: %s\n' "${CALIBRATION_JOB}"
printf 'v79 freeze: %s\n' "${FREEZE_JOB}"
printf 'No held-out campaign was submitted. A future held-out protocol requires both unchanged gates and a v79 freeze.\n'
