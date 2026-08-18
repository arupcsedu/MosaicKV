#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
V86_ROOT=${MOSAICKV_V86_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v86-disjoint-robust-fidelity-calibration-v1}
V86_SPEC=${PACKAGE_ROOT}/configs/experiments/choice_c_v86_disjoint_robust_fidelity_calibration.yaml
FROZEN=${V86_ROOT}/analysis/frozen_candidate.yaml

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"

if [[ -e "${V86_ROOT}" ]]; then
  printf 'refusing to reuse existing v86 output root: %s\n' "${V86_ROOT}" >&2
  exit 2
fi

LEDGER_JOB=$(sbatch --parsable \
  --export="ALL,MOSAICKV_V86_OUTPUT_ROOT=${V86_ROOT}" \
  slurm/choice_c_v86_materialize_robust_calibration.sbatch)
CALIBRATION_JOB=$(sbatch --parsable \
  --dependency="afterok:${LEDGER_JOB}" \
  --export="ALL,MOSAICKV_V86_OUTPUT_ROOT=${V86_ROOT},MOSAICKV_V86_SPEC=${V86_SPEC}" \
  slurm/choice_c_v86_robust_calibration_a100.sbatch)
FREEZE_JOB=$(sbatch --parsable \
  --dependency="afterok:${CALIBRATION_JOB}" \
  --export="ALL,MOSAICKV_V86_OUTPUT_ROOT=${V86_ROOT},MOSAICKV_V86_SPEC=${V86_SPEC},MOSAICKV_V86_FROZEN=${FROZEN}" \
  slurm/choice_c_v86_freeze.sbatch)

printf 'v86 ledger: %s\n' "${LEDGER_JOB}"
printf 'v86 robust calibration: %s\n' "${CALIBRATION_JOB}"
printf 'v86 robust freeze: %s\n' "${FREEZE_JOB}"
printf 'No held-out campaign was submitted. It remains blocked unless every v86 gate passes and a frozen candidate is created.\n'
