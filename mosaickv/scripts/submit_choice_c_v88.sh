#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
V88_ROOT=${MOSAICKV_V88_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v88-initial-query-coverage-calibration-v1}
V88_SPEC=${PACKAGE_ROOT}/configs/experiments/choice_c_v88_initial_query_coverage_calibration.yaml
FROZEN=${V88_ROOT}/analysis/frozen_candidate.yaml

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"

if [[ -e "${V88_ROOT}" ]]; then
  printf 'refusing to reuse existing v88 output root: %s\n' "${V88_ROOT}" >&2
  exit 2
fi
mkdir -p /scratch/djy8hg/runs/mosaickv/slurm

LEDGER_JOB=$(sbatch --parsable \
  --export="ALL,MOSAICKV_V88_OUTPUT_ROOT=${V88_ROOT}" \
  slurm/choice_c_v88_materialize_initial_query_coverage_calibration.sbatch)
CALIBRATION_JOB=$(sbatch --parsable \
  --dependency="afterok:${LEDGER_JOB}" \
  --export="ALL,MOSAICKV_V88_OUTPUT_ROOT=${V88_ROOT},MOSAICKV_V88_SPEC=${V88_SPEC}" \
  slurm/choice_c_v88_initial_query_coverage_a100.sbatch)
FREEZE_JOB=$(sbatch --parsable \
  --dependency="afterok:${CALIBRATION_JOB}" \
  --export="ALL,MOSAICKV_V88_OUTPUT_ROOT=${V88_ROOT},MOSAICKV_V88_SPEC=${V88_SPEC},MOSAICKV_V88_FROZEN=${FROZEN}" \
  slurm/choice_c_v88_freeze.sbatch)

printf 'v88 fresh ledgers: %s\n' "${LEDGER_JOB}"
printf 'v88 initial-query-coverage calibration: %s\n' "${CALIBRATION_JOB}"
printf 'v88 robust freeze analysis: %s\n' "${FREEZE_JOB}"
printf 'No held-out campaign was submitted. It remains blocked unless every unchanged gate passes and a frozen candidate is created.\n'
