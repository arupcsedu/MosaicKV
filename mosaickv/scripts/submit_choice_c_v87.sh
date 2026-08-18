#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
V87_ROOT=${MOSAICKV_V87_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v87-prompt-query-coverage-calibration-v1}
V87_SPEC=${PACKAGE_ROOT}/configs/experiments/choice_c_v87_prompt_query_coverage_development.yaml
FROZEN=${V87_ROOT}/analysis/frozen_candidate.yaml

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"

if [[ -e "${V87_ROOT}" ]]; then
  printf 'refusing to reuse existing v87 output root: %s\n' "${V87_ROOT}" >&2
  exit 2
fi
mkdir -p /scratch/djy8hg/runs/mosaickv/slurm

LEDGER_JOB=$(sbatch --parsable \
  --export="ALL,MOSAICKV_V87_OUTPUT_ROOT=${V87_ROOT}" \
  slurm/choice_c_v87_materialize_query_coverage_calibration.sbatch)
CALIBRATION_JOB=$(sbatch --parsable \
  --dependency="afterok:${LEDGER_JOB}" \
  --export="ALL,MOSAICKV_V87_OUTPUT_ROOT=${V87_ROOT},MOSAICKV_V87_SPEC=${V87_SPEC}" \
  slurm/choice_c_v87_query_coverage_calibration_a100.sbatch)
FREEZE_JOB=$(sbatch --parsable \
  --dependency="afterok:${CALIBRATION_JOB}" \
  --export="ALL,MOSAICKV_V87_OUTPUT_ROOT=${V87_ROOT},MOSAICKV_V87_SPEC=${V87_SPEC},MOSAICKV_V87_FROZEN=${FROZEN}" \
  slurm/choice_c_v87_freeze.sbatch)

printf 'v87 fresh ledgers: %s\n' "${LEDGER_JOB}"
printf 'v87 query-coverage calibration: %s\n' "${CALIBRATION_JOB}"
printf 'v87 robust freeze analysis: %s\n' "${FREEZE_JOB}"
printf 'No held-out campaign was submitted. It remains blocked unless every unchanged gate passes and a frozen candidate is created.\n'
