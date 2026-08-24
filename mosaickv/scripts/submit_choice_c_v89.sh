#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
V89_ROOT=${MOSAICKV_V89_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v89-capped-cumulative-query-coverage-calibration-v1}
V89_SPEC=${PACKAGE_ROOT}/configs/experiments/choice_c_v89_capped_cumulative_query_coverage_calibration.yaml
FROZEN=${V89_ROOT}/analysis/frozen_candidate.yaml
CPU_PARTITION=${MOSAICKV_CPU_PARTITION:-standard}

case "${CPU_PARTITION}" in
  bii|parallel|standard) ;;
  *)
    printf 'MOSAICKV_CPU_PARTITION must be bii, parallel, or standard; got %s\n' "${CPU_PARTITION}" >&2
    exit 2
    ;;
esac

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"

if [[ -e "${V89_ROOT}" ]]; then
  printf 'refusing to reuse existing v89 output root: %s\n' "${V89_ROOT}" >&2
  exit 2
fi
mkdir -p /scratch/djy8hg/runs/mosaickv/slurm

LEDGER_JOB=$(sbatch --parsable \
  --partition="${CPU_PARTITION}" \
  --export="ALL,MOSAICKV_V89_OUTPUT_ROOT=${V89_ROOT}" \
  slurm/choice_c_v89_materialize_capped_cumulative_calibration.sbatch)
CALIBRATION_JOB=$(sbatch --parsable \
  --dependency="afterok:${LEDGER_JOB}" \
  --export="ALL,MOSAICKV_V89_OUTPUT_ROOT=${V89_ROOT},MOSAICKV_V89_SPEC=${V89_SPEC}" \
  slurm/choice_c_v89_capped_cumulative_a100.sbatch)
FREEZE_JOB=$(sbatch --parsable \
  --partition="${CPU_PARTITION}" \
  --dependency="afterok:${CALIBRATION_JOB}" \
  --export="ALL,MOSAICKV_V89_OUTPUT_ROOT=${V89_ROOT},MOSAICKV_V89_SPEC=${V89_SPEC},MOSAICKV_V89_FROZEN=${FROZEN}" \
  slurm/choice_c_v89_freeze.sbatch)

printf 'v89 fresh ledgers: %s (CPU partition %s)\n' "${LEDGER_JOB}" "${CPU_PARTITION}"
printf 'v89 capped-cumulative calibration: %s\n' "${CALIBRATION_JOB}"
printf 'v89 robust freeze analysis: %s (CPU partition %s)\n' "${FREEZE_JOB}" "${CPU_PARTITION}"
printf 'No held-out campaign was submitted. It remains blocked unless every unchanged gate passes and a frozen candidate is created.\n'
