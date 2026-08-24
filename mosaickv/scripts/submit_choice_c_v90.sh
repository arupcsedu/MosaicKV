#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
V90_ROOT=${MOSAICKV_V90_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v90-cross-head-query-byte-allocation-calibration-v1}
V90_SPEC=${PACKAGE_ROOT}/configs/experiments/choice_c_v90_cross_head_query_byte_allocation_calibration.yaml
FROZEN=${V90_ROOT}/analysis/frozen_candidate.yaml
CPU_PARTITION=${MOSAICKV_CPU_PARTITION:-bii}

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
"${PACKAGE_ROOT}/scripts/validate_choice_c_v89_negative_seal.py" \
  --seal "${PACKAGE_ROOT}/configs/ledgers/choice_c_v89_negative_calibration_seal.yaml"

if [[ -e "${V90_ROOT}" ]]; then
  printf 'refusing to reuse existing v90 output root: %s\n' "${V90_ROOT}" >&2
  exit 2
fi
mkdir -p /scratch/djy8hg/runs/mosaickv/slurm

LEDGER_JOB=$(sbatch --parsable \
  --partition="${CPU_PARTITION}" \
  --export="ALL,MOSAICKV_V90_OUTPUT_ROOT=${V90_ROOT}" \
  slurm/choice_c_v90_materialize_cross_head_calibration.sbatch)
CALIBRATION_JOB=$(sbatch --parsable \
  --dependency="afterok:${LEDGER_JOB}" \
  --export="ALL,MOSAICKV_V90_OUTPUT_ROOT=${V90_ROOT},MOSAICKV_V90_SPEC=${V90_SPEC}" \
  slurm/choice_c_v90_cross_head_a100.sbatch)
FREEZE_JOB=$(sbatch --parsable \
  --partition="${CPU_PARTITION}" \
  --dependency="afterok:${CALIBRATION_JOB}" \
  --export="ALL,MOSAICKV_V90_OUTPUT_ROOT=${V90_ROOT},MOSAICKV_V90_SPEC=${V90_SPEC},MOSAICKV_V90_FROZEN=${FROZEN}" \
  slurm/choice_c_v90_freeze.sbatch)

printf 'v90 fresh ledgers: %s (CPU partition %s)\n' "${LEDGER_JOB}" "${CPU_PARTITION}"
printf 'v90 cross-head calibration: %s\n' "${CALIBRATION_JOB}"
printf 'v90 robust freeze analysis: %s (CPU partition %s)\n' "${FREEZE_JOB}" "${CPU_PARTITION}"
printf 'No held-out campaign was submitted. It remains blocked unless every unchanged gate passes and a frozen candidate is created.\n'
