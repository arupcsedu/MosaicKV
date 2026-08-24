#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
V91_ROOT=${MOSAICKV_V91_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v91-conservative-cross-head-stability-calibration-v1}
V91_SPEC=${PACKAGE_ROOT}/configs/experiments/choice_c_v91_conservative_cross_head_stability_calibration.yaml
FROZEN=${V91_ROOT}/analysis/frozen_candidate.yaml
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
PYTHONPATH="${PACKAGE_ROOT}/scripts" "${PYTHON}" \
  "${PACKAGE_ROOT}/scripts/validate_choice_c_v90_negative_seal.py" \
  --seal "${PACKAGE_ROOT}/configs/ledgers/choice_c_v90_negative_calibration_seal.yaml"

if [[ -e "${V91_ROOT}" ]]; then
  printf 'refusing to reuse existing v91 output root: %s\n' "${V91_ROOT}" >&2
  exit 2
fi
mkdir -p /scratch/djy8hg/runs/mosaickv/slurm

LEDGER_JOB=$(sbatch --parsable \
  --partition="${CPU_PARTITION}" \
  --export="ALL,MOSAICKV_V91_OUTPUT_ROOT=${V91_ROOT}" \
  slurm/choice_c_v91_materialize_stability_calibration.sbatch)
CALIBRATION_JOB=$(sbatch --parsable \
  --dependency="afterok:${LEDGER_JOB}" \
  --export="ALL,MOSAICKV_V91_OUTPUT_ROOT=${V91_ROOT},MOSAICKV_V91_SPEC=${V91_SPEC}" \
  slurm/choice_c_v91_stability_a100.sbatch)
FREEZE_JOB=$(sbatch --parsable \
  --partition="${CPU_PARTITION}" \
  --dependency="afterok:${CALIBRATION_JOB}" \
  --export="ALL,MOSAICKV_V91_OUTPUT_ROOT=${V91_ROOT},MOSAICKV_V91_SPEC=${V91_SPEC},MOSAICKV_V91_FROZEN=${FROZEN}" \
  slurm/choice_c_v91_freeze.sbatch)

printf 'v91 fresh ledgers: %s (CPU partition %s)\n' "${LEDGER_JOB}" "${CPU_PARTITION}"
printf 'v91 conservative cross-head calibration: %s\n' "${CALIBRATION_JOB}"
printf 'v91 robust freeze analysis: %s (CPU partition %s)\n' "${FREEZE_JOB}" "${CPU_PARTITION}"
printf 'No held-out campaign was submitted. It remains blocked unless every unchanged gate passes and a frozen candidate is created.\n'
