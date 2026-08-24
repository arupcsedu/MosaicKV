#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
V95_ROOT=${MOSAICKV_V95_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v95-atomic-layout-risk-calibration-v1}
V95_SPEC=${PACKAGE_ROOT}/configs/experiments/choice_c_v95_atomic_layout_risk_calibration.yaml
FROZEN=${V95_ROOT}/analysis/frozen_candidate.yaml
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
  "${PACKAGE_ROOT}/scripts/validate_choice_c_v94_negative_seal.py" \
  --seal "${PACKAGE_ROOT}/configs/ledgers/choice_c_v94_negative_calibration_seal.yaml"

if [[ -e "${V95_ROOT}" ]]; then
  printf 'refusing to reuse existing V95 output root: %s\n' "${V95_ROOT}" >&2
  exit 2
fi
mkdir -p /scratch/djy8hg/runs/mosaickv/slurm

LEDGER_JOB=$(sbatch --parsable \
  --partition="${CPU_PARTITION}" \
  --export="ALL,MOSAICKV_V95_OUTPUT_ROOT=${V95_ROOT}" \
  slurm/choice_c_v95_materialize_atomic_layout_calibration.sbatch)
CALIBRATION_JOB=$(sbatch --parsable \
  --dependency="afterok:${LEDGER_JOB}" \
  --export="ALL,MOSAICKV_V95_OUTPUT_ROOT=${V95_ROOT},MOSAICKV_V95_SPEC=${V95_SPEC}" \
  slurm/choice_c_v95_atomic_layout_a100.sbatch)
FREEZE_JOB=$(sbatch --parsable \
  --partition="${CPU_PARTITION}" \
  --dependency="afterok:${CALIBRATION_JOB}" \
  --export="ALL,MOSAICKV_V95_OUTPUT_ROOT=${V95_ROOT},MOSAICKV_V95_SPEC=${V95_SPEC},MOSAICKV_V95_FROZEN=${FROZEN}" \
  slurm/choice_c_v95_freeze.sbatch)

printf 'V95 fresh ledgers: %s (CPU partition %s)\n' "${LEDGER_JOB}" "${CPU_PARTITION}"
printf 'V95 atomic-layout-risk calibration: %s\n' "${CALIBRATION_JOB}"
printf 'V95 robust freeze analysis: %s (CPU partition %s)\n' "${FREEZE_JOB}" "${CPU_PARTITION}"
printf 'No held-out campaign was submitted. It remains blocked unless every unchanged gate passes and a frozen candidate is created.\n'
