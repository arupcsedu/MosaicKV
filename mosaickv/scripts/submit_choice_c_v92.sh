#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
V92_ROOT=${MOSAICKV_V92_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v92-output-impact-stability-calibration-v1}
V92_SPEC=${PACKAGE_ROOT}/configs/experiments/choice_c_v92_output_impact_stability_calibration.yaml
FROZEN=${V92_ROOT}/analysis/frozen_candidate.yaml
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
  "${PACKAGE_ROOT}/scripts/validate_choice_c_v91_negative_seal.py" \
  --seal "${PACKAGE_ROOT}/configs/ledgers/choice_c_v91_negative_calibration_seal.yaml"

if [[ -e "${V92_ROOT}" ]]; then
  printf 'refusing to reuse existing v92 output root: %s\n' "${V92_ROOT}" >&2
  exit 2
fi
mkdir -p /scratch/djy8hg/runs/mosaickv/slurm

LEDGER_JOB=$(sbatch --parsable \
  --partition="${CPU_PARTITION}" \
  --export="ALL,MOSAICKV_V92_OUTPUT_ROOT=${V92_ROOT}" \
  slurm/choice_c_v92_materialize_output_impact_calibration.sbatch)
CALIBRATION_JOB=$(sbatch --parsable \
  --dependency="afterok:${LEDGER_JOB}" \
  --export="ALL,MOSAICKV_V92_OUTPUT_ROOT=${V92_ROOT},MOSAICKV_V92_SPEC=${V92_SPEC}" \
  slurm/choice_c_v92_output_impact_a100.sbatch)
FREEZE_JOB=$(sbatch --parsable \
  --partition="${CPU_PARTITION}" \
  --dependency="afterok:${CALIBRATION_JOB}" \
  --export="ALL,MOSAICKV_V92_OUTPUT_ROOT=${V92_ROOT},MOSAICKV_V92_SPEC=${V92_SPEC},MOSAICKV_V92_FROZEN=${FROZEN}" \
  slurm/choice_c_v92_freeze.sbatch)

printf 'v92 fresh ledgers: %s (CPU partition %s)\n' "${LEDGER_JOB}" "${CPU_PARTITION}"
printf 'v92 output-impact calibration: %s\n' "${CALIBRATION_JOB}"
printf 'v92 robust freeze analysis: %s (CPU partition %s)\n' "${FREEZE_JOB}" "${CPU_PARTITION}"
printf 'No held-out campaign was submitted. It remains blocked unless every unchanged gate passes and a frozen candidate is created.\n'
