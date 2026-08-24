#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
SOURCE_ROOT=${MOSAICKV_V98_SOURCE_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v98-short-horizon-logit-calibration-v1}
RETRY_ROOT=${MOSAICKV_V98_RETRY_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v98-short-horizon-logit-calibration-v1-retry1}
V98_SPEC=${PACKAGE_ROOT}/configs/experiments/choice_c_v98_short_horizon_logit_calibration.yaml
FROZEN=${RETRY_ROOT}/analysis/frozen_candidate.yaml
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
  "${PACKAGE_ROOT}/scripts/validate_choice_c_v97_negative_seal.py" \
  --seal "${PACKAGE_ROOT}/configs/ledgers/choice_c_v97_negative_calibration_seal.yaml"

if [[ -e "${RETRY_ROOT}" ]]; then
  printf 'refusing to reuse existing V98 retry output root: %s\n' "${RETRY_ROOT}" >&2
  exit 2
fi

validate_ledger() {
  local path=$1
  local expected=$2
  local actual
  if [[ ! -f "${path}" ]]; then
    printf 'required immutable V98 ledger is missing: %s\n' "${path}" >&2
    exit 2
  fi
  actual=$(sha256sum "${path}")
  actual=${actual%% *}
  if [[ "${actual}" != "${expected}" ]]; then
    printf 'V98 ledger SHA mismatch for %s: expected %s, observed %s\n' \
      "${path}" "${expected}" "${actual}" >&2
    exit 2
  fi
}

validate_ledger \
  "${SOURCE_ROOT}/inputs/calibration_docvqa.json" \
  24b7088d3ab6ce383aa298e85479e29d4fb9b9895cfc0f9867cb063ee6a6cad3
validate_ledger \
  "${SOURCE_ROOT}/inputs/calibration_textvqa.json" \
  a67c9342a09841e2dc4ed8a0babaf00aa9b40eb8b78f165d0dd64d314d6daf17

mkdir -p /scratch/djy8hg/runs/mosaickv/slurm
CALIBRATION_JOB=$(sbatch --parsable \
  --export="ALL,MOSAICKV_V98_OUTPUT_ROOT=${RETRY_ROOT},MOSAICKV_V98_SPEC=${V98_SPEC}" \
  slurm/choice_c_v98_short_horizon_logit_a100.sbatch)
FREEZE_JOB=$(sbatch --parsable \
  --partition="${CPU_PARTITION}" \
  --dependency="afterok:${CALIBRATION_JOB}" \
  --export="ALL,MOSAICKV_V98_OUTPUT_ROOT=${RETRY_ROOT},MOSAICKV_V98_SPEC=${V98_SPEC},MOSAICKV_V98_FROZEN=${FROZEN}" \
  slurm/choice_c_v98_freeze.sbatch)

printf 'V98 immutable ledgers reused from failed source root: %s\n' "${SOURCE_ROOT}"
printf 'V98 replacement short-horizon logit calibration: %s\n' "${CALIBRATION_JOB}"
printf 'V98 replacement robust freeze analysis: %s (CPU partition %s)\n' \
  "${FREEZE_JOB}" "${CPU_PARTITION}"
printf 'Failed artifacts remain preserved. No held-out campaign was submitted.\n'
