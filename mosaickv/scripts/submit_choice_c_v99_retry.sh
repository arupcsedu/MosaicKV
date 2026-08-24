#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
LEDGER_ROOT=/scratch/djy8hg/runs/mosaickv/choice-c-v99-progressive-logit-safe-eviction-calibration-v1
RETRY_ROOT=${MOSAICKV_V99_RETRY_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v99-progressive-logit-safe-eviction-calibration-v1-retry1}
V99_SPEC=${PACKAGE_ROOT}/configs/experiments/choice_c_v99_progressive_logit_safe_eviction_calibration.yaml
FROZEN=${RETRY_ROOT}/analysis/frozen_candidate.yaml

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"
PYTHONPATH="${PACKAGE_ROOT}/scripts" "${PYTHON}" \
  "${PACKAGE_ROOT}/scripts/validate_choice_c_v98_negative_seal.py" \
  --seal "${PACKAGE_ROOT}/configs/ledgers/choice_c_v98_negative_calibration_seal.yaml"
PYTHONPATH="${PACKAGE_ROOT}/src:${PACKAGE_ROOT}/scripts" "${PYTHON}" \
  "${PACKAGE_ROOT}/scripts/validate_choice_c_v99_protocol.py" \
  --spec "${V99_SPEC}" \
  --seal "${PACKAGE_ROOT}/configs/ledgers/choice_c_v98_negative_calibration_seal.yaml"

verify_ledger() {
  local path=$1
  local expected=$2
  local observed
  observed=$(sha256sum "${path}" | cut -d' ' -f1)
  if [[ "${observed}" != "${expected}" ]]; then
    printf 'immutable V99 ledger hash mismatch: %s\n' "${path}" >&2
    exit 2
  fi
}

verify_ledger \
  "${LEDGER_ROOT}/inputs/calibration_docvqa.json" \
  c871e44a176fd62988b1bf8a51c93414200832fe424531b39dcf636e33663a73
verify_ledger \
  "${LEDGER_ROOT}/inputs/calibration_textvqa.json" \
  73cb0b3be2cee3ba69174a5403d48a1549dddfec2cdc436ba6544a5f7b9da59a

if [[ -e "${RETRY_ROOT}" ]]; then
  printf 'refusing to reuse existing V99 retry output root: %s\n' "${RETRY_ROOT}" >&2
  exit 2
fi
mkdir -p /scratch/djy8hg/runs/mosaickv/slurm

CALIBRATION_JOB=$(sbatch --parsable \
  --export="ALL,MOSAICKV_V99_OUTPUT_ROOT=${RETRY_ROOT},MOSAICKV_V99_SPEC=${V99_SPEC}" \
  slurm/choice_c_v99_progressive_eviction_a100.sbatch)
FREEZE_JOB=$(sbatch --parsable \
  --partition=bii \
  --dependency="afterok:${CALIBRATION_JOB}" \
  --export="ALL,MOSAICKV_V99_OUTPUT_ROOT=${RETRY_ROOT},MOSAICKV_V99_SPEC=${V99_SPEC},MOSAICKV_V99_FROZEN=${FROZEN}" \
  slurm/choice_c_v99_freeze.sbatch)

printf 'V99 replacement progressive calibration: %s (two one-A100 workers)\n' "${CALIBRATION_JOB}"
printf 'V99 replacement robust freeze: %s (bii, afterok)\n' "${FREEZE_JOB}"
printf 'Immutable ledgers were reused; no ledger materialization was submitted.\n'
printf 'No held-out campaign was submitted. It remains blocked unless every unchanged gate passes and a frozen candidate is created.\n'
