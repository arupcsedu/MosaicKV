#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
RUNTIME_COMMIT=9e300dda54fb5377aefae06d1f28614e426d42fd
RUNTIME_ROOT=/scratch/djy8hg/workdir/mosaickv_v85_runtime_9e300dda54fb
V85_ROOT=${MOSAICKV_V85_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v85-v84-frozen-untouched-heldout-quality-memory-v1}
V85_SPEC=${PACKAGE_ROOT}/configs/experiments/choice_c_v85_v84_frozen_untouched_heldout_quality_memory.yaml
FROZEN=/scratch/djy8hg/runs/mosaickv/choice-c-v84-disjoint-risk-gated-hybrid-calibration-v1/analysis/frozen_candidate.yaml
EXPECTED_FROZEN_SHA=940b124a2c2ab09d8192ab03d6cb2efda85339523e613980d567495a7280cf64

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"

PROTOCOL_COMMIT=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
PROTOCOL_ROOT=/scratch/djy8hg/workdir/mosaickv_v85_protocol_${PROTOCOL_COMMIT:0:12}

prepare_worktree() {
  local destination=$1
  local commit=$2
  if [[ -e "${destination}" ]]; then
    if [[ ! -e "${destination}/.git" ]] || \
      [[ "$(git -C "${destination}" rev-parse HEAD)" != "${commit}" ]] || \
      [[ -n "$(git -C "${destination}" status --porcelain --untracked-files=all)" ]]; then
      printf 'refusing incompatible execution worktree: %s\n' "${destination}" >&2
      exit 2
    fi
    return
  fi
  git -C "${PROJECT_ROOT}" worktree add --detach "${destination}" "${commit}"
}

prepare_worktree "${RUNTIME_ROOT}" "${RUNTIME_COMMIT}"
prepare_worktree "${PROTOCOL_ROOT}" "${PROTOCOL_COMMIT}"
RUNTIME_PACKAGE_ROOT=${RUNTIME_ROOT}/mosaickv
PROTOCOL_PACKAGE_ROOT=${PROTOCOL_ROOT}/mosaickv

if [[ -e "${V85_ROOT}" ]]; then
  printf 'refusing to reuse existing v85 output root: %s\n' "${V85_ROOT}" >&2
  exit 2
fi
"${PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}" \
  "${PROTOCOL_PACKAGE_ROOT}/scripts/validate_choice_c_v85_preregistration.py" \
  --spec "${PROTOCOL_PACKAGE_ROOT}/configs/experiments/choice_c_v85_v84_frozen_untouched_heldout_quality_memory.yaml" \
  --frozen "${FROZEN}"

mkdir -p "${V85_ROOT}"
install -m 0444 \
  "${PROTOCOL_PACKAGE_ROOT}/configs/experiments/choice_c_v85_v84_frozen_untouched_heldout_quality_memory.yaml" \
  "${V85_ROOT}/preregistration.yaml"
PREREG=${V85_ROOT}/preregistration.yaml
PREREG_SHA=$(sha256sum "${PREREG}" | cut -d' ' -f1)

LEDGER_JOB=$(sbatch --parsable \
  --export="ALL,MOSAICKV_V85_OUTPUT_ROOT=${V85_ROOT},MOSAICKV_V85_SPEC=${PREREG},MOSAICKV_V85_SPEC_SHA256=${PREREG_SHA},MOSAICKV_V85_FROZEN=${FROZEN},MOSAICKV_V85_FROZEN_SHA256=${EXPECTED_FROZEN_SHA},MOSAICKV_V85_RUNTIME_PACKAGE_ROOT=${RUNTIME_PACKAGE_ROOT}" \
  slurm/choice_c_v85_materialize_heldout.sbatch)
HELDOUT_JOB=$(sbatch --parsable \
  --dependency="afterok:${LEDGER_JOB}" \
  --export="ALL,MOSAICKV_V85_OUTPUT_ROOT=${V85_ROOT},MOSAICKV_V85_SPEC=${PREREG},MOSAICKV_V85_SPEC_SHA256=${PREREG_SHA},MOSAICKV_V85_FROZEN=${FROZEN},MOSAICKV_V85_FROZEN_SHA256=${EXPECTED_FROZEN_SHA},MOSAICKV_V85_RUNTIME_PACKAGE_ROOT=${RUNTIME_PACKAGE_ROOT}" \
  slurm/choice_c_v85_quality_memory_a100.sbatch)
ANALYSIS_JOB=$(sbatch --parsable \
  --dependency="afterok:${HELDOUT_JOB}" \
  --export="ALL,MOSAICKV_V85_OUTPUT_ROOT=${V85_ROOT},MOSAICKV_V85_SPEC=${PREREG},MOSAICKV_V85_SPEC_SHA256=${PREREG_SHA},MOSAICKV_V85_FROZEN=${FROZEN},MOSAICKV_V85_FROZEN_SHA256=${EXPECTED_FROZEN_SHA},MOSAICKV_V85_PROTOCOL_PACKAGE_ROOT=${PROTOCOL_PACKAGE_ROOT}" \
  slurm/choice_c_v85_analyze.sbatch)

printf 'v85 untouched ledger: %s\n' "${LEDGER_JOB}"
printf 'v85 held-out GPU array: %s\n' "${HELDOUT_JOB}"
printf 'v85 paired analysis: %s\n' "${ANALYSIS_JOB}"
