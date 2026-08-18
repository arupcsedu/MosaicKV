#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
V74_ROOT=${MOSAICKV_V74_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v74-label-blind-calibration-v6}
V75_ROOT=${MOSAICKV_V75_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v75-frozen-heldout-quality-memory-v6}
V75_SPEC=${PACKAGE_ROOT}/configs/experiments/choice_c_v75_frozen_heldout_quality_memory.yaml
FROZEN=${MOSAICKV_V74_FROZEN:-${V74_ROOT}/analysis/frozen_candidate.yaml}

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"
if [[ ! -f "${FROZEN}" ]]; then
  printf 'held-out submission blocked: passing v74 freeze does not exist: %s\n' "${FROZEN}" >&2
  exit 2
fi

HELDOUT_LEDGER_JOB=$(sbatch --parsable \
  --export="ALL,MOSAICKV_V74_OUTPUT_ROOT=${V74_ROOT},MOSAICKV_V75_OUTPUT_ROOT=${V75_ROOT},MOSAICKV_V74_FROZEN=${FROZEN}" \
  slurm/choice_c_v75_materialize_heldout.sbatch)
HELDOUT_JOB=$(sbatch --parsable \
  --dependency="afterok:${HELDOUT_LEDGER_JOB}" \
  --export="ALL,MOSAICKV_V75_OUTPUT_ROOT=${V75_ROOT},MOSAICKV_V75_SPEC=${V75_SPEC}" \
  slurm/choice_c_v75_quality_memory_a100.sbatch)
ANALYSIS_JOB=$(sbatch --parsable \
  --dependency="afterok:${HELDOUT_JOB}" \
  --export="ALL,MOSAICKV_V75_OUTPUT_ROOT=${V75_ROOT},MOSAICKV_V75_SPEC=${V75_SPEC}" \
  slurm/choice_c_v75_analyze.sbatch)

printf 'v75 held-out ledger: %s\n' "${HELDOUT_LEDGER_JOB}"
printf 'v75 held-out GPU: %s\n' "${HELDOUT_JOB}"
printf 'v75 analysis: %s\n' "${ANALYSIS_JOB}"
