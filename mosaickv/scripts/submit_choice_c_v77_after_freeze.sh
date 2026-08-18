#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
V76_ROOT=${MOSAICKV_V76_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v76-disjoint-ocr-calibration-v1}
V77_ROOT=${MOSAICKV_V77_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v77-untouched-heldout-quality-memory-v1}
V77_SPEC=${PACKAGE_ROOT}/configs/experiments/choice_c_v77_untouched_heldout_quality_memory.yaml
FROZEN=${MOSAICKV_V76_FROZEN:-${V76_ROOT}/analysis/frozen_candidate.yaml}

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"
if [[ ! -f "${FROZEN}" ]]; then
  printf 'v77 submission blocked: passing v76 freeze does not exist: %s\n' "${FROZEN}" >&2
  exit 2
fi

LEDGER_JOB=$(sbatch --parsable \
  --export="ALL,MOSAICKV_V76_OUTPUT_ROOT=${V76_ROOT},MOSAICKV_V77_OUTPUT_ROOT=${V77_ROOT},MOSAICKV_V76_FROZEN=${FROZEN}" \
  slurm/choice_c_v77_materialize_heldout.sbatch)
HELDOUT_JOB=$(sbatch --parsable \
  --dependency="afterok:${LEDGER_JOB}" \
  --export="ALL,MOSAICKV_V77_OUTPUT_ROOT=${V77_ROOT},MOSAICKV_V77_SPEC=${V77_SPEC}" \
  slurm/choice_c_v77_quality_memory_a100.sbatch)
ANALYSIS_JOB=$(sbatch --parsable \
  --dependency="afterok:${HELDOUT_JOB}" \
  --export="ALL,MOSAICKV_V77_OUTPUT_ROOT=${V77_ROOT},MOSAICKV_V77_SPEC=${V77_SPEC}" \
  slurm/choice_c_v77_analyze.sbatch)

printf 'v77 untouched held-out ledger: %s\n' "${LEDGER_JOB}"
printf 'v77 untouched held-out GPU: %s\n' "${HELDOUT_JOB}"
printf 'v77 analysis: %s\n' "${ANALYSIS_JOB}"
