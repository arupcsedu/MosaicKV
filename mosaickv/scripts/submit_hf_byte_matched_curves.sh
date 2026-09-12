#!/bin/bash
set -euo pipefail

REPOSITORY=${MOSAICKV_SOURCE_ROOT:-$(git rev-parse --show-toplevel)}
PROJECT_ROOT=${REPOSITORY}/mosaickv
PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python3}
PROTOCOL=${PROJECT_ROOT}/configs/experiments/hf_byte_matched_curves_v1.yaml
: "${MOSAICKV_HF_BYTE_CURVE_INPUT_ROOT:?set a new immutable input directory}"
: "${MOSAICKV_HF_BYTE_CURVE_OUTPUT_ROOT:?set a new result directory under /scratch/djy8hg}"
: "${MOSAICKV_HF_BYTE_CURVE_ANALYSIS_ROOT:?set a new analysis directory under /scratch/djy8hg}"

export MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
# shellcheck source=cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"
cd "${PROJECT_ROOT}"

"${PYTHON}" scripts/prepare_hf_byte_matched_curves.py \
  --protocol "${PROTOCOL}" \
  --destination "${MOSAICKV_HF_BYTE_CURVE_INPUT_ROOT}" \
  --output-root "${MOSAICKV_HF_BYTE_CURVE_OUTPUT_ROOT}" \
  --repository "${REPOSITORY}"

INDEX=${MOSAICKV_HF_BYTE_CURVE_INPUT_ROOT}/jobs.jsonl
if [[ $(wc -l <"${INDEX}") != 48 ]]; then
  printf 'Expected 48 byte-curve jobs\n' >&2
  exit 2
fi

# Run one complete shard per task first. These are canonical campaign shards,
# not throwaway samples; successful outputs are reused by the final analyzer.
PREFLIGHT_JOB_ID=$(sbatch --parsable --array=0,16,32%3 \
  --export="ALL,MOSAICKV_SOURCE_ROOT=${REPOSITORY},MOSAICKV_HF_BYTE_CURVE_INDEX=${INDEX}" \
  "${PROJECT_ROOT}/slurm/hf_byte_matched_curves.sbatch")
REMAINDER_JOB_ID=$(sbatch --parsable --dependency="afterok:${PREFLIGHT_JOB_ID}" \
  --array=1-15,17-31,33-47%4 \
  --export="ALL,MOSAICKV_SOURCE_ROOT=${REPOSITORY},MOSAICKV_HF_BYTE_CURVE_INDEX=${INDEX}" \
  "${PROJECT_ROOT}/slurm/hf_byte_matched_curves.sbatch")
ANALYSIS_JOB_ID=$(sbatch --parsable --dependency="afterok:${REMAINDER_JOB_ID}" \
  --export="ALL,MOSAICKV_SOURCE_ROOT=${REPOSITORY},MOSAICKV_HF_BYTE_CURVE_INDEX=${INDEX},MOSAICKV_HF_BYTE_CURVE_ANALYSIS_ROOT=${MOSAICKV_HF_BYTE_CURVE_ANALYSIS_ROOT}" \
  "${PROJECT_ROOT}/slurm/hf_byte_matched_curves_analysis.sbatch")
printf 'preflight_array_job_id=%s\nremainder_array_job_id=%s\nanalysis_job_id=%s\n' \
  "${PREFLIGHT_JOB_ID}" "${REMAINDER_JOB_ID}" "${ANALYSIS_JOB_ID}"
