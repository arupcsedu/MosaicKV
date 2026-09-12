#!/bin/bash
set -euo pipefail

REPOSITORY_ROOT=$(git rev-parse --show-toplevel)
PROJECT_ROOT=${REPOSITORY_ROOT}/mosaickv
SOURCE_SHA=$(git -C "${REPOSITORY_ROOT}" rev-parse HEAD)
CAMPAIGN_ROOT=${MOSAICKV_CAMPAIGN_ROOT:-/scratch/djy8hg/runs/mosaickv/baseline-fidelity-review-${SOURCE_SHA:0:10}}
INPUT_ROOT=${CAMPAIGN_ROOT}/inputs
OUTPUT_ROOT=${CAMPAIGN_ROOT}/results
PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"

"${PYTHON}" "${PROJECT_ROOT}/scripts/prepare_baseline_fidelity_review.py" \
  --protocol "${PROJECT_ROOT}/configs/experiments/baseline_fidelity_review_v1.yaml" \
  --destination "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --repository "${REPOSITORY_ROOT}"

PREFLIGHT_JOB=$(sbatch --parsable \
  --array=0,8,16,24%4 \
  --export=ALL,MOSAICKV_SOURCE_ROOT="${REPOSITORY_ROOT}",MOSAICKV_JOB_INDEX="${INPUT_ROOT}/jobs.jsonl" \
  "${PROJECT_ROOT}/slurm/baseline_fidelity_review.sbatch")
ARRAY_JOB=$(sbatch --parsable \
  --dependency=afterok:${PREFLIGHT_JOB} \
  --array=1-7,9-15,17-23,25-31%4 \
  --export=ALL,MOSAICKV_SOURCE_ROOT="${REPOSITORY_ROOT}",MOSAICKV_JOB_INDEX="${INPUT_ROOT}/jobs.jsonl" \
  "${PROJECT_ROOT}/slurm/baseline_fidelity_review.sbatch")
ANALYSIS_JOB=$(sbatch --parsable \
  --dependency=afterok:${ARRAY_JOB} \
  --export=ALL,MOSAICKV_SOURCE_ROOT="${REPOSITORY_ROOT}",MOSAICKV_OUTPUT_ROOT="${OUTPUT_ROOT}" \
  "${PROJECT_ROOT}/slurm/baseline_fidelity_review_analysis.sbatch")
printf 'campaign_root=%s\npreflight_job=%s\narray_job=%s\nanalysis_job=%s\n' \
  "${CAMPAIGN_ROOT}" "${PREFLIGHT_JOB}" "${ARRAY_JOB}" "${ANALYSIS_JOB}"
