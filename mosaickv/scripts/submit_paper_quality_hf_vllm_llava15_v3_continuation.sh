#!/bin/bash
set -euo pipefail

REPOSITORY=/scratch/djy8hg/workdir/MosaicKV
PROJECT_ROOT=${REPOSITORY}/mosaickv
PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python3}
PROTOCOL=${PROJECT_ROOT}/configs/experiments/paper_quality_hf_vllm_llava15_v3_continuation.yaml
: "${MOSAICKV_PAPER_FULL_LEDGER_ROOT:?missing V3 full-ledger root}"
: "${MOSAICKV_PAPER_INPUT_ROOT:?missing V3 immutable input root}"
: "${MOSAICKV_PAPER_OUTPUT_ROOT:?missing V3 result root}"
: "${MOSAICKV_PAPER_ANALYSIS_ROOT:?missing V3 analysis root}"

export MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
# shellcheck source=cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"
cd "${PROJECT_ROOT}"

"${PYTHON}" scripts/prepare_paper_quality_hf_vllm_v3_continuation.py \
  --protocol "${PROTOCOL}" \
  --full-ledger "mmstar=${MOSAICKV_PAPER_FULL_LEDGER_ROOT}/mmstar.json" \
  --full-ledger "textvqa=${MOSAICKV_PAPER_FULL_LEDGER_ROOT}/textvqa.json" \
  --full-ledger "docvqa=${MOSAICKV_PAPER_FULL_LEDGER_ROOT}/docvqa.json" \
  --destination "${MOSAICKV_PAPER_INPUT_ROOT}" \
  --output-root "${MOSAICKV_PAPER_OUTPUT_ROOT}"

INDEX=${MOSAICKV_PAPER_INPUT_ROOT}/jobs.jsonl
JOB_COUNT=$(wc -l <"${INDEX}")
if [[ "${JOB_COUNT}" != 260 ]]; then
  printf 'Expected 260 continuation jobs, found %s\n' "${JOB_COUNT}" >&2
  exit 2
fi
GPU_JOB_ID=$(sbatch --parsable --array=0-259%6 \
  --export="ALL,MOSAICKV_PAPER_INDEX=${INDEX}" \
  "${PROJECT_ROOT}/slurm/paper_quality_hf_vllm_llava15_v2.sbatch")
ANALYSIS_JOB_ID=$(sbatch --parsable --dependency="afterok:${GPU_JOB_ID}" \
  --export="ALL,MOSAICKV_PAPER_INDEX=${INDEX},MOSAICKV_PAPER_ANALYSIS_ROOT=${MOSAICKV_PAPER_ANALYSIS_ROOT}" \
  "${PROJECT_ROOT}/slurm/analyze_paper_quality_hf_vllm_llava15_v3_continuation.sbatch")
printf 'gpu_array_job_id=%s\nanalysis_job_id=%s\n' "${GPU_JOB_ID}" "${ANALYSIS_JOB_ID}"
