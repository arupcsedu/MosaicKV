#!/bin/bash
set -euo pipefail

# This launcher never creates sample identities. It accepts three reviewed,
# immutable master ledgers, creates the 12 paired shards/24 backend jobs, and
# caps concurrent GPU use at four one-A100 workers.

REPOSITORY=/scratch/djy8hg/workdir/MosaicKV
PROJECT_ROOT=${REPOSITORY}/mosaickv
PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python3}
PROTOCOL=${PROJECT_ROOT}/configs/experiments/paper_quality_hf_vllm_llava15_v2.yaml
: "${MOSAICKV_PAPER_LEDGER_ROOT:?set the reviewed master-ledger directory}"
: "${MOSAICKV_PAPER_INPUT_ROOT:?set a new immutable campaign-input directory}"
: "${MOSAICKV_PAPER_OUTPUT_ROOT:?set a new result directory under /scratch/djy8hg}"
: "${MOSAICKV_PAPER_ANALYSIS_ROOT:?set a new analysis directory under /scratch/djy8hg}"

export MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
# shellcheck source=cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"
cd "${PROJECT_ROOT}"

"${PYTHON}" scripts/prepare_paper_quality_hf_vllm_v2.py prepare \
  --protocol "${PROTOCOL}" \
  --ledger "mmstar=${MOSAICKV_PAPER_LEDGER_ROOT}/mmstar.json" \
  --ledger "textvqa=${MOSAICKV_PAPER_LEDGER_ROOT}/textvqa.json" \
  --ledger "docvqa=${MOSAICKV_PAPER_LEDGER_ROOT}/docvqa.json" \
  --destination "${MOSAICKV_PAPER_INPUT_ROOT}" \
  --output-root "${MOSAICKV_PAPER_OUTPUT_ROOT}"

INDEX=${MOSAICKV_PAPER_INPUT_ROOT}/jobs.jsonl
JOB_COUNT=$(wc -l <"${INDEX}")
if [[ "${JOB_COUNT}" != 24 ]]; then
  printf 'Expected 24 backend jobs, found %s\n' "${JOB_COUNT}" >&2
  exit 2
fi

GPU_JOB_ID=$(sbatch --parsable --array=0-23%4 \
  --export="ALL,MOSAICKV_PAPER_INDEX=${INDEX}" \
  "${PROJECT_ROOT}/slurm/paper_quality_hf_vllm_llava15_v2.sbatch")
ANALYSIS_JOB_ID=$(sbatch --parsable --dependency="afterok:${GPU_JOB_ID}" \
  --export="ALL,MOSAICKV_PAPER_INDEX=${INDEX},MOSAICKV_PAPER_ANALYSIS_ROOT=${MOSAICKV_PAPER_ANALYSIS_ROOT}" \
  "${PROJECT_ROOT}/slurm/analyze_paper_quality_hf_vllm_llava15_v2.sbatch")
printf 'gpu_array_job_id=%s\nanalysis_job_id=%s\n' "${GPU_JOB_ID}" "${ANALYSIS_JOB_ID}"
