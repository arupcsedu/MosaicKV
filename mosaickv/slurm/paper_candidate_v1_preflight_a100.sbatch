#!/bin/bash
#SBATCH --job-name=mkv-paper-v1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=03:30:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-2%3
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/paper-v1-%A_%a.log
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
PROTOCOL=${PACKAGE_ROOT}/configs/experiments/paper_quality_memory_v1.yaml
: "${MOSAICKV_PAPER_CANDIDATE_EXPECTED_GIT_SHA:?immutable source SHA is required}"
: "${MOSAICKV_PAPER_CANDIDATE_EXPECTED_PROTOCOL_SHA:?protocol SHA is required}"
: "${MOSAICKV_PAPER_CANDIDATE_OUTPUT_ROOT:?preflight output root is required}"

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
export PYTHONPATH="${PACKAGE_ROOT}/src:${PACKAGE_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"
[[ "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)" == "${MOSAICKV_PAPER_CANDIDATE_EXPECTED_GIT_SHA}" ]]
[[ "$(sha256sum "${PROTOCOL}" | cut -d ' ' -f 1)" == "${MOSAICKV_PAPER_CANDIDATE_EXPECTED_PROTOCOL_SHA}" ]]
"${PYTHON}" scripts/prepare_paper_candidate_v1_preflight.py \
  --output-root "${MOSAICKV_PAPER_CANDIDATE_OUTPUT_ROOT}" \
  --validate-only >/dev/null
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,pstate --format=csv,noheader
srun "${PYTHON}" scripts/run_paper_candidate_v1_preflight_worker.py \
  --protocol "${PROTOCOL}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --output-root "${MOSAICKV_PAPER_CANDIDATE_OUTPUT_ROOT}"
