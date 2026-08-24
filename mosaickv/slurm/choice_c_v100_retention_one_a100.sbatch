#!/bin/bash
#SBATCH --job-name=mkv-v100-r1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/v100-r1-%j.log
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
: "${MOSAICKV_V100_EXPECTED_GIT_SHA:?expected immutable source SHA is required}"
OUTPUT_ROOT=${MOSAICKV_V100_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v100-gpu-debug-${MOSAICKV_V100_EXPECTED_GIT_SHA:0:12}}

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
export PYTHONPATH="${PACKAGE_ROOT}/src:${PACKAGE_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"
[[ "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)" == "${MOSAICKV_V100_EXPECTED_GIT_SHA}" ]]
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,pstate,clocks.sm,clocks.mem,power.draw --format=csv,noheader
mkdir -p "${OUTPUT_ROOT}/retention_one"
srun "${PYTHON}" scripts/verify_hf_runtime.py \
  --model-id llava-hf/llava-1.5-7b-hf \
  --block-size 4 \
  --max-new-tokens 4 \
  --logit-atol 1e-4 \
  --output "${OUTPUT_ROOT}/retention_one/report.json"
