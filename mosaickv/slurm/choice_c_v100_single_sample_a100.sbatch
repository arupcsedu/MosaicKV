#!/bin/bash
#SBATCH --job-name=mkv-v100-debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/v100-debug-%j.log
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
export MOSAICKV_ENABLE_V100_GPU_DEBUG=1
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"
[[ "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)" == "${MOSAICKV_V100_EXPECTED_GIT_SHA}" ]]
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,pstate,clocks.sm,clocks.mem,power.draw --format=csv,noheader
srun "${PYTHON}" scripts/run_choice_c_v100_gpu_debug.py \
  --expected-git-sha "${MOSAICKV_V100_EXPECTED_GIT_SHA}" \
  --max-new-tokens 4 \
  --output-root "${OUTPUT_ROOT}/single_sample"
