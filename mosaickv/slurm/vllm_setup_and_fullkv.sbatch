#!/bin/bash
#SBATCH --job-name=mosaickv-common-setup-vllm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=37
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

# Explicit common-environment synchronization followed by the vLLM smoke.
# Ordinary benchmark submissions never mutate their Python environment.

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
ENV_DIR=${MOSAICKV_ENV_DIR:-/scratch/djy8hg/env/mosaickv}
export MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"

MOSAICKV_ENV_DIR="${ENV_DIR}" bash "${PROJECT_ROOT}/scripts/create_envs.sh" --sync common
srun "${ENV_DIR}/bin/python" "${PROJECT_ROOT}/scripts/verify_envs.py" \
  --environment common \
  --lock "${PROJECT_ROOT}/env/common/requirements.lock" \
  --require-cuda

MOSAICKV_VLLM_PYTHON="${ENV_DIR}/bin/python" \
  bash "${PROJECT_ROOT}/slurm/vllm_fullkv.sbatch"
