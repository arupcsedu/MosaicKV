#!/bin/bash
#SBATCH --job-name=mosaickv-env-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --partition=bii-gpu
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
ENV_DIR=${MOSAICKV_ENV_DIR:-/scratch/djy8hg/env/mosaickv}

export MOSAICKV_CACHE_ROOT="${CACHE_ROOT}"
# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  printf 'Environment is absent; create it explicitly before submitting: %s\n' "${ENV_DIR}" >&2
  exit 2
fi

srun "${ENV_DIR}/bin/python" "${PROJECT_ROOT}/scripts/verify_envs.py" \
  --environment common \
  --lock "${PROJECT_ROOT}/env/common/requirements.lock" \
  --require-cuda
