#!/bin/bash
#SBATCH --job-name=mosaickv-three-tier
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=00:10:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
TORCH_PYTHON=${MOSAICKV_TORCH_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
export MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"

if [[ ! -x "${TORCH_PYTHON}" ]]; then
  printf 'Torch environment does not exist or is not executable: %s\n' "${TORCH_PYTHON}" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_ROOT}"
srun "${TORCH_PYTHON}" scripts/verify_three_tier.py
