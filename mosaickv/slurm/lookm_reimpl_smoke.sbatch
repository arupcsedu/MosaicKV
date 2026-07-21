#!/bin/bash
#SBATCH --job-name=mosaickv-lookm-hf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=37
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
HF_PYTHON=${MOSAICKV_HF_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
OUTPUT_ROOT=${MOSAICKV_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/lookm/reimpl-hf7b}
RUN_ID=${MOSAICKV_RUN_ID:-lookm-reimpl-hf7b-${SLURM_JOB_ID}}

export MOSAICKV_CACHE_ROOT="${CACHE_ROOT}"
# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"

if [[ ! -x "${HF_PYTHON}" ]]; then
  printf 'HF environment does not exist or is not executable: %s\n' "${HF_PYTHON}" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TORCH_HOME="${CACHE_ROOT}/torch"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_ROOT}"
srun "${HF_PYTHON}" -m mosaickv.cli evaluate \
  --config configs/hf_lookm_reimpl.yaml \
  --task synthetic_smoke \
  --subset-size 1 \
  --output-dir "${OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --local-files-only \
  --json
