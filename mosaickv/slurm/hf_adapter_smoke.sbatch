#!/bin/bash
#SBATCH --job-name=mosaickv-hf-adapters
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
HF_PYTHON=${MOSAICKV_HF_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}

export MOSAICKV_CACHE_ROOT="${CACHE_ROOT}"
# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"

if [[ ! -x "${HF_PYTHON}" ]]; then
  printf 'HF environment does not exist or is not executable: %s\n' "${HF_PYTHON}" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export MOSAICKV_CACHE_ROOT="${CACHE_ROOT}"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers"
export TORCH_HOME="${CACHE_ROOT}/torch"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_ROOT}"

# Always run no-download, randomly initialized architecture parity first.
srun "${HF_PYTHON}" scripts/verify_hf_adapters.py

# A checkpoint ID opts into the authoritative test. Weights remain offline by
# default; explicitly export MOSAICKV_ALLOW_MODEL_DOWNLOAD=1 to permit fetching.
if [[ -n "${MOSAICKV_HF_MODEL_ID:-}" ]]; then
  CHECKPOINT_ARGS=(--model-id "${MOSAICKV_HF_MODEL_ID}")
  if [[ -n "${MOSAICKV_HF_REVISION:-}" ]]; then
    CHECKPOINT_ARGS+=(--revision "${MOSAICKV_HF_REVISION}")
  fi
  if [[ -n "${MOSAICKV_INTERNVL_PIXEL_VALUES:-}" ]]; then
    CHECKPOINT_ARGS+=(--internvl-pixel-values "${MOSAICKV_INTERNVL_PIXEL_VALUES}")
  fi
  if [[ "${MOSAICKV_ALLOW_MODEL_DOWNLOAD:-0}" == 1 ]]; then
    CHECKPOINT_ARGS+=(--allow-download)
  fi
  srun "${HF_PYTHON}" scripts/verify_hf_adapters.py "${CHECKPOINT_ARGS[@]}"
fi
