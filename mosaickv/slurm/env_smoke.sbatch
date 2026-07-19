#!/bin/bash
#SBATCH --job-name=mosaickv-env-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --partition=bii-gpu
#SBATCH --array=0-2
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
ENV_ROOT=${MOSAICKV_ENV_ROOT:-/scratch/djy8hg/env}
CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
PROFILES=(hf vllm sglang)
PROFILE=${PROFILES["${SLURM_ARRAY_TASK_ID}"]}
ENV_DIR="${ENV_ROOT}/mosaickv_${PROFILE}"

export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers"
export TORCH_HOME="${CACHE_ROOT}/torch"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export SGLANG_CACHE_DIR="${CACHE_ROOT}/sglang"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  printf 'Environment is absent; create it explicitly before submitting: %s\n' "${ENV_DIR}" >&2
  exit 2
fi

srun "${ENV_DIR}/bin/python" "${PROJECT_ROOT}/scripts/verify_envs.py" \
  --environment "${PROFILE}" \
  --lock "${PROJECT_ROOT}/env/${PROFILE}/requirements.lock" \
  --require-cuda
