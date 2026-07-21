#!/bin/bash
#SBATCH --job-name=mosaickv-vllm-fullkv
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
VLLM_PYTHON=${MOSAICKV_VLLM_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
OUTPUT_ROOT=${MOSAICKV_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/vllm-fullkv}
CONFIG=${MOSAICKV_VLLM_CONFIG:-${PROJECT_ROOT}/configs/vllm_fullkv.yaml}
RUN_ID=${MOSAICKV_RUN_ID:-vllm-fullkv-${SLURM_JOB_ID}}

export MOSAICKV_CACHE_ROOT="${CACHE_ROOT}"
# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"

if [[ ! -x "${VLLM_PYTHON}" ]]; then
  printf 'Pinned vLLM environment is absent: %s\n' "${VLLM_PYTHON}" >&2
  printf 'Create it explicitly with mosaickv/scripts/create_envs.sh common\n' >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TORCH_HOME="${CACHE_ROOT}/torch"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${MOSAICKV_ALLOW_MODEL_DOWNLOAD:-0}" != 1 ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

ARGS=(
  evaluate
  --config "${CONFIG}"
  --task synthetic_smoke
  --subset-size 1
  --run-id "${RUN_ID}"
  --output-dir "${OUTPUT_ROOT}"
  --trace-directory "${OUTPUT_ROOT}/traces"
  --cache-probe-repeats "${MOSAICKV_CACHE_PROBE_REPEATS:-2}"
  --gpu-memory-utilization "${MOSAICKV_GPU_MEMORY_UTILIZATION:-0.8}"
  --vllm-max-model-len "${MOSAICKV_VLLM_MAX_MODEL_LEN:-4096}"
  --tensor-parallel-size "${MOSAICKV_TENSOR_PARALLEL_SIZE:-1}"
  --json
)
if [[ "${MOSAICKV_ALLOW_MODEL_DOWNLOAD:-0}" != 1 ]]; then
  ARGS+=(--local-files-only)
fi

srun "${VLLM_PYTHON}" -m mosaickv.cli "${ARGS[@]}"

TRACE=$(find "${OUTPUT_ROOT}/traces/${RUN_ID}" -maxdepth 1 -name '*.json' -print -quit)
if [[ -z "${TRACE}" ]]; then
  printf 'No vLLM trace was produced for run %s\n' "${RUN_ID}" >&2
  exit 1
fi
srun "${VLLM_PYTHON}" scripts/verify_vllm_backend.py \
  --trace "${TRACE}" \
  --require-gpu-measurements
