#!/bin/bash
#SBATCH --job-name=mosaickv-native-gate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:30:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

set -euo pipefail

module load "${MOSAICKV_GCC_MODULE:-gcc/11.4.0}"

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
BACKEND=${MOSAICKV_NATIVE_BACKEND:?set MOSAICKV_NATIVE_BACKEND to vllm or sglang}
OUTPUT_ROOT=${MOSAICKV_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/native-${BACKEND}-${SLURM_JOB_ID}}

export MOSAICKV_CACHE_ROOT="${CACHE_ROOT}"
# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"

if [[ ! -x "${PYTHON}" ]]; then
  printf 'Common environment is absent: %s\n' "${PYTHON}" >&2
  exit 2
fi
if [[ "${BACKEND}" == vllm ]]; then
  FULL_CONFIG=${PROJECT_ROOT}/configs/vllm_fullkv.yaml
  IDENTITY_CONFIG=${PROJECT_ROOT}/configs/native_validation/vllm_qwen25vl3b_identity.yaml
  COMPRESSED_CONFIG=${PROJECT_ROOT}/configs/native_validation/vllm_qwen25vl3b_compressed.yaml
  BACKEND_ARGS=(
    --gpu-memory-utilization "${MOSAICKV_GPU_MEMORY_UTILIZATION:-0.78}"
    --vllm-max-model-len 4096
  )
elif [[ "${BACKEND}" == sglang ]]; then
  FULL_CONFIG=${PROJECT_ROOT}/configs/sglang_fullkv_3b.yaml
  IDENTITY_CONFIG=${PROJECT_ROOT}/configs/native_validation/sglang_qwen25vl3b_identity.yaml
  COMPRESSED_CONFIG=${PROJECT_ROOT}/configs/native_validation/sglang_qwen25vl3b_compressed.yaml
  BACKEND_ARGS=(
    --sglang-mem-fraction-static "${MOSAICKV_SGLANG_MEM_FRACTION_STATIC:-0.72}"
    --sglang-context-length 4096
    --sglang-startup-timeout 1800
  )
else
  printf 'Unsupported native validation backend: %s\n' "${BACKEND}" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TORCH_HOME="${CACHE_ROOT}/torch"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export SGLANG_CACHE_DIR="${CACHE_ROOT}/sglang"
export TVM_FFI_CACHE_DIR="${CACHE_ROOT}/tvm-ffi"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "${OUTPUT_ROOT}" "${TVM_FFI_CACHE_DIR}"
cd "${PROJECT_ROOT}"

run_gate() {
  local label=$1
  local config=$2
  shift 2
  local run_id="${BACKEND}-${label}-${SLURM_JOB_ID}"
  local root="${OUTPUT_ROOT}/${label}"
  srun "${PYTHON}" -m mosaickv.cli evaluate \
    --config "${config}" \
    --task synthetic_smoke \
    --subset-size 2 \
    --run-id "${run_id}" \
    --output-dir "${root}" \
    --trace-directory "${root}/traces" \
    --cache-probe-repeats 2 \
    --tensor-parallel-size 1 \
    --local-files-only \
    --json \
    "${BACKEND_ARGS[@]}" \
    "$@"
}

run_gate full "${FULL_CONFIG}"
run_gate identity "${IDENTITY_CONFIG}" --enable-mosaickv
run_gate compressed "${COMPRESSED_CONFIG}" --enable-mosaickv

srun "${PYTHON}" scripts/verify_native_backend.py \
  --backend "${BACKEND}" \
  --full-dir "${OUTPUT_ROOT}/full/traces/${BACKEND}-full-${SLURM_JOB_ID}" \
  --identity-dir "${OUTPUT_ROOT}/identity/traces/${BACKEND}-identity-${SLURM_JOB_ID}" \
  --compressed-dir "${OUTPUT_ROOT}/compressed/traces/${BACKEND}-compressed-${SLURM_JOB_ID}"

printf 'native_backend_gate_output=%s\n' "${OUTPUT_ROOT}"
