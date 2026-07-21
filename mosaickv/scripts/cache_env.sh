#!/usr/bin/env bash
# Source this file before every MosaicKV setup, test, evaluation, or server run.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'Source this file instead of executing it: source %s\n' "$0" >&2
  exit 2
fi

MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
if [[ "${MOSAICKV_CACHE_ROOT}" != /* ]]; then
  printf 'MOSAICKV_CACHE_ROOT must be absolute: %s\n' "${MOSAICKV_CACHE_ROOT}" >&2
  return 2
fi
case "${MOSAICKV_CACHE_ROOT}/" in
  "${HOME}/"*)
    printf 'MOSAICKV_CACHE_ROOT must not be in the home directory: %s\n' \
      "${MOSAICKV_CACHE_ROOT}" >&2
    return 2
    ;;
esac

export MOSAICKV_CACHE_ROOT
export PYTHONNOUSERSITE=1
export PIP_CONFIG_FILE=/dev/null
export PIP_CACHE_DIR="${MOSAICKV_CACHE_ROOT}/pip"
export UV_CACHE_DIR="${MOSAICKV_CACHE_ROOT}/uv"
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300}
export UV_HTTP_RETRIES=${UV_HTTP_RETRIES:-10}
export UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-4}
export XDG_CACHE_HOME="${MOSAICKV_CACHE_ROOT}/xdg"
export HF_HOME="${MOSAICKV_CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_ASSETS_CACHE="${HF_HOME}/assets"
export HF_DATASETS_CACHE="${MOSAICKV_CACHE_ROOT}/datasets"
export TRANSFORMERS_CACHE="${MOSAICKV_CACHE_ROOT}/transformers"
export TORCH_HOME="${MOSAICKV_CACHE_ROOT}/torch"
export TORCHINDUCTOR_CACHE_DIR="${MOSAICKV_CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${MOSAICKV_CACHE_ROOT}/triton"
export NUMBA_CACHE_DIR="${MOSAICKV_CACHE_ROOT}/numba"
export CUDA_CACHE_PATH="${MOSAICKV_CACHE_ROOT}/cuda"
export FLASHINFER_WORKSPACE_BASE="${MOSAICKV_CACHE_ROOT}/flashinfer"
export VLLM_CACHE_ROOT="${MOSAICKV_CACHE_ROOT}/vllm"
export SGLANG_CACHE_DIR="${MOSAICKV_CACHE_ROOT}/sglang"
export PRE_COMMIT_HOME="${MOSAICKV_CACHE_ROOT}/pre-commit"
export MPLCONFIGDIR="${MOSAICKV_CACHE_ROOT}/matplotlib"
export WANDB_CACHE_DIR="${MOSAICKV_CACHE_ROOT}/wandb/cache"
export WANDB_DATA_DIR="${MOSAICKV_CACHE_ROOT}/wandb/data"
export RAY_TMPDIR="${MOSAICKV_CACHE_ROOT}/ray"
export TMPDIR="${MOSAICKV_CACHE_ROOT}/tmp"

# The PyTorch CUDA wheels keep NVRTC in the environment's site-packages tree.
# Native SGLang kernels use the dynamic loader directly, so expose that exact
# locked library without depending on a cluster-wide CUDA module.
MOSAICKV_ENV_DIR=${MOSAICKV_ENV_DIR:-${VIRTUAL_ENV:-/scratch/djy8hg/env/mosaickv}}
MOSAICKV_NVRTC_LIB="${MOSAICKV_ENV_DIR}/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib"
export MOSAICKV_ENV_DIR
if [[ -d "${MOSAICKV_NVRTC_LIB}" ]]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${MOSAICKV_NVRTC_LIB}:"*) ;;
    *) export LD_LIBRARY_PATH="${MOSAICKV_NVRTC_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
fi

# A value of "0" is truthy to pip. Remove an inherited setting so the explicit
# scratch cache above is honored even on hosts with unusual shell profiles.
unset PIP_NO_CACHE_DIR

mkdir -p \
  "${PIP_CACHE_DIR}" \
  "${UV_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${HF_HUB_CACHE}" \
  "${HF_ASSETS_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${TORCH_HOME}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${NUMBA_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}" \
  "${FLASHINFER_WORKSPACE_BASE}" \
  "${VLLM_CACHE_ROOT}" \
  "${SGLANG_CACHE_DIR}" \
  "${PRE_COMMIT_HOME}" \
  "${MPLCONFIGDIR}" \
  "${WANDB_CACHE_DIR}" \
  "${WANDB_DATA_DIR}" \
  "${RAY_TMPDIR}" \
  "${TMPDIR}"
