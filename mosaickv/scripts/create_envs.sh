#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_ROOT=${MOSAICKV_ENV_ROOT:-/scratch/djy8hg/env}
CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
BASE_PYTHON=${MOSAICKV_PYTHON:-python3}

usage() {
  printf 'Usage: %s {hf|vllm|sglang|mock|all} [...]\n' "$0" >&2
  printf 'Environment overrides: MOSAICKV_ENV_ROOT, MOSAICKV_CACHE_ROOT, MOSAICKV_PYTHON\n' >&2
}

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

if ! "${BASE_PYTHON}" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 11))'; then
  printf 'MOSAICKV_PYTHON must be CPython 3.11: %s\n' "${BASE_PYTHON}" >&2
  exit 2
fi

case "${CACHE_ROOT}/" in
  "${HOME}/"*)
    printf 'MOSAICKV_CACHE_ROOT must be outside the home directory: %s\n' "${CACHE_ROOT}" >&2
    exit 2
    ;;
esac

export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers"
export TORCH_HOME="${CACHE_ROOT}/torch"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export SGLANG_CACHE_DIR="${CACHE_ROOT}/sglang"

mkdir -p \
  "${ENV_ROOT}" \
  "${PIP_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${TORCH_HOME}" \
  "${VLLM_CACHE_ROOT}" \
  "${SGLANG_CACHE_DIR}"

if [[ -n "${HF_TOKEN:-}" ]]; then
  printf 'HF_TOKEN is present in the process environment; it will not be written to disk.\n'
fi

PROFILES=()
add_profile() {
  local candidate=$1
  local existing
  for existing in "${PROFILES[@]}"; do
    if [[ "${existing}" == "${candidate}" ]]; then
      return
    fi
  done
  PROFILES+=("${candidate}")
}

for requested in "$@"; do
  case "${requested}" in
    all)
      add_profile hf
      add_profile vllm
      add_profile sglang
      add_profile mock
      ;;
    hf|vllm|sglang|mock)
      add_profile "${requested}"
      ;;
    *)
      printf 'Unknown environment: %s\n' "${requested}" >&2
      usage
      exit 2
      ;;
  esac
done

for profile in "${PROFILES[@]}"; do
  env_dir="${ENV_ROOT}/mosaickv_${profile}"
  lock_file="${PROJECT_ROOT}/env/${profile}/requirements.lock"
  if [[ -e "${env_dir}" ]]; then
    printf 'Refusing to overwrite existing environment: %s\n' "${env_dir}" >&2
    exit 2
  fi
  if [[ ! -f "${lock_file}" ]]; then
    printf 'Missing lock file: %s\n' "${lock_file}" >&2
    exit 2
  fi

  printf 'Creating %s environment at %s\n' "${profile}" "${env_dir}"
  "${BASE_PYTHON}" -m venv "${env_dir}"
  env_python="${env_dir}/bin/python"

  if [[ "${profile}" == hf ]]; then
    # FlashAttention-2 metadata imports torch during its source build. Install its
    # exactly pinned prerequisites first, then enforce the complete lock.
    "${env_python}" -m pip install \
      'torch==2.11.0' \
      'packaging==26.2' \
      'psutil==7.2.2' \
      'ninja==1.13.0' \
      'wheel==0.47.0'
    MAX_JOBS=${MAX_JOBS:-4} "${env_python}" -m pip install \
      --no-build-isolation -r "${lock_file}"
  else
    "${env_python}" -m pip install -r "${lock_file}"
  fi

  "${env_python}" -m pip install --no-deps --editable "${PROJECT_ROOT}"
  "${env_python}" -m pip check
  "${env_python}" "${PROJECT_ROOT}/scripts/verify_envs.py" \
    --environment "${profile}" --lock "${lock_file}"
  printf '%s created. Backend support remains unverified until CUDA smoke passes.\n' "${profile}"
done
