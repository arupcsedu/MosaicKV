#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_ROOT=${MOSAICKV_ENV_ROOT:-/scratch/djy8hg/env}
ENV_DIR=${MOSAICKV_ENV_DIR:-${ENV_ROOT}/mosaickv}
BASE_PYTHON=${MOSAICKV_PYTHON:-python3}
MAMBA_BIN=${MOSAICKV_MAMBA:-/scratch/djy8hg/tools/micromamba-bin/micromamba}
SYNC_EXISTING=false

usage() {
  printf 'Usage: %s [--sync] common\n' "$0" >&2
}

if [[ "${1:-}" == "--sync" ]]; then
  SYNC_EXISTING=true
  shift
fi
if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi
if [[ "$1" != "common" ]]; then
  usage
  exit 2
fi

# shellcheck source=cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"

LOCK_FILE="${PROJECT_ROOT}/env/common/requirements.lock"
if [[ ! -f "${LOCK_FILE}" ]]; then
  printf 'Missing common lock: %s\n' "${LOCK_FILE}" >&2
  exit 2
fi

if [[ -e "${ENV_DIR}" ]]; then
  if [[ "${SYNC_EXISTING}" != true ]]; then
    printf 'Environment exists; pass --sync to reconcile it explicitly: %s\n' \
      "${ENV_DIR}" >&2
    exit 2
  fi
else
  if [[ -x "${MAMBA_BIN}" ]]; then
    export MAMBA_ROOT_PREFIX="${MOSAICKV_CACHE_ROOT}/micromamba"
    "${MAMBA_BIN}" create --yes --prefix "${ENV_DIR}" python=3.11 pip
  else
    if ! "${BASE_PYTHON}" -c \
      'import sys; raise SystemExit(sys.version_info[:2] != (3, 11))'; then
      printf 'A healthy CPython 3.11 or micromamba is required.\n' >&2
      exit 2
    fi
    "${BASE_PYTHON}" -m venv "${ENV_DIR}"
  fi
fi

ENV_PYTHON="${ENV_DIR}/bin/python"
if ! "${ENV_PYTHON}" -c \
  'import encodings, sys; raise SystemExit(sys.version_info[:2] != (3, 11))'; then
  printf 'Common environment does not contain a healthy CPython 3.11: %s\n' \
    "${ENV_PYTHON}" >&2
  exit 2
fi

# uv is a bootstrap tool and is also present in the common lock. Bootstrapping
# it is the only install performed outside uv's exact synchronization.
if [[ ! -x "${ENV_DIR}/bin/uv" ]]; then
  "${ENV_PYTHON}" -m pip install 'uv==0.11.29'
fi
FLASH_WHEEL_DIR="${MOSAICKV_CACHE_ROOT}/flash-attn-wheelhouse-source"
FLASH_WHEEL_PATH=""
SYNC_ARGS=(
  pip sync
  --python "${ENV_PYTHON}"
  --cache-dir "${UV_CACHE_DIR}"
)
if compgen -G "${FLASH_WHEEL_DIR}/flash_attn-2.8.3-*.whl" >/dev/null; then
  # Prefer the wheel compiled by scripts/build_flash_attn_wheel.sh against
  # this host GLIBC/CUDA stack. The upstream setup script otherwise fetches a
  # binary wheel that can be incompatible with older compute-node GLIBC.
  SYNC_ARGS+=(--find-links "${FLASH_WHEEL_DIR}")
  FLASH_WHEEL_PATH=$(compgen -G "${FLASH_WHEEL_DIR}/flash_attn-2.8.3-*.whl" | head -n 1)
else
  printf 'FlashAttention requires a host-compatible wheel or nvcc. Build it on a GPU node:\n' >&2
  printf '  sbatch mosaickv/slurm/flash_attn_wheel.sbatch\n' >&2
  exit 2
fi
"${ENV_DIR}/bin/uv" "${SYNC_ARGS[@]}" "${LOCK_FILE}"
if [[ -n "${FLASH_WHEEL_PATH}" ]]; then
  # Equal-version candidates from an index and --find-links need not have the
  # same ABI. Force the audited host-built artifact after synchronization.
  if [[ ! -f "${FLASH_WHEEL_PATH}.sha256" ]]; then
    printf 'FlashAttention wheel checksum is absent: %s.sha256\n' \
      "${FLASH_WHEEL_PATH}" >&2
    exit 2
  fi
  (
    cd "${FLASH_WHEEL_DIR}"
    sha256sum -c "$(basename "${FLASH_WHEEL_PATH}").sha256"
  )
  "${ENV_DIR}/bin/uv" pip install \
    --python "${ENV_PYTHON}" \
    --cache-dir "${UV_CACHE_DIR}" \
    --reinstall --no-deps "${FLASH_WHEEL_PATH}"
fi
"${ENV_DIR}/bin/uv" pip install \
  --python "${ENV_PYTHON}" \
  --cache-dir "${UV_CACHE_DIR}" \
  --no-deps --editable "${PROJECT_ROOT}"
"${ENV_PYTHON}" -m pip check

printf 'Common environment synchronized at %s. Imports and GPU support remain ' "${ENV_DIR}"
printf 'unverified until the clean-tree Slurm environment smoke passes.\n'
