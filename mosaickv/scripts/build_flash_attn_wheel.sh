#!/usr/bin/env bash
set -euo pipefail

# Build FlashAttention against the host GLIBC/CUDA stack.  The upstream
# setup.py otherwise downloads a prebuilt wheel whose GLIBC may be newer than
# the compute nodes.  All build/cache artifacts remain under scratch.
MOSAICKV_SCRATCH_ROOT="${MOSAICKV_SCRATCH_ROOT:-/scratch/djy8hg}"
MOSAICKV_ENV="${MOSAICKV_ENV:-${MOSAICKV_SCRATCH_ROOT}/env/mosaickv}"
MOSAICKV_CACHE="${MOSAICKV_CACHE:-${MOSAICKV_SCRATCH_ROOT}/cache/mosaickv}"
WHEEL_DIR="${1:-${MOSAICKV_CACHE}/flash-attn-wheelhouse-source}"
SOURCE_PARENT="${MOSAICKV_CACHE}/flash-attn-source"
SOURCE_ARCHIVE="${SOURCE_PARENT}/flash_attn-2.8.3.tar.gz"
SOURCE_ARCHIVE_SHA256="1e71dd64a9e0280e0447b8a0c2541bad4bf6ac65bdeaa2f90e51a9e57de0370d"
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_PATCH="${PROJECT_ROOT}/env/patches/flash-attn-2.8.3-a100-multimodal-inference.patch"

source "$(dirname "${BASH_SOURCE[0]}")/cache_env.sh"
export PYTHONNOUSERSITE=1
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS="${MAX_JOBS:-16}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-80}"
mkdir -p "${WHEEL_DIR}"

# A fresh replica may not have the common environment yet, while the
# FlashAttention source build imports the exact target PyTorch. Bootstrap only
# the pinned build prerequisites into the eventual common environment; the
# subsequent create_envs.sh --sync common reconciles every transitive package
# against the authoritative lock before any support claim.
if [[ ! -x "${MOSAICKV_ENV}/bin/python" ]]; then
  BASE_PYTHON=${MOSAICKV_PYTHON:-python3.11}
  "${BASE_PYTHON}" -m venv "${MOSAICKV_ENV}"
fi
if ! "${MOSAICKV_ENV}/bin/python" -c \
  'import torch; raise SystemExit(torch.__version__.split("+")[0] != "2.8.0" or torch.version.cuda != "12.8")'; then
  "${MOSAICKV_ENV}/bin/python" -m pip install \
    --cache-dir "${PIP_CACHE_DIR}" \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple \
    torch==2.8.0 \
    packaging==26.2 \
    psutil==7.2.2 \
    ninja==1.13.0 \
    setuptools==83.0.0
fi

if [[ ! -f "${SOURCE_PATCH}" ]]; then
  printf 'FlashAttention inference patch is missing: %s\n' "${SOURCE_PATCH}" >&2
  exit 2
fi
mkdir -p "${SOURCE_PARENT}"
if [[ ! -f "${SOURCE_ARCHIVE}" ]]; then
  "${MOSAICKV_ENV}/bin/python" -m pip download \
    flash-attn==2.8.3 \
    --no-deps \
    --no-binary flash-attn \
    --no-cache-dir \
    --dest "${SOURCE_PARENT}"
fi
if [[ ! -f "${SOURCE_ARCHIVE}" ]]; then
  SOURCE_ARCHIVE=$(find "${SOURCE_PARENT}" -maxdepth 1 -type f \
    -name 'flash*attn*2.8.3*.tar.gz' -print -quit)
fi
if [[ -z "${SOURCE_ARCHIVE}" || ! -f "${SOURCE_ARCHIVE}" ]]; then
  printf 'FlashAttention source archive is unavailable under %s\n' "${SOURCE_PARENT}" >&2
  exit 2
fi
actual_source_sha256=$(sha256sum "${SOURCE_ARCHIVE}" | cut -d' ' -f1)
if [[ "${actual_source_sha256}" != "${SOURCE_ARCHIVE_SHA256}" ]]; then
  printf 'FlashAttention source archive SHA mismatch: expected=%s actual=%s\n' \
    "${SOURCE_ARCHIVE_SHA256}" "${actual_source_sha256}" >&2
  exit 2
fi

# Always build from a fresh extraction. Reusing setup.py's local build/
# directory can silently link objects from a prior source-list specialization.
extract_root=$(mktemp -d "${SOURCE_PARENT}/build-source.XXXXXX")
cleanup_source() {
  rm -rf -- "${extract_root}"
}
trap cleanup_source EXIT
tar -xf "${SOURCE_ARCHIVE}" -C "${extract_root}"
SOURCE_DIR=$(find "${extract_root}" -mindepth 1 -maxdepth 1 -type d -print -quit)
if [[ -z "${SOURCE_DIR}" || ! -f "${SOURCE_DIR}/setup.py" ]]; then
  printf 'FlashAttention source archive had no valid top-level source directory\n' >&2
  exit 2
fi
patch --directory "${SOURCE_DIR}" --strip=1 --forward <"${SOURCE_PATCH}"
if ! grep -Fq '"cxx": compiler_c17_flag + ["-DFLASHATTENTION_DISABLE_BACKWARD"]' \
  "${SOURCE_DIR}/setup.py" || \
  ! grep -q 'sources=\[source for source in \[' "${SOURCE_DIR}/setup.py" || \
  ! grep -q 'MosaicKV inference wheel supports head dimensions <=96 or exactly 128' \
    "${SOURCE_DIR}/csrc/flash_attn/src/static_switch.h"; then
  printf 'FlashAttention source patch did not reach its complete expected state\n' >&2
  exit 2
fi

"${MOSAICKV_ENV}/bin/python" -m pip wheel \
  "${SOURCE_DIR}" \
  --no-build-isolation \
  --no-deps \
  --no-cache-dir \
  --wheel-dir "${WHEEL_DIR}"

wheel=$(find "${WHEEL_DIR}" -maxdepth 1 -type f -name 'flash_attn-2.8.3-*.whl' -print -quit)
if [[ -z "${wheel}" ]]; then
  printf 'FlashAttention source build produced no wheel in %s\n' "${WHEEL_DIR}" >&2
  exit 1
fi
(
  cd "${WHEEL_DIR}"
  sha256sum "$(basename "${wheel}")" >"$(basename "${wheel}").sha256"
)
