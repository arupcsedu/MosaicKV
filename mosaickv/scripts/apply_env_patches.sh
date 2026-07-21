#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_DIR=${1:-${MOSAICKV_ENV_DIR:-/scratch/djy8hg/env/mosaickv}}
ENV_PYTHON="${ENV_DIR}/bin/python"
EXPECTED_SGLANG=0.4.3.post4
PATCH_FILE="${PROJECT_ROOT}/env/patches/sglang-0.4.3.post4-transformers-4.49.patch"

if [[ ! -x "${ENV_PYTHON}" ]]; then
  printf 'Environment Python is unavailable: %s\n' "${ENV_PYTHON}" >&2
  exit 2
fi
ACTUAL_SGLANG=$("${ENV_PYTHON}" -c \
  'from importlib.metadata import version; print(version("sglang"))')
if [[ "${ACTUAL_SGLANG}" != "${EXPECTED_SGLANG}" ]]; then
  printf 'Refusing SGLang patch: expected %s, found %s\n' \
    "${EXPECTED_SGLANG}" "${ACTUAL_SGLANG}" >&2
  exit 2
fi

SITE_PACKAGES=$("${ENV_PYTHON}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
TARGET="${SITE_PACKAGES}/sglang/srt/configs/qwen2_5_vl_config.py"
ORIGINAL_IMAGE='AutoImageProcessor.register(Qwen2_5_VLConfig, None, Qwen2_5_VLImageProcessor, None)'
ORIGINAL_PROCESSOR='AutoProcessor.register(Qwen2_5_VLConfig, Qwen2_5_VLProcessor)'
PATCHED_IMAGE='Qwen2_5_VLConfig, None, Qwen2_5_VLImageProcessor, None, exist_ok=True'
PATCHED_PROCESSOR='Qwen2_5_VLConfig, Qwen2_5_VLProcessor, exist_ok=True'

if [[ ! -f "${TARGET}" || ! -f "${PATCH_FILE}" ]]; then
  printf 'SGLang target or patch is missing: %s %s\n' "${TARGET}" "${PATCH_FILE}" >&2
  exit 2
fi
if grep -Fqx "${ORIGINAL_IMAGE}" "${TARGET}" \
  && grep -Fqx "${ORIGINAL_PROCESSOR}" "${TARGET}"; then
  patch --batch --forward --directory="${SITE_PACKAGES}" -p0 < "${PATCH_FILE}"
elif grep -Fq "${PATCHED_IMAGE}" "${TARGET}" \
  && grep -Fq "${PATCHED_PROCESSOR}" "${TARGET}"; then
  printf 'environment_patch_already_applied=%s\n' "$(basename "${PATCH_FILE}")"
else
  printf 'Refusing SGLang patch: installed source does not match known states: %s\n' \
    "${TARGET}" >&2
  exit 2
fi

if ! grep -Fq "${PATCHED_IMAGE}" "${TARGET}" \
  || ! grep -Fq "${PATCHED_PROCESSOR}" "${TARGET}"; then
  printf 'SGLang patch verification failed: %s\n' "${TARGET}" >&2
  exit 1
fi
printf 'environment_patch_sha256=%s\n' "$(sha256sum "${PATCH_FILE}" | cut -d' ' -f1)"
