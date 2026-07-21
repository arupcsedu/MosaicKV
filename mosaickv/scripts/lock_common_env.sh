#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"

ENV_DIR=${MOSAICKV_ENV_DIR:-/scratch/djy8hg/env/mosaickv}
UV_BIN=${MOSAICKV_UV:-${ENV_DIR}/bin/uv}
OUTPUT="${MOSAICKV_CACHE_ROOT}/tmp/requirements.common.lock"

if [[ ! -x "${UV_BIN}" ]]; then
  printf 'uv is unavailable: %s\n' "${UV_BIN}" >&2
  exit 2
fi

"${UV_BIN}" pip compile "${PROJECT_ROOT}/env/common/requirements.in" \
  --python "${ENV_DIR}/bin/python" \
  --python-platform x86_64-manylinux_2_28 \
  --no-annotate \
  --emit-index-url \
  --custom-compile-command 'mosaickv/scripts/lock_common_env.sh' \
  --output-file "${OUTPUT}"

if ! cmp --silent "${OUTPUT}" "${PROJECT_ROOT}/env/common/requirements.lock"; then
  printf 'Common lock is stale. Inspect: diff -u %s %s\n' \
    "${PROJECT_ROOT}/env/common/requirements.lock" "${OUTPUT}" >&2
  exit 1
fi
printf 'common_lock_verified=%s\n' "$(sha256sum "${OUTPUT}" | cut -d' ' -f1)"
