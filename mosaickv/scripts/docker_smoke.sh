#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"

ARGS=(
  --environment common
  --lock "${PROJECT_ROOT}/env/common/requirements.lock"
)
if [[ "${MOSAICKV_REQUIRE_CUDA:-0}" == 1 ]]; then
  ARGS+=(--require-cuda)
fi

python "${PROJECT_ROOT}/scripts/verify_envs.py" "${ARGS[@]}"
mosaickv doctor --json
mosaickv smoke --json
