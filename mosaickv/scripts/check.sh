#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"
cd "${PROJECT_ROOT}"

PYTHON_BIN=${PYTHON_BIN:-/scratch/djy8hg/env/mosaickv/bin/python}

CHECK_PATHS=(
  src
  tests
  scripts
)

"${PYTHON_BIN}" -m ruff check "${CHECK_PATHS[@]}"
"${PYTHON_BIN}" -m ruff format --check "${CHECK_PATHS[@]}"
"${PYTHON_BIN}" -m mypy "${CHECK_PATHS[@]}"
"${PYTHON_BIN}" -m pytest -q
