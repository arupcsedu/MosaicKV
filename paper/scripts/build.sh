#!/usr/bin/env bash
set -euo pipefail
PAPER_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "${PAPER_DIR}/../mosaickv/scripts/cache_env.sh"
export TEXMFVAR="${MOSAICKV_CACHE_ROOT}/texlive/var"
export TEXMFCONFIG="${MOSAICKV_CACHE_ROOT}/texlive/config"
mkdir -p "${TEXMFVAR}" "${TEXMFCONFIG}"
if ! command -v latexmk >/dev/null 2>&1 && type module >/dev/null 2>&1; then
  module load texlive/2025
fi
command -v latexmk >/dev/null 2>&1 || {
  printf '%s\n' 'TeX Live with latexmk is required; no packages were installed.' >&2
  exit 1
}
cd "${PAPER_DIR}"
make "${@:-all}"
