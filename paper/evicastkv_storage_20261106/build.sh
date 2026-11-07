#!/usr/bin/env bash
set -euo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY=$(cd -- "$PROJECT/../.." && pwd)
source "$REPOSITORY/mosaickv/scripts/cache_env.sh"
export TEXMFVAR="$MOSAICKV_CACHE_ROOT/texlive/var"
export TEXMFCONFIG="$MOSAICKV_CACHE_ROOT/texlive/config"
export TEXINPUTS="$PROJECT:$PROJECT/../vendor:${TEXINPUTS:-}"
export BSTINPUTS="$PROJECT:$PROJECT/../vendor:${BSTINPUTS:-}"
mkdir -p "$TEXMFVAR" "$TEXMFCONFIG" "$PROJECT/build"
if ! command -v latexmk >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh
  module load texlive/2025
fi
cd "$PROJECT"
for document in main appendix fox_response; do
  latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
    -outdir=build "$document.tex"
done
if command -v pdfunite >/dev/null 2>&1; then
  pdfunite build/main.pdf build/appendix.pdf build/fox_response.pdf \
    build/complete_reading_copy.pdf
fi
