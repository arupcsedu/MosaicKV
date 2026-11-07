# EviCastKV / PrefixKV / Prof. Fox review report

Standalone retrospective report. It does not change the submission manuscript,
raw results, candidate algorithms, or experiment authorization.

Updated October 30, 2026: `main.tex` includes
`storage_engineering_20261030.tex`, with three artifact-derived tables and a
TikZ storage-boundary figure. This update separates the completed benchmark
campaigns from the two-consumed-input lossless-history diagnostics. Its final
engineering source is `d8eb51be69c5d6469ba12283f0cf811ba352345c`; the
2,631-test CPU pass is not an independent benchmark-quality confirmation.
Initial storage is unchanged, final/decode-owned peak improve, source history
is slower than ordinary K/V history, and whole-request peak is uncertified.
The historical 192-example MileBench E6 curves are also included with separate
Accuracy/ROUGE-L columns and the failed lower-memory quality gate preserved.

From this directory:

```bash
source ../../../mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
/scratch/djy8hg/env/mosaickv/bin/python export_report.py
module load texlive/2025
export TEXMFVAR="$MOSAICKV_CACHE_ROOT/texlive/var"
export TEXMFCONFIG="$MOSAICKV_CACHE_ROOT/texlive/config"
mkdir -p "$TEXMFVAR" "$TEXMFCONFIG" "$MOSAICKV_CACHE_ROOT/latex/evicastkv_prefixkv_fox_review"
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir="$MOSAICKV_CACHE_ROOT/latex/evicastkv_prefixkv_fox_review" main.tex
cp "$MOSAICKV_CACHE_ROOT/latex/evicastkv_prefixkv_fox_review/main.pdf" evicastkv_prefixkv_fox_review.pdf
/scratch/djy8hg/env/mosaickv/bin/python verify_report.py \
  --pdf evicastkv_prefixkv_fox_review.pdf \
  --log "$MOSAICKV_CACHE_ROOT/latex/evicastkv_prefixkv_fox_review/main.log"
```

All numeric tables and plot files are exported from the immutable source artifacts
listed in `provenance.json`. No GPU inference or new statistical analysis runs.
Positive evidence is emphasized without treating native-policy points as
byte-matched, development cohorts as fresh confirmation, or phase profiling as
a general latency win. EviCastKV and MosaicKV refer to the same system.
