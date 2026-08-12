#!/bin/bash
#SBATCH --job-name=mosaickv-prefixkv-finalize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=bii
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/llava-prefixkv-repair-finalize-v1-%j.log

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
OVERLAY_INDEX=/scratch/djy8hg/runs/mosaickv/campaigns/llava-full-quality-prefixkv-repair-v1/full_overlay/jobs.jsonl
OUTPUT_ROOT=/scratch/djy8hg/runs/mosaickv/llava-full-quality-prefixkv-repair-v1

# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

printf '%s  %s\n' \
  adeea1d7782894e01401f0d54a75c0f14acc5c0df6cb7d2345889a02c138f3dc \
  scripts/validate_llava_quality_continuation.py | sha256sum --check --strict
exec "${PYTHON}" scripts/validate_llava_quality_continuation.py \
  --index "${OVERLAY_INDEX}" \
  --output-root "${OUTPUT_ROOT}"
