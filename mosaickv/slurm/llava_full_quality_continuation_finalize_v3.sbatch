#!/bin/bash
#SBATCH --job-name=mosaickv-llava-cont-finalize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=bii
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/llava-full-quality-cont-finalize-%j.log

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
export MOSAICKV_PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
export MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
CONTINUATION_INDEX=/scratch/djy8hg/runs/mosaickv/campaigns/llava-full-quality-continuation-6gpu-v1/jobs.jsonl
OUTPUT_ROOT=/scratch/djy8hg/runs/mosaickv/llava-full-quality-continuation-6gpu-v1

# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

printf '%s  %s\n' \
  ff324e336ca914cd17ebfac0420d0d4c098f46ecbdbcfd93666c3ec51883ad54 \
  "${CONTINUATION_INDEX}" | sha256sum --check --strict
exec "${MOSAICKV_PYTHON}" scripts/validate_llava_quality_continuation.py \
  --index "${CONTINUATION_INDEX}" \
  --output-root "${OUTPUT_ROOT}"
