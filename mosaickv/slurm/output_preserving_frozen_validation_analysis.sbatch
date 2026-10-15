#!/bin/bash
# Analyze the preregistered 128-sample frozen-candidate validation on CPU.
#SBATCH --job-name=mk-op-frozen-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=bii
set -euo pipefail
source_root=$1
run_root=$2
plan_sha=$3
cd "$source_root"
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$source_root/mosaickv/src:$source_root/mosaickv/scripts"
exec /scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/analyze_output_preserving_frozen_validation.py \
  --root "$run_root" --plan-sha256 "$plan_sha"
