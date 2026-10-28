#!/bin/bash
# Finite afterany callback: no polling, no perpetual agent, no success-only dependency.
#SBATCH --job-name=evicast-analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_campaign.py analyze --root "$2" --stage "$3"
