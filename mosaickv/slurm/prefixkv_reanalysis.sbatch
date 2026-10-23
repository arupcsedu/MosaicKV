#!/bin/bash
#SBATCH --job-name=mk-prefix-reanalysis
#SBATCH --partition=bii
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=00:10:00
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
bash mosaickv/scripts/assert_clean_worktree.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$1/mosaickv/src:$1/mosaickv/scripts"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/analyze_prefixkv_full_campaign.py \
  --root "$2" --plan-sha "$3" --authorization "$4" --output "$5"
