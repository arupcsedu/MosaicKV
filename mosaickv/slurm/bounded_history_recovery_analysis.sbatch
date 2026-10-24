#!/bin/bash
#SBATCH --job-name=mk-v4-recovery-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --partition=bii
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
bash mosaickv/scripts/assert_clean_worktree.sh
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$1/mosaickv/src:$1/mosaickv/scripts"
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/recover_bounded_history_v4.py \
  analyze --root "$2" --plan-sha "$3"
