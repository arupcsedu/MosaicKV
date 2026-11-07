#!/bin/bash
#SBATCH --job-name=mk-modest-analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=standard
set -euo pipefail
source_root=$1
run_root=$2
plan_sha=$3
analysis_tooling=${4:-"$run_root/tooling"}
cd "$source_root"
source mosaickv/scripts/cache_env.sh
bash mosaickv/scripts/assert_clean_worktree.sh
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$source_root/mosaickv/src:$source_root/mosaickv/scripts"
exec /scratch/djy8hg/env/mosaickv/bin/python \
  "$analysis_tooling/analyze_modest_memory.py" --root "$run_root" --plan-sha "$plan_sha"
