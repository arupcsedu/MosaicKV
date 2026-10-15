#!/bin/bash
# Retrospective development evidence only; each array element is one method/cap arm.
#SBATCH --job-name=mk-op-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
set -euo pipefail
source_root=$1
run_root=$2
plan_sha=$3
cd "$source_root"
source mosaickv/scripts/cache_env.sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
export PYTHONPATH="$source_root/mosaickv/src:$source_root/mosaickv/scripts"
exec /scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/run_output_preserving_sweep.py \
  --root "$run_root" --plan-sha256 "$plan_sha" \
  --index "${SLURM_ARRAY_TASK_ID:?}"
