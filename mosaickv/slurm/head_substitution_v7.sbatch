#!/bin/bash
#SBATCH --job-name=mk-head-v7
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$1/mosaickv/src:$1/mosaickv/scripts"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/head_substitution_v7.py \
  worker --root "$2" --plan-sha "$3" --index "${SLURM_ARRAY_TASK_ID:?}"
