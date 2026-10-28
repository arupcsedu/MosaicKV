#!/bin/bash
#SBATCH --job-name=mk-anchor-cpu-v3
#SBATCH --partition=bii
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/study_anchor_compaction_v3.py \
  worker --root "$2" --plan-sha "$3" --index "${SLURM_ARRAY_TASK_ID:?}"
