#!/bin/bash
#SBATCH --job-name=evicast-media
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --partition=bii
set -euo pipefail
cd "$1"
test "$(git rev-parse HEAD)" = "$3"
bash mosaickv/scripts/assert_clean_worktree.sh
source mosaickv/scripts/cache_env.sh
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_media_audit.py \
  --scope /scratch/djy8hg/runs/evicastkv/fox-review-20261028-milebench-additional-v1 --output "$2"
