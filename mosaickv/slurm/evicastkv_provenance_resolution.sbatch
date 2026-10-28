#!/bin/bash
#SBATCH --job-name=evicast-provenance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=bii
set -euo pipefail
cd "$1"
test "$(git rev-parse HEAD)" = "$3"
bash mosaickv/scripts/assert_clean_worktree.sh
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Remote reads are limited to four pinned Parquet input projections. No images,
# response columns, models, archive downloads, or dataset preparation is allowed.
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_provenance_resolution.py \
  --history-review /scratch/djy8hg/runs/evicastkv/fox-review-20261028-history-two-roots-v1/review.json \
  --scope /scratch/djy8hg/runs/evicastkv/fox-review-20261028-milebench-additional-v1 \
  --media-root /scratch/djy8hg/runs/evicastkv/fox-review-20261028-milebench-media-content-v1 \
  --iedit-root /scratch/djy8hg/runs/evicastkv/fox-review-20261028-iedit-media-recovery-v1 \
  --output "$2"
