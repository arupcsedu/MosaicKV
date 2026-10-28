#!/bin/bash
#SBATCH --job-name=evicast-transport-score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
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
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_transport_scores.py analyze --root "$2"
