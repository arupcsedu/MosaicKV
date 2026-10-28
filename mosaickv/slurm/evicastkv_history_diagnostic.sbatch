#!/bin/bash
#SBATCH --job-name=evicast-history
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
set -euo pipefail
cd "$1"
test "$(git rev-parse HEAD)" = "$3"
bash mosaickv/scripts/assert_clean_worktree.sh
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_history_diagnostic.py worker --root "$2"
