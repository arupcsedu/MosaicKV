#!/usr/bin/env bash
#SBATCH --job-name=fox36-history
#SBATCH --account=bii_dsc_community
#SBATCH --partition=bii,standard,interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:10:00
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
bash mosaickv/scripts/assert_clean_worktree.sh
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_global_history_v36.py "${3:-controller}" --root "$2"
