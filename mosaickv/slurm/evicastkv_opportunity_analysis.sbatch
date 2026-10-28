#!/bin/bash
#SBATCH --job-name=evicast-opportunity-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=bii
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
bash mosaickv/scripts/assert_clean_worktree.sh
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_opportunity_study.py analyze --root "$2"
