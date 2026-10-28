#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --partition=bii
#SBATCH --job-name=evicast-repair-summary
set -euo pipefail
cd "$1"
bash mosaickv/scripts/assert_clean_worktree.sh
source mosaickv/scripts/cache_env.sh
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_followup_summary.py --root "$2"
