#!/bin/bash
#SBATCH --job-name=mk-op-promote
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --partition=bii
set -euo pipefail
source_root=$1
run_root=$2
plan_sha=$3
cd "$source_root"
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_root/mosaickv/src:$source_root/mosaickv/scripts"
exec /scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/promote_output_preserving_sweep.py \
  --source "$source_root" --root "$run_root" --plan-sha256 "$plan_sha"
