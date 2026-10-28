#!/bin/bash
#SBATCH --job-name=mk-attention-bias-analysis
#SBATCH --partition=bii
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/attention_compaction_hf_gate.py \
  analyze --root "$2" --plan-sha "$3"
