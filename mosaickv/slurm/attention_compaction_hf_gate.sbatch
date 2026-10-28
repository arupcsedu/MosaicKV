#!/bin/bash
#SBATCH --job-name=mk-attention-bias
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=00:30:00
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/attention_compaction_hf_gate.py \
  worker --root "$2" --plan-sha "$3" --index "${SLURM_ARRAY_TASK_ID:?}"
