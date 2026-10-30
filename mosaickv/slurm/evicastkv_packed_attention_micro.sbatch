#!/bin/bash
#SBATCH --job-name=ev-packed-micro
#SBATCH --partition=bii-gpu
#SBATCH --account=bii_dsc_community
#SBATCH --reservation=bi_fox_dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=00:20:00
set -euo pipefail
: "${MOSAICKV_SOURCE:?explicit immutable worktree required}"
: "${MOSAICKV_OUTPUT:?new immutable output required}"
cd "$MOSAICKV_SOURCE"
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TRITON_CACHE_DIR="$MOSAICKV_CACHE_ROOT/triton/packed-attention-${SLURM_JOB_ID}"
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
bash mosaickv/scripts/assert_clean_worktree.sh
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_packed_attention_micro.py --root "$MOSAICKV_OUTPUT"
