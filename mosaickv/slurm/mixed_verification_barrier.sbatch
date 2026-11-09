#!/usr/bin/env bash
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export TRITON_CACHE_DIR="/scratch/djy8hg/cache/mosaickv/triton-private/${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-cpu}"
mkdir -p "$TRITON_CACHE_DIR"
bash mosaickv/scripts/assert_clean_worktree.sh
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/mixed_verification_barrier_study.py "$4" --root "$2" --plan-sha "$3" --stage "$5"
