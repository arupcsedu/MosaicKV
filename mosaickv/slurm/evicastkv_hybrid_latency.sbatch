#!/usr/bin/env bash
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
ulimit -v unlimited
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TRITON_CACHE_DIR="$MOSAICKV_CACHE_ROOT/triton/hybrid-${SLURM_JOB_ID:?}-${SLURM_ARRAY_TASK_ID:-cpu}"
bash mosaickv/scripts/assert_clean_worktree.sh
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_hybrid_latency.py "$4" --root "$2" --plan-sha "$3"
