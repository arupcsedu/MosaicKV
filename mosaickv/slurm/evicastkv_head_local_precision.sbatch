#!/usr/bin/env bash
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TRITON_CACHE_DIR="$MOSAICKV_CACHE_ROOT/triton/head-local-${SLURM_JOB_ID:?}-${SLURM_ARRAY_TASK_ID:-cpu}"
if [[ "$4" != worker ]]; then
 export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
fi
bash mosaickv/scripts/assert_clean_worktree.sh
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_head_local_precision.py "$4" --root "$2" --plan-sha "$3"
