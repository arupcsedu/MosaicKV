#!/usr/bin/env bash
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TRITON_CACHE_DIR="$MOSAICKV_CACHE_ROOT/triton/source-geometry-${SLURM_JOB_ID:?}-${SLURM_ARRAY_TASK_ID:-cpu}"
if [[ "$4" == worker ]]; then
 export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
fi
bash mosaickv/scripts/assert_clean_worktree.sh
if [[ "$4" == start || "$4" == defer ]]; then
 exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_row_geometry.py "$4" --root "$2" --source-sha "$3"
fi
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_row_geometry.py "$4" --root "$2" --plan-sha "$3"
