#!/usr/bin/env bash
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TRITON_CACHE_DIR="${MOSAICKV_CACHE_ROOT}/triton/predictive-${SLURM_JOB_ID:?}-${SLURM_ARRAY_TASK_ID:-analysis}"
mkdir -p "$TRITON_CACHE_DIR"
bash mosaickv/scripts/assert_clean_worktree.sh
args=("$4" --root "$2" --plan-sha "$3")
if [[ "$4" == worker || "$4" == codec_worker ]]; then
  args+=(--index "${SLURM_ARRAY_TASK_ID:?}")
fi
if [[ "$4" == worker ]]; then
  export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
fi
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_predictive_feasibility.py "${args[@]}"
