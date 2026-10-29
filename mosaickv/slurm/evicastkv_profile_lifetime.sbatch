#!/usr/bin/env bash
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
bash mosaickv/scripts/assert_clean_worktree.sh
args=("$4" --root "$2" --plan-sha "$3")
if [[ "$4" == worker ]]; then
  args+=(--index "${SLURM_ARRAY_TASK_ID:?}")
else
  export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
  if [[ "$4" == conditional_submit ]]; then
    args+=(--parent "${5:?}")
  fi
fi
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_profile_lifetime.py "${args[@]}"
