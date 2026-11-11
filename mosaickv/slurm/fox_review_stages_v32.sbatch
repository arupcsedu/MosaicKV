#!/usr/bin/env bash
set -euo pipefail
SOURCE_ROOT=$1
RUN_ROOT=$2
MODE=$3
EXPECTED_SHA=$4
cd "$SOURCE_ROOT"
source mosaickv/scripts/cache_env.sh
export PYTHONPATH="$SOURCE_ROOT/mosaickv/src:$SOURCE_ROOT/mosaickv/scripts"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ATEN_CPU_CAPABILITY=avx2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
bash mosaickv/scripts/assert_clean_worktree.sh
[[ "$(git rev-parse HEAD)" == "$EXPECTED_SHA" ]]
if [[ "$MODE" == tests ]]; then
  module load gcc/11.4.0
  export CXX="$(command -v g++)"
  exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_review_closure_v2.py \
    tests --root "$RUN_ROOT/full-cpu" --cpu-timeout-seconds 840 --cpu-capture-mode sys
fi
if [[ "$MODE" == after-whole ]]; then
  export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
fi
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_review_stages_v32.py \
  "$MODE" --root "$RUN_ROOT"
