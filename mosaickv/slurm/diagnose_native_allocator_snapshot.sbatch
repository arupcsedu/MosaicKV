#!/usr/bin/env bash
set -euo pipefail
SOURCE_ROOT=$1
RUN_ROOT=$2
EXPECTED_SHA=$3
cd "$SOURCE_ROOT"
source mosaickv/scripts/cache_env.sh
export PYTHONPATH="$SOURCE_ROOT/mosaickv/src:$SOURCE_ROOT/mosaickv/scripts"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_ALLOC_CONF=backend:native,expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_ALLOC_CONF"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
bash mosaickv/scripts/assert_clean_worktree.sh
[[ "$(git rev-parse HEAD)" == "$EXPECTED_SHA" ]]
exec /scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/diagnose_native_allocator_snapshot.py --root "$RUN_ROOT/capture"
