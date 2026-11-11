#!/usr/bin/env bash
set -euo pipefail
SOURCE_ROOT=$1
RUN_ROOT=$2
MODE=$3
EXPECTED_SHA=$4
cd "$SOURCE_ROOT"
source mosaickv/scripts/cache_env.sh
export PYTHONPATH="$SOURCE_ROOT/mosaickv/src:$SOURCE_ROOT/mosaickv/scripts"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=backend:native,expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_ALLOC_CONF"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 ATEN_CPU_CAPABILITY=avx2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
bash mosaickv/scripts/assert_clean_worktree.sh
[[ "$(git rev-parse HEAD)" == "$EXPECTED_SHA" ]]
if [[ "$MODE" == worker ]]; then
  [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[01]$ ]]
  exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_coco_boundary_v28.py \
    worker --root "$RUN_ROOT" --index "$SLURM_ARRAY_TASK_ID"
fi
export CUDA_VISIBLE_DEVICES=""
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_coco_boundary_v28.py "$MODE" --root "$RUN_ROOT"
