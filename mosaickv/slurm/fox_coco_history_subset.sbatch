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
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
bash mosaickv/scripts/assert_clean_worktree.sh
[[ "$(git rev-parse HEAD)" == "$EXPECTED_SHA" ]]
if [[ "$MODE" == worker ]]; then
  [[ "${SLURM_ARRAY_TASK_ID:-}" == 1 || "${SLURM_ARRAY_TASK_ID:-}" == 7 ]]
  exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_coco_history_subset.py \
    worker --root "$RUN_ROOT" --index "$SLURM_ARRAY_TASK_ID"
fi
[[ "$MODE" == analyze ]]
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_coco_history_subset.py \
  analyze --root "$RUN_ROOT"
