#!/usr/bin/env bash
# Resource flags are specified by the immutable launch receipt.
set -euo pipefail
SOURCE_ROOT=$1
RUN_ROOT=$2
MODE=$3
EXPECTED_SHA=$4
cd "$SOURCE_ROOT"
source mosaickv/scripts/cache_env.sh
export PYTHONPATH="$SOURCE_ROOT/mosaickv/src:$SOURCE_ROOT/mosaickv/scripts"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=backend:native,expandable_segments:True,roundup_power2_divisions:0
export PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_ALLOC_CONF"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 ATEN_CPU_CAPABILITY=avx2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
bash mosaickv/scripts/assert_clean_worktree.sh
[[ "$(git rev-parse HEAD)" == "$EXPECTED_SHA" ]]
if [[ "$MODE" == tests ]]; then
  export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  module load gcc/11.4.0
  export CXX="$(command -v g++)"
  exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_review_closure_v2.py \
    tests --root "$RUN_ROOT/full-cpu" --cpu-timeout-seconds 1200 --cpu-capture-mode sys
fi
if [[ "$MODE" == worker ]]; then
  [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[01]$ ]]
  exec /scratch/djy8hg/env/mosaickv/bin/python \
    mosaickv/scripts/evicastkv_bounded_reader_v26.py "$MODE" --root "$RUN_ROOT" \
    --index "$SLURM_ARRAY_TASK_ID"
fi
if [[ "$MODE" != operator ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi
exec /scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/evicastkv_bounded_reader_v26.py "$MODE" --root "$RUN_ROOT"



