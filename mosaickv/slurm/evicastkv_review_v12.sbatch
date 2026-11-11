#!/usr/bin/env bash
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
if [[ "$3" == "ownership-worker" ]]; then
  module load gcc/11.4.0
  export CXX="$(command -v g++)"
fi
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 ATEN_CPU_CAPABILITY=avx2
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
case "$3" in
  precision-worker|lookm-worker|lossless-worker|ownership-worker) ;;
  *) export CUDA_VISIBLE_DEVICES="" ;;
esac
bash mosaickv/scripts/assert_clean_worktree.sh
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_review_v12.py "$3" --root "$2"
