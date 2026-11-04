#!/usr/bin/env bash
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
bash mosaickv/scripts/assert_clean_worktree.sh
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_strict_latency.py "$6" --root "$2" --plan-sha "$3" --analyzer-sha "$4" --runtime-source "$5" --kind "$7"
