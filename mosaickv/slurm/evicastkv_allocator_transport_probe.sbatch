#!/usr/bin/env bash
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 ATEN_CPU_CAPABILITY=avx2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
module load gcc/11.4.0
export CXX="$(command -v g++)"
bash mosaickv/scripts/assert_clean_worktree.sh
if [[ "$3" == "tests" ]]; then
    export CUDA_VISIBLE_DEVICES=""
    exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_review_closure_v2.py tests --root "$2/full-cpu" --cpu-timeout-seconds 1200
fi
if [[ "$3" != "worker" ]]; then
    export CUDA_VISIBLE_DEVICES=""
fi
if [[ "$3" == "snapshot-audit" ]]; then
    # The dedicated CLI installs the bounded address-space/CPU/wall limits
    # before reading the saved archives. No semantic DAG is expanded.
    exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_snapshot_state_audit.py --root "$2/snapshot-state" --cpu-gate-root "$2"
fi
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_allocator_transport_probe.py "$3" --root "$2"
