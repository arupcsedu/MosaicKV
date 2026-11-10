#!/usr/bin/env bash
# Finite resources/afterany dependencies are bound by the immutable submitter.
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export ATEN_CPU_CAPABILITY=avx2 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export JAVA_TOOL_OPTIONS="-Djava.io.tmpdir=$TMPDIR -Djava.util.prefs.userRoot=$XDG_CACHE_HOME/java/prefs"
case "$3" in
  rmsnorm-worker|protection-worker) ;;
  *) export CUDA_VISIBLE_DEVICES="" ;;
esac
bash mosaickv/scripts/assert_clean_worktree.sh
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_review_v8_diagnostics.py "$3" --root "$2"
