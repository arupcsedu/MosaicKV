#!/usr/bin/env bash
# One aggregate array; resources are recorded by the immutable submitter.
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export JAVA_TOOL_OPTIONS="-Djava.io.tmpdir=$TMPDIR -Djava.util.prefs.userRoot=$XDG_CACHE_HOME/java/prefs"
if [[ "$3" == worker ]]; then
  export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
else
  export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
fi
bash mosaickv/scripts/assert_clean_worktree.sh
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_review_parallel.py "$3" --root "$2"
