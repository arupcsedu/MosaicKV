#!/usr/bin/env bash
# Resource requests are explicit in the immutable controller submission ledger.
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export JAVA_TOOL_OPTIONS="-Djava.io.tmpdir=$TMPDIR -Djava.util.prefs.userRoot=$XDG_CACHE_HOME/java/prefs"
case "$3" in
    graph-worker|window-worker|boundary-worker)
        export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
        ;;
    *)
        export CUDA_VISIBLE_DEVICES=""
        export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
        ;;
esac
bash mosaickv/scripts/assert_clean_worktree.sh
shift
task_root="$1"
task_mode="$2"
shift 2
exec /scratch/djy8hg/env/mosaickv/bin/python \
    mosaickv/scripts/evicastkv_review_followups.py "$task_mode" --root "$task_root" "$@"
