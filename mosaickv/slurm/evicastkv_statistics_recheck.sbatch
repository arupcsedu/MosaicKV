#!/bin/bash
# One bounded afterany callback, never a polling loop.
set -euo pipefail
cd "$1"
test "$(git rev-parse HEAD)" = "$5"
bash mosaickv/scripts/assert_clean_worktree.sh
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export JAVA_TOOL_OPTIONS="-Djava.io.tmpdir=$TMPDIR -Djava.util.prefs.userRoot=$XDG_CACHE_HOME/java/prefs"
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_statistics_recheck.py follow --primary "$2" --root "$3" --stage "$4" --finalizer-source-sha "${6:-$5}"
