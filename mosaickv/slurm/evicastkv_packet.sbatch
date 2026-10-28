#!/bin/bash
# Finite event callback: no queue/log polling; report failed stages too.
set -euo pipefail
cd "$1"
test "$(git rev-parse HEAD)" = "$4"
bash mosaickv/scripts/assert_clean_worktree.sh
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export JAVA_TOOL_OPTIONS="-Djava.io.tmpdir=$TMPDIR -Djava.util.prefs.userRoot=$XDG_CACHE_HOME/java/prefs"
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_packet.py watch --root "$2" --stage "$3"
