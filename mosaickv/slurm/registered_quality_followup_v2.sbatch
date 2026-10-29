#!/usr/bin/env bash
# Resource and dependency arguments are recorded by the finite submitter.
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAVA_TOOL_OPTIONS="-Djava.io.tmpdir=$TMPDIR -Djava.util.prefs.userRoot=$XDG_CACHE_HOME/java/prefs"
bash mosaickv/scripts/assert_clean_worktree.sh
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/registered_quality_followup_v2.py "$4" --root "$2" --plan-sha "$3"
