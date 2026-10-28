#!/usr/bin/env bash
#SBATCH --job-name=mk-prefix-isoquality
# Resource arguments are supplied by the immutable, bounded submitter.
set -euo pipefail
source_root=$1
run_root=$2
plan_sha=$3
mode=$4
cd "$source_root"
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$source_root/mosaickv/src:$source_root/mosaickv/scripts"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=12
export JAVA_TOOL_OPTIONS="-Djava.io.tmpdir=$TMPDIR -Djava.util.prefs.userRoot=$XDG_CACHE_HOME/java/prefs"
bash mosaickv/scripts/assert_clean_worktree.sh
args=("$mode" --root "$run_root" --plan-sha "$plan_sha")
if [[ "$mode" == worker || "$mode" == timing ]]; then
  args+=(--index "${SLURM_ARRAY_TASK_ID:?}")
fi
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/prefixkv_isoquality_development.py "${args[@]}"
