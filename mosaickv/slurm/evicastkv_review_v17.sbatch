#!/usr/bin/env bash
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 ATEN_CPU_CAPABILITY=avx2
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export JAVA_TOOL_OPTIONS="-Djava.io.tmpdir=$TMPDIR -Djava.util.prefs.userRoot=$XDG_CACHE_HOME/java/prefs"
bash mosaickv/scripts/assert_clean_worktree.sh
if [[ "$3" == "tests" ]]; then
    module load gcc/11.4.0
    export CXX="$(command -v g++)"
    # Preserve the canonical unit-suite numerical environment. The scheduler
    # still reserves twelve CPUs for compiler/test subprocesses, as before.
    export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
    exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_review_closure_v2.py tests --root "$2/full-cpu"
fi
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_review_v17.py "$3" --root "$2"
