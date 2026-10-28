#!/bin/bash
#SBATCH --job-name=evicast-review-closure
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --partition=bii
set -euo pipefail
cd "$1"
test "$(git rev-parse HEAD)" = "$4"
bash mosaickv/scripts/assert_clean_worktree.sh
source mosaickv/scripts/cache_env.sh
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export JAVA_TOOL_OPTIONS="-Djava.io.tmpdir=$TMPDIR -Djava.util.prefs.userRoot=$XDG_CACHE_HOME/java/prefs"
if [[ "$5" == schema ]]; then
  exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_source_schema.py --output "$2"
fi
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_review_closure.py --output "$2" --schema-root "$3"
