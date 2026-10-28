#!/bin/bash
#SBATCH --job-name=evicast-evidence-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=bii
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
bash mosaickv/scripts/assert_clean_worktree.sh
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export JAVA_TOOL_OPTIONS="-Djava.io.tmpdir=$TMPDIR -Djava.util.prefs.userRoot=$XDG_CACHE_HOME/java/prefs"
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_opportunity_development.py analyze --recorded-source --root "$2"
