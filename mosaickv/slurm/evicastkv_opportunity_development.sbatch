#!/bin/bash
#SBATCH --job-name=evicast-evidence-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
bash mosaickv/scripts/assert_clean_worktree.sh
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export JAVA_TOOL_OPTIONS="-Djava.io.tmpdir=$TMPDIR -Djava.util.prefs.userRoot=$XDG_CACHE_HOME/java/prefs"
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_opportunity_development.py "$3" --root "$2" --spec-sha "$4"
