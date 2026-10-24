#!/bin/bash
#SBATCH --job-name=mk-v4-ppl-recovery
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
bash mosaickv/scripts/assert_clean_worktree.sh
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$1/mosaickv/src:$1/mosaickv/scripts"
if [[ "${4:-}" == "analyze" ]]; then
  exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/recover_bounded_history_v4.py \
    analyze --root "$2" --plan-sha "$3"
fi
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/recover_bounded_history_v4.py \
  worker --root "$2" --plan-sha "$3" --index "${SLURM_ARRAY_TASK_ID:?}"
