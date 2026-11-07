#!/bin/bash
# One immutable method x budget x 16-image calibration shard.
#SBATCH --job-name=mk-modest-cal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
set -euo pipefail
source_root=$1
run_root=$2
plan_sha=$3
cd "$source_root"
source mosaickv/scripts/cache_env.sh
bash mosaickv/scripts/assert_clean_worktree.sh
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
export CUBLAS_WORKSPACE_CONFIG=:4096:8
[[ "$CUBLAS_WORKSPACE_CONFIG" == ':4096:8' ]] || exit 2
export PYTHONPATH="$source_root/mosaickv/src:$source_root/mosaickv/scripts"
exec /scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/run_coco_shared_bytes.py --root "$run_root" \
  --plan-sha "$plan_sha" --index "${SLURM_ARRAY_TASK_ID:?}"
