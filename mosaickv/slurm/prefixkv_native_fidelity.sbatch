#!/bin/bash
# Submit only registered FullKV shard indices; shared controller must admit
# this array under the global four-A100 ceiling (not four per array).
#SBATCH --job-name=mk-prefix-native-fidelity
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
export PYTHONPATH="$source_root/mosaickv/src${PYTHONPATH:+:$PYTHONPATH}"
exec /scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/verify_prefixkv_native_fidelity.py \
  --root "$run_root" --plan-sha "$plan_sha" --index "${SLURM_ARRAY_TASK_ID:?}"
