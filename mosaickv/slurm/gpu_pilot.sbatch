#!/bin/bash
#SBATCH --job-name=mosaickv-gpu-pilot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=37
# 3B HF workers fit comfortably within this bound. Requesting --mem=0 would
# reserve the whole DGX host and serialize otherwise independent array tasks.
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --export=ALL
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/gpu-pilot-%A_%a.log

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
MOSAICKV_PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
export MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}

# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"
: "${MOSAICKV_PILOT_INDEX:?set MOSAICKV_PILOT_INDEX to immutable jobs.jsonl}"
: "${SLURM_ARRAY_TASK_ID:?submit this file as a Slurm array}"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_ROOT}"
srun "${MOSAICKV_PYTHON}" scripts/run_gpu_pilot_worker.py \
  --index "${MOSAICKV_PILOT_INDEX}" \
  --array-index "${SLURM_ARRAY_TASK_ID}"
