#!/bin/bash
#SBATCH --job-name=mosaickv-ledger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/sample-ledger-%j.log

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
: "${MOSAICKV_LEDGER_TASK:?set MOSAICKV_LEDGER_TASK}"
: "${MOSAICKV_LEDGER_DATASET_REVISION:?set MOSAICKV_LEDGER_DATASET_REVISION}"
: "${MOSAICKV_LEDGER_MODEL:?set MOSAICKV_LEDGER_MODEL}"
: "${MOSAICKV_LEDGER_OUTPUT:?set MOSAICKV_LEDGER_OUTPUT}"

source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

srun "${PYTHON}" "${PROJECT_ROOT}/scripts/materialize_sample_ledger.py" \
  --task "${MOSAICKV_LEDGER_TASK}" \
  --dataset-revision "${MOSAICKV_LEDGER_DATASET_REVISION}" \
  --model-name "${MOSAICKV_LEDGER_MODEL}" \
  --seed 17 \
  --output "${MOSAICKV_LEDGER_OUTPUT}" \
  --resume
