#!/bin/bash
#SBATCH --job-name=mosaickv-v65-route
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-2%3
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/v65-route-%A_%a.log
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
OUTPUT_ROOT=${MOSAICKV_V65_OUTPUT_ROOT:?set MOSAICKV_V65_OUTPUT_ROOT}
BASE_CONFIG=${MOSAICKV_V65_BASE_CONFIG:?set MOSAICKV_V65_BASE_CONFIG}
LEDGER=${MOSAICKV_V65_LEDGER:?set MOSAICKV_V65_LEDGER}

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
export PYTHONPATH="${PACKAGE_ROOT}/src:${PACKAGE_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export MOSAICKV_ALLOW_DIRTY_EXPLORATORY=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
srun "${PYTHON}" scripts/run_choice_c_v65_routed_quality_worker.py \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --base-config "${BASE_CONFIG}" \
  --ledger "${LEDGER}" \
  --output-root "${OUTPUT_ROOT}"
