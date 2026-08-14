#!/bin/bash
#SBATCH --job-name=mosaickv-v69-compact
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/v69-compact-%j.log
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
OUTPUT_ROOT=${MOSAICKV_V69_OUTPUT_ROOT:?set MOSAICKV_V69_OUTPUT_ROOT}
BASE_CONFIG=${MOSAICKV_V69_BASE_CONFIG:?set MOSAICKV_V69_BASE_CONFIG}
LEDGER=${MOSAICKV_V69_LEDGER:?set MOSAICKV_V69_LEDGER}

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
export PYTHONPATH="${PACKAGE_ROOT}/src:${PACKAGE_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export MOSAICKV_ALLOW_DIRTY_EXPLORATORY=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export MOSAICKV_FIXED_CLOCK_STATUS=not_permitted_or_not_requested

cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,pstate,clocks.sm,clocks.mem,power.draw --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
srun "${PYTHON}" scripts/run_choice_c_v67_matched_sdpa_worker.py \
  --base-config "${BASE_CONFIG}" \
  --ledger "${LEDGER}" \
  --output-root "${OUTPUT_ROOT}" \
  --warmups 2 \
  --repetitions 5 \
  --routed-head-only
"${PYTHON}" scripts/analyze_choice_c_v67_matched_sdpa.py \
  --output-root "${OUTPUT_ROOT}"
