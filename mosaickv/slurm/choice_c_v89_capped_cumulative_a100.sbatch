#!/bin/bash
#SBATCH --job-name=mosaickv-v89-cal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/v89-cal-%A_%a.log
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
OUTPUT_ROOT=${MOSAICKV_V89_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v89-capped-cumulative-query-coverage-calibration-v1}
SPEC=${MOSAICKV_V89_SPEC:-${PACKAGE_ROOT}/configs/experiments/choice_c_v89_capped_cumulative_query_coverage_calibration.yaml}

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
export PYTHONPATH="${PACKAGE_ROOT}/src:${PACKAGE_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export MOSAICKV_FIXED_CLOCK_STATUS=not_permitted_or_not_requested
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,pstate,clocks.sm,clocks.mem,power.draw --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
srun "${PYTHON}" scripts/run_choice_c_v74_label_blind_calibration_worker.py \
  --spec "${SPEC}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-prefix v89 \
  --expected-experiment-id choice_c_v89_capped_cumulative_query_coverage_calibration
