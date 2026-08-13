#!/bin/bash
#SBATCH --job-name=mosaickv-v38-bottlenecks
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/v38-bottlenecks-%A_%a.log
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
OUTPUT_ROOT=${MOSAICKV_V38_OUTPUT_ROOT:?set MOSAICKV_V38_OUTPUT_ROOT}
SPEC=${MOSAICKV_V38_SPEC:?set MOSAICKV_V38_SPEC}
PROFILE=${MOSAICKV_V38_PROFILE:?set MOSAICKV_V38_PROFILE}
LEDGER=${MOSAICKV_V38_LEDGER:?set MOSAICKV_V38_LEDGER}

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
export PYTHONPATH="${PACKAGE_ROOT}/src:${PACKAGE_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export MOSAICKV_ALLOW_DIRTY_EXPLORATORY=1

cd "${PACKAGE_ROOT}"
printf 'classification=exploratory_dirty_tree_retrospective_non_claim_validation\n'
printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID}"
printf 'slurm_array_task_id=%s\n' "${SLURM_ARRAY_TASK_ID}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader

srun "${PYTHON}" scripts/run_choice_c_v38_bottleneck_worker.py \
  --rank "${SLURM_ARRAY_TASK_ID}" \
  --spec "${SPEC}" \
  --profile "${PROFILE}" \
  --ledger "${LEDGER}" \
  --output-root "${OUTPUT_ROOT}"
