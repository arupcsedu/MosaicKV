#!/bin/bash
# One immutable task: one lane, task, cap and <=32 IDs. Submit held through
# the shared A100 controller; never release independent arrays wholesale.
#SBATCH --job-name=mk-hf-matched
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --account=bii_dsc_community
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/scratch/djy8hg/slurm-logs/%x-%A_%a.out
#SBATCH --error=/scratch/djy8hg/slurm-logs/%x-%A_%a.err
set -euo pipefail
module load gcc/11.4.0
cd "${MOSAICKV_SOURCE_ROOT:?clean immutable worktree required}"
source mosaickv/scripts/cache_env.sh
mosaickv/scripts/assert_clean_worktree.sh
export PYTHONPATH="${MOSAICKV_SOURCE_ROOT}/mosaickv/src:${MOSAICKV_SOURCE_ROOT}/mosaickv/scripts:${MOSAICKV_SOURCE_ROOT}/mosaickv"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
exec srun /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/run_hf_matched_validation.py \
  --root "${MOSAICKV_RUN_ROOT:?}" --plan-sha "${MOSAICKV_PLAN_SHA:?}" \
  --lane "${MOSAICKV_LANE:?}" --stage "${MOSAICKV_STAGE:?}" \
  --index "${SLURM_ARRAY_TASK_ID:?}"
