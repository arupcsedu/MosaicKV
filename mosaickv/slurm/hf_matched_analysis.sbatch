#!/bin/bash
#SBATCH --job-name=mk-hf-matched-analysis
#SBATCH --partition=bii
#SBATCH --account=bii_dsc_community
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/djy8hg/slurm-logs/%x-%j.out
#SBATCH --error=/scratch/djy8hg/slurm-logs/%x-%j.err
set -euo pipefail
cd "${MOSAICKV_SOURCE_ROOT:?}"
source mosaickv/scripts/cache_env.sh
mosaickv/scripts/assert_clean_worktree.sh
export PYTHONPATH="${MOSAICKV_SOURCE_ROOT}/mosaickv/src:${MOSAICKV_SOURCE_ROOT}/mosaickv/scripts:${MOSAICKV_SOURCE_ROOT}/mosaickv"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/run_hf_matched_validation.py \
  --root "${MOSAICKV_RUN_ROOT:?}" --plan-sha "${MOSAICKV_PLAN_SHA:?}" \
  --lane "${MOSAICKV_LANE:?}" --stage "${MOSAICKV_STAGE:?}" --analyze
