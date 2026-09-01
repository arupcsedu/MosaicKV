#!/bin/bash
#SBATCH --job-name=mk-pubbase-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=bii
#SBATCH --output=/scratch/djy8hg/slurm-logs/%x-%j.out
#SBATCH --error=/scratch/djy8hg/slurm-logs/%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

REPOSITORY=/scratch/djy8hg/workdir/MosaicKV
PROJECT_ROOT=${REPOSITORY}/mosaickv
PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python3}
CAMPAIGN_ROOT=${MOSAICKV_CAMPAIGN_ROOT:?MOSAICKV_CAMPAIGN_ROOT is required}
REUSED_COMPOSITE=${MOSAICKV_REUSED_COMPOSITE:?MOSAICKV_REUSED_COMPOSITE is required}

export MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
cd "${PROJECT_ROOT}"

"${PYTHON}" scripts/analyze_vllm_published_baseline_full.py \
  --campaign-root "${CAMPAIGN_ROOT}" --reused-composite "${REUSED_COMPOSITE}"
