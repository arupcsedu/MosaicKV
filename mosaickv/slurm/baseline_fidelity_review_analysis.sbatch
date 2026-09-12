#!/bin/bash
#SBATCH --job-name=mk-base-fidelity-a
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --partition=bii
#SBATCH --export=ALL

set -euo pipefail

REPOSITORY_ROOT=${MOSAICKV_SOURCE_ROOT:?MOSAICKV_SOURCE_ROOT is required}
PROJECT_ROOT=${REPOSITORY_ROOT}/mosaickv
PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"
cd "${PROJECT_ROOT}"
srun "${PYTHON}" scripts/analyze_baseline_fidelity_review.py \
  --input-root "${MOSAICKV_OUTPUT_ROOT:?MOSAICKV_OUTPUT_ROOT is required}" \
  --output-root "${MOSAICKV_OUTPUT_ROOT}/analysis"

