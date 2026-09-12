#!/bin/bash
#SBATCH --job-name=mk-hf-byte-analysis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=bii
#SBATCH --output=/scratch/djy8hg/slurm-logs/%x-%j.out
#SBATCH --error=/scratch/djy8hg/slurm-logs/%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

REPOSITORY=${MOSAICKV_SOURCE_ROOT:?MOSAICKV_SOURCE_ROOT is required}
PROJECT_ROOT=${REPOSITORY}/mosaickv
PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python3}
: "${MOSAICKV_HF_BYTE_CURVE_INDEX:?set the immutable byte-curve jobs.jsonl path}"
: "${MOSAICKV_HF_BYTE_CURVE_ANALYSIS_ROOT:?set a new analysis output directory}"

export MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
cd "${PROJECT_ROOT}"

srun "${PYTHON}" scripts/analyze_hf_byte_matched_curves.py \
  --index "${MOSAICKV_HF_BYTE_CURVE_INDEX}" \
  --analysis-root "${MOSAICKV_HF_BYTE_CURVE_ANALYSIS_ROOT}"
