#!/bin/bash
#SBATCH --job-name=mk-paper-v3-analyze
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --partition=bii
#SBATCH --output=/scratch/djy8hg/slurm-logs/%x-%j.out
#SBATCH --error=/scratch/djy8hg/slurm-logs/%x-%j.err
#SBATCH --export=ALL

set -euo pipefail
PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python3}
: "${MOSAICKV_PAPER_INDEX:?missing V3 immutable index}"
: "${MOSAICKV_PAPER_ANALYSIS_ROOT:?missing V3 analysis output root}"
export MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"
"${PYTHON}" scripts/analyze_paper_quality_hf_vllm_v3_continuation.py \
  --index "${MOSAICKV_PAPER_INDEX}" \
  --protocol configs/experiments/paper_quality_hf_vllm_llava15_v3_continuation.yaml \
  --output-directory "${MOSAICKV_PAPER_ANALYSIS_ROOT}"
