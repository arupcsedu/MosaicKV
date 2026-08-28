#!/bin/bash
#SBATCH --job-name=mkv-paper-v1-analysis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --partition=bii
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/paper-v1-analysis-%j.log
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
: "${MOSAICKV_PAPER_CANDIDATE_EXPECTED_GIT_SHA:?immutable source SHA is required}"
: "${MOSAICKV_PAPER_CANDIDATE_OUTPUT_ROOT:?preflight output root is required}"

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
export PYTHONPATH="${PACKAGE_ROOT}/src:${PACKAGE_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"
[[ "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)" == "${MOSAICKV_PAPER_CANDIDATE_EXPECTED_GIT_SHA}" ]]
"${PYTHON}" scripts/analyze_paper_candidate_v1_preflight.py \
  --output-root "${MOSAICKV_PAPER_CANDIDATE_OUTPUT_ROOT}"
