#!/bin/bash
#SBATCH --job-name=mosaickv-v77-analyze
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --partition=parallel
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/v77-analyze-%j.log
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
OUTPUT_ROOT=${MOSAICKV_V77_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v77-untouched-heldout-quality-memory-v1}
SPEC=${MOSAICKV_V77_SPEC:-${PACKAGE_ROOT}/configs/experiments/choice_c_v77_untouched_heldout_quality_memory.yaml}

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
export PYTHONPATH="${PACKAGE_ROOT}/src:${PACKAGE_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"
"${PYTHON}" scripts/analyze_choice_c_v73_clean_quality_memory.py \
  --spec "${SPEC}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-prefix v77
