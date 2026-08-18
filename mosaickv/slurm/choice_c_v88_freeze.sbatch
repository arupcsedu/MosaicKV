#!/bin/bash
#SBATCH --job-name=mosaickv-v88-freeze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --partition=bii
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/v88-freeze-%j.log
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
OUTPUT_ROOT=${MOSAICKV_V88_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v88-initial-query-coverage-calibration-v1}
SPEC=${MOSAICKV_V88_SPEC:-${PACKAGE_ROOT}/configs/experiments/choice_c_v88_initial_query_coverage_calibration.yaml}
FROZEN=${MOSAICKV_V88_FROZEN:-${OUTPUT_ROOT}/analysis/frozen_candidate.yaml}

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
export PYTHONPATH="${PACKAGE_ROOT}/src:${PACKAGE_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"

set +e
"${PYTHON}" scripts/analyze_choice_c_v86_robust_fidelity.py \
  --spec "${SPEC}" \
  --output-root "${OUTPUT_ROOT}" \
  --frozen-output "${FROZEN}" \
  --run-prefix v88
ANALYSIS_STATUS=$?
set -e

if [[ "${ANALYSIS_STATUS}" -eq 2 && ! -e "${FROZEN}" ]]; then
  printf 'V88 completed as a negative calibration: no candidate passed every unchanged gate.\n'
  exit 0
fi
exit "${ANALYSIS_STATUS}"
