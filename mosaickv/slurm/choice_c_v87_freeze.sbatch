#!/bin/bash
#SBATCH --job-name=mosaickv-v87-freeze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --partition=bii
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/v87-freeze-%j.log
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
OUTPUT_ROOT=${MOSAICKV_V87_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v87-prompt-query-coverage-calibration-v1}
SPEC=${MOSAICKV_V87_SPEC:-${PACKAGE_ROOT}/configs/experiments/choice_c_v87_prompt_query_coverage_development.yaml}
FROZEN=${MOSAICKV_V87_FROZEN:-${OUTPUT_ROOT}/analysis/frozen_candidate.yaml}

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
export PYTHONPATH="${PACKAGE_ROOT}/src:${PACKAGE_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"

set +e
"${PYTHON}" scripts/analyze_choice_c_v86_robust_fidelity.py \
  --spec "${SPEC}" \
  --output-root "${OUTPUT_ROOT}" \
  --frozen-output "${FROZEN}" \
  --run-prefix v87
ANALYSIS_STATUS=$?
set -e

if [[ "${ANALYSIS_STATUS}" -eq 2 && ! -e "${FROZEN}" ]]; then
  printf 'V87 completed as a negative calibration: no candidate passed every unchanged gate.\n'
  exit 0
fi
exit "${ANALYSIS_STATUS}"
