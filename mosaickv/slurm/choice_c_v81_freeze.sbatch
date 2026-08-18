#!/bin/bash
#SBATCH --job-name=mosaickv-v81-freeze
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --partition=parallel
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/v81-freeze-%j.log
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
OUTPUT_ROOT=${MOSAICKV_V81_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v81-disjoint-exact-future-rerank-calibration-v1}
SPEC=${MOSAICKV_V81_SPEC:-${PACKAGE_ROOT}/configs/experiments/choice_c_v81_disjoint_exact_future_rerank_calibration.yaml}
FROZEN=${MOSAICKV_V81_FROZEN:-${OUTPUT_ROOT}/analysis/frozen_candidate.yaml}

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
export PYTHONPATH="${PACKAGE_ROOT}/src:${PACKAGE_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"
"${PYTHON}" scripts/analyze_choice_c_v74_label_blind_calibration.py \
  --spec "${SPEC}" \
  --output-root "${OUTPUT_ROOT}" \
  --frozen-output "${FROZEN}" \
  --run-prefix v81
