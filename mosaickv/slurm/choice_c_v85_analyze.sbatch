#!/bin/bash
#SBATCH --job-name=mosaickv-v85-analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --partition=bii
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/v85-analyze-%j.log
#SBATCH --export=ALL

set -euo pipefail

PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
PACKAGE_ROOT=${MOSAICKV_V85_PROTOCOL_PACKAGE_ROOT:?missing pinned v85 protocol package root}
OUTPUT_ROOT=${MOSAICKV_V85_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v85-v84-frozen-untouched-heldout-quality-memory-v1}
SPEC=${MOSAICKV_V85_SPEC:?missing immutable v85 preregistration}
EXPECTED_SPEC_SHA=${MOSAICKV_V85_SPEC_SHA256:?missing v85 preregistration hash}
FROZEN=${MOSAICKV_V85_FROZEN:?missing v84 frozen candidate}
EXPECTED_FROZEN_SHA=${MOSAICKV_V85_FROZEN_SHA256:?missing v84 frozen-candidate hash}

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
export PYTHONPATH="${PACKAGE_ROOT}/src:${PACKAGE_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"
[[ "$(sha256sum "${SPEC}" | cut -d' ' -f1)" == "${EXPECTED_SPEC_SHA}" ]]
[[ "$(sha256sum "${FROZEN}" | cut -d' ' -f1)" == "${EXPECTED_FROZEN_SHA}" ]]
"${PYTHON}" scripts/analyze_choice_c_v85_heldout_quality_memory.py \
  --spec "${SPEC}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-prefix v85 \
  --candidate-effective-method mosaickv_exact
