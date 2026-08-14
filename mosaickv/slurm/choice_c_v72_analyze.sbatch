#!/bin/bash
#SBATCH --job-name=mosaickv-v72-analyze
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --partition=bii
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/v72-analyze-%j.log
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
OUTPUT_ROOT=${MOSAICKV_V72_OUTPUT_ROOT:?set MOSAICKV_V72_OUTPUT_ROOT}
SPEC=${MOSAICKV_V72_SPEC:?set MOSAICKV_V72_SPEC}

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
export PYTHONPATH="${PACKAGE_ROOT}/src:${PACKAGE_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PACKAGE_ROOT}"
"${PYTHON}" scripts/analyze_choice_c_v72_heldout_quality_memory.py \
  --spec "${SPEC}" \
  --output-root "${OUTPUT_ROOT}"
