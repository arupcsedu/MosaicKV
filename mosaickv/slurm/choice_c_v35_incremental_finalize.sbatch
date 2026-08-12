#!/bin/bash
#SBATCH --job-name=mosaickv-v35-fin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --partition=bii
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/choice-c-v35-finalize-%j.log

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
export MOSAICKV_PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
export MOSAICKV_CACHE_ROOT=/scratch/djy8hg/cache/mosaickv
PARENT_JOB_ID=${MOSAICKV_PARENT_JOB_ID:?MOSAICKV_PARENT_JOB_ID is required}
OUTPUT_ROOT=/scratch/djy8hg/runs/mosaickv/choice-c-incremental-v35-${PARENT_JOB_ID}

# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

"${MOSAICKV_PYTHON}" scripts/analyze_choice_c_incremental.py \
  --output-root "${OUTPUT_ROOT}"
