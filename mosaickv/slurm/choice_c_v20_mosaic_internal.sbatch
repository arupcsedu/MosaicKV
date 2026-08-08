#!/bin/bash
#SBATCH --job-name=mosaickv-v20-internal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=00:25:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/choice-c-v20-internal-%j.log

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
MOSAICKV_PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
SPEC=configs/experiments/choice_c_forecast_stability_design_v15.yaml
export MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}

# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

sha256sum \
  "${SPEC}" \
  scripts/run_choice_c_gate.py \
  src/mosaickv/backends/hf_runtime.py \
  src/mosaickv/cache_state.py \
  src/mosaickv/graph/builder.py \
  src/mosaickv/graph/pooling.py \
  src/mosaickv/prototypes/construction.py \
  src/mosaickv/prototypes/rope.py

"${MOSAICKV_PYTHON}" scripts/prewarm_choice_c_ledger.py \
  --spec "${SPEC}" \
  --ledger-key calibration_validation_ledger

"${MOSAICKV_PYTHON}" scripts/run_choice_c_gate.py \
  --spec "${SPEC}" \
  --phase internal-method \
  --variant mosaickv_full
