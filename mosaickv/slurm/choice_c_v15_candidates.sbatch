#!/bin/bash
#SBATCH --job-name=mosaickv-v15-design
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=18
#SBATCH --mem=300G
#SBATCH --time=00:50:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:2
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/choice-c-v15-design-%j.log

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
  scripts/prewarm_choice_c_ledger.py \
  scripts/run_choice_c_gate.py \
  src/mosaickv/backends/hf_runtime.py \
  src/mosaickv/forecasting/core.py \
  src/mosaickv/prototypes/construction.py

"${MOSAICKV_PYTHON}" scripts/prewarm_choice_c_ledger.py --spec "${SPEC}"

pids=()
for variant in draft4_prompt16 draft8_prompt16_draft_trust; do
  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=18 --mem=140G \
    --gres=gpu:1 \
    "${MOSAICKV_PYTHON}" scripts/run_choice_c_gate.py \
      --spec "${SPEC}" \
      --phase calibration-candidate \
      --variant "${variant}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
