#!/bin/bash
#SBATCH --job-name=mosaickv-v11-design
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=520G
#SBATCH --time=00:45:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:4
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/choice-c-v11-design-%j.log

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
MOSAICKV_PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
SPEC=${MOSAICKV_CHOICE_C_SPEC:-configs/experiments/choice_c_uncertainty_system_design_v11.yaml}
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
  src/mosaickv/prototypes/construction.py

pids=()
for variant in \
  forecast_mass_medoid \
  uncertainty_exact_1pct \
  uncertainty_exact_2pct \
  uncertainty_exact_2pct_graph4; do
  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=12 --mem=125G \
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
