#!/bin/bash
#SBATCH --job-name=mosaickv-v13-design
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=10
#SBATCH --mem=600G
#SBATCH --time=00:50:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:5
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/choice-c-v13-design-%j.log

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
MOSAICKV_PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
SPEC=${MOSAICKV_CHOICE_C_SPEC:-configs/experiments/choice_c_group_geometry_design_v13.yaml}
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
  base_g15_graph8 \
  base_g15_graph4 \
  g11_span255_graph4 \
  g11_span255_graph8 \
  g9_span255_graph4; do
  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=10 --mem=115G \
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
