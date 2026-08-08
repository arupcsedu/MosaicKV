#!/bin/bash
#SBATCH --job-name=mosaickv-choice-c-cal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=12
#SBATCH --mem=450G
#SBATCH --time=01:30:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:3
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/choice-c-cal-%j.log

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
MOSAICKV_PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
SPEC=${MOSAICKV_CHOICE_C_SPEC:-configs/experiments/choice_c_corrective_gate_v6.yaml}
export MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}

# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

pids=()
for candidate in prompt_anchored_15 evidence_anchored_15 balanced_23; do
  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=12 --mem=140G \
    --gres=gpu:1 \
    "${MOSAICKV_PYTHON}" scripts/run_choice_c_gate.py \
      --spec "${SPEC}" \
      --phase calibration-candidate \
      --variant "${candidate}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
