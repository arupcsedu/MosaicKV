#!/bin/bash
#SBATCH --job-name=mosaickv-internal-gate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=10
#SBATCH --mem=680G
#SBATCH --time=01:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:5
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/choice-c-internal-gate-%j.log

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
MOSAICKV_PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
SPEC=${MOSAICKV_CHOICE_C_SPEC:-configs/experiments/choice_c_forecast_mass_design_v10.yaml}
export MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
export MOSAICKV_BASELINE_BUDGET_PROFILE=${MOSAICKV_BASELINE_BUDGET_PROFILE:-${PROJECT_ROOT}/configs/experiments/choice_c_baseline_budget_profile_v10.yaml}

# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

sha256sum "${SPEC}" "${MOSAICKV_BASELINE_BUDGET_PROFILE}" scripts/run_choice_c_gate.py \
  src/mosaickv/backends/hf_runtime.py src/mosaickv/prototypes/construction.py

pids=()
for method in \
  full_kv \
  prompt_attention_topk \
  lookm_reimpl \
  vl_cache_reimpl \
  mosaickv_full; do
  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=10 --mem=132G \
    --gres=gpu:1 \
    "${MOSAICKV_PYTHON}" scripts/run_choice_c_gate.py \
      --spec "${SPEC}" \
      --phase internal-method \
      --variant "${method}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
