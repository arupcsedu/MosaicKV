#!/bin/bash
#SBATCH --job-name=mosaickv-tail-repair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=10
#SBATCH --mem=330G
#SBATCH --time=00:15:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:3
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/choice-c-tail-repair-%j.log

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
MOSAICKV_PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
SOURCE_SPEC=configs/experiments/choice_c_uncertainty_system_design_v11.yaml
LEDGER=configs/ledgers/choice_c_v11_tail_diagnostic_docvqa.json
OUTPUT_ROOT=/scratch/djy8hg/runs/mosaickv/choice-c-v11-tail-repair-diagnostic-v2
export MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}

# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

sha256sum \
  "${SOURCE_SPEC}" \
  "${LEDGER}" \
  scripts/run_choice_c_repair_diagnostic.py \
  scripts/run_choice_c_gate.py \
  src/mosaickv/prototypes/refresh.py \
  src/mosaickv/repair/core.py \
  src/mosaickv/backends/hf_runtime.py

pids=()
for restored in 1 2 4; do
  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=10 --mem=105G \
    --gres=gpu:1 \
    "${MOSAICKV_PYTHON}" scripts/run_choice_c_repair_diagnostic.py \
      --source-spec "${SOURCE_SPEC}" \
      --ledger "${LEDGER}" \
      --max-blocks "${restored}" \
      --output-root "${OUTPUT_ROOT}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
