#!/bin/bash
#SBATCH --job-name=mosaickv-v19-heldout
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=10
#SBATCH --mem=600G
#SBATCH --time=01:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:5
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/choice-c-v19-heldout-%j.log

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
MOSAICKV_PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
SPEC=configs/experiments/choice_c_forecast_stability_design_v15.yaml
OUTPUT_ROOT=${MOSAICKV_OUTPUT_ROOT_OVERRIDE:?set MOSAICKV_OUTPUT_ROOT_OVERRIDE}
export MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}

# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

"${MOSAICKV_PYTHON}" - "${OUTPUT_ROOT}/internal_validation_gate.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists() or json.loads(path.read_text()).get("pass") is not True:
    raise SystemExit("held-out execution is forbidden until the optimized internal gate passes")
PY

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
  --ledger-key evaluation_ledger

pids=()
for method in full_kv prompt_attention_topk lookm_reimpl vl_cache_reimpl mosaickv_full; do
  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=10 --mem=115G \
    --gres=gpu:1 \
    "${MOSAICKV_PYTHON}" scripts/run_choice_c_gate.py \
      --spec "${SPEC}" \
      --phase heldout-method \
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
