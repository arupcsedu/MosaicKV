#!/bin/bash
#SBATCH --job-name=mosaickv-v11-heldout
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=10
#SBATCH --mem=600G
#SBATCH --time=01:15:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:5
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/choice-c-v11-heldout-%j.log

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

gate=/scratch/djy8hg/runs/mosaickv/choice-c-uncertainty-system-design-v11/internal_validation_gate.json
"${MOSAICKV_PYTHON}" - "${gate}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists() or json.loads(path.read_text()).get("pass") is not True:
    raise SystemExit("held-out execution is forbidden until the internal gate passes")
PY

sha256sum \
  "${SPEC}" \
  scripts/run_choice_c_gate.py \
  src/mosaickv/backends/hf_runtime.py \
  src/mosaickv/prototypes/construction.py

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
