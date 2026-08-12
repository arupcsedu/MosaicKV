#!/bin/bash
#SBATCH --job-name=mosaickv-v23-latency
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --time=00:30:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:2
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/choice-c-v23-latency-%j.log

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
MOSAICKV_PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
SPEC=configs/experiments/choice_c_batched_prototype_system_v16.yaml
PROFILE=/scratch/djy8hg/runs/mosaickv/choice-c-forecast-stability-design-v15/frozen_fidelity_profile.json
LEDGER=configs/ledgers/choice_c_v16_system_parity_docvqa.json
OUTPUT_ROOT=/scratch/djy8hg/runs/mosaickv/choice-c-host-transfer-latency-v23
export MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}

# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

sha256sum \
  "${SPEC}" \
  "${PROFILE}" \
  "${LEDGER}" \
  scripts/run_choice_c_latency_worker.py \
  scripts/analyze_choice_c_latency_ab.py \
  src/mosaickv/config.py \
  src/mosaickv/graph/builder.py \
  src/mosaickv/graph/pooling.py

"${MOSAICKV_PYTHON}" scripts/prewarm_choice_c_ledger.py \
  --spec "${SPEC}" \
  --ledger-key calibration_validation_ledger

srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 \
  "${MOSAICKV_PYTHON}" scripts/run_choice_c_latency_worker.py \
  --mode per_group_reference \
  --spec "${SPEC}" \
  --profile "${PROFILE}" \
  --ledger "${LEDGER}" \
  --output-root "${OUTPUT_ROOT}" &
REFERENCE_PID=$!

srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 \
  "${MOSAICKV_PYTHON}" scripts/run_choice_c_latency_worker.py \
  --mode deferred_layer \
  --spec "${SPEC}" \
  --profile "${PROFILE}" \
  --ledger "${LEDGER}" \
  --output-root "${OUTPUT_ROOT}" &
CANDIDATE_PID=$!

wait "${REFERENCE_PID}"
wait "${CANDIDATE_PID}"

"${MOSAICKV_PYTHON}" scripts/analyze_choice_c_latency_ab.py \
  --output-root "${OUTPUT_ROOT}" \
  --required-speedup 0.03
