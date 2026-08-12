#!/bin/bash
#SBATCH --job-name=mosaickv-v25-latency
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=340G
#SBATCH --time=00:30:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:2
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/choice-c-v25-latency-%j.log

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
export MOSAICKV_PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
export MOSAICKV_LATENCY_SPEC=configs/experiments/choice_c_batched_prototype_system_v16.yaml
export MOSAICKV_LATENCY_PROFILE=/scratch/djy8hg/runs/mosaickv/choice-c-forecast-stability-design-v15/frozen_fidelity_profile.json
export MOSAICKV_LATENCY_LEDGER=configs/ledgers/choice_c_v16_system_parity_docvqa.json
OUTPUT_ROOT=/scratch/djy8hg/runs/mosaickv/choice-c-batched-source-gather-latency-v25
export MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}

# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

sha256sum \
  "${MOSAICKV_LATENCY_SPEC}" \
  "${MOSAICKV_LATENCY_PROFILE}" \
  "${MOSAICKV_LATENCY_LEDGER}" \
  scripts/run_choice_c_latency_worker.py \
  scripts/run_choice_c_latency_round.sh \
  scripts/analyze_choice_c_latency_ab.py \
  src/mosaickv/cache_state.py \
  src/mosaickv/config.py \
  src/mosaickv/graph/builder.py \
  src/mosaickv/graph/pooling.py \
  src/mosaickv/prototypes/construction.py

"${MOSAICKV_PYTHON}" scripts/prewarm_choice_c_ledger.py \
  --spec "${MOSAICKV_LATENCY_SPEC}" \
  --ledger-key calibration_validation_ledger

# Each round assigns one 160-GiB task and one GPU to each worker. Reversing the
# mode-to-rank mapping in round 1 counterbalances device and task-rank effects.
# The candidate combines deferred descriptor/attention host transfer with
# batched per-layer source gathering; the reference preserves both old paths.
for ROUND in 0 1; do
  srun --exact --exclusive --nodes=1 --ntasks=2 --cpus-per-task=16 \
    --gpus-per-task=1 --gpu-bind=single:1 --mem-per-cpu=10G \
    scripts/run_choice_c_latency_round.sh "${ROUND}" "${OUTPUT_ROOT}"
done

"${MOSAICKV_PYTHON}" scripts/analyze_choice_c_latency_ab.py \
  --output-root "${OUTPUT_ROOT}" \
  --required-speedup 0.03
