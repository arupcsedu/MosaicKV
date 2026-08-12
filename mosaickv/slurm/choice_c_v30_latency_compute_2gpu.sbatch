#!/bin/bash
#SBATCH --job-name=mosaickv-v30-compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=340G
#SBATCH --time=00:30:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:2
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/choice-c-v30-compute-%j.log

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
export MOSAICKV_PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
export MOSAICKV_LATENCY_SPEC=configs/experiments/choice_c_batched_prototype_system_v16.yaml
export MOSAICKV_LATENCY_PROFILE=/scratch/djy8hg/runs/mosaickv/choice-c-forecast-stability-design-v15/frozen_fidelity_profile.json
export MOSAICKV_LATENCY_LEDGER=configs/ledgers/choice_c_v16_system_parity_docvqa.json
OUTPUT_ROOT=/scratch/djy8hg/runs/mosaickv/choice-c-online-latency-compute-v30-2gpu
export MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}

# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

if [[ -e ${OUTPUT_ROOT} ]]; then
  echo "refusing to overwrite immutable output root: ${OUTPUT_ROOT}" >&2
  exit 2
fi

sha256sum \
  "${MOSAICKV_LATENCY_SPEC}" \
  "${MOSAICKV_LATENCY_PROFILE}" \
  "${MOSAICKV_LATENCY_LEDGER}" \
  scripts/verify_choice_c_gpu_binding.py \
  scripts/run_choice_c_latency_compute_worker.py \
  scripts/run_choice_c_latency_compute_pair.sh \
  scripts/analyze_choice_c_latency_compute.py \
  src/mosaickv/backends/hf_runtime.py \
  src/mosaickv/config.py \
  src/mosaickv/forecasting/core.py \
  src/mosaickv/selection/utility.py

"${MOSAICKV_PYTHON}" scripts/prewarm_choice_c_ledger.py \
  --spec "${MOSAICKV_LATENCY_SPEC}" \
  --ledger-key calibration_validation_ledger

GPU_SELECTOR=${CUDA_VISIBLE_DEVICES:-}
if [[ -n ${GPU_SELECTOR} ]]; then
  IFS=',' read -r -a ALLOCATED_GPUS <<<"${GPU_SELECTOR}"
  CLOCK_RC=0
  CLOCK_OUTPUT=""
  for GPU_ID in "${ALLOCATED_GPUS[@]}"; do
    DETAIL=$(nvidia-smi -i "${GPU_ID}" -lgc 1095,1095 2>&1) || CLOCK_RC=$?
    CLOCK_OUTPUT+="gpu=${GPU_ID}: ${DETAIL}; "
  done
else
  ALLOCATED_GPUS=()
  CLOCK_OUTPUT="CUDA_VISIBLE_DEVICES was not set in the batch process"
  CLOCK_RC=2
fi
cleanup_clocks() {
  for GPU_ID in "${ALLOCATED_GPUS[@]}"; do
    nvidia-smi -i "${GPU_ID}" -rgc >/dev/null 2>&1 || true
  done
}
trap cleanup_clocks EXIT
if [[ ${CLOCK_RC} -eq 0 ]]; then
  export MOSAICKV_FIXED_CLOCK_STATUS=locked_1095_mhz
else
  export MOSAICKV_FIXED_CLOCK_STATUS="not_permitted_or_unavailable_rc_${CLOCK_RC}"
fi
echo "clock_status=${MOSAICKV_FIXED_CLOCK_STATUS} detail=${CLOCK_OUTPUT}"
nvidia-smi --query-gpu=index,name,uuid,pstate,clocks.current.sm,clocks.current.memory,power.draw \
  --format=csv,noheader,nounits

for ROUND in 0 1; do
  srun --exact --exclusive --nodes=1 --ntasks=2 --cpus-per-task=16 \
    --gpus-per-task=1 --gpu-bind=verbose,map_gpu:0,1 \
    --cpu-bind=cores --mem-bind=local --distribution=block:block \
    --kill-on-bad-exit=1 --mem-per-cpu=10G \
    scripts/run_choice_c_latency_compute_pair.sh \
    "${ROUND}" "${OUTPUT_ROOT}"
done

# This tolerance is fixed before the run and matches the selector's existing
# float64 marginal-gain unit-test contract. Identity/rank/reason remain exact.
"${MOSAICKV_PYTHON}" scripts/analyze_choice_c_latency_compute.py \
  --output-root "${OUTPUT_ROOT}" \
  --required-speedup 0.03 \
  --selection-score-atol 1e-12 \
  --selection-score-rtol 1e-12
