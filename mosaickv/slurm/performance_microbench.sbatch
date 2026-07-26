#!/bin/bash
#SBATCH --job-name=mkv-exp2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:30:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --array=0-2%3
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/performance-%A_%a.out
#SBATCH --error=/scratch/djy8hg/runs/mosaickv/slurm/performance-%A_%a.err

set -euo pipefail

REPO=/scratch/djy8hg/workdir/MosaicKV/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
RUN_ROOT=/scratch/djy8hg/runs/mosaickv/performance-exp2
CAMPAIGN=${MOSAICKV_PERFORMANCE_CAMPAIGN:?set MOSAICKV_PERFORMANCE_CAMPAIGN}

source "${REPO}/scripts/cache_env.sh"
mkdir -p "${RUN_ROOT}/${CAMPAIGN}" /scratch/djy8hg/runs/mosaickv/slurm

BACKENDS=(hf_eager hf_eager hf_eager)
MODELS=(qwen3b qwen7b llava7b)
BACKEND=${MOSAICKV_PERFORMANCE_BACKEND_OVERRIDE:-${BACKENDS[${SLURM_ARRAY_TASK_ID}]}}
MODEL=${MODELS[${SLURM_ARRAY_TASK_ID}]}
OUTPUT_ROOT="${RUN_ROOT}/${CAMPAIGN}/${SLURM_ARRAY_TASK_ID}-${BACKEND}-${MODEL}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

GPU_SELECTOR=${CUDA_VISIBLE_DEVICES%%,*}
CLOCK_OUTPUT=$(nvidia-smi -i "${GPU_SELECTOR}" -lgc 1095,1095 2>&1) && CLOCK_RC=0 || CLOCK_RC=$?
if [[ ${CLOCK_RC} -eq 0 ]]; then
  export MOSAICKV_FIXED_CLOCK_STATUS="locked_1095_mhz"
  cleanup_clock() {
    nvidia-smi -i "${GPU_SELECTOR}" -rgc >/dev/null 2>&1 || true
  }
  trap cleanup_clock EXIT
else
  export MOSAICKV_FIXED_CLOCK_STATUS="not_permitted_rc_${CLOCK_RC}"
fi
printf 'clock_status=%s detail=%s\n' "${MOSAICKV_FIXED_CLOCK_STATUS}" "${CLOCK_OUTPUT}"
nvidia-smi --query-gpu=name,uuid,pstate,clocks.current.sm,clocks.current.memory,power.draw \
  --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits || true

EXTRA_ARGS=()
if [[ -n "${MOSAICKV_PERFORMANCE_POINT_IDS:-}" ]]; then
  IFS=',' read -r -a REQUESTED_POINTS <<<"${MOSAICKV_PERFORMANCE_POINT_IDS}"
  for point_id in "${REQUESTED_POINTS[@]}"; do
    EXTRA_ARGS+=(--point-id "${point_id}")
  done
fi
if [[ -n "${MOSAICKV_PERFORMANCE_METHOD_OVERRIDE:-}" ]]; then
  EXTRA_ARGS+=(--method-override "${MOSAICKV_PERFORMANCE_METHOD_OVERRIDE}")
fi

srun "${PYTHON}" "${REPO}/scripts/run_performance_microbench.py" \
  --backend "${BACKEND}" \
  --model "${MODEL}" \
  --campaign "${CAMPAIGN}" \
  --output-root "${OUTPUT_ROOT}" \
  --warmups 5 \
  --trials 20 \
  "${EXTRA_ARGS[@]}"
