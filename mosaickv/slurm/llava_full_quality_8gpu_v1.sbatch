#!/bin/bash
#SBATCH --job-name=mosaickv-llava-full-q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:a100:8
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/llava-full-quality-8gpu-v1-%j.log

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
export MOSAICKV_PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
export MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
export MOSAICKV_QUALITY_INDEX=${MOSAICKV_QUALITY_INDEX:-/scratch/djy8hg/runs/mosaickv/campaigns/llava-full-quality-8gpu-v1/jobs.jsonl}
OUTPUT_ROOT=/scratch/djy8hg/runs/mosaickv/llava-full-quality-8gpu-v1

# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "${PROJECT_ROOT}"

if [[ ! -r ${MOSAICKV_QUALITY_INDEX} ]]; then
  echo "immutable campaign index is missing: ${MOSAICKV_QUALITY_INDEX}" >&2
  exit 2
fi
if [[ -f ${OUTPUT_ROOT}/completion_report.json ]] && \
  "${MOSAICKV_PYTHON}" -c \
    'import json,sys; sys.exit(json.load(open(sys.argv[1]))["status"] != "passed")' \
    "${OUTPUT_ROOT}/completion_report.json"; then
  echo "campaign already has a passing completion report: ${OUTPUT_ROOT}" >&2
  exit 0
fi
mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/logs"

sha256sum \
  "${MOSAICKV_QUALITY_INDEX}" \
  configs/experiments/llava_full_quality_8gpu_v1.yaml \
  scripts/run_gpu_pilot_worker.py \
  scripts/run_llava_full_quality_worker.sh \
  scripts/validate_llava_full_quality_run.py \
  src/mosaickv/backends/hf_runtime.py \
  src/mosaickv/config.py
nvidia-smi --query-gpu=index,name,uuid,pstate,memory.total,memory.free \
  --format=csv,noheader,nounits

set +e
srun --exact --exclusive --nodes=1 --ntasks=8 --cpus-per-task=16 \
  --gpus-per-task=1 --gpu-bind=verbose,map_gpu:0,1,2,3,4,5,6,7 \
  --cpu-bind=cores --mem-bind=local --distribution=block:block \
  --kill-on-bad-exit=0 \
  --output="${OUTPUT_ROOT}/logs/worker-%t-%j.log" \
  scripts/run_llava_full_quality_worker.sh
WORKER_STATUS=$?
set -e

set +e
"${MOSAICKV_PYTHON}" scripts/validate_llava_full_quality_run.py \
  --index "${MOSAICKV_QUALITY_INDEX}" \
  --output-root "${OUTPUT_ROOT}"
VALIDATION_STATUS=$?
set -e

if (( WORKER_STATUS != 0 )); then
  echo "one or more workers reported errors: status=${WORKER_STATUS}" >&2
fi
if (( VALIDATION_STATUS != 0 )); then
  echo "campaign validation failed: status=${VALIDATION_STATUS}" >&2
fi
if (( WORKER_STATUS != 0 || VALIDATION_STATUS != 0 )); then
  exit 1
fi
