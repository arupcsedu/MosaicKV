#!/bin/bash
#SBATCH --job-name=mosaickv-llava-cont-array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-5%6
#SBATCH --requeue
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/llava-full-quality-cont-array-%A_%a.log

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
export MOSAICKV_PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
export MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
export MOSAICKV_CONTINUATION_INDEX=${MOSAICKV_CONTINUATION_INDEX:-/scratch/djy8hg/runs/mosaickv/campaigns/llava-full-quality-continuation-6gpu-v1/jobs.jsonl}
OUTPUT_ROOT=/scratch/djy8hg/runs/mosaickv/llava-full-quality-continuation-6gpu-v1
LAUNCH_SPEC=configs/experiments/llava_full_quality_continuation_array_launch_v3.yaml

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

: "${SLURM_ARRAY_TASK_ID:?this launcher must run as a Slurm array task}"
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= 6 )); then
  echo "invalid continuation array index: ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi
if [[ ! -r ${MOSAICKV_CONTINUATION_INDEX} ]]; then
  echo "immutable continuation index is missing: ${MOSAICKV_CONTINUATION_INDEX}" >&2
  exit 2
fi
if [[ -f ${OUTPUT_ROOT}/completion_report.json ]] && \
  "${MOSAICKV_PYTHON}" -c \
    'import json,sys; sys.exit(json.load(open(sys.argv[1]))["status"] != "passed")' \
    "${OUTPUT_ROOT}/completion_report.json"; then
  echo "continuation already has a passing completion report: ${OUTPUT_ROOT}" >&2
  exit 0
fi
mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/logs"

printf '%s  %s\n' \
  bb5d3db9092f2e5e5e6aec10715cb236a06901957836ada4009a2f89f8bd80b8 \
  src/mosaickv/backends/hf_runtime.py \
  9559feb1bdd2baa187f25695c3aa8e8b64ded0822cf07d7102ed8e6166a82c93 \
  src/mosaickv/config.py | sha256sum --check --strict
printf '%s  %s\n' \
  ff324e336ca914cd17ebfac0420d0d4c098f46ecbdbcfd93666c3ec51883ad54 \
  "${MOSAICKV_CONTINUATION_INDEX}" | sha256sum --check --strict
sha256sum \
  "${LAUNCH_SPEC}" \
  scripts/run_llava_quality_continuation_worker.py \
  scripts/validate_llava_quality_continuation.py
nvidia-smi --query-gpu=index,name,uuid,pstate,memory.total,memory.free \
  --format=csv,noheader,nounits

exec "${MOSAICKV_PYTHON}" scripts/run_llava_quality_continuation_worker.py \
  --index "${MOSAICKV_CONTINUATION_INDEX}" \
  --worker-index "${SLURM_ARRAY_TASK_ID}"
