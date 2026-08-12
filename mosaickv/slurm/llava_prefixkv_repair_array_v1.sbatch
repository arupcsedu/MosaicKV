#!/bin/bash
#SBATCH --job-name=mosaickv-prefixkv-repair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-5%6
#SBATCH --requeue
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/llava-prefixkv-repair-v1-%A_%a.log

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
export MOSAICKV_PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
export MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
export MOSAICKV_CONTINUATION_INDEX=${MOSAICKV_CONTINUATION_INDEX:-/scratch/djy8hg/runs/mosaickv/campaigns/llava-full-quality-prefixkv-repair-v1/jobs.jsonl}
OUTPUT_ROOT=/scratch/djy8hg/runs/mosaickv/llava-full-quality-prefixkv-repair-v1

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
  echo "invalid PrefixKV repair worker index: ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi
if [[ ! -r ${MOSAICKV_CONTINUATION_INDEX} ]]; then
  echo "immutable PrefixKV repair index is missing: ${MOSAICKV_CONTINUATION_INDEX}" >&2
  exit 2
fi
if [[ -f ${OUTPUT_ROOT}/completion_report.json ]] && \
  "${MOSAICKV_PYTHON}" -c \
    'import json,sys; sys.exit(json.load(open(sys.argv[1]))["status"] != "passed")' \
    "${OUTPUT_ROOT}/completion_report.json"; then
  echo "repair campaign already has a passing completion report" >&2
  exit 0
fi

printf '%s  %s\n' \
  c6098a2e7223a884979b4e516c0007a008daed67c85038e7d2cabe0415938663 \
  src/mosaickv/backends/hf_runtime.py \
  9559feb1bdd2baa187f25695c3aa8e8b64ded0822cf07d7102ed8e6166a82c93 \
  src/mosaickv/config.py \
  b5c12f85bf03a91a7815645adea59e37a7f3f7308fe08711d47979437b475836 \
  scripts/run_llava_quality_continuation_worker.py | sha256sum --check --strict
sha256sum "${MOSAICKV_CONTINUATION_INDEX}"
nvidia-smi --query-gpu=index,name,uuid,pstate,memory.total,memory.free \
  --format=csv,noheader,nounits

exec "${MOSAICKV_PYTHON}" scripts/run_llava_quality_continuation_worker.py \
  --index "${MOSAICKV_CONTINUATION_INDEX}" \
  --worker-index "${SLURM_ARRAY_TASK_ID}"
