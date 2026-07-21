#!/bin/bash
#SBATCH --job-name=mosaickv-prefixkv-parity
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=37
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:a100:1
#SBATCH --export=ALL

set -euo pipefail

REPOSITORY_ROOT=/scratch/djy8hg/workdir/MosaicKV
PARITY_PYTHON=${MOSAICKV_PREFIXKV_PARITY_PYTHON:-/scratch/djy8hg/env/mosaickv_prefixkv_parity/bin/python}
MODEL_PATH=${MOSAICKV_PREFIXKV_MODEL_PATH:-/scratch/djy8hg/models/mosaickv/prefixkv/Zuyan-ElasticCache/llava-v1.5-7b}
DATA_PATH=${MOSAICKV_PREFIXKV_DATA_PATH:-/scratch/djy8hg/data/mosaickv/prefixkv/detail_1k.json}
MEDIA_ROOT=${MOSAICKV_PREFIXKV_MEDIA_ROOT:-/scratch/djy8hg/data/mosaickv/prefixkv}
OUTPUT_ROOT=${MOSAICKV_PREFIXKV_PARITY_OUTPUT:-/scratch/djy8hg/runs/mosaickv/prefixkv/official-parity}
RUN_ID=${MOSAICKV_RUN_ID:-prefixkv-official-parity-${SLURM_JOB_ID}}
PROJECT_ROOT=${REPOSITORY_ROOT}/mosaickv
export MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"

if [[ ! -x "${PARITY_PYTHON}" ]]; then
  printf 'PrefixKV parity environment is missing: %s\n' "${PARITY_PYTHON}" >&2
  exit 2
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=${REPOSITORY_ROOT}/mosaickv/src${PYTHONPATH:+:${PYTHONPATH}}

mkdir -p "${XDG_CACHE_HOME}/torch/kernels"

cd "${REPOSITORY_ROOT}"
srun "${PARITY_PYTHON}" mosaickv/scripts/run_prefixkv_llava_parity.py \
  --repository-root "${REPOSITORY_ROOT}" \
  --model-path "${MODEL_PATH}" \
  --data-path "${DATA_PATH}" \
  --media-root "${MEDIA_ROOT}" \
  --output-directory "${OUTPUT_ROOT}/${RUN_ID}" \
  --calibration-index 1 \
  --evaluation-index 0 \
  --retention-ratio 0.5 \
  --max-new-tokens 16 \
  --ppl-tokens 16 \
  --seed 0
