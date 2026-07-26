#!/bin/bash
#SBATCH --job-name=mosaickv-hf-fullkv-synthetic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
PYTHON=${MOSAICKV_HF_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
OUTPUT_ROOT=${MOSAICKV_OUTPUT_ROOT:?set MOSAICKV_OUTPUT_ROOT}
RUN_ID=${MOSAICKV_RUN_ID:?set MOSAICKV_RUN_ID}
ATTENTION=${MOSAICKV_HF_ATTENTION_IMPLEMENTATION:-sdpa}

export MOSAICKV_CACHE_ROOT="${CACHE_ROOT}"
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"
srun "${PYTHON}" -m mosaickv.cli evaluate \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --model-revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --backend hf \
  --attention-backend "${ATTENTION}" \
  --method full_kv \
  --task synthetic_smoke \
  --dataset-revision schema-v1 \
  --retention-ratio 1.0 \
  --max-new-tokens 16 \
  --precision bf16 \
  --subset-size 2 \
  --run-id "${RUN_ID}" \
  --output-dir "${OUTPUT_ROOT}" \
  --trace-directory "${OUTPUT_ROOT}/traces" \
  --local-files-only \
  --json
