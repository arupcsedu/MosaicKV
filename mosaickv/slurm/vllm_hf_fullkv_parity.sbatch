#!/bin/bash
#SBATCH --job-name=mosaickv-vllm-hf-parity
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=37
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

# Produce the HF FullKV side of the controlled vLLM/HF output comparison.

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
HF_PYTHON=${MOSAICKV_HF_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
OUTPUT_ROOT=${MOSAICKV_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/vllm-fullkv/hf-reference}
RUN_ID=${MOSAICKV_RUN_ID:-hf-fullkv-parity-${SLURM_JOB_ID}}

export MOSAICKV_CACHE_ROOT="${CACHE_ROOT}"
# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"

export PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers"
export TORCH_HOME="${CACHE_ROOT}/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"
srun "${HF_PYTHON}" -m mosaickv.cli evaluate \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --model-revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --dataset-revision schema-v1 \
  --backend hf \
  --attention-backend eager \
  --method full_kv \
  --task synthetic_smoke \
  --retention-ratio 1.0 \
  --block-size 16 \
  --max-new-tokens 16 \
  --precision bf16 \
  --seed 0 \
  --subset-size 1 \
  --run-id "${RUN_ID}" \
  --output-dir "${OUTPUT_ROOT}" \
  --trace-directory "${OUTPUT_ROOT}/traces" \
  --local-files-only \
  --json
