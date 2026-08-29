#!/bin/bash
#SBATCH --job-name=mk-hf-quality-gate
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --time=00:20:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/djy8hg/slurm-logs/%x-%A_%a.out
#SBATCH --error=/scratch/djy8hg/slurm-logs/%x-%A_%a.err
#SBATCH --export=ALL

# Retrospective development-only paired quality/memory gate. Paper evaluation
# must replace this seeded subset with a reviewed immutable sample ledger.

set -euo pipefail
module load "${MOSAICKV_GCC_MODULE:-gcc/11.4.0}"

REPOSITORY=/scratch/djy8hg/workdir/MosaicKV
PROJECT_ROOT=${REPOSITORY}/mosaickv
PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python3}
CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
OUTPUT_ROOT=${MOSAICKV_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/hf-quality-development-gate-${SLURM_ARRAY_JOB_ID}}

TASKS=(mmstar textvqa docvqa)
DATASET_REVISIONS=(
  bc98d668301da7b14f648724866e57302778ab27
  9c0699cd19768ac5ab97568f6b3cbac4c0062884
  539088ef8a8ada01ac8e2e6d4e372586748a265e
)
TASK=${TASKS[${SLURM_ARRAY_TASK_ID}]}
DATASET_REVISION=${DATASET_REVISIONS[${SLURM_ARRAY_TASK_ID}]}
TASK_ROOT=${OUTPUT_ROOT}/${TASK}

export MOSAICKV_CACHE_ROOT=${CACHE_ROOT}
# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
export MOSAICKV_ALLOW_DIRTY_EXPLORATORY=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_CUDA_ARCH_LIST=8.0
mkdir -p "${TASK_ROOT}" "${TMPDIR}"
cd "${PROJECT_ROOT}"

COMMON=(
  --backend hf --model llava-hf/llava-1.5-7b-hf
  --model-revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9
  --dataset-revision "${DATASET_REVISION}" --attention-backend eager
  --task "${TASK}" --block-size 4 --max-new-tokens 64 --precision fp16
  --seed 20270911 --subset-size 8 --local-files-only --json
)

srun "${PYTHON}" -m mosaickv.cli evaluate \
  "${COMMON[@]}" --method prompt_attention_topk --retention-ratio 0.13 \
  --cache-budget 9223372036854775807 --budget-unit bytes \
  --repair-policy none --run-id "hf-gate-prompt-${TASK}" \
  --output-dir "${TASK_ROOT}/prompt_attention_topk" \
  --trace-directory "${TASK_ROOT}/prompt_attention_topk/traces"

srun "${PYTHON}" -m mosaickv.cli evaluate \
  "${COMMON[@]}" --method mosaickv_full --retention-ratio 0.075 \
  --cache-budget 28500000 --budget-unit bytes --forecast hybrid \
  --prompt-window 16 --draft-tokens 4 \
  --repair-policy entropy_or_prototype_risk --repair-blocks 4 \
  --run-id "hf-gate-mosaic-${TASK}" \
  --output-dir "${TASK_ROOT}/mosaickv_full" \
  --trace-directory "${TASK_ROOT}/mosaickv_full/traces"
