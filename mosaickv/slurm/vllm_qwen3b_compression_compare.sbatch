#!/bin/bash
#SBATCH --job-name=mosaickv-vllm-compare
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/djy8hg/slurm-logs/%x-%A_%a.out
#SBATCH --error=/scratch/djy8hg/slurm-logs/%x-%A_%a.err
#SBATCH --export=ALL

# Exploratory common-selector/vLLM comparison. Each task uses one isolated A100
# and runs FullKV plus equal-budget Random, Prompt Top-K, and MosaicKV-exact.

set -euo pipefail

module load "${MOSAICKV_GCC_MODULE:-gcc/11.4.0}"

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python3}
CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is required}
ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
OUTPUT_ROOT=${MOSAICKV_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/vllm-qwen3b-comparison-${ARRAY_JOB_ID}}
EXPECTED_SAMPLES=${MOSAICKV_EXPECTED_SAMPLES:-2}

TASKS=(mmstar textvqa docvqa)
DATASET_REVISIONS=(
  bc98d668301da7b14f648724866e57302778ab27
  9c0699cd19768ac5ab97568f6b3cbac4c0062884
  539088ef8a8ada01ac8e2e6d4e372586748a265e
)
TASK=${TASKS[${ARRAY_TASK_ID}]}
DATASET_REVISION=${DATASET_REVISIONS[${ARRAY_TASK_ID}]}
MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct
MODEL_REVISION=66285546d2b821cf421d4f5eb2576359d3770cd3
RETENTION_RATIO=${MOSAICKV_RETENTION_RATIO:-0.5}
TASK_ROOT=${OUTPUT_ROOT}/${TASK}

export MOSAICKV_CACHE_ROOT=${CACHE_ROOT}
# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
# This bridge is intentionally exploratory until the user authorizes an
# immutable commit. Preserve dirty-source provenance rather than pretending
# the rows are canonical.
export MOSAICKV_ALLOW_DIRTY_EXPLORATORY=1
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export HF_HOME=${CACHE_ROOT}/huggingface
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${CACHE_ROOT}/datasets
export XDG_CACHE_HOME=${CACHE_ROOT}/xdg
export TORCH_HOME=${CACHE_ROOT}/torch
export VLLM_CACHE_ROOT=${CACHE_ROOT}/vllm
export TMPDIR=${CACHE_ROOT}/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_CUDA_ARCH_LIST=8.0
mkdir -p "${TASK_ROOT}" "${TMPDIR}"
cd "${PROJECT_ROOT}"

COMMON_ARGS=(
  --model "${MODEL_ID}"
  --model-revision "${MODEL_REVISION}"
  --dataset-revision "${DATASET_REVISION}"
  --attention-backend eager
  --task "${TASK}"
  --block-size 16
  --max-new-tokens 16
  --precision bf16
  --seed 17
  --subset-size "${EXPECTED_SAMPLES}"
  --local-files-only
  --json
)

# Ordinary vLLM FullKV reference.
FULL_RUN=vllm-${TASK}-full-${ARRAY_JOB_ID}-${ARRAY_TASK_ID}
srun "${PYTHON}" -m mosaickv.cli evaluate \
  --backend vllm \
  --method full_kv \
  --retention-ratio 1.0 \
  "${COMMON_ARGS[@]}" \
  --run-id "${FULL_RUN}" \
  --output-dir "${TASK_ROOT}/full_kv" \
  --trace-directory "${TASK_ROOT}/full_kv/traces" \
  --cache-probe-repeats 2 \
  --gpu-memory-utilization 0.75 \
  --vllm-max-model-len 4096 \
  --tensor-parallel-size 1

for METHOD in random_kv prompt_attention_topk mosaickv_exact; do
  HF_RUN=hf-${TASK}-${METHOD}-${ARRAY_JOB_ID}-${ARRAY_TASK_ID}
  HF_ROOT=${TASK_ROOT}/${METHOD}/hf
  PLAN=${TASK_ROOT}/${METHOD}/selector_plan.json
  VLLM_RUN=vllm-${TASK}-${METHOD}-${ARRAY_JOB_ID}-${ARRAY_TASK_ID}

  EXTRA_HF_ARGS=(--repair-policy none)
  if [[ "${METHOD}" == mosaickv_exact ]]; then
    EXTRA_HF_ARGS+=(--forecast hybrid --draft-tokens 4 --prompt-window 8)
  elif [[ "${METHOD}" == prompt_attention_topk ]]; then
    EXTRA_HF_ARGS+=(--forecast prompt_window --prompt-window 8)
  fi

  srun "${PYTHON}" -m mosaickv.cli evaluate \
    --backend hf \
    --method "${METHOD}" \
    --retention-ratio "${RETENTION_RATIO}" \
    "${COMMON_ARGS[@]}" \
    "${EXTRA_HF_ARGS[@]}" \
    --run-id "${HF_RUN}" \
    --output-dir "${HF_ROOT}" \
    --trace-directory "${HF_ROOT}/traces"

  srun "${PYTHON}" scripts/build_vllm_selector_plan.py \
    --manifest "${HF_ROOT}/${HF_RUN}/manifest.json" \
    --trace-root "${HF_ROOT}/traces/${HF_RUN}" \
    --task "${TASK}" \
    --output "${PLAN}"
  PLAN_SHA=$(sha256sum "${PLAN}" | cut -d' ' -f1)

  srun "${PYTHON}" -m mosaickv.cli evaluate \
    --backend vllm \
    --method "${METHOD}" \
    --retention-ratio "${RETENTION_RATIO}" \
    "${COMMON_ARGS[@]}" \
    --enable-mosaickv \
    --vllm-selector-plan "${PLAN}" \
    --vllm-selector-plan-sha "${PLAN_SHA}" \
    --run-id "${VLLM_RUN}" \
    --output-dir "${TASK_ROOT}/${METHOD}/vllm" \
    --trace-directory "${TASK_ROOT}/${METHOD}/vllm/traces" \
    --cache-probe-repeats 2 \
    --gpu-memory-utilization 0.75 \
    --vllm-max-model-len 4096 \
    --tensor-parallel-size 1
done
