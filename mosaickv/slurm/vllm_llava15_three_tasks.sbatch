#!/bin/bash
#SBATCH --job-name=mosaickv-vllm-llava15
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/djy8hg/slurm-logs/%x-%A_%a.out
#SBATCH --error=/scratch/djy8hg/slurm-logs/%x-%A_%a.err
#SBATCH --export=ALL

# Paired retention-1 HF/vLLM verification on the three user-selected tasks.
# This script intentionally contains no SGLang invocation and emits no native
# compressed-MosaicKV row.

set -euo pipefail

module load "${MOSAICKV_GCC_MODULE:-gcc/11.4.0}"

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python3}
CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is required}
ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
OUTPUT_ROOT=${MOSAICKV_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/vllm-llava15-three-tasks-${ARRAY_JOB_ID}}
EXPECTED_SAMPLES=${MOSAICKV_EXPECTED_SAMPLES:-2}

TASKS=(mmstar textvqa docvqa)
DATASET_REVISIONS=(
  bc98d668301da7b14f648724866e57302778ab27
  9c0699cd19768ac5ab97568f6b3cbac4c0062884
  539088ef8a8ada01ac8e2e6d4e372586748a265e
)
TASK=${TASKS[${ARRAY_TASK_ID}]}
DATASET_REVISION=${DATASET_REVISIONS[${ARRAY_TASK_ID}]}
MODEL_ID=llava-hf/llava-1.5-7b-hf
MODEL_REVISION=b234b804b114d9e37bb655e11cbbb5f5e971b7a9
TASK_ROOT=${OUTPUT_ROOT}/${TASK}
HF_RUN_ID=hf-${TASK}-${ARRAY_JOB_ID}-${ARRAY_TASK_ID}
VLLM_RUN_ID=vllm-${TASK}-${ARRAY_JOB_ID}-${ARRAY_TASK_ID}

export MOSAICKV_CACHE_ROOT=${CACHE_ROOT}
# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"

if [[ ! -x "${PYTHON}" ]]; then
  printf 'Common MosaicKV interpreter is unavailable: %s\n' "${PYTHON}" >&2
  exit 2
fi
if (( EXPECTED_SAMPLES < 2 )); then
  printf 'At least two samples are required for the A-B-A isolation probe.\n' >&2
  exit 2
fi

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
mkdir -p "${TASK_ROOT}/hf" "${TASK_ROOT}/vllm" "${TMPDIR}"
cd "${PROJECT_ROOT}"

COMMON_ARGS=(
  --model "${MODEL_ID}"
  --model-revision "${MODEL_REVISION}"
  --dataset-revision "${DATASET_REVISION}"
  --attention-backend eager
  --method full_kv
  --task "${TASK}"
  --retention-ratio 1.0
  --block-size 16
  --max-new-tokens 16
  --precision bf16
  --seed 17
  --subset-size "${EXPECTED_SAMPLES}"
  --local-files-only
  --json
)

# The HF side establishes identical processor, prompt, media, task-scoring,
# precision, generation, and sample-selection controls.
srun "${PYTHON}" -m mosaickv.cli evaluate \
  --backend hf \
  "${COMMON_ARGS[@]}" \
  --run-id "${HF_RUN_ID}" \
  --output-dir "${TASK_ROOT}/hf" \
  --trace-directory "${TASK_ROOT}/hf/traces"

# vLLM uses one persistent V1 eager engine, CUDA graphs and prefix caching
# disabled, two identical trials per sample, and an A-B-A isolation recheck.
srun "${PYTHON}" -m mosaickv.cli evaluate \
  --backend vllm \
  "${COMMON_ARGS[@]}" \
  --run-id "${VLLM_RUN_ID}" \
  --output-dir "${TASK_ROOT}/vllm" \
  --trace-directory "${TASK_ROOT}/vllm/traces" \
  --cache-probe-repeats 2 \
  --gpu-memory-utilization 0.80 \
  --vllm-max-model-len 4096 \
  --tensor-parallel-size 1

srun "${PYTHON}" scripts/verify_vllm_llava15_three_tasks.py \
  --root "${OUTPUT_ROOT}" \
  --task "${TASK}" \
  --expected-samples "${EXPECTED_SAMPLES}" \
  --output "${TASK_ROOT}/verification.json"
