#!/bin/bash
#SBATCH --job-name=mosaickv-vllm-pareto
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

# Retrospective development reproduction of the HF quality-memory gate.  The
# frozen 4-token selector is projected to supported 16-token vLLM cache blocks.

set -euo pipefail

module load "${MOSAICKV_GCC_MODULE:-gcc/11.4.0}"

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python3}
CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
SOURCE_ROOT=/scratch/djy8hg/runs/mosaickv/paper-candidate-v1-retrospective-99a412bf0d2c
ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is required}
ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
OUTPUT_ROOT=${MOSAICKV_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/vllm-llava15-pareto-${ARRAY_JOB_ID}}

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
PROMPT_RETENTION=0.13
MOSAIC_SEED_RETENTION=0.075
SERVING_BLOCK_SIZE=16
TASK_ROOT=${OUTPUT_ROOT}/${TASK}
SAMPLE_FILE=${SOURCE_ROOT}/inputs/${TASK}.json
SAMPLE_SHA=$(sha256sum "${SAMPLE_FILE}" | cut -d' ' -f1)

export MOSAICKV_CACHE_ROOT=${CACHE_ROOT}
# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
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

VLLM_COMMON=(
  --model "${MODEL_ID}"
  --model-revision "${MODEL_REVISION}"
  --dataset-revision "${DATASET_REVISION}"
  --attention-backend eager
  --task "${TASK}"
  --block-size "${SERVING_BLOCK_SIZE}"
  --max-new-tokens 64
  --precision fp16
  --seed 17
  --local-files-only
  --json
  --cache-probe-repeats 2
  --gpu-memory-utilization 0.75
  --vllm-max-model-len 4096
  --tensor-parallel-size 1
)

# Ordinary vLLM FullKV reference on the exact two retrospective identities.
FULL_RUN=vllm-${TASK}-full-${ARRAY_JOB_ID}-${ARRAY_TASK_ID}
srun "${PYTHON}" -m mosaickv.cli evaluate \
  --backend vllm --method full_kv --retention-ratio 1.0 \
  "${VLLM_COMMON[@]}" \
  --sample-id-file "${SAMPLE_FILE}" --sample-id-file-sha "${SAMPLE_SHA}" \
  --run-id "${FULL_RUN}" \
  --output-dir "${TASK_ROOT}/full_kv/vllm" \
  --trace-directory "${TASK_ROOT}/full_kv/vllm/traces"

# Prompt Top-K at the frozen HF 13% operating point.
mapfile -t PROMPT_CONFIGS < <(find "${SOURCE_ROOT}/configs" -maxdepth 1 \
  -name "paper-v1-retro-${TASK}-99a412bf0d-prompt_attention_topk-*.yaml" | sort)
[[ ${#PROMPT_CONFIGS[@]} -eq 1 ]]
PROMPT_CONFIG=${PROMPT_CONFIGS[0]}
PROMPT_HF_RUN=hf-${TASK}-prompt-${ARRAY_JOB_ID}-${ARRAY_TASK_ID}
PROMPT_HF_ROOT=${TASK_ROOT}/prompt_attention_topk/hf
PROMPT_PLAN=${TASK_ROOT}/prompt_attention_topk/selector_plan.json
srun "${PYTHON}" -m mosaickv.cli evaluate \
  --config "${PROMPT_CONFIG}" --task "${TASK}" --local-files-only --json \
  --sample-id-file "${SAMPLE_FILE}" --sample-id-file-sha "${SAMPLE_SHA}" \
  --run-id "${PROMPT_HF_RUN}" --output-dir "${PROMPT_HF_ROOT}" \
  --trace-directory "${PROMPT_HF_ROOT}/traces"
srun "${PYTHON}" scripts/build_vllm_selector_plan.py \
  --manifest "${PROMPT_HF_ROOT}/${PROMPT_HF_RUN}/manifest.json" \
  --trace-root "${PROMPT_HF_ROOT}/traces/${PROMPT_HF_RUN}" \
  --task "${TASK}" --target-block-size "${SERVING_BLOCK_SIZE}" \
  --target-retention-ratio "${PROMPT_RETENTION}" --output "${PROMPT_PLAN}"
PROMPT_PLAN_SHA=$(sha256sum "${PROMPT_PLAN}" | cut -d' ' -f1)
PROMPT_VLLM_RUN=vllm-${TASK}-prompt-${ARRAY_JOB_ID}-${ARRAY_TASK_ID}
srun "${PYTHON}" -m mosaickv.cli evaluate \
  --backend vllm --method prompt_attention_topk \
  --retention-ratio "${PROMPT_RETENTION}" "${VLLM_COMMON[@]}" \
  --sample-id-file "${SAMPLE_FILE}" --sample-id-file-sha "${SAMPLE_SHA}" \
  --enable-mosaickv --vllm-selector-plan "${PROMPT_PLAN}" \
  --vllm-selector-plan-sha "${PROMPT_PLAN_SHA}" \
  --run-id "${PROMPT_VLLM_RUN}" \
  --output-dir "${TASK_ROOT}/prompt_attention_topk/vllm" \
  --trace-directory "${TASK_ROOT}/prompt_attention_topk/vllm/traces"

# MosaicKV uses each frozen per-sample selector configuration. The serving
# projection then fills the largest whole-block capacity that is still at
# least 30% below the paired Prompt Top-K active-byte budget. This is a
# label-blind physical-budget rule, not a task-answer-dependent adjustment.
mapfile -t MOSAIC_CONFIGS < <(find "${SOURCE_ROOT}/configs" -maxdepth 1 \
  -name "paper-v1-retro-${TASK}-99a412bf0d-sample-*-mosaickv-full-*.yaml" | sort)
mapfile -t SAMPLE_LEDGERS < <(find "${SOURCE_ROOT}/sample_ledgers/${TASK}" -maxdepth 1 \
  -name '*.json' | sort)
[[ ${#MOSAIC_CONFIGS[@]} -eq 2 && ${#SAMPLE_LEDGERS[@]} -eq 2 ]]
MOSAIC_PLANS=()
MOSAIC_PLAN_SHAS=()
MOSAIC_RETENTIONS=()
for SAMPLE_INDEX in 0 1; do
  LEDGER=${SAMPLE_LEDGERS[${SAMPLE_INDEX}]}
  LEDGER_SHA=$(sha256sum "${LEDGER}" | cut -d' ' -f1)
  DERIVED_CONFIG=${TASK_ROOT}/mosaickv_exact/config-${SAMPLE_INDEX}.yaml
  srun "${PYTHON}" scripts/derive_exact_selector_config.py \
    --input "${MOSAIC_CONFIGS[${SAMPLE_INDEX}]}" --output "${DERIVED_CONFIG}"
  HF_RUN=hf-${TASK}-mosaic-${SAMPLE_INDEX}-${ARRAY_JOB_ID}-${ARRAY_TASK_ID}
  HF_ROOT=${TASK_ROOT}/mosaickv_exact/hf/sample-${SAMPLE_INDEX}
  PLAN=${TASK_ROOT}/mosaickv_exact/selector-plan-${SAMPLE_INDEX}.json
  srun "${PYTHON}" -m mosaickv.cli evaluate \
    --config "${DERIVED_CONFIG}" --task "${TASK}" --local-files-only --json \
    --sample-id-file "${LEDGER}" --sample-id-file-sha "${LEDGER_SHA}" \
    --run-id "${HF_RUN}" --output-dir "${HF_ROOT}" \
    --trace-directory "${HF_ROOT}/traces"
  srun "${PYTHON}" scripts/build_vllm_selector_plan.py \
    --manifest "${HF_ROOT}/${HF_RUN}/manifest.json" \
    --trace-root "${HF_ROOT}/traces/${HF_RUN}" \
    --task "${TASK}" --target-block-size "${SERVING_BLOCK_SIZE}" \
    --target-retention-ratio "${MOSAIC_SEED_RETENTION}" \
    --competitor-catalog "${PROMPT_PLAN}" \
    --competitor-catalog-sha256 "${PROMPT_PLAN_SHA}" \
    --minimum-reduction-fraction 0.30 --output "${PLAN}"
  PLAN_SHA=$(sha256sum "${PLAN}" | cut -d' ' -f1)
  MOSAIC_RETENTION=$("${PYTHON}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["entries"][0]["projection"]["retention_ratio"])' \
    "${PLAN}")
  MOSAIC_PLANS+=("${PLAN}")
  MOSAIC_PLAN_SHAS+=("${PLAN_SHA}")
  MOSAIC_RETENTIONS+=("${MOSAIC_RETENTION}")
  VLLM_RUN=vllm-${TASK}-mosaic-${SAMPLE_INDEX}-${ARRAY_JOB_ID}-${ARRAY_TASK_ID}
  srun "${PYTHON}" -m mosaickv.cli evaluate \
    --backend vllm --method mosaickv_exact \
    --retention-ratio "${MOSAIC_RETENTION}" "${VLLM_COMMON[@]}" \
    --sample-id-file "${LEDGER}" --sample-id-file-sha "${LEDGER_SHA}" \
    --enable-mosaickv --vllm-selector-plan "${PLAN}" \
    --vllm-selector-plan-sha "${PLAN_SHA}" \
    --run-id "${VLLM_RUN}" \
    --output-dir "${TASK_ROOT}/mosaickv_exact/vllm/sample-${SAMPLE_INDEX}" \
    --trace-directory "${TASK_ROOT}/mosaickv_exact/vllm/sample-${SAMPLE_INDEX}/traces"
done

# Exercise two distinct compressed requests in one persistent vLLM engine.
# TextVQA's paired plans have the same physical block ratio, so they can share
# the version-pinned engine setting while retaining distinct selected blocks.
if [[ "${TASK}" == textvqa ]]; then
  [[ "${MOSAIC_RETENTIONS[0]}" == "${MOSAIC_RETENTIONS[1]}" ]]
  MERGED_PLAN=${TASK_ROOT}/mosaickv_isolation/selector-plan.json
  srun "${PYTHON}" scripts/merge_vllm_selector_plans.py \
    --catalog "${MOSAIC_PLANS[0]}" --catalog "${MOSAIC_PLANS[1]}" \
    --catalog-sha256 "${MOSAIC_PLAN_SHAS[0]}" \
    --catalog-sha256 "${MOSAIC_PLAN_SHAS[1]}" \
    --model-id "${MODEL_ID}" --model-revision "${MODEL_REVISION}" \
    --task "${TASK}" --dataset-revision "${DATASET_REVISION}" \
    --method mosaickv_exact --output "${MERGED_PLAN}"
  MERGED_PLAN_SHA=$(sha256sum "${MERGED_PLAN}" | cut -d' ' -f1)
  ISOLATION_RUN=vllm-${TASK}-mosaic-isolation-${ARRAY_JOB_ID}-${ARRAY_TASK_ID}
  srun "${PYTHON}" -m mosaickv.cli evaluate \
    --backend vllm --method mosaickv_exact \
    --retention-ratio "${MOSAIC_RETENTIONS[0]}" "${VLLM_COMMON[@]}" \
    --sample-id-file "${SAMPLE_FILE}" --sample-id-file-sha "${SAMPLE_SHA}" \
    --enable-mosaickv --vllm-selector-plan "${MERGED_PLAN}" \
    --vllm-selector-plan-sha "${MERGED_PLAN_SHA}" \
    --run-id "${ISOLATION_RUN}" \
    --output-dir "${TASK_ROOT}/mosaickv_isolation/vllm" \
    --trace-directory "${TASK_ROOT}/mosaickv_isolation/vllm/traces"
fi
