#!/bin/bash
#SBATCH --job-name=mk-vllm-pubbase
#SBATCH --array=0-2%3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/scratch/djy8hg/slurm-logs/%x-%A_%a.out
#SBATCH --error=/scratch/djy8hg/slurm-logs/%x-%A_%a.err
#SBATCH --export=ALL

# Development pilot: two identical samples per task and one native vLLM
# request-wide whole-block projection for each selector. LOOK-M's merged-KV
# operation is deliberately not represented by these serving rows.

set -euo pipefail

module load "${MOSAICKV_GCC_MODULE:-gcc/11.4.0}"

REPOSITORY=/scratch/djy8hg/workdir/MosaicKV
PROJECT_ROOT=${REPOSITORY}/mosaickv
PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python3}
CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
SOURCE_ROOT=/scratch/djy8hg/runs/mosaickv/paper-candidate-v1-retrospective-99a412bf0d2c
OUTPUT_ROOT=${MOSAICKV_OUTPUT_ROOT:?MOSAICKV_OUTPUT_ROOT is required}
ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}

TASKS=(mmstar textvqa docvqa)
DATASET_REVISIONS=(
  bc98d668301da7b14f648724866e57302778ab27
  9c0699cd19768ac5ab97568f6b3cbac4c0062884
  539088ef8a8ada01ac8e2e6d4e372586748a265e
)
TASK=${TASKS[${ARRAY_TASK_ID}]}
DATASET_REVISION=${DATASET_REVISIONS[${ARRAY_TASK_ID}]}
TASK_ROOT=${OUTPUT_ROOT}/${TASK}
SAMPLE_FILE=${SOURCE_ROOT}/inputs/${TASK}.json
SAMPLE_SHA=$(sha256sum "${SAMPLE_FILE}" | cut -d' ' -f1)
MODEL_ID=llava-hf/llava-1.5-7b-hf
MODEL_REVISION=b234b804b114d9e37bb655e11cbbb5f5e971b7a9
BASELINE_RETENTION=0.13
SERVING_BLOCK_SIZE=16
SOURCE_COMMIT=$(git -C "${REPOSITORY}" rev-parse HEAD)

export MOSAICKV_CACHE_ROOT=${CACHE_ROOT}
# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
export MOSAICKV_ALLOW_DIRTY_EXPLORATORY=1
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_CUDA_ARCH_LIST=8.0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
mkdir -p "${TASK_ROOT}" "${TMPDIR}"
cd "${PROJECT_ROOT}"

HF_COMMON=(
  --backend hf --model "${MODEL_ID}" --model-revision "${MODEL_REVISION}"
  --dataset-revision "${DATASET_REVISION}" --attention-backend eager
  --task "${TASK}" --max-new-tokens 64 --precision fp16 --seed 17
  --retention-ratio "${BASELINE_RETENTION}" --block-size 1
  --budget-unit blocks --cache-budget 2147483647 --local-files-only --json
  --sample-id-file "${SAMPLE_FILE}" --sample-id-file-sha "${SAMPLE_SHA}"
)
VLLM_COMMON=(
  --backend vllm --model "${MODEL_ID}" --model-revision "${MODEL_REVISION}"
  --dataset-revision "${DATASET_REVISION}" --attention-backend eager
  --task "${TASK}" --block-size "${SERVING_BLOCK_SIZE}"
  --max-new-tokens 64 --precision fp16 --seed 17 --local-files-only --json
  --cache-probe-repeats 1 --gpu-memory-utilization 0.75
  --vllm-max-model-len 4096 --tensor-parallel-size 1
  --sample-id-file "${SAMPLE_FILE}" --sample-id-file-sha "${SAMPLE_SHA}"
)

run_hf_source() {
  local method=$1
  shift
  local run_id=hf-${TASK}-${method}-${SLURM_ARRAY_JOB_ID}-${ARRAY_TASK_ID}
  local root=${TASK_ROOT}/${method}/hf
  srun "${PYTHON}" -m mosaickv.cli evaluate \
    "${HF_COMMON[@]}" --method "${method}" "$@" \
    --run-id "${run_id}" --output-dir "${root}" \
    --trace-directory "${root}/traces"
}

build_plan() {
  local method=$1
  local run_id=hf-${TASK}-${method}-${SLURM_ARRAY_JOB_ID}-${ARRAY_TASK_ID}
  local root=${TASK_ROOT}/${method}/hf
  local output=${TASK_ROOT}/${method}/selector-plan.json
  srun "${PYTHON}" scripts/build_vllm_selector_plan.py \
    --manifest "${root}/${run_id}/manifest.json" \
    --trace-root "${root}/traces/${run_id}" --task "${TASK}" \
    --target-block-size "${SERVING_BLOCK_SIZE}" \
    --target-retention-ratio "${BASELINE_RETENTION}" --output "${output}"
}

run_vllm_plan() {
  local method=$1
  shift
  local plan=${TASK_ROOT}/${method}/selector-plan.json
  local plan_sha
  plan_sha=$(sha256sum "${plan}" | cut -d' ' -f1)
  local run_id=vllm-${TASK}-${method}-${SLURM_ARRAY_JOB_ID}-${ARRAY_TASK_ID}
  local root=${TASK_ROOT}/${method}/vllm
  srun "${PYTHON}" -m mosaickv.cli evaluate \
    "${VLLM_COMMON[@]}" --method "${method}" \
    --retention-ratio "${BASELINE_RETENTION}" --enable-mosaickv \
    --vllm-selector-plan "${plan}" --vllm-selector-plan-sha "${plan_sha}" \
    "$@" --run-id "${run_id}" --output-dir "${root}" \
    --trace-directory "${root}/traces"
}

# Recompute all HF selector sources from this implementation so every catalog
# contains the new hash-bound projection field.
run_hf_source prompt_attention_topk
run_hf_source lookm_reimpl \
  --lookm-recent-ratio 0.065 --lookm-important-ratio 0.065 \
  --lookm-merge-strategy pivotal
run_hf_source prefixkv_reimpl --prefixkv-profile-mode offline_profile \
  --prefixkv-profile "${REPOSITORY}/third_party/PrefixKV/confs/prefixkv_llava-v1.5-7b_0.8.json"
run_hf_source vl_cache_reimpl --vl-cache-sparsity-threshold 0.01 \
  --vl-cache-min-layer-retention 0.01 --vl-cache-max-layer-retention 1.0 \
  --vl-cache-recent-window-fraction 0.1

for method in prompt_attention_topk lookm_reimpl prefixkv_reimpl vl_cache_reimpl; do
  build_plan "${method}"
done

# Build the exact MosaicKV selector independently for each sample, then merge
# the two hash-bound entries. Its serving budget is capped at 70% of the paired
# Prompt Top-K block count, which is the preregistered 30% active-byte gate.
PROMPT_PLAN=${TASK_ROOT}/prompt_attention_topk/selector-plan.json
PROMPT_PLAN_SHA=$(sha256sum "${PROMPT_PLAN}" | cut -d' ' -f1)
MOSAIC_PLANS=()
MOSAIC_SHAS=()
mapfile -t MOSAIC_CONFIGS < <(find "${SOURCE_ROOT}/configs" -maxdepth 1 \
  -name "paper-v1-retro-${TASK}-99a412bf0d-sample-*-mosaickv-full-*.yaml" | sort)
mapfile -t SAMPLE_LEDGERS < <(find "${SOURCE_ROOT}/sample_ledgers/${TASK}" -maxdepth 1 \
  -name '*.json' | sort)
[[ ${#MOSAIC_CONFIGS[@]} -eq 2 && ${#SAMPLE_LEDGERS[@]} -eq 2 ]]
for index in 0 1; do
  ledger=${SAMPLE_LEDGERS[${index}]}
  ledger_sha=$(sha256sum "${ledger}" | cut -d' ' -f1)
  derived=${TASK_ROOT}/mosaickv_exact/config-${index}.yaml
  srun "${PYTHON}" scripts/derive_exact_selector_config.py \
    --input "${MOSAIC_CONFIGS[${index}]}" --output "${derived}"
  hf_run=hf-${TASK}-mosaickv_exact-${index}-${SLURM_ARRAY_JOB_ID}-${ARRAY_TASK_ID}
  hf_root=${TASK_ROOT}/mosaickv_exact/hf/sample-${index}
  srun "${PYTHON}" -m mosaickv.cli evaluate --config "${derived}" --task "${TASK}" \
    --local-files-only --json --sample-id-file "${ledger}" \
    --sample-id-file-sha "${ledger_sha}" --run-id "${hf_run}" \
    --output-dir "${hf_root}" --trace-directory "${hf_root}/traces"
  plan=${TASK_ROOT}/mosaickv_exact/selector-plan-${index}.json
  srun "${PYTHON}" scripts/build_vllm_selector_plan.py \
    --manifest "${hf_root}/${hf_run}/manifest.json" \
    --trace-root "${hf_root}/traces/${hf_run}" --task "${TASK}" \
    --target-block-size "${SERVING_BLOCK_SIZE}" --target-retention-ratio 0.075 \
    --competitor-catalog "${PROMPT_PLAN}" \
    --competitor-catalog-sha256 "${PROMPT_PLAN_SHA}" \
    --minimum-reduction-fraction 0.30 --minimum-competitor-anchor-blocks 1 \
    --output "${plan}"
  MOSAIC_PLANS+=("${plan}")
  MOSAIC_SHAS+=("$(sha256sum "${plan}" | cut -d' ' -f1)")
done
MERGED_PLAN=${TASK_ROOT}/mosaickv_exact/selector-plan.json
srun "${PYTHON}" scripts/merge_vllm_selector_plans.py \
  --catalog "${MOSAIC_PLANS[0]}" --catalog "${MOSAIC_PLANS[1]}" \
  --catalog-sha256 "${MOSAIC_SHAS[0]}" --catalog-sha256 "${MOSAIC_SHAS[1]}" \
  --model-id "${MODEL_ID}" --model-revision "${MODEL_REVISION}" \
  --task "${TASK}" --dataset-revision "${DATASET_REVISION}" \
  --method mosaickv_exact --output "${MERGED_PLAN}"

# Same vLLM model, precision, processor, task samples, output length, seed,
# eager execution, block size, and measurement path for all six methods.
FULL_RUN=vllm-${TASK}-full_kv-${SLURM_ARRAY_JOB_ID}-${ARRAY_TASK_ID}
srun "${PYTHON}" -m mosaickv.cli evaluate "${VLLM_COMMON[@]}" \
  --method full_kv --retention-ratio 1.0 --run-id "${FULL_RUN}" \
  --output-dir "${TASK_ROOT}/full_kv/vllm" \
  --trace-directory "${TASK_ROOT}/full_kv/vllm/traces"
run_vllm_plan prompt_attention_topk
run_vllm_plan lookm_reimpl \
  --lookm-recent-ratio 0.065 --lookm-important-ratio 0.065
run_vllm_plan prefixkv_reimpl --prefixkv-profile-mode offline_profile \
  --prefixkv-profile "${REPOSITORY}/third_party/PrefixKV/confs/prefixkv_llava-v1.5-7b_0.8.json"
run_vllm_plan vl_cache_reimpl

MOSAIC_PLAN_SHA=$(sha256sum "${MERGED_PLAN}" | cut -d' ' -f1)
MOSAIC_RUN=vllm-${TASK}-mosaickv_exact-${SLURM_ARRAY_JOB_ID}-${ARRAY_TASK_ID}
srun "${PYTHON}" -m mosaickv.cli evaluate "${VLLM_COMMON[@]}" \
  --method mosaickv_exact --retention-ratio 0.075 --enable-mosaickv \
  --vllm-selector-plan "${MERGED_PLAN}" \
  --vllm-selector-plan-sha "${MOSAIC_PLAN_SHA}" --run-id "${MOSAIC_RUN}" \
  --output-dir "${TASK_ROOT}/mosaickv_exact/vllm" \
  --trace-directory "${TASK_ROOT}/mosaickv_exact/vllm/traces"

"${PYTHON}" - <<PY
import json
from pathlib import Path
root = Path(${TASK_ROOT@Q})
payload = {
    "schema_version": 1,
    "status": "completed",
    "classification": "development_pilot_non_paper",
    "task": ${TASK@Q},
    "source_commit": ${SOURCE_COMMIT@Q},
    "methods": ["full_kv", "prompt_attention_topk", "lookm_reimpl", "prefixkv_reimpl", "vl_cache_reimpl", "mosaickv_exact"],
}
(root / "worker_complete.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
