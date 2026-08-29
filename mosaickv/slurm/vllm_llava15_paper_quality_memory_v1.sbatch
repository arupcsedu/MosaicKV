#!/bin/bash
#SBATCH --job-name=mosaickv-paper-qm-v1
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=07:30:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/djy8hg/slurm-logs/%x-%A_%a.out
#SBATCH --error=/scratch/djy8hg/slurm-logs/%x-%A_%a.err
#SBATCH --export=ALL

# Fixed paired paper-quality cohort. One array element owns one task and one
# A100; all methods within it consume the same immutable 32-sample ledger.

set -euo pipefail

module load "${MOSAICKV_GCC_MODULE:-gcc/11.4.0}"

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
REPOSITORY=/scratch/djy8hg/workdir/MosaicKV
PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python3}
CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is required}
ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
OUTPUT_ROOT=${MOSAICKV_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/paper-quality-memory-vllm-v1-${ARRAY_JOB_ID}}
PROTOCOL=${PROJECT_ROOT}/configs/experiments/paper_quality_memory_llava_vllm_v1.yaml
TEMPLATE=${PROJECT_ROOT}/configs/paper_candidate_llava_v1.yaml

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
SAMPLE_COUNT=32
SEED=20270901
OUTPUT_LENGTH=64
SERVING_BLOCK_SIZE=16
PROMPT_RETENTION=0.13
MOSAIC_SEED_RETENTION=0.075
TASK_ROOT=${OUTPUT_ROOT}/${TASK}
LEDGER=${OUTPUT_ROOT}/inputs/${TASK}.json

export MOSAICKV_CACHE_ROOT=${CACHE_ROOT}
# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_CUDA_ARCH_LIST=8.0
mkdir -p "${TASK_ROOT}" "$(dirname "${LEDGER}")" "${TMPDIR}"
cd "${PROJECT_ROOT}"

# Materialize one deterministic, hash-bound cohort. Prior dataset exposure is
# explicitly disclosed by the protocol; no claim of untouched samples is made.
srun "${PYTHON}" scripts/materialize_sample_ledger.py \
  --task "${TASK}" --dataset-revision "${DATASET_REVISION}" \
  --model-name "${MODEL_ID}" --seed "${SEED}" --subset-size "${SAMPLE_COUNT}" \
  --classification fixed_preregistered_paper_quality_memory \
  --selection-freeze "${PROTOCOL}" --output "${LEDGER}"
LEDGER_SHA=$(sha256sum "${LEDGER}" | cut -d' ' -f1)

srun "${PYTHON}" scripts/prepare_vllm_paper_quality_memory.py \
  --protocol "${PROTOCOL}" --template "${TEMPLATE}" --task "${TASK}" \
  --seed "${SEED}" --planner-output-length 1 \
  --output-root "${TASK_ROOT}/planning_configs"
PROMPT_CONFIG=${TASK_ROOT}/planning_configs/prompt_attention_topk.yaml
MOSAIC_CONFIG=${TASK_ROOT}/planning_configs/mosaickv_exact.yaml

VLLM_COMMON=(
  --model "${MODEL_ID}"
  --model-revision "${MODEL_REVISION}"
  --dataset-revision "${DATASET_REVISION}"
  --attention-backend eager
  --task "${TASK}"
  --block-size "${SERVING_BLOCK_SIZE}"
  --max-new-tokens "${OUTPUT_LENGTH}"
  --precision fp16
  --seed "${SEED}"
  --local-files-only
  --json
  --cache-probe-repeats 1
  --gpu-memory-utilization 0.75
  --vllm-max-model-len 4096
  --tensor-parallel-size 1
  --sample-id-file "${LEDGER}"
  --sample-id-file-sha "${LEDGER_SHA}"
)

FULL_RUN=paper-v1-${TASK}-full-${ARRAY_JOB_ID}-${ARRAY_TASK_ID}
srun "${PYTHON}" -m mosaickv.cli evaluate \
  --backend vllm --method full_kv --retention-ratio 1.0 \
  "${VLLM_COMMON[@]}" --run-id "${FULL_RUN}" \
  --output-dir "${TASK_ROOT}/full_kv/vllm" \
  --trace-directory "${TASK_ROOT}/full_kv/vllm/traces"

# HF is used only to construct common-interface selector plans. Primary
# answers, scores, timings, and active bytes all come from the vLLM arms below.
PROMPT_HF_RUN=paper-v1-${TASK}-prompt-plan-${ARRAY_JOB_ID}-${ARRAY_TASK_ID}
PROMPT_HF_ROOT=${TASK_ROOT}/prompt_attention_topk/hf
srun "${PYTHON}" -m mosaickv.cli evaluate \
  --config "${PROMPT_CONFIG}" --task "${TASK}" --local-files-only --json \
  --sample-id-file "${LEDGER}" --sample-id-file-sha "${LEDGER_SHA}" \
  --run-id "${PROMPT_HF_RUN}" --output-dir "${PROMPT_HF_ROOT}" \
  --trace-directory "${PROMPT_HF_ROOT}/traces"
PROMPT_PLAN=${TASK_ROOT}/prompt_attention_topk/selector_plan.json
srun "${PYTHON}" scripts/build_vllm_selector_plan.py \
  --manifest "${PROMPT_HF_ROOT}/${PROMPT_HF_RUN}/manifest.json" \
  --trace-root "${PROMPT_HF_ROOT}/traces/${PROMPT_HF_RUN}" \
  --task "${TASK}" --target-block-size "${SERVING_BLOCK_SIZE}" \
  --target-retention-ratio "${PROMPT_RETENTION}" --output "${PROMPT_PLAN}"
PROMPT_PLAN_SHA=$(sha256sum "${PROMPT_PLAN}" | cut -d' ' -f1)

PROMPT_RUN=paper-v1-${TASK}-prompt-${ARRAY_JOB_ID}-${ARRAY_TASK_ID}
srun "${PYTHON}" -m mosaickv.cli evaluate \
  --backend vllm --method prompt_attention_topk \
  --retention-ratio "${PROMPT_RETENTION}" "${VLLM_COMMON[@]}" \
  --enable-mosaickv --vllm-selector-plan "${PROMPT_PLAN}" \
  --vllm-selector-plan-sha "${PROMPT_PLAN_SHA}" --run-id "${PROMPT_RUN}" \
  --output-dir "${TASK_ROOT}/prompt_attention_topk/vllm" \
  --trace-directory "${TASK_ROOT}/prompt_attention_topk/vllm/traces"

MOSAIC_HF_RUN=paper-v1-${TASK}-mosaic-plan-${ARRAY_JOB_ID}-${ARRAY_TASK_ID}
MOSAIC_HF_ROOT=${TASK_ROOT}/mosaickv_exact/hf
srun "${PYTHON}" -m mosaickv.cli evaluate \
  --config "${MOSAIC_CONFIG}" --task "${TASK}" --local-files-only --json \
  --sample-id-file "${LEDGER}" --sample-id-file-sha "${LEDGER_SHA}" \
  --run-id "${MOSAIC_HF_RUN}" --output-dir "${MOSAIC_HF_ROOT}" \
  --trace-directory "${MOSAIC_HF_ROOT}/traces"
MOSAIC_PLAN=${TASK_ROOT}/mosaickv_exact/selector_plan.json
srun "${PYTHON}" scripts/build_vllm_selector_plan.py \
  --manifest "${MOSAIC_HF_ROOT}/${MOSAIC_HF_RUN}/manifest.json" \
  --trace-root "${MOSAIC_HF_ROOT}/traces/${MOSAIC_HF_RUN}" \
  --task "${TASK}" --target-block-size "${SERVING_BLOCK_SIZE}" \
  --target-retention-ratio "${MOSAIC_SEED_RETENTION}" \
  --competitor-catalog "${PROMPT_PLAN}" \
  --competitor-catalog-sha256 "${PROMPT_PLAN_SHA}" \
  --minimum-reduction-fraction 0.30 --output "${MOSAIC_PLAN}"
MOSAIC_PLAN_SHA=$(sha256sum "${MOSAIC_PLAN}" | cut -d' ' -f1)

MOSAIC_RUN=paper-v1-${TASK}-mosaic-${ARRAY_JOB_ID}-${ARRAY_TASK_ID}
srun "${PYTHON}" -m mosaickv.cli evaluate \
  --backend vllm --method mosaickv_exact \
  --retention-ratio "${MOSAIC_SEED_RETENTION}" "${VLLM_COMMON[@]}" \
  --enable-mosaickv --vllm-selector-plan "${MOSAIC_PLAN}" \
  --vllm-selector-plan-sha "${MOSAIC_PLAN_SHA}" --run-id "${MOSAIC_RUN}" \
  --output-dir "${TASK_ROOT}/mosaickv_exact/vllm" \
  --trace-directory "${TASK_ROOT}/mosaickv_exact/vllm/traces"

"${PYTHON}" - <<'PY' "${TASK_ROOT}" "${TASK}" "${LEDGER_SHA}"
import hashlib, json, pathlib, sys
root, task, ledger_sha = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
payload = {
    "schema_version": 1,
    "status": "task_complete",
    "task": task,
    "ledger_sha256": ledger_sha,
    "source_git_sha": __import__("subprocess").run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip(),
}
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
path = root / "task_complete.json"
path.write_bytes(encoded)
print(json.dumps({"path": str(path), "sha256": hashlib.sha256(encoded).hexdigest()}))
PY
