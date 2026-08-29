#!/bin/bash
#SBATCH --job-name=mk-paper-hf-vllm-v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --time=03:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/djy8hg/slurm-logs/%x-%A_%a.out
#SBATCH --error=/scratch/djy8hg/slurm-logs/%x-%A_%a.err
#SBATCH --export=ALL

# Submit with --array=0-23%4. The immutable index contains three tasks, four
# sample shards per task, and separate HF/vLLM jobs. Each job owns one A100.

set -euo pipefail
module load "${MOSAICKV_GCC_MODULE:-gcc/11.4.0}"

REPOSITORY=/scratch/djy8hg/workdir/MosaicKV
PROJECT_ROOT=${REPOSITORY}/mosaickv
PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python3}
CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
: "${MOSAICKV_PAPER_INDEX:?set MOSAICKV_PAPER_INDEX to immutable jobs.jsonl}"
: "${SLURM_ARRAY_TASK_ID:?submit this script as a Slurm array}"

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
mkdir -p "${TMPDIR}" /scratch/djy8hg/slurm-logs
cd "${PROJECT_ROOT}"

# Validate every hash and the current clean source before reading job fields.
"${PYTHON}" scripts/prepare_paper_quality_hf_vllm_v2.py validate-job \
  --index "${MOSAICKV_PAPER_INDEX}" \
  --array-index "${SLURM_ARRAY_TASK_ID}" >/dev/null

mapfile -d '' -t JOB_FIELDS < <(
  "${PYTHON}" - "${MOSAICKV_PAPER_INDEX}" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
row = rows[int(sys.argv[2])]
backend = row["backend_config"]
values = (
    row["backend"], row["task"], row["dataset_revision"],
    row["sample_ledger_path"], row["sample_ledger_sha256"],
    row["selection_seed"], row["output_length"], row["precision"],
    row["output_root"], row["run_namespace"], row["protocol_path"],
    backend.get("block_size", backend.get("serving_block_size", 0)),
    backend["prompt_topk_retention_ratio"],
    backend["mosaickv_seed_retention_ratio"],
    backend.get("mosaickv_cache_budget_bytes", 0),
    backend.get("minimum_competitor_anchor_blocks", 0),
    backend.get("competitor_tail_dominance_threshold", 0.0),
    backend.get("coherent_evidence_minimum_route_ratio", 0.0),
)
for value in values:
    sys.stdout.write(str(value) + "\0")
PY
)

BACKEND=${JOB_FIELDS[0]}
TASK=${JOB_FIELDS[1]}
DATASET_REVISION=${JOB_FIELDS[2]}
LEDGER=${JOB_FIELDS[3]}
LEDGER_SHA=${JOB_FIELDS[4]}
SEED=${JOB_FIELDS[5]}
OUTPUT_LENGTH=${JOB_FIELDS[6]}
PRECISION=${JOB_FIELDS[7]}
OUTPUT_ROOT=${JOB_FIELDS[8]}
RUN_NAMESPACE=${JOB_FIELDS[9]}
PROTOCOL=${JOB_FIELDS[10]}
BLOCK_SIZE=${JOB_FIELDS[11]}
PROMPT_RETENTION=${JOB_FIELDS[12]}
MOSAIC_RETENTION=${JOB_FIELDS[13]}
MOSAIC_BUDGET=${JOB_FIELDS[14]}
MINIMUM_ANCHORS=${JOB_FIELDS[15]}
TAIL_DOMINANCE=${JOB_FIELDS[16]}
COHERENT_EVIDENCE_RATIO=${JOB_FIELDS[17]}
MODEL_ID=llava-hf/llava-1.5-7b-hf
MODEL_REVISION=b234b804b114d9e37bb655e11cbbb5f5e971b7a9
JOB_ROOT=${OUTPUT_ROOT}/${BACKEND}/${TASK}/${RUN_NAMESPACE}
COMPLETION=${JOB_ROOT}/job_complete.json

if [[ -f "${COMPLETION}" ]]; then
  "${PYTHON}" - "${COMPLETION}" "${RUN_NAMESPACE}" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("status") != "complete" or value.get("run_namespace") != sys.argv[2]:
    raise SystemExit("existing completion seal has the wrong identity")
PY
  exit 0
fi
mkdir -p "${JOB_ROOT}"

COMMON=(
  --model "${MODEL_ID}"
  --model-revision "${MODEL_REVISION}"
  --dataset-revision "${DATASET_REVISION}"
  --attention-backend eager
  --task "${TASK}"
  --max-new-tokens "${OUTPUT_LENGTH}"
  --precision "${PRECISION}"
  --seed "${SEED}"
  --local-files-only
  --json
  --sample-id-file "${LEDGER}"
  --sample-id-file-sha "${LEDGER_SHA}"
)

if [[ "${BACKEND}" == hf_eager ]]; then
  FULL_RUN=${RUN_NAMESPACE}-full-kv
  srun "${PYTHON}" -m mosaickv.cli evaluate \
    --backend hf --method full_kv --retention-ratio 1.0 \
    --block-size "${BLOCK_SIZE}" --cache-budget 9223372036854775807 \
    --budget-unit bytes --repair-policy none "${COMMON[@]}" \
    --run-id "${FULL_RUN}" --output-dir "${JOB_ROOT}/full_kv" \
    --trace-directory "${JOB_ROOT}/full_kv/traces"

  PROMPT_RUN=${RUN_NAMESPACE}-prompt-attention-topk
  srun "${PYTHON}" -m mosaickv.cli evaluate \
    --backend hf --method prompt_attention_topk \
    --retention-ratio "${PROMPT_RETENTION}" --block-size "${BLOCK_SIZE}" \
    --cache-budget 9223372036854775807 --budget-unit bytes \
    --repair-policy none "${COMMON[@]}" --run-id "${PROMPT_RUN}" \
    --output-dir "${JOB_ROOT}/prompt_attention_topk" \
    --trace-directory "${JOB_ROOT}/prompt_attention_topk/traces"

  MOSAIC_RUN=${RUN_NAMESPACE}-mosaickv-full
  srun "${PYTHON}" -m mosaickv.cli evaluate \
    --backend hf --method mosaickv_full \
    --retention-ratio "${MOSAIC_RETENTION}" --block-size "${BLOCK_SIZE}" \
    --cache-budget "${MOSAIC_BUDGET}" --budget-unit bytes \
    --forecast hybrid --prompt-window 16 --draft-tokens 4 \
    --repair-policy entropy_or_prototype_risk --repair-blocks 4 \
    "${COMMON[@]}" --run-id "${MOSAIC_RUN}" \
    --output-dir "${JOB_ROOT}/mosaickv_full" \
    --trace-directory "${JOB_ROOT}/mosaickv_full/traces"

elif [[ "${BACKEND}" == vllm ]]; then
  VLLM_COMMON=(
    "${COMMON[@]}"
    --block-size "${BLOCK_SIZE}"
    --cache-probe-repeats 1
    --gpu-memory-utilization 0.75
    --vllm-max-model-len 4096
    --tensor-parallel-size 1
  )
  FULL_RUN=${RUN_NAMESPACE}-full-kv
  srun "${PYTHON}" -m mosaickv.cli evaluate \
    --backend vllm --method full_kv --retention-ratio 1.0 \
    "${VLLM_COMMON[@]}" --run-id "${FULL_RUN}" \
    --output-dir "${JOB_ROOT}/full_kv/vllm" \
    --trace-directory "${JOB_ROOT}/full_kv/vllm/traces"

  PLANNING_ROOT=${JOB_ROOT}/planning_configs
  if [[ ! -f "${PLANNING_ROOT}/descriptor.json" ]]; then
    srun "${PYTHON}" scripts/prepare_vllm_paper_quality_memory.py \
      --protocol "${PROTOCOL}" \
      --template "${PROJECT_ROOT}/configs/paper_candidate_llava_v1.yaml" \
      --task "${TASK}" --seed "${SEED}" --planner-output-length 1 \
      --output-root "${PLANNING_ROOT}"
  fi

  PROMPT_CONFIG=${PLANNING_ROOT}/prompt_attention_topk.yaml
  PROMPT_HF_RUN=${RUN_NAMESPACE}-prompt-plan
  PROMPT_HF_ROOT=${JOB_ROOT}/prompt_attention_topk/hf_planning
  srun "${PYTHON}" -m mosaickv.cli evaluate \
    --config "${PROMPT_CONFIG}" --task "${TASK}" --local-files-only --json \
    --sample-id-file "${LEDGER}" --sample-id-file-sha "${LEDGER_SHA}" \
    --run-id "${PROMPT_HF_RUN}" --output-dir "${PROMPT_HF_ROOT}" \
    --trace-directory "${PROMPT_HF_ROOT}/traces"
  PROMPT_PLAN=${JOB_ROOT}/prompt_attention_topk/selector_plan.json
  if [[ ! -f "${PROMPT_PLAN}" ]]; then
    srun "${PYTHON}" scripts/build_vllm_selector_plan.py \
      --manifest "${PROMPT_HF_ROOT}/${PROMPT_HF_RUN}/manifest.json" \
      --trace-root "${PROMPT_HF_ROOT}/traces/${PROMPT_HF_RUN}" \
      --task "${TASK}" --target-block-size "${BLOCK_SIZE}" \
      --target-retention-ratio "${PROMPT_RETENTION}" --output "${PROMPT_PLAN}"
  fi
  PROMPT_PLAN_SHA=$(sha256sum "${PROMPT_PLAN}" | cut -d' ' -f1)

  PROMPT_RUN=${RUN_NAMESPACE}-prompt-attention-topk
  srun "${PYTHON}" -m mosaickv.cli evaluate \
    --backend vllm --method prompt_attention_topk \
    --retention-ratio "${PROMPT_RETENTION}" "${VLLM_COMMON[@]}" \
    --enable-mosaickv --vllm-selector-plan "${PROMPT_PLAN}" \
    --vllm-selector-plan-sha "${PROMPT_PLAN_SHA}" --run-id "${PROMPT_RUN}" \
    --output-dir "${JOB_ROOT}/prompt_attention_topk/vllm" \
    --trace-directory "${JOB_ROOT}/prompt_attention_topk/vllm/traces"

  MOSAIC_CONFIG=${PLANNING_ROOT}/mosaickv_exact.yaml
  MOSAIC_HF_RUN=${RUN_NAMESPACE}-mosaic-plan
  MOSAIC_HF_ROOT=${JOB_ROOT}/mosaickv_exact/hf_planning
  srun "${PYTHON}" -m mosaickv.cli evaluate \
    --config "${MOSAIC_CONFIG}" --task "${TASK}" --local-files-only --json \
    --sample-id-file "${LEDGER}" --sample-id-file-sha "${LEDGER_SHA}" \
    --run-id "${MOSAIC_HF_RUN}" --output-dir "${MOSAIC_HF_ROOT}" \
    --trace-directory "${MOSAIC_HF_ROOT}/traces"
  MOSAIC_PLAN=${JOB_ROOT}/mosaickv_exact/selector_plan.json
  if [[ ! -f "${MOSAIC_PLAN}" ]]; then
    srun "${PYTHON}" scripts/build_vllm_selector_plan.py \
      --manifest "${MOSAIC_HF_ROOT}/${MOSAIC_HF_RUN}/manifest.json" \
      --trace-root "${MOSAIC_HF_ROOT}/traces/${MOSAIC_HF_RUN}" \
      --task "${TASK}" --target-block-size "${BLOCK_SIZE}" \
      --target-retention-ratio "${MOSAIC_RETENTION}" \
      --competitor-catalog "${PROMPT_PLAN}" \
      --competitor-catalog-sha256 "${PROMPT_PLAN_SHA}" \
      --minimum-reduction-fraction 0.30 \
      --minimum-competitor-anchor-blocks "${MINIMUM_ANCHORS}" \
      --competitor-tail-dominance-threshold "${TAIL_DOMINANCE}" \
      --coherent-evidence-minimum-route-ratio "${COHERENT_EVIDENCE_RATIO}" \
      --output "${MOSAIC_PLAN}"
  fi
  MOSAIC_PLAN_SHA=$(sha256sum "${MOSAIC_PLAN}" | cut -d' ' -f1)

  MOSAIC_RUN=${RUN_NAMESPACE}-mosaickv-exact
  srun "${PYTHON}" -m mosaickv.cli evaluate \
    --backend vllm --method mosaickv_exact \
    --retention-ratio "${MOSAIC_RETENTION}" "${VLLM_COMMON[@]}" \
    --enable-mosaickv --vllm-selector-plan "${MOSAIC_PLAN}" \
    --vllm-selector-plan-sha "${MOSAIC_PLAN_SHA}" --run-id "${MOSAIC_RUN}" \
    --output-dir "${JOB_ROOT}/mosaickv_exact/vllm" \
    --trace-directory "${JOB_ROOT}/mosaickv_exact/vllm/traces"
else
  printf 'Unsupported paper backend in job record: %s\n' "${BACKEND}" >&2
  exit 2
fi

"${PYTHON}" - "${COMPLETION}" "${RUN_NAMESPACE}" "${BACKEND}" "${TASK}" \
  "${LEDGER_SHA}" <<'PY'
import json
import os
import pathlib
import subprocess
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "status": "complete",
    "run_namespace": sys.argv[2],
    "backend": sys.argv[3],
    "task": sys.argv[4],
    "sample_ledger_sha256": sys.argv[5],
    "source_git_sha": subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
}
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
