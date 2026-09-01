#!/bin/bash
#SBATCH --job-name=mk-pubbase-full
#SBATCH --array=0-141%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/scratch/djy8hg/slurm-logs/%x-%A_%a.out
#SBATCH --error=/scratch/djy8hg/slurm-logs/%x-%A_%a.err
#SBATCH --export=ALL

set -euo pipefail

module load "${MOSAICKV_GCC_MODULE:-gcc/11.4.0}"
REPOSITORY=/scratch/djy8hg/workdir/MosaicKV
PROJECT_ROOT=${REPOSITORY}/mosaickv
PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python3}
CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
CAMPAIGN_ROOT=${MOSAICKV_CAMPAIGN_ROOT:?MOSAICKV_CAMPAIGN_ROOT is required}
INDEX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
JOB_SPEC=$(printf '%s/inputs/jobs/%04d.json' "${CAMPAIGN_ROOT}" "${INDEX}")

export MOSAICKV_CACHE_ROOT=${CACHE_ROOT}
# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
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
cd "${PROJECT_ROOT}"

mapfile -t FIELDS < <("${PYTHON}" - "${JOB_SPEC}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
for key in ('task','shard_index','sample_ledger_path','sample_ledger_sha256',
            'dataset_revision','source_git_sha','seed','output_length'):
    print(x[key])
PY
)
TASK=${FIELDS[0]}
SHARD=${FIELDS[1]}
LEDGER=${FIELDS[2]}
LEDGER_SHA=${FIELDS[3]}
DATASET_REVISION=${FIELDS[4]}
SOURCE_SHA=${FIELDS[5]}
SEED=${FIELDS[6]}
OUTPUT_LENGTH=${FIELDS[7]}
[[ "$(git -C "${REPOSITORY}" rev-parse HEAD)" == "${SOURCE_SHA}" ]]
SHARD_ROOT=$(printf '%s/results/%s/%03d' "${CAMPAIGN_ROOT}" "${TASK}" "${SHARD}")
HF_ROOT=${SHARD_ROOT}/hf
mkdir -p "${HF_ROOT}"

srun "${PYTHON}" scripts/run_vllm_published_baseline_full_hf_worker.py \
  --job-spec "${JOB_SPEC}" --output-root "${HF_ROOT}"

MODEL_ID=llava-hf/llava-1.5-7b-hf
MODEL_REVISION=b234b804b114d9e37bb655e11cbbb5f5e971b7a9
RETENTION=0.13
SERVING_BLOCK_SIZE=16
PREFIX_PROFILE=${REPOSITORY}/third_party/PrefixKV/confs/prefixkv_llava-v1.5-7b_0.8.json
NAMESPACE=$(printf 'baseline-full-%s-%s-%03d' "${SOURCE_SHA:0:10}" "${TASK}" "${SHARD}")

for METHOD in lookm_reimpl prefixkv_reimpl vl_cache_reimpl; do
  RUN_ID=${NAMESPACE}-${METHOD}
  PLAN_ROOT=${SHARD_ROOT}/plans/${METHOD}
  PLAN=${PLAN_ROOT}/selector-plan.json
  mkdir -p "${PLAN_ROOT}"
  if [[ ! -f "${PLAN}" ]]; then
    srun "${PYTHON}" scripts/build_vllm_selector_plan.py \
      --manifest "${HF_ROOT}/${RUN_ID}/manifest.json" \
      --trace-root "${HF_ROOT}/traces/${RUN_ID}" --task "${TASK}" \
      --target-block-size "${SERVING_BLOCK_SIZE}" \
      --target-retention-ratio "${RETENTION}" --output "${PLAN}"
  fi
  PLAN_SHA=$(sha256sum "${PLAN}" | cut -d' ' -f1)
  VLLM_ROOT=${SHARD_ROOT}/vllm/${METHOD}
  EXTRA=()
  if [[ "${METHOD}" == lookm_reimpl ]]; then
    EXTRA+=(--lookm-recent-ratio 0.065 --lookm-important-ratio 0.065)
  elif [[ "${METHOD}" == prefixkv_reimpl ]]; then
    EXTRA+=(--prefixkv-profile-mode offline_profile --prefixkv-profile "${PREFIX_PROFILE}")
  fi
  srun "${PYTHON}" -m mosaickv.cli evaluate \
    --backend vllm --model "${MODEL_ID}" --model-revision "${MODEL_REVISION}" \
    --dataset-revision "${DATASET_REVISION}" --attention-backend eager \
    --task "${TASK}" --method "${METHOD}" --retention-ratio "${RETENTION}" \
    --block-size "${SERVING_BLOCK_SIZE}" --max-new-tokens "${OUTPUT_LENGTH}" \
    --precision fp16 --seed "${SEED}" --local-files-only --json \
    --cache-probe-repeats 1 --gpu-memory-utilization 0.75 \
    --vllm-max-model-len 4096 --tensor-parallel-size 1 --enable-mosaickv \
    --sample-id-file "${LEDGER}" --sample-id-file-sha "${LEDGER_SHA}" \
    --vllm-selector-plan "${PLAN}" --vllm-selector-plan-sha "${PLAN_SHA}" \
    --run-id "${RUN_ID}-vllm" --output-dir "${VLLM_ROOT}" \
    --trace-directory "${VLLM_ROOT}/traces" "${EXTRA[@]}"
done

"${PYTHON}" - "${SHARD_ROOT}/job_complete.json" "${JOB_SPEC}" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
path=Path(sys.argv[1]); spec=Path(sys.argv[2])
payload={
  'schema_version':1,'status':'completed','job_spec':str(spec),
  'job_spec_sha256':hashlib.sha256(spec.read_bytes()).hexdigest(),
  'slurm_job_id':os.environ['SLURM_JOB_ID'],
  'slurm_array_task_id':os.environ['SLURM_ARRAY_TASK_ID'],
}
encoded=json.dumps(payload,indent=2,sort_keys=True)+'\n'
if path.exists():
    if path.read_text()!=encoded:
        raise RuntimeError(f'inconsistent completion marker: {path}')
else:
    path.write_text(encoded)
PY
