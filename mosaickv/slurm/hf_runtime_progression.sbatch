#!/bin/bash
#SBATCH --job-name=mosaickv-hf-runtime
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=37
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
HF_PYTHON=${MOSAICKV_HF_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
OUTPUT_ROOT=${MOSAICKV_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/hf-runtime}

export MOSAICKV_CACHE_ROOT="${CACHE_ROOT}"
# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"

if [[ ! -x "${HF_PYTHON}" ]]; then
  printf 'HF environment does not exist or is not executable: %s\n' "${HF_PYTHON}" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TORCH_HOME="${CACHE_ROOT}/torch"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

# Gate 1 and 2: backend-independent synthetic cache tests are covered by the
# normal test suite; this no-download gate exercises three real tiny HF model
# architectures, all MosaicKV methods, retention-1 parity, and trace emission.
srun "${HF_PYTHON}" scripts/verify_hf_runtime.py \
  --max-new-tokens 16 \
  --output "${OUTPUT_ROOT}/tiny-runtime-${SLURM_JOB_ID}.json"

DOWNLOAD_ARGS=()
PARITY_DOWNLOAD_ARGS=()
if [[ "${MOSAICKV_ALLOW_MODEL_DOWNLOAD:-0}" != 1 ]]; then
  DOWNLOAD_ARGS+=(--local-files-only)
else
  PARITY_DOWNLOAD_ARGS+=(--allow-download)
fi

# Gate 3: one Qwen2.5-VL-3B image/prompt. Set the opt-in variable because this
# can download several GB when the exact revision is absent from shared cache.
if [[ "${MOSAICKV_RUN_QWEN3B:-0}" == 1 ]]; then
  srun "${HF_PYTHON}" -m mosaickv.cli evaluate \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --backend hf \
    --attention-backend eager \
    --method mosaickv_full \
    --task synthetic_smoke \
    --retention-ratio 0.5 \
    --block-size 16 \
    --forecast hybrid \
    --draft-tokens 4 \
    --repair-policy entropy_or_prototype_risk \
    --subset-size 1 \
    --output-dir "${OUTPUT_ROOT}/qwen3b" \
    "${DOWNLOAD_ARGS[@]}"
  srun "${HF_PYTHON}" scripts/verify_hf_runtime.py \
    --model-id Qwen/Qwen2.5-VL-3B-Instruct \
    --block-size 16 \
    --max-new-tokens 16 \
    --output "${OUTPUT_ROOT}/qwen3b-retention-one-${SLURM_JOB_ID}.json" \
    "${PARITY_DOWNLOAD_ARGS[@]}"
fi

# Gate 4: one LLaVA-1.5 image/prompt at its audited immutable revision.
if [[ "${MOSAICKV_RUN_LLAVA15:-0}" == 1 ]]; then
  srun "${HF_PYTHON}" -m mosaickv.cli evaluate \
    --model llava-hf/llava-1.5-7b-hf \
    --backend hf \
    --attention-backend eager \
    --method mosaickv_full \
    --task synthetic_smoke \
    --retention-ratio 0.5 \
    --block-size 16 \
    --forecast hybrid \
    --draft-tokens 4 \
    --repair-policy entropy_or_prototype_risk \
    --subset-size 1 \
    --output-dir "${OUTPUT_ROOT}/llava15" \
    "${DOWNLOAD_ARGS[@]}"
  srun "${HF_PYTHON}" scripts/verify_hf_runtime.py \
    --model-id llava-hf/llava-1.5-7b-hf \
    --block-size 16 \
    --max-new-tokens 16 \
    --output "${OUTPUT_ROOT}/llava15-retention-one-${SLURM_JOB_ID}.json" \
    "${PARITY_DOWNLOAD_ARGS[@]}"
fi

# Gate 5: 20 deterministic MMStar development examples. A dataset revision is
# mandatory and must be supplied from the resolved dataset cache/Hub metadata.
if [[ "${MOSAICKV_RUN_DEV20:-0}" == 1 ]]; then
  if [[ -z "${MOSAICKV_MMSTAR_REVISION:-}" ]]; then
    printf 'MOSAICKV_RUN_DEV20=1 requires MOSAICKV_MMSTAR_REVISION\n' >&2
    exit 2
  fi
  srun "${HF_PYTHON}" -m mosaickv.cli evaluate \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset-revision "${MOSAICKV_MMSTAR_REVISION}" \
    --backend hf \
    --attention-backend eager \
    --method mosaickv_full \
    --task mmstar \
    --retention-ratio 0.5 \
    --block-size 16 \
    --forecast hybrid \
    --draft-tokens 4 \
    --repair-policy entropy_or_prototype_risk \
    --subset-size "${MOSAICKV_DEV_SUBSET_SIZE:-20}" \
    --output-dir "${OUTPUT_ROOT}/mmstar-dev20" \
    "${DOWNLOAD_ARGS[@]}"
fi
