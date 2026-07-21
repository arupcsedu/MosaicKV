#!/bin/bash
#SBATCH --job-name=mosaickv-fullkv
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
OUTPUT_DIR=${MOSAICKV_OUTPUT_DIR:-/scratch/djy8hg/results/mosaickv/fullkv}
MODE=${MOSAICKV_FULLKV_MODE:-fullkv-debug}

export MOSAICKV_CACHE_ROOT="${CACHE_ROOT}"
# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"

: "${MOSAICKV_CONFIG:?export MOSAICKV_CONFIG to a FullKV TOML/JSON config}"
: "${MOSAICKV_WORKLOAD:?export MOSAICKV_WORKLOAD to a local JSONL workload}"
: "${MOSAICKV_RUN_ID:?export MOSAICKV_RUN_ID to an immutable run identifier}"

if [[ ! -x "${HF_PYTHON}" ]]; then
  printf 'HF environment does not exist or is not executable: %s\n' "${HF_PYTHON}" >&2
  exit 2
fi
case "${MODE}" in
  fullkv-debug|fullkv-smoke|fullkv-run|fullkv-latency) ;;
  *) printf 'Unsupported MOSAICKV_FULLKV_MODE: %s\n' "${MODE}" >&2; exit 2 ;;
esac

export PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers"
export TORCH_HOME="${CACHE_ROOT}/torch"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUTPUT_DIR}"
ARGS=(
  "${MODE}"
  --config "${MOSAICKV_CONFIG}"
  --workload "${MOSAICKV_WORKLOAD}"
  --run-id "${MOSAICKV_RUN_ID}"
  --raw-output "${OUTPUT_DIR}/${MOSAICKV_RUN_ID}.raw.jsonl"
  --aggregate-output "${OUTPUT_DIR}/${MOSAICKV_RUN_ID}.aggregate.json"
  --log-output "${OUTPUT_DIR}/${MOSAICKV_RUN_ID}.log.json"
  --manifest "${OUTPUT_DIR}/${MOSAICKV_RUN_ID}.manifest.json"
  --cache-root "${CACHE_ROOT}"
  --json
)
if [[ "${MOSAICKV_ALLOW_MODEL_DOWNLOAD:-0}" == 1 ]]; then
  ARGS+=(--allow-download)
fi

cd "${PROJECT_ROOT}"
srun "${HF_PYTHON}" -m mosaickv.cli "${ARGS[@]}"
