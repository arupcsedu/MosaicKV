#!/bin/bash
#SBATCH --job-name=mosaickv-sglang-fullkv
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

# SGLang 0.5.10 JIT kernels include C++20 standard-library headers.  The
# cluster default GCC 8.5 does not provide them; load the audited compiler
# before spawning the server.  An override remains explicit in the job env.
module load "${MOSAICKV_GCC_MODULE:-gcc/11.4.0}"

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
SGLANG_PYTHON=${MOSAICKV_SGLANG_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
OUTPUT_ROOT=${MOSAICKV_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/sglang-fullkv}
CONFIG=${MOSAICKV_SGLANG_CONFIG:-${PROJECT_ROOT}/configs/sglang_fullkv_3b.yaml}
RUN_ID=${MOSAICKV_RUN_ID:-sglang-fullkv-${SLURM_JOB_ID}}

export MOSAICKV_CACHE_ROOT="${CACHE_ROOT}"
# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"

if [[ ! -x "${SGLANG_PYTHON}" ]]; then
  printf 'Pinned SGLang environment is absent: %s\n' "${SGLANG_PYTHON}" >&2
  printf 'Create it explicitly with mosaickv/scripts/create_envs.sh common\n' >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TORCH_HOME="${CACHE_ROOT}/torch"
export SGLANG_CACHE_DIR="${CACHE_ROOT}/sglang"
export TVM_FFI_CACHE_DIR="${CACHE_ROOT}/tvm-ffi"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$(dirname "${SGLANG_PYTHON}"):${PATH}"

if [[ "${MOSAICKV_ALLOW_MODEL_DOWNLOAD:-0}" != 1 ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${TVM_FFI_CACHE_DIR}"
cd "${PROJECT_ROOT}"

ARGS=(
  evaluate
  --config "${CONFIG}"
  --task synthetic_smoke
  --subset-size "${MOSAICKV_SUBSET_SIZE:-2}"
  --run-id "${RUN_ID}"
  --output-dir "${OUTPUT_ROOT}"
  --trace-directory "${OUTPUT_ROOT}/traces"
  --cache-probe-repeats "${MOSAICKV_CACHE_PROBE_REPEATS:-2}"
  --sglang-mem-fraction-static "${MOSAICKV_SGLANG_MEM_FRACTION_STATIC:-0.8}"
  --sglang-context-length "${MOSAICKV_SGLANG_CONTEXT_LENGTH:-4096}"
  --sglang-page-size "${MOSAICKV_SGLANG_PAGE_SIZE:-1}"
  --tensor-parallel-size "${MOSAICKV_TENSOR_PARALLEL_SIZE:-1}"
  --json
)
if [[ "${MOSAICKV_ALLOW_MODEL_DOWNLOAD:-0}" != 1 ]]; then
  ARGS+=(--local-files-only)
fi

srun "${SGLANG_PYTHON}" -m mosaickv.cli "${ARGS[@]}"

mapfile -t TRACES < <(find "${OUTPUT_ROOT}/traces/${RUN_ID}" -maxdepth 1 -name '*.json' -print | sort)
if [[ ${#TRACES[@]} -eq 0 ]]; then
  printf 'No SGLang trace was produced for run %s\n' "${RUN_ID}" >&2
  exit 1
fi
VERIFY_ARGS=()
for trace in "${TRACES[@]}"; do
  VERIFY_ARGS+=(--trace "${trace}")
done
srun "${SGLANG_PYTHON}" scripts/verify_sglang_backend.py \
  "${VERIFY_ARGS[@]}" \
  --results-jsonl "${OUTPUT_ROOT}/${RUN_ID}/results.jsonl" \
  --require-gpu-measurements
