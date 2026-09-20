#!/bin/bash
#SBATCH --job-name=mk-published-parity
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:a100:1
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=${MOSAICKV_PROJECT_ROOT:?Set MOSAICKV_PROJECT_ROOT to an immutable checkout}
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
OUTPUT=${MOSAICKV_PARITY_OUTPUT:?Set MOSAICKV_PARITY_OUTPUT to a write-once scratch path}

# shellcheck source=../scripts/cache_env.sh
source "${PACKAGE_ROOT}/scripts/cache_env.sh"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"

if [[ ! -x "${PYTHON}" ]]; then
  printf 'MosaicKV common environment is unavailable: %s\n' "${PYTHON}" >&2
  exit 2
fi
if [[ -e "${OUTPUT}" ]]; then
  printf 'Refusing to overwrite parity artifact: %s\n' "${OUTPUT}" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PACKAGE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PACKAGE_ROOT}"
srun "${PYTHON}" scripts/verify_published_baseline_checkpoint_parity.py \
  --model-id llava-hf/llava-1.5-7b-hf \
  --revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9 \
  --max-new-tokens 16 \
  --logit-atol 1e-4 \
  --output "${OUTPUT}"
