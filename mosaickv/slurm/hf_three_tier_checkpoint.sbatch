#!/bin/bash
#SBATCH --job-name=mosaickv-hf-three-tier
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
PYTHON=${MOSAICKV_HF_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
export MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"

if [[ ! -x "${PYTHON}" ]]; then
  printf 'Common MosaicKV environment is unavailable: %s\n' "${PYTHON}" >&2
  exit 2
fi
if [[ -z "${MOSAICKV_HF_MODEL_ID:-}" ]]; then
  printf 'Set MOSAICKV_HF_MODEL_ID to one audited LLaVA checkpoint.\n' >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MOSAICKV_VERIFY_THREE_TIER=1

cd "${PROJECT_ROOT}"
srun "${PYTHON}" -m pytest -q -s \
  tests/gpu/test_hf_checkpoint_parity.py::test_pinned_checkpoint_16_token_parity
