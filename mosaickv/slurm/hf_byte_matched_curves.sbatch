#!/bin/bash
#SBATCH --job-name=mk-hf-byte-curve
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/scratch/djy8hg/slurm-logs/%x-%A_%a.out
#SBATCH --error=/scratch/djy8hg/slurm-logs/%x-%A_%a.err
#SBATCH --export=ALL

set -euo pipefail
module load "${MOSAICKV_GCC_MODULE:-gcc/11.4.0}"

REPOSITORY=${MOSAICKV_SOURCE_ROOT:?MOSAICKV_SOURCE_ROOT is required}
PROJECT_ROOT=${REPOSITORY}/mosaickv
PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python3}
: "${MOSAICKV_HF_BYTE_CURVE_INDEX:?set the immutable byte-curve jobs.jsonl path}"
: "${SLURM_ARRAY_TASK_ID:?submit this script as a Slurm array}"

export MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_CUDA_ARCH_LIST=8.0
mkdir -p "${TMPDIR}" /scratch/djy8hg/slurm-logs
cd "${PROJECT_ROOT}"

mapfile -t JOB_FIELDS < <(
  "${PYTHON}" - "${MOSAICKV_HF_BYTE_CURVE_INDEX}" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
row = rows[int(sys.argv[2])]
print(row["job_spec_path"])
print(row["job_spec_sha256"])
PY
)

srun "${PYTHON}" scripts/run_hf_byte_matched_curves_worker.py \
  --job-spec "${JOB_FIELDS[0]}" \
  --job-spec-sha256 "${JOB_FIELDS[1]}"
