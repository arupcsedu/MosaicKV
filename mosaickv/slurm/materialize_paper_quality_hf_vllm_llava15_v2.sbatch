#!/bin/bash
#SBATCH --job-name=mk-paper-ledgers-v2
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --partition=bii
#SBATCH --output=/scratch/djy8hg/slurm-logs/%x-%A_%a.out
#SBATCH --error=/scratch/djy8hg/slurm-logs/%x-%A_%a.err
#SBATCH --export=ALL

# CPU-only deterministic input preparation. This script does not load model
# weights and is intentionally separate from the 24 GPU jobs.

set -euo pipefail
module load "${MOSAICKV_GCC_MODULE:-gcc/11.4.0}"

REPOSITORY=/scratch/djy8hg/workdir/MosaicKV
PROJECT_ROOT=${REPOSITORY}/mosaickv
PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python3}
CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
PROTOCOL=${PROJECT_ROOT}/configs/experiments/paper_quality_hf_vllm_llava15_v2.yaml
: "${MOSAICKV_PAPER_LEDGER_ROOT:?set a new write-once ledger root under /scratch/djy8hg}"

TASKS=(mmstar textvqa docvqa)
REVISIONS=(
  bc98d668301da7b14f648724866e57302778ab27
  9c0699cd19768ac5ab97568f6b3cbac4c0062884
  539088ef8a8ada01ac8e2e6d4e372586748a265e
)
TASK=${TASKS[${SLURM_ARRAY_TASK_ID}]}
REVISION=${REVISIONS[${SLURM_ARRAY_TASK_ID}]}
OUTPUT=${MOSAICKV_PAPER_LEDGER_ROOT}/${TASK}.json

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
mkdir -p "${MOSAICKV_PAPER_LEDGER_ROOT}" "${TMPDIR}"
cd "${PROJECT_ROOT}"

"${PYTHON}" - "${PROTOCOL}" <<'PY'
import sys, yaml
value = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
if value.get("execution_authorized") is not True:
    raise SystemExit("paper protocol execution is not authorized")
PY

srun "${PYTHON}" scripts/materialize_sample_ledger.py \
  --task "${TASK}" --dataset-revision "${REVISION}" \
  --model-name llava-hf/llava-1.5-7b-hf \
  --seed 20271001 --subset-size 256 \
  --classification fixed_preregistered_paper_quality_memory \
  --selection-freeze "${PROTOCOL}" --output "${OUTPUT}" --resume
