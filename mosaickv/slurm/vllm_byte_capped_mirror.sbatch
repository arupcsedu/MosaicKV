#!/bin/bash
# Submit held, then register with the aggregate A100 controller before release.
#SBATCH --job-name=mk-vllm-bytecap
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
set -euo pipefail
: "${MOSAICKV_SOURCE_ROOT:?isolated source snapshot required}"
: "${MOSAICKV_VLLM_BYTE_INDEX:?immutable jobs index required}"
: "${MOSAICKV_VLLM_BYTE_INDEX_SHA:?index hash required}"
source "${MOSAICKV_SOURCE_ROOT}/mosaickv/scripts/cache_env.sh"
module load gcc/11.4.0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${MOSAICKV_SOURCE_ROOT}/mosaickv/src:${MOSAICKV_SOURCE_ROOT}/mosaickv"
export PATH="/scratch/djy8hg/env/mosaickv/bin:${PATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false VLLM_ATTENTION_BACKEND=FLASH_ATTN
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mapfile -t fields < <(python - <<'PY'
import hashlib, json, os
from pathlib import Path
p=Path(os.environ['MOSAICKV_VLLM_BYTE_INDEX'])
if hashlib.sha256(p.read_bytes()).hexdigest() != os.environ['MOSAICKV_VLLM_BYTE_INDEX_SHA']:
    raise RuntimeError('job index changed')
row=json.loads(p.read_text())['jobs'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
print(row['path'])
print(row['sha256'])
PY
)
test "${#fields[@]}" -eq 2
cd "${MOSAICKV_SOURCE_ROOT}/mosaickv"
exec srun python scripts/run_vllm_byte_capped_mirror.py \
  --job "${fields[0]}" --job-sha256 "${fields[1]}"
