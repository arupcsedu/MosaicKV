#!/bin/bash
#SBATCH --job-name=mkv-exp5-logits
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=120G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/generalization-logits-%j.out
#SBATCH --error=/scratch/djy8hg/runs/mosaickv/slurm/generalization-logits-%j.err

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
INDEX=/scratch/djy8hg/runs/mosaickv/generalization-evaluation-matrix-v1/jobs.jsonl
DESTINATION=/scratch/djy8hg/runs/mosaickv/generalization-backend-logits-v1

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
export PYTHONPATH="${PACKAGE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "${PROJECT_ROOT}"
srun "${PYTHON}" "${PACKAGE_ROOT}/scripts/run_hf_backend_logit_parity.py" \
  --index "${INDEX}" \
  --destination "${DESTINATION}"
