#!/bin/bash
#SBATCH --job-name=mosaickv-quality
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/code-quality-%j.log

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
ENV_DIR=/scratch/djy8hg/env/mosaickv
source "${PROJECT_ROOT}/scripts/cache_env.sh"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

srun "${ENV_DIR}/bin/ruff" check src scripts tests
srun "${ENV_DIR}/bin/ruff" format --check src scripts tests
srun "${ENV_DIR}/bin/mypy" src scripts/verify_envs.py
