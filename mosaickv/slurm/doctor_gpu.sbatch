#!/bin/bash
#SBATCH --job-name=ragbench-multinode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=37
#SBATCH --mem=0
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx

set -euo pipefail

MOSAICKV_VENV=/scratch/djy8hg/env/mosaickv_dev
export PYTHONNOUSERSITE=1

srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  "${MOSAICKV_VENV}/bin/mosaickv" doctor --json --require-gpu
srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  "${MOSAICKV_VENV}/bin/mosaickv" smoke --json
srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  "${MOSAICKV_VENV}/bin/python" -m pytest -q -p no:cacheprovider -m gpu tests/gpu
