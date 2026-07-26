#!/bin/bash
#SBATCH --job-name=mkv-fa2-wheel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --partition=bii
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/fa2-wheel-%j.out
#SBATCH --error=/scratch/djy8hg/runs/mosaickv/slurm/fa2-wheel-%j.err

set -euo pipefail

module load gcc/11.4.0 cuda/12.8.0
export CC=$(command -v gcc)
export CXX=$(command -v g++)
export MAX_JOBS=16
export TORCH_CUDA_ARCH_LIST=8.0
export FLASH_ATTN_CUDA_ARCHS=80
/scratch/djy8hg/workdir/MosaicKV/mosaickv/scripts/build_flash_attn_wheel.sh
