#!/bin/bash
#SBATCH --job-name=mk-readout-audit
#SBATCH --partition=bii
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:10:00
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/audit_readout_v8_2.py \
  worker --root "$2" --plan-sha "$3"
