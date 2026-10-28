#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
exec bash "$1/mosaickv/slurm/evicastkv_operator_audit.sbatch" "$@"
