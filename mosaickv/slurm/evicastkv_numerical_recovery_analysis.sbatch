#!/bin/bash
#SBATCH --job-name=evicast-num-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --partition=bii
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
# SBATCH directives in the invoked file are shell comments, not GPU requests.
exec bash "$1/mosaickv/slurm/evicastkv_numerical_recovery.sbatch" "$1" "$2" "$3" analyze
