#!/usr/bin/env bash
#SBATCH --job-name=fox63-packet
#SBATCH --account=bii_dsc_community
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
set -euo pipefail
cd "$1"
unset MOSAICKV_ALLOW_DIRTY_EXPLORATORY
source mosaickv/scripts/cache_env.sh
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ "$(git rev-parse HEAD)" != "$3" ]]; then
  exit 2
fi
bash mosaickv/scripts/assert_clean_worktree.sh
exec /scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/prepare_fox_review_packet_v63.py --output "$2/packet"
