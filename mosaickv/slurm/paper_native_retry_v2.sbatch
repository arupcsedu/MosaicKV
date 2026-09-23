#!/bin/bash
#SBATCH --job-name=mk-native-v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:a100:1

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: sbatch paper_native_retry_v2.sbatch LANE METHOD" >&2
  exit 2
fi

exec /scratch/djy8hg/workdir/MosaicKV/mosaickv/scripts/run_paper_native_retry_v2.sh "$1" "$2"
