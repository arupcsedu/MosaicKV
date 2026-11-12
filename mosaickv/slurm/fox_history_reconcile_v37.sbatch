#!/usr/bin/env bash
#SBATCH --job-name=fox37-reconcile
#SBATCH --account=bii_dsc_community
#SBATCH --partition=bii,standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
module load gcc/11.4.0
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export ATEN_CPU_CAPABILITY=avx2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
bash mosaickv/scripts/assert_clean_worktree.sh
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_review_closure_v2.py tests \
  --root "$3/cpu/full-cpu" --cpu-timeout-seconds 840 --cpu-capture-mode sys
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_history_reconcile_v37.py \
  --history-root "$2" --output "$3/analysis" --cpu-gate-root "$3/cpu"
