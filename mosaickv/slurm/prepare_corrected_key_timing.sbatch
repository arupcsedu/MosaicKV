#!/usr/bin/env bash
# Finite conditional follow-up: all numerical controls are checked before any
# timing workers are submitted. Failure never widens the scope or cohort.
set -euo pipefail
cd "$1"
source mosaickv/scripts/cache_env.sh
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
bash mosaickv/scripts/assert_clean_worktree.sh
exec /scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/source_hybrid_latency.py \
  prepare --root "$2" --bitmap-prefix --shared-suffix --planner-exact-fast \
  --cpu-torch-threads 1 --guard-tail --manual-guard-capture \
  --native-readout-handoff --structural-templates --device-guard-records \
  --direct-metadata-key --full-head-parent
