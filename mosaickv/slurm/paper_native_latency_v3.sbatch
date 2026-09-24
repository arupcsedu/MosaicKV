#!/bin/bash
#SBATCH --job-name=mk-native-lat-v3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:a100:1

set -euo pipefail

repo=/scratch/djy8hg/workdir/MosaicKV
package="$repo/mosaickv"
source "$package/scripts/cache_env.sh"
cd "$repo"
"$package/scripts/assert_clean_worktree.sh"

exec /scratch/djy8hg/env/mosaickv/bin/python \
  "$package/scripts/run_paper_native_latency_v3.py" \
  --protocol "$package/configs/experiments/paper_native_latency_v3.yaml" \
  --published-protocol "$package/configs/experiments/published_paper_native_metrics_v1.yaml" \
  --ledger "${MOSAICKV_PREFIX_LEDGER:?set MOSAICKV_PREFIX_LEDGER}" \
  --ledger-sha "${MOSAICKV_PREFIX_LEDGER_SHA:?set MOSAICKV_PREFIX_LEDGER_SHA}" \
  --output-root "${MOSAICKV_OUTPUT_ROOT:?set MOSAICKV_OUTPUT_ROOT}" \
  --source-git-sha "${MOSAICKV_SOURCE_SHA:?set MOSAICKV_SOURCE_SHA}"
