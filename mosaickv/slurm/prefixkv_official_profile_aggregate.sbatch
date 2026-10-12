#!/bin/bash
# Run only after every array worker from prefixkv_official_profile_array.sbatch
# completed successfully.  This CPU reducer validates the exact 1,000-ID union
# and creates the sole eligible official-profile artifact.
#SBATCH --job-name=mosaickv-prefixkv-aggregate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --partition=bii
#SBATCH --export=ALL

set -euo pipefail

REPOSITORY_ROOT=/scratch/djy8hg/workdir/MosaicKV
PROJECT_ROOT=${REPOSITORY_ROOT}/mosaickv
OUTPUT_ROOT=${MOSAICKV_PREFIXKV_OFFICIAL_PROFILE_OUTPUT:-/scratch/djy8hg/runs/mosaickv/prefixkv/official-profile-r050}
export MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}
# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"
export PYTHONNOUSERSITE=1
export PYTHONPATH=${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}

cd "${REPOSITORY_ROOT}"
/scratch/djy8hg/env/mosaickv_prefixkv_parity/bin/python \
  mosaickv/scripts/aggregate_prefixkv_official_profile.py \
  --shard-root "${OUTPUT_ROOT}" \
  --data-path /scratch/djy8hg/data/mosaickv/prefixkv/detail_1k.json \
  --official-profile-path "${REPOSITORY_ROOT}/third_party/PrefixKV/confs/prefixkv_llava-v1.5-7b_0.5.json" \
  --output "${OUTPUT_ROOT}/official_profile_reproduction_gate.json"
