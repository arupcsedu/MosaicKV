#!/bin/bash
# Canonical official PrefixKV profile reproduction, sharded for the two-hour
# A100 limit.  Every worker is intentionally ineligible on its own; use
# prefixkv_official_profile_aggregate.sbatch only after all 63 shards finish.
#SBATCH --job-name=mosaickv-prefixkv-official
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-62%8
#SBATCH --export=ALL

set -euo pipefail

REPOSITORY_ROOT=/scratch/djy8hg/workdir/MosaicKV
PROJECT_ROOT=${REPOSITORY_ROOT}/mosaickv
TOTAL_SAMPLES=1000
SAMPLES_PER_SHARD=16
OUTPUT_ROOT=${MOSAICKV_PREFIXKV_OFFICIAL_PROFILE_OUTPUT:-/scratch/djy8hg/runs/mosaickv/prefixkv/official-profile-r050}
SHARD_INDEX=${SLURM_ARRAY_TASK_ID:?SLURM array task ID is required}
START_INDEX=$((SHARD_INDEX * SAMPLES_PER_SHARD))
END_INDEX=$((START_INDEX + SAMPLES_PER_SHARD - 1))
if (( END_INDEX >= TOTAL_SAMPLES )); then
  END_INDEX=$((TOTAL_SAMPLES - 1))
fi
if (( START_INDEX >= TOTAL_SAMPLES )); then
  printf 'Shard %s starts outside registered evaluation universe\n' "${SHARD_INDEX}" >&2
  exit 2
fi
EVALUATION_INDICES=$(seq -s, "${START_INDEX}" "${END_INDEX}")

export MOSAICKV_PREFIXKV_OFFICIAL_PROFILE_REPRODUCTION=1
export MOSAICKV_PREFIXKV_OFFICIAL_PROFILE_SHARD=1
export MOSAICKV_PREFIXKV_EVALUATION_INDICES="${EVALUATION_INDICES}"
export MOSAICKV_PREFIXKV_PARITY_OUTPUT="${OUTPUT_ROOT}"
export MOSAICKV_RUN_ID="official-profile-r050-shard-$(printf '%03d' "${SHARD_INDEX}")"
# shellcheck source=prefixkv_official_parity.sbatch
exec bash "${PROJECT_ROOT}/slurm/prefixkv_official_parity.sbatch"
