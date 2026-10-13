#!/bin/bash
#SBATCH --job-name=mk-a100-policy
#SBATCH --partition=bii
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch/djy8hg/slurm-logs/%x-%j.out
#SBATCH --error=/scratch/djy8hg/slurm-logs/%x-%j.err
set -euo pipefail
: "${MOSAICKV_POLICY_ROOT:?immutable controller snapshot required}"
source "${MOSAICKV_POLICY_ROOT}/cache_env.sh"
export PYTHONDONTWRITEBYTECODE=1
exec /scratch/djy8hg/env/mosaickv/bin/python \
  "${MOSAICKV_POLICY_ROOT}/control_a100_schedule.py" \
  --policy "${MOSAICKV_POLICY_ROOT}/a100_schedule.json" \
  --state "${MOSAICKV_POLICY_ROOT}/state.json" --apply --watch
