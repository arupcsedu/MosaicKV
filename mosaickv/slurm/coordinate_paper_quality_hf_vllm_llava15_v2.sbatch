#!/bin/bash
#SBATCH --job-name=mk-paper-coordinate-v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=parallel
#SBATCH --output=/scratch/djy8hg/slurm-logs/%x-%j.out
#SBATCH --error=/scratch/djy8hg/slurm-logs/%x-%j.err
#SBATCH --export=ALL

# Runs only after all three master-ledger workers succeed. It creates the
# immutable 12-shard/24-job index, submits the four-A100-capped array, and
# submits analysis after successful completion of the entire array.

set -euo pipefail
REPOSITORY=/scratch/djy8hg/workdir/MosaicKV
PROJECT_ROOT=${REPOSITORY}/mosaickv
: "${MOSAICKV_PAPER_LEDGER_ROOT:?missing reviewed ledger root}"
: "${MOSAICKV_PAPER_INPUT_ROOT:?missing immutable input root}"
: "${MOSAICKV_PAPER_OUTPUT_ROOT:?missing result root}"
: "${MOSAICKV_PAPER_ANALYSIS_ROOT:?missing analysis root}"

"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"
cd "${REPOSITORY}"
bash "${PROJECT_ROOT}/scripts/submit_paper_quality_hf_vllm_llava15_v2.sh"
