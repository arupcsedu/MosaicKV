#!/bin/bash
#SBATCH --job-name=mk-paper-v3-coordinate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=bii
#SBATCH --output=/scratch/djy8hg/slurm-logs/%x-%j.out
#SBATCH --error=/scratch/djy8hg/slurm-logs/%x-%j.err
#SBATCH --export=ALL

set -euo pipefail
PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
: "${MOSAICKV_PAPER_FULL_LEDGER_ROOT:?missing V3 full-ledger root}"
: "${MOSAICKV_PAPER_INPUT_ROOT:?missing V3 immutable input root}"
: "${MOSAICKV_PAPER_OUTPUT_ROOT:?missing V3 result root}"
: "${MOSAICKV_PAPER_ANALYSIS_ROOT:?missing V3 analysis root}"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"
bash "${PROJECT_ROOT}/scripts/submit_paper_quality_hf_vllm_llava15_v3_continuation.sh"
