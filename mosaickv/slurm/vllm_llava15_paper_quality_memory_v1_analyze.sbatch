#!/bin/bash
#SBATCH --job-name=mosaickv-paper-qm-analysis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=bii
#SBATCH --output=/scratch/djy8hg/slurm-logs/%x-%j.out
#SBATCH --error=/scratch/djy8hg/slurm-logs/%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python3}
INPUT_ROOT=${MOSAICKV_INPUT_ROOT:?MOSAICKV_INPUT_ROOT is required}
source "${PROJECT_ROOT}/scripts/cache_env.sh"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

mkdir -p "${INPUT_ROOT}/analysis"
"${PYTHON}" scripts/analyze_vllm_paper_quality_memory.py \
  --root "${INPUT_ROOT}" --expected-samples 32 --minimum-reduction 0.30 \
  --bootstrap-resamples 10000 --bootstrap-seed 20270902 \
  --protocol "${PROJECT_ROOT}/configs/experiments/paper_quality_memory_llava_vllm_v1.yaml" \
  --output "${INPUT_ROOT}/analysis/quality_memory_gate.json" \
  --markdown "${INPUT_ROOT}/analysis/quality_memory_gate.md"
