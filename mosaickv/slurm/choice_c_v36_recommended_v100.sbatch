#!/bin/bash
#SBATCH --job-name=mosaickv-v36-v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=01:30:00
#SBATCH --partition=bii-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --array=0-1
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/choice-c-v36-v100-%A-%a.log

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
export MOSAICKV_PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
export MOSAICKV_CACHE_ROOT=/scratch/djy8hg/cache/mosaickv
SPEC=configs/experiments/choice_c_batched_prototype_system_v16.yaml
PROFILE=/scratch/djy8hg/runs/mosaickv/choice-c-forecast-stability-design-v15/frozen_fidelity_profile.json
LEDGER=configs/ledgers/choice_c_v36_retrospective_docvqa.json
OUTPUT_ROOT=/scratch/djy8hg/runs/mosaickv/choice-c-v36-recommended-v100-${SLURM_ARRAY_JOB_ID}

# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

mkdir -p "${OUTPUT_ROOT}"
sha256sum \
  "${SPEC}" \
  "${PROFILE}" \
  "${LEDGER}" \
  scripts/run_choice_c_v36_recommended_worker.py \
  scripts/analyze_choice_c_v36_recommended.py \
  src/mosaickv/backends/hf_runtime.py \
  src/mosaickv/config.py \
  src/mosaickv/graph/builder.py \
  src/mosaickv/graph/pooling.py \
  src/mosaickv/prototypes/__init__.py \
  src/mosaickv/prototypes/construction.py \
  src/mosaickv/residual/storage.py \
  src/mosaickv/selection/selector.py \
  >"${OUTPUT_ROOT}/source_sha256_rank${SLURM_ARRAY_TASK_ID}.txt"

"${MOSAICKV_PYTHON}" scripts/run_choice_c_v36_recommended_worker.py \
  --rank "${SLURM_ARRAY_TASK_ID}" \
  --spec "${SPEC}" \
  --profile "${PROFILE}" \
  --ledger "${LEDGER}" \
  --output-root "${OUTPUT_ROOT}"
