#!/bin/bash
#SBATCH --job-name=mosaickv-v85-qmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/v85-qmem-%A_%a.log
#SBATCH --export=ALL

set -euo pipefail

PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
PACKAGE_ROOT=${MOSAICKV_V85_RUNTIME_PACKAGE_ROOT:?missing pinned v85 runtime package root}
OUTPUT_ROOT=${MOSAICKV_V85_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v85-v84-frozen-untouched-heldout-quality-memory-v1}
SPEC=${MOSAICKV_V85_SPEC:?missing immutable v85 preregistration}
EXPECTED_SPEC_SHA=${MOSAICKV_V85_SPEC_SHA256:?missing v85 preregistration hash}
FROZEN=${MOSAICKV_V85_FROZEN:?missing v84 frozen candidate}
EXPECTED_FROZEN_SHA=${MOSAICKV_V85_FROZEN_SHA256:?missing v84 frozen-candidate hash}

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
export PYTHONPATH="${PACKAGE_ROOT}/src:${PACKAGE_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export MOSAICKV_FIXED_CLOCK_STATUS=not_permitted_or_not_requested
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"
[[ "$(sha256sum "${SPEC}" | cut -d' ' -f1)" == "${EXPECTED_SPEC_SHA}" ]]
[[ "$(sha256sum "${FROZEN}" | cut -d' ' -f1)" == "${EXPECTED_FROZEN_SHA}" ]]
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,pstate,clocks.sm,clocks.mem,power.draw --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
srun "${PYTHON}" scripts/run_choice_c_v75_frozen_quality_memory_worker.py \
  --spec "${SPEC}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-prefix v85 \
  --expected-experiment-id choice_c_v85_v84_frozen_untouched_heldout_quality_memory \
  --expected-frozen-classification frozen_v84_label_blind_risk_gated_hybrid_candidate \
  --expected-ledger-classification fresh_disjoint_heldout_quality_validation
