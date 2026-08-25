#!/bin/bash
#SBATCH --job-name=mkv-v105-preflight
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:30:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/v105-preflight-v2-%A_%a.log
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
SPEC=/scratch/djy8hg/runs/mosaickv/choice-c-v105-reserved-beam-preflight-v2/inputs/gpu_preflight_worker_spec.json
OUTPUT_ROOT=/scratch/djy8hg/runs/mosaickv/choice-c-v105-reserved-beam-preflight-v2/preflight
: "${MOSAICKV_V105_EXPECTED_GIT_SHA:?expected immutable V105 source SHA is required}"
: "${MOSAICKV_V105_EXPECTED_SPEC_SHA:?expected immutable V105 worker-spec SHA is required}"

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
export PYTHONPATH="${PACKAGE_ROOT}/src:${PACKAGE_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export MOSAICKV_FIXED_CLOCK_STATUS=not_permitted_or_not_requested
export MOSAICKV_ENABLE_V100_GPU_DEBUG=1
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"
[[ "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)" == "${MOSAICKV_V105_EXPECTED_GIT_SHA}" ]]
[[ "$(sha256sum "${SPEC}" | cut -d ' ' -f 1)" == "${MOSAICKV_V105_EXPECTED_SPEC_SHA}" ]]
"${PYTHON}" scripts/prepare_choice_c_v105_preflight.py --validate-only >/dev/null
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,pstate,clocks.sm,clocks.mem,power.draw --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
srun "${PYTHON}" scripts/run_choice_c_v74_label_blind_calibration_worker.py \
  --spec "${SPEC}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-prefix v105-preflight \
  --expected-experiment-id choice_c_v105_reserved_beam_preflight \
  --expected-ledger-classification fresh_disjoint_label_blind_calibration
