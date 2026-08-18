#!/bin/bash
# CPU-only one-node array: bii avoids parallel's mandatory two-node allocation.
#SBATCH --job-name=mosaickv-v74-ledger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=bii
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/v74-ledger-%A_%a.log
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
OUTPUT_ROOT=${MOSAICKV_V74_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v74-label-blind-calibration-v6}

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
export PYTHONPATH="${PACKAGE_ROOT}/src:${PACKAGE_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    TASK=docvqa
    REVISION=539088ef8a8ada01ac8e2e6d4e372586748a265e
    ;;
  1)
    TASK=chartqa
    REVISION=9e63b7df1592a1c2158e735cc1725454aef0d6d9
    ;;
  *)
    printf 'unsupported array index: %s\n' "${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

EXCLUSIONS=(--exclude-ledger-directory "${PACKAGE_ROOT}/configs/ledgers")
for directory in \
  /scratch/djy8hg/runs/mosaickv/choice-c-v73-clean-heldout-quality-memory-v1/inputs \
  /scratch/djy8hg/runs/mosaickv/choice-c-v74-label-blind-calibration-v1/inputs \
  /scratch/djy8hg/runs/mosaickv/choice-c-v74-label-blind-calibration-v2/inputs \
  /scratch/djy8hg/runs/mosaickv/choice-c-v74-label-blind-calibration-v3/inputs \
  /scratch/djy8hg/runs/mosaickv/choice-c-v74-label-blind-calibration-v4/inputs \
  /scratch/djy8hg/runs/mosaickv/choice-c-v74-label-blind-calibration-v5/inputs \
  /scratch/djy8hg/runs/mosaickv/generalization-evaluation-matrix-v1/ledgers \
  /scratch/djy8hg/runs/mosaickv/ledgers
do
  if [[ -d "${directory}" ]]; then
    EXCLUSIONS+=(--exclude-ledger-directory "${directory}")
  fi
done

OBSERVED_RESULTS=/scratch/djy8hg/runs/mosaickv/experiments/main-quality-2d9e316-n8-v1
if [[ -d "${OBSERVED_RESULTS}" ]]; then
  EXCLUSIONS+=(--exclude-result-directory "${OBSERVED_RESULTS}")
fi

"${PYTHON}" scripts/materialize_sample_ledger.py \
  --task "${TASK}" \
  --dataset-revision "${REVISION}" \
  --model-name llava-hf/llava-1.5-7b-hf \
  --seed 20270821 \
  --subset-size 16 \
  --classification fresh_disjoint_label_blind_calibration \
  "${EXCLUSIONS[@]}" \
  --output "${OUTPUT_ROOT}/inputs/calibration_${TASK}.json"
