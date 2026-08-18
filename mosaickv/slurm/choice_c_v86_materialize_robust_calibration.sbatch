#!/bin/bash
#SBATCH --job-name=mosaickv-v86-ledger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=bii
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/v86-ledger-%A_%a.log
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
OUTPUT_ROOT=${MOSAICKV_V86_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v86-disjoint-robust-fidelity-calibration-v1}

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
    TASK=textvqa
    REVISION=9c0699cd19768ac5ab97568f6b3cbac4c0062884
    ;;
  *)
    printf 'unsupported array index: %s\n' "${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

V85_INPUTS=/scratch/djy8hg/runs/mosaickv/choice-c-v85-v84-frozen-untouched-heldout-quality-memory-v1/inputs
if [[ ! -d "${V85_INPUTS}" ]]; then
  printf 'required sealed v85 exclusion directory is missing: %s\n' "${V85_INPUTS}" >&2
  exit 2
fi

EXCLUSIONS=(
  --exclude-ledger-directory "${PACKAGE_ROOT}/configs/ledgers"
  --exclude-ledger-directory "${V85_INPUTS}"
)
shopt -s nullglob
for directory in /scratch/djy8hg/runs/mosaickv/choice-c-v*/inputs; do
  if [[ "${directory}" != "${OUTPUT_ROOT}/inputs" && "${directory}" != "${V85_INPUTS}" ]]; then
    EXCLUSIONS+=(--exclude-ledger-directory "${directory}")
  fi
done
shopt -u nullglob

for directory in \
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
  --seed 20280203 \
  --subset-size 128 \
  --classification fresh_disjoint_label_blind_calibration \
  "${EXCLUSIONS[@]}" \
  --output "${OUTPUT_ROOT}/inputs/calibration_${TASK}.json"
