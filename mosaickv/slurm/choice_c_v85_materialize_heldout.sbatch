#!/bin/bash
#SBATCH --job-name=mosaickv-v85-ledger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=bii
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/v85-ledger-%A_%a.log
#SBATCH --export=ALL

set -euo pipefail

PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
PACKAGE_ROOT=${MOSAICKV_V85_RUNTIME_PACKAGE_ROOT:?missing pinned v85 runtime package root}
OUTPUT_ROOT=${MOSAICKV_V85_OUTPUT_ROOT:-/scratch/djy8hg/runs/mosaickv/choice-c-v85-v84-frozen-untouched-heldout-quality-memory-v1}
SPEC=${MOSAICKV_V85_SPEC:?missing immutable v85 preregistration}
EXPECTED_SPEC_SHA=${MOSAICKV_V85_SPEC_SHA256:?missing v85 preregistration hash}
FROZEN=${MOSAICKV_V85_FROZEN:?missing v84 frozen candidate}
EXPECTED_FROZEN_SHA=${MOSAICKV_V85_FROZEN_SHA256:?missing v84 frozen-candidate hash}
V83_INPUTS=/scratch/djy8hg/runs/mosaickv/choice-c-v83-v82-frozen-untouched-heldout-quality-memory-v1/inputs
V84_INPUTS=/scratch/djy8hg/runs/mosaickv/choice-c-v84-disjoint-risk-gated-hybrid-calibration-v1/inputs

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
export PYTHONPATH="${PACKAGE_ROOT}/src:${PACKAGE_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"
[[ "$(sha256sum "${SPEC}" | cut -d' ' -f1)" == "${EXPECTED_SPEC_SHA}" ]] || {
  printf 'v85 preregistration hash mismatch\n' >&2
  exit 2
}
[[ "$(sha256sum "${FROZEN}" | cut -d' ' -f1)" == "${EXPECTED_FROZEN_SHA}" ]] || {
  printf 'v84 frozen-candidate hash mismatch\n' >&2
  exit 2
}
for required in "${V83_INPUTS}" "${V84_INPUTS}"; do
  [[ -d "${required}" ]] || {
    printf 'required held-out exclusion directory is missing: %s\n' "${required}" >&2
    exit 2
  }
done

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

EXCLUSIONS=(
  --exclude-ledger-directory "${PACKAGE_ROOT}/configs/ledgers"
  --exclude-ledger-directory "${V83_INPUTS}"
  --exclude-ledger-directory "${V84_INPUTS}"
)
for directory in \
  /scratch/djy8hg/runs/mosaickv/choice-c-v73-clean-heldout-quality-memory-v1/inputs \
  /scratch/djy8hg/runs/mosaickv/choice-c-v74-label-blind-calibration-v1/inputs \
  /scratch/djy8hg/runs/mosaickv/choice-c-v74-label-blind-calibration-v2/inputs \
  /scratch/djy8hg/runs/mosaickv/choice-c-v74-label-blind-calibration-v3/inputs \
  /scratch/djy8hg/runs/mosaickv/choice-c-v74-label-blind-calibration-v4/inputs \
  /scratch/djy8hg/runs/mosaickv/choice-c-v74-label-blind-calibration-v5/inputs \
  /scratch/djy8hg/runs/mosaickv/choice-c-v74-label-blind-calibration-v6/inputs \
  /scratch/djy8hg/runs/mosaickv/choice-c-v75-frozen-heldout-quality-memory-v6/inputs \
  /scratch/djy8hg/runs/mosaickv/choice-c-v76-disjoint-ocr-calibration-v1/inputs \
  /scratch/djy8hg/runs/mosaickv/choice-c-v78-disjoint-visual-balance-calibration-v1/inputs \
  /scratch/djy8hg/runs/mosaickv/choice-c-v79-disjoint-adaptive-swap-calibration-v1/inputs \
  /scratch/djy8hg/runs/mosaickv/choice-c-v80-disjoint-risk-gated-swap-calibration-v1/inputs \
  /scratch/djy8hg/runs/mosaickv/choice-c-v81-disjoint-exact-future-rerank-calibration-v1/inputs \
  /scratch/djy8hg/runs/mosaickv/choice-c-v82-disjoint-hybrid-divergence-calibration-v1/inputs \
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
  --seed 20280113 \
  --subset-size 128 \
  --classification fresh_disjoint_heldout_quality_validation \
  --selection-freeze "${FROZEN}" \
  "${EXCLUSIONS[@]}" \
  --output "${OUTPUT_ROOT}/inputs/heldout_${TASK}.json"
