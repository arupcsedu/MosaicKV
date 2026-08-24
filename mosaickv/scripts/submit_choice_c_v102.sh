#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
OUTPUT_ROOT=/scratch/djy8hg/runs/mosaickv/choice-c-v102-v100-fresh-cross-dataset-calibration-v1-retry1
INPUT_ROOT=/scratch/djy8hg/runs/mosaickv/choice-c-v102-v100-fresh-cross-dataset-calibration-v1
SPEC=${PACKAGE_ROOT}/configs/experiments/choice_c_v102_v100_fresh_cross_dataset_calibration.yaml
REPORT=${INPUT_ROOT}/inputs/disjointness_report.json
FROZEN=${OUTPUT_ROOT}/analysis/frozen_candidate.yaml

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
cd "${PACKAGE_ROOT}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"
"${PYTHON}" scripts/validate_choice_c_v102_preregistration.py \
  --spec "${SPEC}" \
  --verify-exclusion-manifest \
  --verify-projected-catalogs \
  --require-calibration-authorized >/dev/null

EXPECTED_REPORT_SHA=2f7f947ca016b8fb5ad425d9d9cfb8e304b9e3ffd95391216e6cdaf8117d36ac
OBSERVED_REPORT_SHA=$(sha256sum "${REPORT}" | cut -d' ' -f1)
if [[ "${OBSERVED_REPORT_SHA}" != "${EXPECTED_REPORT_SHA}" ]]; then
  printf 'V102 disjointness report differs: %s\n' "${OBSERVED_REPORT_SHA}" >&2
  exit 2
fi
if [[ -e "${FROZEN}" || -d "${OUTPUT_ROOT}/telemetry" || -d "${OUTPUT_ROOT}/raw" ]]; then
  printf 'refusing to reuse V102 calibration outputs under %s\n' "${OUTPUT_ROOT}" >&2
  exit 2
fi

CALIBRATION_JOB=$(sbatch --parsable slurm/choice_c_v102_calibration_a100.sbatch)
FREEZE_JOB=$(sbatch --parsable \
  --dependency="afterok:${CALIBRATION_JOB}" \
  slurm/choice_c_v102_freeze.sbatch)

printf 'V102 two-worker A100 calibration: %s\n' "${CALIBRATION_JOB}"
printf 'V102 conditional robust freeze: %s\n' "${FREEZE_JOB}"
printf 'No held-out materialization or evaluation was submitted.\n'
