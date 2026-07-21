#!/bin/bash
#SBATCH --job-name=mosaickv-array
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=37
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm-%A_%a.submit.log

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
MOSAICKV_PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}

export MOSAICKV_CACHE_ROOT
# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"

: "${MOSAICKV_MATRIX_INDEX:?set MOSAICKV_MATRIX_INDEX to an immutable jobs.jsonl}"
: "${SLURM_ARRAY_TASK_ID:?submit this script with --array=0-N}"

if [[ ! -x "${MOSAICKV_PYTHON}" ]]; then
  printf 'Common MosaicKV environment is unavailable: %s\n' "${MOSAICKV_PYTHON}" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${MOSAICKV_PYTHON}" "${PROJECT_ROOT}/scripts/validate_matrix.py" \
  --expanded-index "${MOSAICKV_MATRIX_INDEX}" \
  --array-index "${SLURM_ARRAY_TASK_ID}" \
  --json >/dev/null

mapfile -d '' -t JOB_FIELDS < <(
  "${MOSAICKV_PYTHON}" - "${MOSAICKV_MATRIX_INDEX}" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json
import sys
from pathlib import Path

index_path = Path(sys.argv[1])
array_index = int(sys.argv[2])
rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line]
if not 0 <= array_index < len(rows):
    raise SystemExit(f"array index {array_index} is outside [0, {len(rows) - 1}]")
row = rows[array_index]
for key in (
    "config_path",
    "run_id",
    "output_root",
    "trace_directory",
    "task",
    "subset_size",
    "measurement_mode",
    "warmups",
    "timed_repetitions",
):
    value = row.get(key)
    sys.stdout.write(("" if value is None else str(value)) + "\0")
PY
)

CONFIG_PATH=${JOB_FIELDS[0]}
RUN_ID=${JOB_FIELDS[1]}
OUTPUT_ROOT=${JOB_FIELDS[2]}
TRACE_DIRECTORY=${JOB_FIELDS[3]}
TASK=${JOB_FIELDS[4]}
SUBSET_SIZE=${JOB_FIELDS[5]}
MEASUREMENT_MODE=${JOB_FIELDS[6]}
WARMUPS=${JOB_FIELDS[7]}
TIMED_REPETITIONS=${JOB_FIELDS[8]}

if [[ "${MEASUREMENT_MODE}" == microbenchmark ]]; then
  if (( WARMUPS < 5 || TIMED_REPETITIONS < 20 )); then
    printf 'Microbenchmark controls violate the 5/20 minimum.\n' >&2
    exit 2
  fi
  printf 'Compressed-method microbenchmark execution is not yet exposed by the unified CLI.\n' >&2
  printf 'The validator must keep this matrix disabled until that runner exists.\n' >&2
  exit 2
fi

RUN_DIRECTORY="${OUTPUT_ROOT}/${RUN_ID}"
ATTEMPT_DIRECTORY="${RUN_DIRECTORY}/attempts/${SLURM_JOB_ID}"
mkdir -p "${ATTEMPT_DIRECTORY}" "${TRACE_DIRECTORY}"

ARGS=(
  evaluate
  --config "${CONFIG_PATH}"
  --task "${TASK}"
  --run-id "${RUN_ID}"
  --output-dir "${OUTPUT_ROOT}"
  --trace-directory "${TRACE_DIRECTORY}"
  --local-files-only
  --json
)
if [[ -n "${SUBSET_SIZE}" ]]; then
  ARGS+=(--subset-size "${SUBSET_SIZE}")
fi

cd "${PROJECT_ROOT}"
srun "${MOSAICKV_PYTHON}" -m mosaickv.cli "${ARGS[@]}" \
  >"${ATTEMPT_DIRECTORY}/stdout.jsonl" \
  2>"${ATTEMPT_DIRECTORY}/stderr.log"
