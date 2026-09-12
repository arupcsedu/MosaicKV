#!/bin/bash
#SBATCH --job-name=mk-base-fidelity
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:a100:1
#SBATCH --export=ALL
#SBATCH --output=/scratch/djy8hg/slurm-logs/%x-%A_%a.out
#SBATCH --error=/scratch/djy8hg/slurm-logs/%x-%A_%a.err

set -euo pipefail

REPOSITORY_ROOT=${MOSAICKV_SOURCE_ROOT:?MOSAICKV_SOURCE_ROOT is required}
PROJECT_ROOT=${REPOSITORY_ROOT}/mosaickv
PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
INDEX=${MOSAICKV_JOB_INDEX:?MOSAICKV_JOB_INDEX is required}
source "${PROJECT_ROOT}/scripts/cache_env.sh"
"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

readarray -t JOB < <(
  "${PYTHON}" - "${INDEX}" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json, pathlib, sys
rows = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
row = json.loads(rows[int(sys.argv[2])])
print(row["job_spec_path"])
print(row["job_spec_sha256"])
PY
)

cd "${PROJECT_ROOT}"
srun "${PYTHON}" scripts/run_baseline_fidelity_review_worker.py \
  --job-spec "${JOB[0]}" \
  --job-spec-sha256 "${JOB[1]}"
