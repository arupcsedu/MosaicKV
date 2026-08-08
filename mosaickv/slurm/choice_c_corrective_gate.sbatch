#!/bin/bash
#SBATCH --job-name=mosaickv-choice-c-v5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=12
# Three isolated LLaVA-1.5 processes remain comfortably below this ceiling.
# Avoid --mem=0: requesting the whole 2-TB DGX prevents safe backfill whenever
# even a small unrelated allocation is present.
#SBATCH --mem=450G
#SBATCH --time=04:00:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:3
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/choice-c-v5-%j.log

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
MOSAICKV_PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python}
SPEC=${MOSAICKV_CHOICE_C_SPEC:-configs/experiments/choice_c_corrective_gate.yaml}
export MOSAICKV_CACHE_ROOT=${MOSAICKV_CACHE_ROOT:-/scratch/djy8hg/cache/mosaickv}

# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_ROOT}"

# These three candidates are independent and use one isolated model replica
# per GPU. No evidence crosses candidates until every task completes.
pids=()
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=12 --mem=140G \
  --gres=gpu:1 \
  "${MOSAICKV_PYTHON}" scripts/run_choice_c_gate.py \
    --spec "${SPEC}" \
    --phase calibration-candidate \
    --variant prompt_anchored_15 &
pids+=("$!")
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=12 --mem=140G \
  --gres=gpu:1 \
  "${MOSAICKV_PYTHON}" scripts/run_choice_c_gate.py \
    --spec "${SPEC}" \
    --phase calibration-candidate \
    --variant evidence_anchored_15 &
pids+=("$!")
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=12 --mem=140G \
  --gres=gpu:1 \
  "${MOSAICKV_PYTHON}" scripts/run_choice_c_gate.py \
    --spec "${SPEC}" \
    --phase calibration-candidate \
    --variant balanced_23 &
pids+=("$!")

candidate_status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    candidate_status=1
  fi
done
if [[ "${candidate_status}" -ne 0 ]]; then
  printf 'One or more independent calibration candidates failed.\n' >&2
  exit 1
fi

# The coordinator resumes the immutable candidate outputs, freezes one
# profile, and alone executes adaptive routing, repair calibration, internal
# validation, and the untouched held-out comparison.
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=12 --mem=140G \
  --gres=gpu:1 \
  "${MOSAICKV_PYTHON}" scripts/run_choice_c_gate.py \
    --spec "${SPEC}" \
    --phase all
