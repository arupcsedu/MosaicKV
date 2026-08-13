#!/bin/bash
#SBATCH --job-name=mosaickv-readiness-verify
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/verification/readiness-%j.log
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
RUFF=/scratch/djy8hg/env/mosaickv/bin/ruff
MYPY=/scratch/djy8hg/env/mosaickv/bin/mypy

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
export PYTHONPATH="${PACKAGE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MOSAICKV_ALLOW_DIRTY_EXPLORATORY=1

cd "${PACKAGE_ROOT}"

printf 'classification=exploratory_dirty_tree_not_paper_evidence\n'
printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID}"
printf 'slurm_node=%s\n' "${SLURMD_NODENAME}"
printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-not_set}"
"${PACKAGE_ROOT}/scripts/assert_clean_worktree.sh"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader

srun "${PYTHON}" scripts/verify_envs.py \
  --environment common \
  --lock env/common/requirements.lock \
  --require-cuda
srun "${RUFF}" check src tests scripts
srun "${MYPY}" --config-file pyproject.toml
srun "${PYTHON}" -m pytest -q -p no:cacheprovider

printf 'verification_status=passed\n'
