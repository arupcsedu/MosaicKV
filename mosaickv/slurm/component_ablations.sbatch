#!/bin/bash
#SBATCH --job-name=mkv-exp3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=120G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:30:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --array=0-3%4
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/component-ablation-%A_%a.out
#SBATCH --error=/scratch/djy8hg/runs/mosaickv/slurm/component-ablation-%A_%a.err

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
INDEX=${MOSAICKV_ABLATION_INDEX:?set MOSAICKV_ABLATION_INDEX to immutable jobs.jsonl}

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
if [[ -n "$(git -C "${PROJECT_ROOT}" status --porcelain=v1)" ]]; then
  if [[ "${MOSAICKV_ALLOW_EXPLORATORY_DIRTY:-0}" != 1 ]]; then
    printf 'Dirty-tree Experiment 3 requires MOSAICKV_ALLOW_EXPLORATORY_DIRTY=1.\n' >&2
    exit 2
  fi
  printf 'classification=exploratory_dirty_tree_component_ablation\n'
fi

export PYTHONPATH="${PACKAGE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

nvidia-smi --query-gpu=name,uuid,pstate,clocks.current.sm,clocks.current.memory,power.draw \
  --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits || true

cd "${PROJECT_ROOT}"
srun "${PYTHON}" "${PACKAGE_ROOT}/scripts/run_component_ablation_worker.py" \
  --index "${INDEX}" \
  --array-index "${SLURM_ARRAY_TASK_ID}"
