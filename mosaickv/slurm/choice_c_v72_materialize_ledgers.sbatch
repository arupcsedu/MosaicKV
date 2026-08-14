#!/bin/bash
#SBATCH --job-name=mosaickv-v72-ledger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=bii
#SBATCH --array=0-2%3
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/v72-ledger-%A_%a.log
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
LEDGER_DIR=${PACKAGE_ROOT}/configs/ledgers

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
export PYTHONPATH="${PACKAGE_ROOT}/src:${PACKAGE_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
cd "${PACKAGE_ROOT}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    TASK=docvqa
    REVISION=539088ef8a8ada01ac8e2e6d4e372586748a265e
    ;;
  1)
    TASK=mmstar
    REVISION=bc98d668301da7b14f648724866e57302778ab27
    ;;
  2)
    TASK=chartqa
    REVISION=9e63b7df1592a1c2158e735cc1725454aef0d6d9
    ;;
  *)
    printf 'unsupported array index: %s\n' "${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

"${PYTHON}" scripts/materialize_sample_ledger.py \
  --task "${TASK}" \
  --dataset-revision "${REVISION}" \
  --model-name llava-hf/llava-1.5-7b-hf \
  --seed 20270814 \
  --subset-size 8 \
  --exclude-ledger-directory "${LEDGER_DIR}" \
  --allow-dirty-exploratory \
  --output "${LEDGER_DIR}/choice_c_v72_heldout_${TASK}.json"
