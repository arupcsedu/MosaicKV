#!/bin/bash
#SBATCH --job-name=mosaickv-v44-ledger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=bii
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/v44-ledger-%j.log
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV
PACKAGE_ROOT=${PROJECT_ROOT}/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
OUTPUT=${MOSAICKV_V44_LEDGER_OUTPUT:?set MOSAICKV_V44_LEDGER_OUTPUT}

source "${PACKAGE_ROOT}/scripts/cache_env.sh"
export PYTHONPATH="${PACKAGE_ROOT}/src:${PACKAGE_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
cd "${PACKAGE_ROOT}"

"${PYTHON}" scripts/materialize_sample_ledger.py \
  --task docvqa \
  --dataset-revision 539088ef8a8ada01ac8e2e6d4e372586748a265e \
  --model-name llava-hf/llava-1.5-7b-hf \
  --seed 20270812 \
  --subset-size 8 \
  --exclude-ledger-directory "${PACKAGE_ROOT}/configs/ledgers" \
  --allow-dirty-exploratory \
  --output "${OUTPUT}"
