#!/bin/bash
#SBATCH --job-name=mosaickv-prefixkv-fix-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/scratch/djy8hg/runs/mosaickv/slurm/prefixkv-fix-smoke-v1-%j.log

set -euo pipefail

PROJECT_ROOT=/scratch/djy8hg/workdir/MosaicKV/mosaickv
PYTHON=/scratch/djy8hg/env/mosaickv/bin/python
CLI=/scratch/djy8hg/env/mosaickv/bin/mosaickv
CONFIG=/scratch/djy8hg/runs/mosaickv/campaigns/llava-full-quality-continuation-6gpu-v1/configs/070-d2748988ee3b166ab194f0ce6dd3c0f3541be97fec27276aaf5229215c1191f8.yaml
SAMPLE_LEDGER=configs/experiments/prefixkv_fix_smoke_sample_v1.json
OUTPUT_ROOT=/scratch/djy8hg/runs/mosaickv/prefixkv-fix-smoke-v1
RUN_ID=prefixkv-fix-smoke-v1

# shellcheck source=../scripts/cache_env.sh
source "${PROJECT_ROOT}/scripts/cache_env.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "${PROJECT_ROOT}"

if [[ -e ${OUTPUT_ROOT} ]]; then
  echo "refusing to overwrite immutable smoke output: ${OUTPUT_ROOT}" >&2
  exit 2
fi
mkdir -p /scratch/djy8hg/runs/mosaickv/slurm
printf '%s  %s\n' \
  c6098a2e7223a884979b4e516c0007a008daed67c85038e7d2cabe0415938663 \
  src/mosaickv/backends/hf_runtime.py \
  9559feb1bdd2baa187f25695c3aa8e8b64ded0822cf07d7102ed8e6166a82c93 \
  src/mosaickv/config.py \
  1bd500a40a8050054d78820fe10125f5d968d1dcebefc806f2337e386757345d \
  "${SAMPLE_LEDGER}" | sha256sum --check --strict
"${PYTHON}" -c \
  'import sys; from mosaickv.config import config_sha256,load_config; sys.exit(config_sha256(load_config(sys.argv[1])) != sys.argv[2])' \
  "${CONFIG}" d2748988ee3b166ab194f0ce6dd3c0f3541be97fec27276aaf5229215c1191f8
nvidia-smi --query-gpu=index,name,uuid,pstate,memory.total,memory.free \
  --format=csv,noheader,nounits

"${CLI}" evaluate \
  --config "${CONFIG}" \
  --task chartqa \
  --run-id "${RUN_ID}" \
  --output-dir "${OUTPUT_ROOT}" \
  --sample-id-file "${SAMPLE_LEDGER}" \
  --sample-id-file-sha 1bd500a40a8050054d78820fe10125f5d968d1dcebefc806f2337e386757345d \
  --local-files-only \
  --json

"${PYTHON}" - "${OUTPUT_ROOT}/${RUN_ID}/results.jsonl" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
if len(rows) != 1:
    raise SystemExit(f"expected one smoke row, found {len(rows)}")
row = rows[0]
if row.get("status") != "completed" or row.get("error") is not None:
    raise SystemExit(f"PrefixKV smoke failed: {row.get('error')}")
if not str(row.get("method", "")).startswith("prefixkv_reimpl"):
    raise SystemExit(f"unexpected method label: {row.get('method')}")
if float(row.get("retention_ratio")) != 0.1:
    raise SystemExit("smoke retention ratio changed")
for field in ("active_kv_bytes", "compression_time", "decode_time", "end_to_end_time"):
    value = row.get(field)
    if value is None or not math.isfinite(float(value)):
        raise SystemExit(f"nonfinite {field}: {value}")
if int(row.get("generated_tokens")) != 64:
    raise SystemExit("smoke output length changed")
print("prefixkv_fix_smoke=passed")
PY
