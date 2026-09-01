#!/bin/bash

set -euo pipefail

REPOSITORY=/scratch/djy8hg/workdir/MosaicKV
PROJECT_ROOT=${REPOSITORY}/mosaickv
PYTHON=${MOSAICKV_PYTHON:-/scratch/djy8hg/env/mosaickv/bin/python3}
REUSED_COMPOSITE=${MOSAICKV_REUSED_COMPOSITE:-/scratch/djy8hg/runs/mosaickv/paper-hf-vllm-llava15-v3-corrective-d414992703/analysis_after_complete_19043530_19049448/composite_9000/paired_rows_9000.parquet}

"${PROJECT_ROOT}/scripts/assert_clean_worktree.sh"
SOURCE_SHA=$(git -C "${REPOSITORY}" rev-parse HEAD)
CAMPAIGN_ROOT=${MOSAICKV_CAMPAIGN_ROOT:-/scratch/djy8hg/runs/mosaickv/vllm-published-baseline-full-9000-${SOURCE_SHA:0:10}}
mkdir -p "${CAMPAIGN_ROOT}"

source "${PROJECT_ROOT}/scripts/cache_env.sh"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON}" "${PROJECT_ROOT}/scripts/prepare_vllm_published_baseline_full.py" \
  --source "${REUSED_COMPOSITE}" --output-root "${CAMPAIGN_ROOT}" \
  --source-git-sha "${SOURCE_SHA}" --shard-size 64

JOB_COUNT=$("${PYTHON}" - "${CAMPAIGN_ROOT}/inputs/job_index.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
if x.get('job_count') != 142 or len(x.get('jobs', [])) != 142:
    raise SystemExit('expected exactly 142 jobs')
print(x['job_count'])
PY
)
ARRAY_JOB=$(sbatch --parsable --array="0-$((JOB_COUNT - 1))%8" \
  --export="ALL,MOSAICKV_CAMPAIGN_ROOT=${CAMPAIGN_ROOT}" \
  "${PROJECT_ROOT}/slurm/vllm_published_baseline_full.sbatch")
ANALYSIS_JOB=$(sbatch --parsable --dependency="afterok:${ARRAY_JOB}" \
  --export="ALL,MOSAICKV_CAMPAIGN_ROOT=${CAMPAIGN_ROOT},MOSAICKV_REUSED_COMPOSITE=${REUSED_COMPOSITE}" \
  "${PROJECT_ROOT}/slurm/vllm_published_baseline_full_analysis.sbatch")

"${PYTHON}" - "${CAMPAIGN_ROOT}/submission.json" "${ARRAY_JOB}" "${ANALYSIS_JOB}" \
  "${SOURCE_SHA}" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1])
payload={
    'schema_version':1,
    'source_git_sha':sys.argv[4],
    'array_job_id':sys.argv[2],
    'analysis_job_id':sys.argv[3],
    'maximum_concurrent_a100s':8,
    'cpus_per_job':12,
    'per_shard_time_limit':'02:00:00',
}
encoded=json.dumps(payload,indent=2,sort_keys=True)+'\n'
if path.exists() and path.read_text()!=encoded:
    raise RuntimeError(f'refusing to replace submission record: {path}')
if not path.exists():
    path.write_text(encoded)
print(encoded,end='')
PY
