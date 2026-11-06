#!/usr/bin/env bash
set -euo pipefail
source_root=$(git rev-parse --show-toplevel)
cd "$source_root"
source mosaickv/scripts/cache_env.sh
bash mosaickv/scripts/assert_clean_worktree.sh
test "$#" -eq 1
run_root="$1"
case "$run_root" in
  /scratch/djy8hg/runs/evicastkv/*) ;;
  *) echo 'Use a new absolute EviCastKV scratch output root.' >&2; exit 2 ;;
esac
test ! -e "$run_root"
launch_root="${run_root}-launch"
test ! -e "$launch_root"
mkdir -p "$launch_root"
revision=$(git rev-parse HEAD)
wrapper="$source_root/mosaickv/slurm/evicastkv_uniform_core_operator.sbatch"
common=(--parsable --account=bii_dsc_community --nodes=1 --ntasks=1)
cpu=$(sbatch "${common[@]}" --partition=bii --nodelist=udc-aj36-2c0 --cpus-per-task=2 --mem=16G \
  --time=00:25:00 --job-name=mk-uniform-core-cpu \
  --output="$launch_root/cpu-%j.out" --error="$launch_root/cpu-%j.err" \
  "$wrapper" "$source_root" "$run_root" "$revision" cpu)
gpu=$(sbatch "${common[@]}" --partition=bii-gpu --reservation=bi_fox_dgx \
  --gres=gpu:a100:1 --cpus-per-task=12 --mem=48G --time=00:10:00 \
  --dependency="afterany:$cpu" --job-name=mk-uniform-core-operator \
  --output="$launch_root/gpu-%j.out" --error="$launch_root/gpu-%j.err" \
  "$wrapper" "$source_root" "$run_root" "$revision" run)
analysis=$(sbatch "${common[@]}" --partition=bii --cpus-per-task=2 --mem=16G \
  --time=00:05:00 --dependency="afterany:$gpu" --job-name=mk-uniform-core-analysis \
  --output="$launch_root/analysis-%j.out" --error="$launch_root/analysis-%j.err" \
  "$wrapper" "$source_root" "$run_root" "$revision" analyze)
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$source_root/mosaickv/src"
/scratch/djy8hg/env/mosaickv/bin/python - "$launch_root" "$revision" "$cpu" "$gpu" "$analysis" <<'PY'
import sys
from pathlib import Path
from mosaickv.evaluation.fox_campaign import write_once
root, revision, cpu, gpu, analysis = sys.argv[1:]
write_once(Path(root) / 'submission.json', dict(
    source_sha=revision, cpu=cpu, gpu=gpu, analysis=analysis,
    maximum_GPU_workers=1, synthetic_operator_only=True,
    no_automatic_model_or_quality_submission=True,
))
print(dict(cpu=cpu, gpu=gpu, analysis=analysis))
PY
