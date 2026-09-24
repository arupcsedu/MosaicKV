#!/bin/bash
set -euo pipefail

repo=/scratch/djy8hg/workdir/MosaicKV
package="$repo/mosaickv"
source "$package/scripts/cache_env.sh"
cd "$repo"
"$package/scripts/assert_clean_worktree.sh"

source_sha="$(git rev-parse HEAD)"
output_root="${MOSAICKV_OUTPUT_ROOT:?set MOSAICKV_OUTPUT_ROOT}"
prefix_ledger="${MOSAICKV_PREFIX_LEDGER:?set MOSAICKV_PREFIX_LEDGER}"
prefix_sha="${MOSAICKV_PREFIX_LEDGER_SHA:?set MOSAICKV_PREFIX_LEDGER_SHA}"
case "$output_root" in
  /scratch/djy8hg/*) ;;
  *) echo "MOSAICKV_OUTPUT_ROOT must be under /scratch/djy8hg" >&2; exit 2 ;;
esac
[[ "$(sha256sum "$prefix_ledger" | awk '{print $1}')" == "$prefix_sha" ]]
if [[ -d "$output_root" ]] && [[ -n "$(find "$output_root" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "output root must be absent or empty: $output_root" >&2
  exit 2
fi

log_root="${output_root}.slurm-logs"
mkdir -p "$log_root"
exports="ALL,MOSAICKV_SOURCE_SHA=$source_sha,MOSAICKV_OUTPUT_ROOT=$output_root,MOSAICKV_PREFIX_LEDGER=$prefix_ledger,MOSAICKV_PREFIX_LEDGER_SHA=$prefix_sha"
gpu_job="$(sbatch --parsable \
  --export="$exports" \
  --output="$log_root/worker-%j.out" \
  --error="$log_root/worker-%j.err" \
  "$package/slurm/paper_native_latency_v3.sbatch")"
analysis_job="$(sbatch --parsable \
  --job-name=mk-native-lat-v3-analysis \
  --partition=bii --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:10:00 \
  --dependency="afterok:$gpu_job" \
  --output="$log_root/analysis-%j.out" \
  --error="$log_root/analysis-%j.err" \
  --wrap="set -euo pipefail; source $package/scripts/cache_env.sh; cd $repo; observed=\$(git rev-parse HEAD); [[ \"\$observed\" == $source_sha ]]; /scratch/djy8hg/env/mosaickv/bin/python $package/scripts/analyze_paper_native_latency_v3.py --protocol $package/configs/experiments/paper_native_latency_v3.yaml --output-root $output_root --output $output_root/analysis/report.json")"
printf '{"analysis_job":"%s","gpu_job":"%s","source_git_sha":"%s"}\n' \
  "$analysis_job" "$gpu_job" "$source_sha"
