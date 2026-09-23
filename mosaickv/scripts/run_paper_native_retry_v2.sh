#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {prefixkv|milebench} METHOD" >&2
  exit 2
fi

lane="$1"
method="$2"
repo="/scratch/djy8hg/workdir/MosaicKV"
package="$repo/mosaickv"
source "$package/scripts/cache_env.sh"
source_sha="${MOSAICKV_SOURCE_SHA:?set MOSAICKV_SOURCE_SHA}"
output_root="${MOSAICKV_OUTPUT_ROOT:?set MOSAICKV_OUTPUT_ROOT}"

cd "$repo"
"$package/scripts/assert_clean_worktree.sh"
observed_sha="$(git rev-parse HEAD)"
if [[ "$observed_sha" != "$source_sha" ]]; then
  echo "source SHA mismatch: expected $source_sha, observed $observed_sha" >&2
  exit 2
fi
case "$output_root" in
  /scratch/djy8hg/*) ;;
  *) echo "MOSAICKV_OUTPUT_ROOT must be under /scratch/djy8hg" >&2; exit 2 ;;
esac

protocol="$package/configs/experiments/published_paper_native_metrics_v1.yaml"
lane_root="$output_root/$lane"
mkdir -p "$lane_root/traces"

if [[ "$lane" == "prefixkv" ]]; then
  ledger="${MOSAICKV_PREFIX_LEDGER:?set MOSAICKV_PREFIX_LEDGER}"
  ledger_sha="${MOSAICKV_PREFIX_LEDGER_SHA:?set MOSAICKV_PREFIX_LEDGER_SHA}"
  protocol_id="prefixkv_hf_reimpl_native_metrics_v1"
  task="llava_description"
  model="llava-hf/llava-1.5-7b-hf"
  model_revision="b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
  dataset_revision="407c2d2f8fc13611d13f72a5e18b8c33bf9c44e6e05278bf0b4da41deeacbe6a"
  max_new_tokens=144
  case "$method" in
    full_kv) ratio="1.0" ;;
    prefixkv_reimpl|mosaickv_exact) ratio="0.2" ;;
    *) echo "unsupported PrefixKV lane method: $method" >&2; exit 2 ;;
  esac
elif [[ "$lane" == "milebench" ]]; then
  ledger="${MOSAICKV_MILEBENCH_LEDGER:?set MOSAICKV_MILEBENCH_LEDGER}"
  ledger_sha="${MOSAICKV_MILEBENCH_LEDGER_SHA:?set MOSAICKV_MILEBENCH_LEDGER_SHA}"
  protocol_id="lookm_hf_reimpl_native_metrics_v1"
  task="milebench"
  model="llava-hf/llava-1.5-7b-hf"
  model_revision="b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
  dataset_revision="7f40a34031af3dc364330184a78b07ddf09b4f94d"
  max_new_tokens=32
  case "$method" in
    full_kv) ratio="1.0" ;;
    lookm_reimpl|mosaickv_exact) ratio="0.2" ;;
    *) echo "unsupported MileBench lane method: $method" >&2; exit 2 ;;
  esac
else
  echo "unsupported lane: $lane" >&2
  exit 2
fi

observed_ledger_sha="$(sha256sum "$ledger" | awk '{print $1}')"
if [[ "$observed_ledger_sha" != "$ledger_sha" ]]; then
  echo "ledger SHA mismatch: expected $ledger_sha, observed $observed_ledger_sha" >&2
  exit 2
fi

command=(
  /scratch/djy8hg/env/mosaickv/bin/mosaickv evaluate
  --task "$task"
  --model "$model"
  --model-revision "$model_revision"
  --dataset-revision "$dataset_revision"
  --backend hf
  --attention-backend eager
  --precision bf16
  --method "$method"
  --retention-ratio "$ratio"
  --budget-unit blocks
  --block-size 1
  --forecast prompt_window
  --prompt-window 16
  --draft-tokens 0
  --repair-policy none
  --max-new-tokens "$max_new_tokens"
  --seed 0
  --local-files-only
  --output-dir "$lane_root"
  --trace-directory "$lane_root/traces"
  --run-id "paper-native-retry-v2-${lane}-${method}"
  --raw-output "$lane_root/${method}-raw.jsonl"
  --parquet-output "$lane_root/${method}-results.parquet"
  --metric-output "$lane_root/${method}-metrics.jsonl"
  --metric-observations-parquet-output "$lane_root/${method}-observations.parquet"
  --metric-aggregates-parquet-output "$lane_root/${method}-aggregates.parquet"
  --published-protocol "$protocol"
  --published-protocol-id "$protocol_id"
  --paper-native-ledger "$ledger"
  --paper-native-ledger-sha "$ledger_sha"
  --manifest "$lane_root/${method}-manifest.json"
  --json
)

if [[ "$lane" == "prefixkv" ]]; then
  command+=(--prefixkv-profile-mode fixed_global)
  reference="$lane_root/fullkv-reference.json"
  if [[ "$method" == "full_kv" ]]; then
    command+=(--fullkv-reference-output "$reference")
  else
    if [[ ! -f "$reference" ]]; then
      echo "compressed PrefixKV lane requires completed FullKV reference: $reference" >&2
      exit 2
    fi
    reference_sha="$(sha256sum "$reference" | awk '{print $1}')"
    command+=(--fullkv-reference-input "$reference" --fullkv-reference-sha "$reference_sha")
  fi
fi

"${command[@]}"
