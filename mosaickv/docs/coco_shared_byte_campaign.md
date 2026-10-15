# COCO/CIDEr shared-physical-byte campaign

Status: implementation and CPU verification, **not a completed GPU experiment**.
The clean-source/commit gate is still required. No historical score is relabeled
as a shared-byte result. No FullKV latency comparison is made.

## Frozen comparison

Use `shared_byte_comparison_v2.yaml` and the unchanged MosaicKV exact-selection
policy (prompt window 16, routed graph 1/256, no prototypes/residual/repair).
The new instrumentation and runner require a new source/content seal; this is
not selection-policy tuning.

| Control | Value |
| --- | --- |
| Model | `llava-hf/llava-v1.6-mistral-7b-hf` |
| Model revision | `2424fdd47412fccc66d91719126b420e9fbd7065` |
| Dataset | `lmms-lab/COCO-Caption`, **2014 validation**, not COCO 2017 |
| Dataset revision | `c3213b54e173ae86517d481de17f31439d6f8c72` |
| Backend | HF eager, FP16, deterministic greedy, 64 timed tokens |
| Prompt | Installed lmms-eval COCO one-sentence-caption prompt |
| Inputs | One RGB image; all five annotation captions; hashes frozen before inference |
| Blocks / mandatory | One token; protect last prompt position for every method |
| Methods | FullKV reference, Prompt Top-K, `vl_cache_reimpl`, `mosaickv_exact` |
| Statistics | CIDEr ×100 score points, 10,000 paired bootstrap draws, seed 20271014 |
| Resources | At most **four A100s total**; one A100, 12 CPUs, two hours per worker |

CIDEr uses the pinned `lmms_eval.tasks.coco_cap.utils.PTBTokenizer` and `Cider`
(`lmms-eval==0.7.2`, `pycocoevalcap==1.2`). It is a corpus metric, not per-example
accuracy. Tokenization is cached, but reference document frequencies and CIDEr
scores are recomputed over the complete cohort and every resampled corpus.
Repeated bootstrap identities are assigned distinct evaluator-local IDs.
**Never average shard CIDEr scores.** Singleton metric artifacts produced during
resumable execution are diagnostics, not the reported score.

## Reuse audit

Read-only audit found 64 completed COCO FullKV raw shards and 4,096 unique IDs at
`/scratch/djy8hg/runs/mosaickv/paper-native-7day-v1-61e5a68/vl_cache/coco_caption/full_kv`.
Their manifests record **BF16**, not the new FP16 control. Reuse the identities
only, not those predictions or timings. These previously evaluated identities
are disclosed as a historical benchmark cohort, never untouched held-out data.
Original nominal-retention results remain unchanged.

Within the new campaign, completed samples are reused only with identical
source/config hashes and intact raw output, trace, manifest and metric hashes.
Timeouts/failures leave immutable attempts; retries execute missing/failed
samples only. FullKV is run once per cohort, not once per byte cap. The final
cohort explicitly reuses compatible 32/256-example pilot completions.

## Stages and admission gates

1. Materialize the historical ID pool using only the fourteen original cached
   validation Arrow shards. Freeze lossless PNGs, five references, processor
   signature, tokenizer vocabulary/chat templates and source asset hashes.
   There are no downloads or unrequested split preparations.
2. **32 distinct examples, 40 MiB**: eight method-specific shards, sixteen
   examples each. Each worker runs ordinary-generation and retention-one
   extraction/reinjection parity (16 tokens, full token agreement, max absolute
   logit difference ≤1e-4), then five recorded warmups and one quality trial.
3. Analyzer requires complete captions/references/timing/trace fields, exact
   pairing, no fallback/repair/eviction, and matching initial/final/peak unique
   backing storage. An independent FullKV-shape certificate proves the largest
   feasible initial cache below each cap. Padding and preallocation count.
4. Only a successful immutable validation report can promote **256 different
   examples at 10/20/40/80 MiB**: 208 method-specific shard entries, including
   only sixteen FullKV shards. Failed analyses do not occupy the success-gate
   path; their attempt reports remain preserved.
5. Use the widest measured paired CIDEr interval and worker wall time to
   register final sample size/runtime before the final run. Default planning
   precision is ±2 CIDEr points; the square-root pilot extrapolation is a
   heuristic, **not guaranteed statistical power**. The target is the existing
   4,096-example pool if available. If its projected precision is insufficient,
   report that limitation rather than weakening a confidence interval.
6. Final preparation uses the same frozen policy, caps and cohort, with explicit
   pilot reuse. Final CIDEr and its bootstrap are recomputed over the entire
   final cohort, not combined from intermediate scores.

Passing execution/byte gates is not a superiority claim. Report negative quality
or latency differences. A 30%-smaller-at-equal-quality claim requires the curves
to support it; it is not inferred from one unmatched operating point.

The byte-only route uses the floating-point predecessor of 1 as its retention
sentinel, avoiding the retention-one identity fast path without imposing the
old additional 1% pruning. For supported caches, the byte cap alone determines
capacity. If the full input fits below a cap, retain all feasible positions and
report the achieved bytes/shortfall, not the nominal target as measured storage.

## Commands after scoped commit and clean worktree creation

Use an isolated clean worktree so unrelated `paper/`, vLLM, scheduler and user
edits remain untouched. Replace `SOURCE`, `SHA`, `POOL_ROOT` and `RUN_ROOT` with
the clean worktree, its SHA and **new external scratch paths**.

```bash
cd "$SOURCE"
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SOURCE/mosaickv/src"
PY=/scratch/djy8hg/env/mosaickv/bin/python
bash mosaickv/scripts/assert_clean_worktree.sh

$PY mosaickv/scripts/materialize_coco_shared_bytes.py --source-sha "$SHA" \
  --historical-fullkv-root /scratch/djy8hg/runs/mosaickv/paper-native-7day-v1-61e5a68/vl_cache/coco_caption/full_kv \
  --arrow-cache /scratch/djy8hg/cache/mosaickv/datasets/lmms-lab___coco-caption/default/0.0.0/c3213b54e173ae86517d481de17f31439d6f8c72 \
  --model-snapshot /scratch/djy8hg/cache/mosaickv/huggingface/hub/models--llava-hf--llava-v1.6-mistral-7b-hf/snapshots/2424fdd47412fccc66d91719126b420e9fbd7065 \
  --root "$POOL_ROOT" --prior-use 'Previously evaluated 4096-example nominal-ratio cohort; IDs only, no BF16 prediction reuse'
$PY mosaickv/scripts/prepare_native_metric_shared_bytes.py --source-sha "$SHA" \
  --task coco_caption --ledger "$POOL_ROOT/ledger.json" --root "$RUN_ROOT"
```

Preparation prints `PLAN_SHA`, the file hash required by every worker/analyzer.
Submission is dry-run unless `--submit` is present. Actual submission requires
the common controller's policy and creates a **held** array. Register the emitted
policy with the existing shared controller under its lock before admitting any
worker; do not release all pending tasks manually. A per-array `%4` alone is not
a global four-GPU guarantee. Controller code must enforce four at every time.

```bash
$PY mosaickv/scripts/manage_coco_shared_bytes.py --root "$RUN_ROOT" \
  --plan-sha "$PLAN_SHA" submit --stage validation
# Add --submit --controller-policy <shared-policy.json> only after review.

$PY mosaickv/scripts/analyze_coco_shared_bytes.py --root "$RUN_ROOT" \
  --plan-sha "$PLAN_SHA" --stage validation
$PY mosaickv/scripts/manage_coco_shared_bytes.py --root "$RUN_ROOT" \
  --plan-sha "$PLAN_SHA" promote
$PY mosaickv/scripts/manage_coco_shared_bytes.py --root "$RUN_ROOT" \
  --plan-sha "$PLAN_SHA" submit --stage larger
# After the larger array completes:
$PY mosaickv/scripts/analyze_coco_shared_bytes.py --root "$RUN_ROOT" \
  --plan-sha "$PLAN_SHA" --stage larger
$PY mosaickv/scripts/manage_coco_shared_bytes.py --root "$RUN_ROOT" \
  --plan-sha "$PLAN_SHA" register-final --cider-ci-half-width 2
# Review the measured runtime/precision registration before enabling final work.
$PY mosaickv/scripts/prepare_coco_final_curve.py --parent "$RUN_ROOT" \
  --parent-sha "$PLAN_SHA" --pool "$POOL_ROOT/ledger.json" --root "$FINAL_ROOT"
# --authorize-final enables only this frozen final plan; preparation never submits.
```

The final plan uses the same submit/analyze commands with its own hash and
`--stage final`. Outputs: immutable raw JSONL/manifests/traces for each sample;
pooled `corpus_metrics.parquet`, `paired_samples.parquet`,
`paired_comparisons.parquet`, capacity certificates and stage `report.json`.
The analyzer also writes `cider_memory.svg` and
`compressed_latency_memory.svg` from measured, validated points, without adding
plotting packages to the locked common environment. FullKV appears
only as a horizontal quality-reference line, never in the latency plot.
CIDEr bootstrap is CPU work and can take substantially longer than a simple
mean reduction; GPU runtime estimates do not include CPU queue/analysis time.

Actual large-run completion, measured variance/runtime, numerical parity and
quality/latency findings remain pending. Do not describe this setup document or
its CPU synthetic tests as paper-ready empirical evidence.
