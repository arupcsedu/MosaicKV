# Focused HF completion: preserve old results, measure shared bytes

This is an execution plan and implementation status, not a new measured-results
report. Do not rerun the original nominal-retention campaign. Its raw predictions,
manifests, and aggregate tables remain unchanged.

## Evidence retained

- The original 7-day campaign contains 1,691 completed shards and 82,254
  evaluation rows: LLaVA-Description (1,000 examples), MM-Vet (218), COCO Caption
  (4,096), and DocVQA (4,096), across their registered method/ratio points.
  Source: `/scratch/djy8hg/runs/mosaickv/paper-native-7day-v1-61e5a68/aggregate/campaign-report.json`.
  These are separate nominal-ratio curves, not equal-byte comparisons.
- The official PrefixKV 1,000-example profile reproduction achieved corpus PPL
  3.401169 versus the registered 3.41 target (absolute tolerance 0.25).
  Its FullKV-reference ROUGE-L was 0.828348. The latter has no registered
  published numeric ROUGE-L target and is not an additional reproduction pass.
  Source: `/scratch/djy8hg/runs/mosaickv/prefixkv/official-profile-r050-2c7a266/official_profile_reproduction_gate.json`.
- The recovered shared-byte VL-Cache OCR campaign has 256 examples per task,
  TextVQA and DocVQA, at 10/20/40/80 MiB; its 128 shards contain 8,192 completed
  rows. This does not supply the missing shared-byte COCO/CIDEr evidence.
  Source: `/scratch/djy8hg/runs/mosaickv/hf-vlcache-recovery-20261014-v1`.
- The old LOOK-M lane covered only 20 unique ActionSequence examples. Repeating
  those examples did not create a 40-example cohort or broad MileBench evidence.

Do not reuse old FullKV rows in a new comparison unless the exact model,
precision, prompts/media/processor, IDs, and generation controls match. In
particular, the original BF16 runs are not references for the new FP16 lanes.

## Priority 1: corrected PrefixKV validation

Follow [hf_completion_protocol.md](hf_completion_protocol.md): 64 examples each
from TextVQA and DocVQA, four paired workers, frozen control and C-only profiles,
40 MiB initial physical storage, common append-only decode, and all existing
quality gates. C-only is `prefixkv_question_scored_selection_only_reimpl`, not
official PrefixKV. Its alpha remains 0.5; new evaluation labels cannot tune it.

The byte-calibrated control previously rejected append-only mode while the
C-only profile accepted it. The correction permits the same decode policy in
both arms without changing scoring, profile allocation, or the upstream path.
The regression checks both arms through 64 decode steps, initial/final storage,
no eviction, finite logits, logical positions, and unchanged source cache.
CPU tests are not evidence of real-checkpoint quality recovery.

## Priority 2: COCO/CIDEr at shared physical bytes

Use `llava-hf/llava-v1.6-mistral-7b-hf` at revision
`2424fdd47412fccc66d91719126b420e9fbd7065`, with `lmms-lab/COCO-Caption`, `val`,
revision `c3213b54e173ae86517d481de17f31439d6f8c72`.

Compare Prompt Top-K, `vl_cache_reimpl`, and `mosaickv_exact`. FullKV supplies
quality context only, never the latency comparison. First use 32 ID-selected
examples at 40 MiB, split into two 16-example shards per method. Prepare a
separate, disjoint 256-example cohort at 10/20/40/80 MiB, but leave its execution
disabled until the small validation is reviewed.

CIDEr is recomputed from all predictions/references in the selected cohort with
the pinned lmms-eval/COCO evaluator. Never average shard CIDEr or require a fake
per-sample `task_score`. Each paired bootstrap replicate reruns the corpus
reducer. Keep raw captions and reference sets, evaluator identity, and hashes.

## Priority 3: representative MileBench at shared physical bytes

Use LLaVA-1.5-7B at revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`.
Compare Prompt Top-K, `lookm_reimpl`, and `mosaickv_exact`, with FullKV as a
quality reference. Keep official LOOK-M disabled; its upstream execution blocker
has not been resolved by this experiment.

Preregister four subtasks: ActionSequence (temporal), CLEVR-Change and
Spot-the-Diff (semantic), and TextNeedleInAHaystack (diagnostic). Start with eight
examples per subtask, then prepare 64 additional examples per subtask for the
larger stage. Select by a fixed hash of seed, subtask, and ID, not score. Balance
each shard across subtasks and exclude the original pilot IDs explicitly.

Report Accuracy and ROUGE-L separately with the official hierarchy over these
selected datasets. Label it a representative subset, not the complete 29-dataset
MileBench score. The new `--unified-milebench` route preserves the pinned scorers
without mislabeling a byte-matched experiment as paper reproduction. Results,
metrics, failures, and manifests must agree on `unified_comparison`.

**Input readiness remains incomplete:** the cache has an embedded-media
ActionSequence preview and four other annotation previews. That is not yet a
reviewed representative media ledger. Do not mix source revisions under one
dataset SHA or silently substitute combined images for native multi-image
inputs. Materialize and hash the exact common representation before preparing
GPU workers; declare combined-image experiments as such if that route is used.

## Common controls and promotion

- HF eager, FP16, deterministic 64-token output, single-token blocks, no
  residual/repair, bounded 1/256 MosaicKV graph routing. Mandatory policy is
  last-token for the COCO lane and first/last for the MileBench lane; it is
  identical across all compressed methods within each lane.
- Shared initial physical caps; equal append-only growth during decode. Require
  exact per-sample realized initial **and final** bytes across compressed methods.
  Record failure/underfill rather than silently comparing unequal capacities.
- A 16-token FullKV/generate and extraction/reinjection parity check precedes
  measured inference (identical greedy tokens, maximum logit error <=1e-4).
- Five stored warmups and one quality trial per sample. Report paired timing as
  informational, not as a repeated systems microbenchmark. Never hide failures.
- Require complete canonical rows, input/config/provenance hashes, finite
  timings and metric outputs, and matching identities before interpreting curves.
  A positive quality claim must follow the paired intervals, not a forced gate.
- A >=30% smaller-at-equal-quality claim can only come from quality-eligible
  points on the resulting curves. An equal-byte test cannot itself demonstrate
  30% smaller memory. Negative outcomes remain in the report.

`scripts/prepare_native_metric_shared_bytes.py` consumes reviewed hash-bound
input ledgers and requires a clean source SHA. It writes new immutable ID/media
ledgers, resolved configurations, and one-method/one-shard job specifications;
it does not download data, execute inference, submit jobs, or authorize the
larger stage. FullKV appears once per shard rather than once per cap. The common
worker supports separate metric Parquet exports and the unified MileBench route.

All future submissions use the common admission controller: at most four A100s
in aggregate, one A100 and 12 CPUs per worker, two hours maximum,
`bii-gpu` / `bi_fox_dgx`. Use external scratch output roots. Preserve unrelated
worktree changes and obtain the requested narrowly scoped commit authorization
before creating a canonical source snapshot. No new GPU result is claimed here.
