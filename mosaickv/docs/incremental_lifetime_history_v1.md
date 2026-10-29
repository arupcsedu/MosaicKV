# Incremental lifetime-history V1: experimental exact-cache candidate

EviCastKV and MosaicKV are the same system. This is a new, explicitly labeled
history/ownership variant, not a revision of historical measured results. No
quality improvement, 30% saving, whole-request peak reduction, or paper readiness
is assumed. Existing exact prompt selection, original K/V, logical positions,
mandatory policy, and model weights remain unchanged within each control pair.

## Why this version

The native PrefixKV campaign's append-only EviCastKV path adds 511.5 MiB during
1,024-token generation. Earlier bounded-history V1--V6 experiments proved
storage bounds but not consistent quality preservation. This successor does
not repeat their prompt/layer exchange search. It changes actual history
allocation and retention and retires unused source-cache ownership.

PrefixKV does **not** merely keep the chronological prefix: its pinned path
selects important positions using ranked attention and layer profiles. Native
PrefixKV and its byte-rescaled append-only adaptations remain different rows.

## Implementation sequence

1. `KVOwnershipLedger` deduplicates explicitly registered backing allocations
   across original prefill, snapshot, packing, decode, and growth-copy roles.
   Weak references do not extend cache lifetime. Its measured boundary is
   explicit: it does not certify every internal prefill transient or process
   memory. Existing `CacheLifecycle` fields keep their historical post-compression
   decode-cache meaning and are never silently relabeled as whole-request peak.
2. `IncrementalHistoryLayer` initially wraps only the privately packed prompt.
   It allocates generated slots in configured chunks, not a full reserve at
   initialization. Buffers grow one layer at a time; before publishing a
   replacement, the ledger includes both old and new allocations. A conservative
   largest-layer scratch reserve is deducted before prompt selection.
3. The bounded `sink_recent` control protects generated sinks and uses a recent
   FIFO ring. `recent_importance` additionally protects a recent window and
   evicts the lowest accumulated causal-attention score among eligible older
   positions. Scores are updated using actual decode attention, not benchmark
   references or FullKV probes. Layers can retain different original generated
   positions but have equal physical length and no per-head padding.
4. Before measured decode, selection metadata and scalar planner measurements
   are preserved, and unused private FullKV/snapshot/plan references are retired.
   Registered source tensors still alive after retirement fail explicitly.
   The retirement work is charged to compression; diagnostic trace serialization
   remains separately excluded as before. No CPU offload or quantization hides
   source memory.
5. Retention-1 bypasses the new eviction/allocation path. Eager LLaVA-1.5 with
   Transformers 4.55.2, batch one, one-token decoding is the audited first scope.
   Unsupported variants fail configuration validation. Native PrefixKV is
   unmodified; its reference rows are not passed through this new policy.

The attention-score mechanism is related to H2O's important-plus-recent eviction
(https://arxiv.org/abs/2306.14048) and StreamingLLM's sink/recent preservation
(https://arxiv.org/abs/2309.17453); neither principle is claimed as novel here.
It is not a paper-faithful port of either baseline. Prompt evidence still comes
from EviCastKV's original forecast/graph/utility exact selector.

## Physical budgeting

Let `c_l` be K+V bytes for one position across all KV heads in layer `l`.
With `C=sum(c_l)` and `c_max=max(c_l)`, the conservative complete position
capacity is `floor(B/(C+c_max))`. Reserve the configured history positions
inside that capacity; remaining positions are the prompt budget. Thus maximum
resident KV plus one layer's growth-copy scratch stays below the hard cap.

For the existing LLaVA-1.5 FP16/BF16 geometry, **planning arithmetic only**:
192 MiB hard transaction cap and 192 history positions yield a 90 MiB prompt,
186 MiB maximum resident storage, and 5.8125 MiB conservative scratch reserve.
These are not pretrained-model quality or measured-performance results.

Protected recent tokens plus generated sinks must leave room for an optional
important-token region. When a short teacher-forced target needs fewer updates,
the same registered planning rule reserves only those updates for every arm;
generation and PPL storage therefore remain separately recorded. Targets are
never truncated or passed to the selector. The online controller sees the
ordinary previous-token input and its actual attention, as in teacher forcing.

## First GPU diagnostic, frozen before execution

Two one-A100 workers, 12 CPUs and at most one hour each; aggregate A100 limit
six, explicit `bi_fox_dgx` reservation. Each uses the first already-consumed
input in the immutable intermediate Description/MM-Vet ledgers, never selected
using an answer or failure. This is not fresh or confirmatory evidence.

Each worker executes seven arms with matched BF16 weights, processor, prompt,
media, eager backend, deterministic seed, and 1,024-token generation:

- FullKV quality reference (excluded from compressed latency comparisons).
- Pinned native-policy PrefixKV reimplementation as a separate native control.
- EviCastKV append-only anchor at 90 MiB prompt storage.
- Prompt Top-K and EviCastKV under the same 192 MiB transaction cap, each with
  `sink_recent` and `recent_importance` history policies.

For every arm, save generation and complete teacher-forced NLL/token records,
reference ROUGE-L, initial/final/peak decode storage, known-owner peak including
source/scratch buffers, timings, raw answers, errors, hashes, and full provenance.
Native-versus-lifetime comparisons have different budgets and policies and are
explicitly not byte-matched. The four new shared-cap arms use the identical
transport and planning rule. No microbenchmark/statistical latency win can be
inferred from zero warmups and one quality repetition.

Run FullKV generate/reinjection and configured retention-1 parity first. On
failure, retain failed rows and analyze with a finite CPU `afterany` job. Do not
rerun successful work or automatically enlarge the cohort. There is one input
per task: no paired confidence interval or quality-preservation claim is valid.

## Subsequent gates

Only meaningful correctness and memory results justify a separately frozen
disjoint-development protocol. PPL and ROUGE-L must pass separately; no margin
is weakened after seeing data. A powered independent quality test and common
budget grid are required before lower-memory-at-equal-quality paper claims.

Whole-request peak remains a separate engineering gate. Full-cache prefill
creates a shared floor; a future layer-streamed construction must preserve
prefill outputs and position semantics before that stronger claim is possible.
This implementation must not claim that source retirement alone resolves that
prefill floor. Prototype merging and residual repair remain outside this study.
