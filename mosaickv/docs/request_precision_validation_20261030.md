# Complete-request and broader-cohort validation, version 2 (unprofiled)

The user's October 30 request authorizes this separately versioned, bounded
implementation and validation. Preserve all prior artifacts. Never rewrite the
two consumed diagnostics as independent evidence. Do not change the frozen
precision rule, correction fraction or quality thresholds after seeing scores.

## Scope frozen before new predictions

Pinned LLaVA-1.5-7B, revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9;
BF16 weights, HF eager, Transformers 4.55.2, common environment only. Exactly
128 predictions per timing request; score only before the first model EOS.
All requests recompute media loading, processor, prefill, online selection,
construction, source retirement, decode and detokenization. No saved prefix,
prefill, selection or draft rollout. Model loading, artifact writes, provenance
hashing and official metric scoring are outside the inference timer.

This online variant uses the attributed pinned PrefixKV profile SHAPE and a
fixed 0.5 full-prompt shared-head / 0.5 last-16 per-head attention mixture.
It is NOT the original graph-based mosaickv_exact paper candidate or official
PrefixKV. Its explicit label is online_profiled_shared_prior_precision_v1.
The public legacy method ID is retained only as a routing key. No graph,
prototype, residual repair or training is active. This is a design variant.

Compressed source-slot ratio is 0.5. Physical storage includes metadata and
materialization. Encoded bytes vary with data and selection: these are common
source-capacity Pareto points, NOT exact-byte-matched comparisons. No direct
same-budget claim is permitted. All five compressed selection methods receive
BOTH the common lossless K/V and common query-weighted 3-bit variants, with
exact text/boundaries and lossless append-only generated history. Exceptions
remain <=0.5% per component and must be funded by exact-text codec savings.
LOOK-M's actual merged K/V must be preserved. PrefixKV and VL-Cache here are
explicit append-only/generalized local reimplementations, not paper reproduction.
Top-K and EviCastKV store only exact last-16 probability rows; EviCastKV also
stores the exact shared-head full-prompt prior. Those statistics are reduced
inside read-only eager hooks without retaining complete QxK tensors. Other
baseline implementations retain the complete maps they currently consume.
Each method runs only its own selector, not both Top-K and EviCastKV selectors.
All published-method local planners borrow validated read-only source views
instead of making a redundant full-KV snapshot copy. CPU parity tests require
identical selected positions and actual LOOK-M merged values, and unchanged
original source tensors. The legacy diagnostic default still clones snapshots.

## Measurement contract

Reset PyTorch peak counters before EACH complete request, after loaded-model
warmup and common GC/synchronize/empty-allocator policy. Record baseline model
allocation, whole-request max_memory_allocated/max_memory_reserved, incremental
allocated peak, and decode-owned initial/final/peak unique backing separately.
The allocator peak includes weights, original prefill, activations, capture,
construction and decode. It is NOT an exact KV-only whole-request peak and NOT
an NVML-certified total process-HBM high-water mark. Never relabel it as either.
Request boundaries synchronize CUDA; CUDA Events accompany monotonic wall time.
TorchDispatch tensor-workspace auditing MUST be disabled for all measured
requests: its overhead made the first canary unsuitable for latency claims.
Independent storage enumeration and source ownership checks remain enabled,
as do CUDA allocator peaks and numerical identity controls. The first canary
is preserved as a profiled diagnostic, never reused as unprofiled timing.
FullKV collects no unused query/attention/projection capture, performs no
compression, and remains excluded from compressed-method latency claims.

## Bounded sequence

1. CPU regression suite and complete-request parity/retirement tests.
2. Same two consumed inputs, all eleven arms, with ordinary generate/reinjection
   and unmodified numeric guard checks. Not independent quality evidence.
3. If correctness passes, select 128 COCO IDs by a frozen input-only SHA order,
   excluding every ID in the reviewed historical COCO inventories and the two
   consumed source IDs. Preserve all references. Use source/image hashes.
   Also compare decoded RGB content against all 4,096 historical COCO images
   and the two consumed diagnostic inputs; PNG container differences cannot
   hide exact pixel reuse. Fail rather than replace a selected overlapping image.
   Exact-ID disjointness is mandatory. Unresolved global cross-dataset media
   history prevents a globally untouched/paper-confirmatory label: report that
   limitation rather than fabricate a freshness certificate.
4. All eleven arms on the same 128 images, counterbalanced within each worker.
   No adaptive stopping, favorable-image selection, tuning or cohort enlargement.
5. Five warmups plus twenty full-request paired repetitions on two fixed
   input-only timing identities, after the quality workers finish. The comparator
   is selected by the preregistered maximum measured baseline corpus CIDEr rule,
   with smaller initial bytes and method ID as deterministic tie-breakers.
   Both baseline precision policies are eligible; never pick an easier opponent.
   Repeated
   timing observations do not increase the unique quality sample count.
6. Finite afterany CPU analyses preserve errors and validate all denominators,
   controls, source retirement, budgets, provenance and output lengths. Recompute
   corpus CIDEr using the pinned lmms-eval/COCO scorer and rebuild corpus IDF in
   each of 10,000 paired bootstrap replicates. Correct the nine other compressed
   point comparisons (eight baseline points plus the same-method lossless
   ablation; Bonferroni simultaneous intervals). Never average
   per-shard CIDEr. No combined average with ROUGE/PPL.

## Decisions

Primary quality comparisons use the new candidate vs every compressed baseline
under both common representations. A superiority result requires adjusted
paired CIDEr lower bound >=0, measured smaller initial/final/decode-peak storage,
complete provenance/rows, no fallback and paired end-to-end latency upper bound
<=1.0. Whole-request allocated peaks are separately tested/reported. These are
cross-budget equal-quality tests, not byte-matched tests. Failure on the strongest
eligible baseline remains a failure; do not pick an easier comparator afterward.
The noncomparative same-method codec ablation is reported separately.

At most six aggregate A100s; one GPU, twelve CPUs per worker; <=2 hours per shard.
No vLLM/SGLang campaign, prototype/repair claim, paper-scale expansion or LaTeX
report is authorized by a mere execution pass. A negative result is retained.
