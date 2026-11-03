# Prompt-source operation attribution on consumed inputs

This is an engineering diagnostic of the existing source-precision/storage
VARIANT, not a performance claim for all six MosaicKV/EviCastKV mechanisms.
Only the same two consumed LLaVA-Description/MM-Vet inputs may run. No labels,
reference answers, new examples, calibration, quality campaign or baseline
reproduction is authorized. Baselines are local precision/storage adaptations.

The preceding descriptor trial preserved every numerical/storage control but
failed its preregistered adjusted <=0.99 end-to-end ratio on both inputs.
Neither descriptor nor compiled-dispatch reuse is promoted. Retain the
previously passed carry_literals reader, original descriptors, original
dispatcher, all existing frozen K/V/weights/positions, and 128 predictions.

Before any optimization, subdivide ACTUAL prompt reconstruction into:
validation; indexing/argument/view preparation; projection dispatch;
protected-row restoration dispatch; numerical-correction dispatch; and named
explicit temporary-allocation entrypoints. All original validation predicates,
kernel parameters, arithmetic and correction ordering remain unchanged.
No check is disabled or cached. No reconstructed K/V is retained.

Nested timing uses exclusive host leaves and current-stream CUDA events.
Event intervals include host gaps and are not pure GPU-active kernel costs.
Never add overlapping parent reconstruction/projection and leaf intervals.
The allocation watcher covers named Python entrypoints, NOT all allocator
operations; zero calls does not prove zero CUDA allocation or GPU peak.
All contexts and allocation wrappers restore on exceptions and retain no
tensor outputs. Instrumentation is never part of comparative latency trials.

One excluded warmup and two instrumented full requests per method per input:
mosaickv_exact and prefixkv_reimpl, alternating order. Audited, uninstrumented
companions must first exactly reproduce frozen packet bits, 128 logits,
initial/final and declared decode-owned peak storage. Profile requests must
also match logits/endpoints and their separately registered fast counters;
fast counters must never be relabeled physical peak. Every early/middle/late
window must contain all source operations for the actual source variant.
The PrefixKV adaptation does not use shared-input source projection; report
its actual applicable operation counts, without fabricating source activity.

Use clean immutable source and complete unit/integration CPU proof. Exactly
two one-A100/12-CPU workers, at most 15 minutes each, afterany CPU analysis,
registered six-A100 aggregate ceiling, scratch-only caches. Preserve failures
and write-once outputs. Rank measured exclusive host operations only after
both workers complete and hash/numerical/accounting checks pass. The analyzer
does not promote an optimization or submit another GPU campaign. A successor
must target the dominant measured operation, retain all controls and receive
its own bounded numerical and paired uninstrumented verification.

Independent quality N=0; whole-request GPU peak remains uncertified. The
different storage points are not byte-matched algorithm comparisons.
