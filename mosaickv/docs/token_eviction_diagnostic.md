# Consumed-input per-token NLL / history-eviction diagnostic

This is evaluation instrumentation, not a new EviCastKV/MosaicKV candidate.
Preserve V10 as negative. Do not optimize prior weights, select new examples,
materialize data, or authorize calibration or a larger campaign.

Reuse the two consumed Description and MM-Vet identities, all existing 1024-token
greedy outputs, EOS records, configurations and cache budgets. Record first
greedy divergence from FullKV, native PrefixKV and ordinary matched Top-K for
every saved V10 schedule. Distinguish post-EOS differences from scored-answer
changes. The gold teacher trajectory is different from a free greedy trajectory;
do not conflate their token positions as the same linguistic prefix.

Missing data: old traces retain aggregate NLL but not token losses. Run only
teacher-forced replay for FullKV, native prefixkv_reimpl, window Top-K, uniform
mixed-prior EviCastKV, and uniform full-prior EviCastKV, for both inputs. Ten
one-arm A100 workers, one GPU/12 CPUs/16 GiB each, at most six aggregate GPUs,
20-minute limits, bi_fox_dgx reservation. No new generation-quality run or sweep.
Finite afterany CPU analysis; all failures retained. All caches remain on scratch.

Sideband adapter wrappers record prefill and each decode's target NLL, greedy
argmax, logical position, observed generated-history eviction count, removed
history positions, and native PrefixKV post-forward cleanup events. Wrappers
restore on exceptions and never change K/V, masks, weights or selection. Targets
are used only for offline scoring; no observer values feed the online selector.

Token positions are one-based. Prediction 1 uses the original prefill logits,
before compression. For FIFO capacity G, the first history replacement occurs
while feeding generated token G+1, affecting prediction G+2 (258 for G=256).
Verify that boundary with actual counters rather than only inferring it. Native
PrefixKV evicts after computing a step's logits, so its cleanup can affect the
next prediction, not the logits just scored. Classify native prompt and generated
evictions separately by original logical position.

Require complete targets (97/593 tokens for these identities), unchanged selected
positions and physical cache lifecycle, and replay NLL sum matching both runtime
and historical aggregate within 1e-8 nats absolute. The tolerance is an accounting
check, not a changed quality criterion. Log per-token deltas against FullKV,
native PrefixKV (a separate operating point), and matched Top-K. Report first
positive delta above a fixed descriptive 1e-6-nat noise filter, first positive
cumulative delta, nonoverlapping 16-token windows, and summed deltas before/from
the first observed generated-history eviction. Preserve raw deltas too; do not
choose thresholds or windows after seeing outcomes.

The observer repeats CE and scalar transfers. Its timing/memory is diagnostic,
not a systems performance comparison. A temporal alignment alone does not prove
causality. In particular, a divergence before any generated-history eviction
rules out that FIFO event as the cause, but does not identify an individual
evicted prompt block. Two consumed inputs cannot establish a paper-level win.
