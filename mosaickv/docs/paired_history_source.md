# Immediate K/V construction-source reuse

The mixed reader's constructor retains three complete verification passes. Each
pair formerly decoded an identical source independently for K and V. Reuse that
exact source **only within one immediate pair call**, with no intervening prompt
reconstruction. The same optimization applies to compatible native exact-history
construction in storage-only and baseline variants; it is common transport.

Original native GEMMs, RoPE, counted patches, output clearing, scratch sentinels,
phase/owner checks and all three verification passes remain. Reject a source
alias of either output. Reconstruct again on every next pair/verification pass.
Ordinary sequential model K/V decoding never reuses this buffer: prompt work
between the components can overwrite it. No persistent source, GPU pointer table
or other GPU storage is added.

CPU regressions -> actual operator pairs -> same two consumed pretrained inputs
with bitwise all-128-logit comparison -> conditional randomized direct timing.
The valid algorithm-vs-storage latency comparison applies the common reuse to
**both** arms. Uninstrumented 5-warmup/20-trial results are distinct from audited
single-request allocator traces. All failures remain negative; no thresholds,
precision rules or byte boundaries change. Independent quality and paper-scale
execution remain blocked.
