# Native request-private descriptor and restoration dispatch follow-up

Preserve the preceding fused-only operator result as NEGATIVE: numerical and
storage controls passed, but early-layer ratios of approximately 0.92--0.93
missed the registered <=0.90 operator gate. No full-request job was launched.
Do not change that outcome or loosen its thresholds. Its saved scopes show
remaining host time in native projection/restore validation, views and dispatch.

The new `kv_fused_prepared` candidate reuses immutable projection WEIGHT VIEWS
and the exact compiled restoration kernel/ABI within one request. No destination
or source-workspace view is retained in the plan; each launch passes the current
destination and head stride. On every hit verify input owners, metadata, kernel
options and counts; for weights also verify the model parameter version. Require
Triton 3.4.0 and zero compiled global scratch. No new GPU buffers or weights are
created. Keep all existing per-call validation and clear aliases at request end,
including on failure. At most 64 weight and 64 restoration entries per request.

One separately labeled operator attempt only. Use the SAME two consumed inputs,
layers 0/16/31 and padding 0/64/127, five warmups/twenty randomized repetitions.
Compare original, unfused BMM, fused-only and prepared-fused. Only the prepared
arm activates a private plan; all other arms explicitly suspend it. The layer
microbenchmark session spans repeated replay of that layer, not independent
requests, so it excludes request-start amortization and cannot establish request
latency. Record real reuse hits/misses and teardown. Require the same <=0.90
ratio against original and BMM for EVERY geometry and both inputs, PLUS <=0.99
against fused-only, exact parent bits, untouched history and no output workspace.

Only after that operator gate passes, run audited complete requests. Require all
128 parent logits bitwise, original packet hashes and measured original-reader
storage unchanged, 3 MiB repair cap, minimum eight converted layers, and smaller
initial/final/decode-owned tensor peak than the smallest baseline boundary.
For 32 converted layers require exactly 64 projection misses, 64 restore misses,
and 8,064 hits of each kind per full 127-step decode. Construction and all cache
planning remain inside request latency. Then use the unchanged paired randomized
five-warmup/twenty-trial adjusted <=0.99 own-parent and <=1.05 Prefix-adaptation
gates; if passed, the existing all-five <=1.0 gate follows. Afterany callbacks
stop on failure. No second policy, new identities, quality sweep or source edits
are authorized by this finite chain.

CPU preparation may use fifteen minutes (the previous complete suite required
7.5 minutes plus provenance verification); other CPU callbacks use ten minutes.
Operator workers: one A100, 12 CPUs, 48 GiB, ten minutes. Conditional complete
request workers: same resources, sixty minutes. At most two active workers,
maximum six aggregate A100s, reservation bi_fox_dgx, common environment, scratch
caches only. Do not continuously poll logs or edit the immutable worker snapshot.

These remain consumed development inputs with independent quality N=0. The
six main mechanisms are disabled in this source-precision/storage variant.
Baseline adaptations are local, not official; different measured memory points
are not byte-matched algorithm comparisons. Decode-owned tensor peak excludes
opaque library scratch and is not whole-request GPU peak. No paper superiority
claim follows from operator or own-parent numerical parity.
