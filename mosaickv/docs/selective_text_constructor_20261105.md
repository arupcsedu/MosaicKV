# Exact selective cached-text construction

The native reader and direct-fitting successor are preserved as negative.
This default-off constructor optimization uses the original fused reader,
original selected K/V, source headers, protected values and logical positions.
No numerical tolerance or output is changed. All six main mechanisms remain
disabled in this precision-storage diagnostic. Independent quality N=0.

The frozen cached-text constructor materializes all original prompt K/V when
any requested text row is approximate. The successor computes only whole,
originally aligned MMA-row tiles covering requested text positions, retaining
the same kernel, row alignment, head groups, arithmetic and RoPE coordinates.
It restores the same protected values and corrections. Only text rows may
be read from these partially initialized buffers; visual protection is gathered
directly from the already exact protected bodies and re-encoded identically.
No uninitialized visual value may enter a packet. Temporary pair backing is
the same full shape as before, not hidden compact/mis-strided storage.

CPU protected-row indices originate in the already CPU-generated protection
mask and are copied unchanged to the authoritative GPU tensor. This avoids
a redundant device round trip during exact-body filtering. The new CPU index
backing is separately reported; it contains no keys/values and is not residual
offloading. GPU tensor words and physical storage must remain identical.
The legacy `parent_materialization_layers` includes partial materialization;
the new `selective_text_construction` counters identify its exact extent.

Full CPU tests precede two one-A100/12-CPU consumed-input workers. Require the
entire frozen packet tensor digest, all 128 original vocabulary logits bitwise,
no fallback, and identical initial/final/decode-owned-peak GPU storage before
the short screen. Require genuine reduction in projected rows and index reuse.
The old EviCastKV graph/arena and both fastest PrefixKV graph controls remain.
Five warmups and three randomized paired measured requests identify engineering
progress only. A <=0.96 POINT ratio is not acceptance; the unchanged adjusted
twenty-repetition <=0.96 gate is still needed before any latency superiority
claim. No automatic formal, fresh-cohort, calibration or paper submission.
Every producer has afterany terminal analysis, including failures. Aggregate
A100 cap six, at most two used here. All caches remain outside home.
