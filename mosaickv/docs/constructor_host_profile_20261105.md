# Constructor attribution after common host optimization

The shared counter optimization improved both EviCastKV and PrefixKV but did
not pass the comparative latency gate. Profile unique pre-decode work rather
than repeat a decoder-kernel sweep. Same consumed inputs and unchanged original,
native EviCastKV and original PrefixKV paths; one excluded full warmup each.

cProfile starts at prepare_inputs and stops when the first decode starts. This
includes preprocessing, media/model prefill, fitting, retirement and initial
cache setup; it is **not pure compression time or latency acceptance**. Preserve
the binary caller graph to attribute nonzero/item/allocation calls to their
actual callers. No model arithmetic, values, thresholds, positions or GPU
workspace changes. All 128 own-parent logit vectors remain bitwise required.

Full clean CPU tests, then two one-A100/12-CPU consumed-input workers, 20-minute
limits, six A100s maximum aggregate, afterany terminal analysis. CPU tests use
udc-aj36-3c0 (bii), observed with six unallocated CPUs at submission preparation,
because its prior complete CPU suite finished in about three minutes. No GPU
resources are allocated for the CPU checks. No new identities, automatic
timing retry or quality campaign. Independent quality N remains zero.
