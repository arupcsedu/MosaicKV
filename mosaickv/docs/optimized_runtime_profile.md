# Optimized runtime attribution (consumed diagnostics only)

The previous event-updated ownership accounting removes redundant sums for
EviCastKV **and both PrefixKV readers**. Its three-repeat screen still misses
the requested 4% full-request advantage. Earlier CPU-call profiles predate that
optimization. Do not infer the remaining bottleneck from those old counters.

This finite chain first runs the complete CPU suite, then two one-A100 workers,
then CPU analysis even when a worker fails. Workers reuse only the same two
consumed inputs. They compare original EviCastKV, native-source EviCastKV with
the already audited direct constructor, and original PrefixKV reader; **all
use event-sum accounting**. One excluded warmup precedes each measured profile.
No calibration or larger sweep is authorized by this attribution.

Nonoverlapping cProfile scopes cover (a) preprocessing, prefill, construction,
and setup until first decode and (b) decode updates 1–126, excluding the first
capture update. These are instrumentation diagnostics, not latency acceptance.
C function self time can contain GPU waiting; cumulative times are nested.
Full 128-vector parent-bitwise logits, no fallback, exact storage accounting,
and existing validation barriers remain mandatory. Whole-request GPU peak and
independent benchmark-quality superiority remain unestablished.
