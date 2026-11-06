# Common prefill ownership catalog

Warmed profiling finds repeated full-prefix storage-identity scans during
prefill: about 0.10--0.11 seconds in EviCastKV's `observe` and 0.24 seconds in
the optimized PrefixKV control. These are instrumented main-thread times, not
latency wins. Optimizing only EviCastKV would be unfair. Apply this fix to both
RetiringPrefillCache and SelectedRetiringCache and all compared methods.

Keep metadata segments for immutable sealed layers; invalidate only the known
changed layer on native K/V installation, sealing, or lossless-archive
replacement. Generic `owners_changed()` still invalidates everything. Extra
replacement invalidates only its own segment: StreamingBaseline calls `_extra`
on every layer, and discarding all sealed-layer metadata there would preserve
the baseline's quadratic rescan. Recombine current segment keys to deduplicate aliases
across layers, extras and observed temporaries. Compute each tensor identity
once, not twice. Segments hold only device/pointer/size tuples and byte counts,
never tensor/view/storage owners. Peak-observation cadence is unchanged.

An independent original full enumeration and `backing_bytes` check remain at
the prefill phase boundary before deleting the cache. A stale/unannounced
owner change fails closed. Report that check and catalog work counters for
every method. The CPU metadata is instrumentation workspace, not CPU residual
KV. Existing exclusions of Python workspace and uncertified whole-request
GPU peak remain explicit; do not present decode-owned peak as whole-request
peak.

Preserve every selected K/V word, packet digest, logical position and initial,
final and decode-owned peak storage. CPU tests cover aliasing, invalidation,
owner retirement, stale-owner rejection and linear identity work even with
the per-layer extra updates actually used by baselines. Extra replacement
must never conceal an unannounced sealed-layer mutation. The GPU
screen rechecks all 128 parent logits and the existing outer packet audits
before five warmups and three randomized paired repetitions on the same two
consumed inputs. Use two one-A100/twelve-CPU workers, twenty-minute limits,
full same-SHA CPU tests and automatic terminal analysis. No new data or
paper-quality campaign is released. All six primary algorithm mechanisms
remain disabled in this precision/storage variant.

Freeze `same_buffer` as the faster previously measured constructor, without
using benchmark answers. Apply common metadata optimization to every arm even
if it narrows EviCastKV's relative advantage. The 4% point screen and eventual
adjusted twenty-repeat gate are unchanged. No larger timing confirmation
follows a negative point result.
