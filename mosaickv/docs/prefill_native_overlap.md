# Overlap unchanged native repairs with later prefill layers

The saved Source construction profile attributes about 0.17 s to serialized
native conversion. This is CPU/instrumented attribution, not a speedup promise.
After each original attention layer has finished, its source packet has been
finalized, validated, installed and detached from retired FullKV storage.
Submit precisely this immutable packet to one private worker/stream/handle.
At most two layers are pending. The worker waits on the main-stream readiness
event, then performs the unchanged native fit, full word validation, bounded
sparse-repair construction and (on excluded audits) all stride checks. The main
thread may continue original prefill; no second prefill or changed model math.

At the original conversion boundary, consume results in layer order and require
the final packet's source object to be the identical finalized object that was
submitted. Any intervening representation change rejects, never recomputes or
silently falls back. Synchronize and retire the fit stream, close its handle on
its owner thread, join the executor and drop all temporary/future references
before decoding. Each request gets its own resources. All failures also join
and retire. Existing decode/storage checks, mandatory positions and parent
vocabulary words remain unchanged.

Audit worker temporary storage with weak per-ATen-result owners plus explicit
native workspace; report its peak separately, as well as process-wide Torch
allocated/reserved peaks. Do not sum overlapping worker wall time into request
wall time or call this whole-request peak certification: prefill concurrency
and vendor-internal resources remain outside that certification. Decode-owned
initial/final/peak bytes must remain identical to this variant's own control.

Only the Source precision representation is eligible; offer the requested flag
to all readers and report ineligible methods as no-op, without artificial work.
This disables/replaces the six mechanisms of the complete EviCastKV algorithm.
Two consumed inputs, 128 predictions, independent benchmark-quality N=0,
four paired arms, five warmups and three timed repetitions; two one-A100
workers, 12 CPUs each, 20-minute limits. Full CPU suite precedes GPU work;
afterany completeness analysis runs on failures. No fresh or larger campaign.
