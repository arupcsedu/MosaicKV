# Consumed-input transport controls

This bounded diagnostic extends—not replaces—the immutable 128-image study
at e7d7fcda7bca568fb28325fee4982d12088aa936. Do not edit its active source.
No new identities, benchmark references, or answer-based selection are allowed.

The eleven study arms offer common lossless and three-bit encoded storage.
They do not include each selector's ordinary BF16 cache. Therefore even a
positive result among encoded arms cannot establish dominance of the fastest
ordinary HF baseline. This control closes that measurement gap on the same
two consumed inputs first, not as independent benchmark evidence.

All five selectors run with their unchanged 0.5 source-slot capacity, original
K/V values, checkpoint, processor, media, seed, eager backend and 128-prediction
schedule. Native caches retain original logical positions. Per-layer extents
use the existing audited ragged mask interface; a representative first-head
position list supplies only each layer's mask extent. The complete original
per-head prompt maps remain separately recorded, and all prompt keys precede
each new query. No prototype, repair, history eviction, or model.generate is
enabled. Snapshot owners are released before native append so immutable prompt
copies do not inflate live storage. The complete-request CUDA/wall timers and
allocator peaks are those of the frozen study. Transient native torch.cat
allocations are included in the whole-request allocator peak, but native
KV-only transaction peaks are not certified by boundary snapshots.

Compare:

- ordinary BF16 selected cache for each selector;
- frozen EviCastKV three-bit prompt plus source-reconstructed lossless history;
- identical EviCastKV prompt plus raw lossless BF16 generated history.

The third arm tests reconstruction overhead rather than changing evidence.
Its larger final history storage must be reported. Three timed repetitions and
one warmup are cost-screening diagnostics only, not paper latency evidence.
Every request must be deterministic. Require all 128 greedy tokens to match
the saved corresponding lossless/three-bit output and maximum logit difference
at most 0.001 before treating the transport as equivalent. Also retain the
unchanged 32-token token/KL/logit guard against the saved lossless anchor:
agreement 1.0, KL at most 0.02, maximum logit difference at most 1.0.

Function substitutions are isolated to one diagnostic request and restored on
error; concurrent requests explicitly fail. This is an ordinary HF transport
diagnostic, not native vLLM, SGLang, or a production integration. It cannot
promote a paper claim or authorize a fresh cohort.

Two new one-A100/12-CPU workers may start only after all first twelve shards of
22208958 finish. At most four parent workers can then remain, so the aggregate
limit remains six A100s. Once the parent cohort ends, its two timing workers
and these two controls still total at most four. Jobs have a 30-minute bound
and an afterany CPU completeness/numerical analysis. All caches stay in scratch.
