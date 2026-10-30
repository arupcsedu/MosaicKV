# V2: isolated exact source/KV transport, not a paper-quality experiment

V1 array `22183334` and analysis `22183335` completed. Every retained K/V bit
matched at all 32 layers, for both source formats, both saved selectors, and
both consumed inputs. These were component measurements, not a live cache.
Cold one-shot timings include compilation effects and cannot support speedup.

The next step installs those same representations only in an isolated decode
control. Keep original selected positions, BF16 values, RoPE, mandatory tokens
and raw append-only history. Reconstruct a layer's selected prompt K/V from
its archived original normalized inputs, copy into the layer lease, and
discard the reconstruction. No prototypes, repaired evidence, training,
new queries or new selection decisions are introduced.

Compare three common policies on BOTH Top-K and EviCastKV:

- `kv_only`: the already verified lossless KV format, without unused
  source-format construction during this control's decode or constructor.
- `hybrid`: the V1 conservative per-layer smaller-component rule.
- `bounded_sources`: at most **eight** source-reconstructed layers, with the
  remaining layers using lossless KV. Rank positive byte savings, break ties
  by layer, and charge the shared rotary storage once. Activate source storage
  only if total savings exceed that charge. With shared original RoPE this is
  the exact cardinality-constrained representation rule for measured component
  costs. It is not a selection-utility change or a byte-budget paper comparison.

The bound controls added projection work identically for both methods; eight
is frozen before this runtime test, not selected using task answers. Timing
improvement is unproven. The previous unlimited component envelope is retained.

## Memory controls

Count normalized-input/code buffers, union and per-head indices, truly shared
rotary backing, optional encoded KV, raw history, growth overlap and full layer
leases. During reconstruction, a weak-reference ATen-result audit observes
live tensor backing at every operation, deduplicates aliases and excludes only
already-counted owners and model weights. This includes padded GEMM canvases,
decoded input, projections, RoPE temporaries and gathered outputs, rather than
claiming the smaller persistent archive as peak storage. Shared rotary bytes
are charged once, not once per Python view.

The boundary excludes inactive original KV/full normalized capture, model
weights, attention activations, internal GEMM workspaces and allocator reserve.
Whole-request/HBM peak remains uncertified. The audit is instrumentation; its
wall time is **not** a comparable latency benchmark. A later repeated,
uninstrumented timing study is justified only after correctness/memory pass.

## Numerical controls and stopping

On the same two consumed inputs, verify source-format retention-one greedy
logits/tokens for 32 tokens, then every policy against ordinary exact-cache
32-token own-greedy decoding for both frozen selections. Require finite,
bitwise-equal logits, all 31 appends, correct logical positions, source
immutability and no leaked lease. Do not waive any old absolute FullKV-fidelity
failure; exact representation cannot recover previously evicted evidence.

No new identities, reference labels, task-score measurements, fresh calibration,
confirmation or paper campaign are authorized. Two one-A100/12-CPU workers,
ten-minute limits, scratch-only common environment, immutable source, and a
finite `afterany` analysis. A pass would justify longer/timed storage controls,
not benchmark superiority or a claim of equal-quality smaller memory.
