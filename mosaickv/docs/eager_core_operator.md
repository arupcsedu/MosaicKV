# Exact eager-core execution: operator feasibility before model integration

The fused transport-predicate request check completed with exact cache words
and bytes but failed the four-percent latency gate. Its isolated operator win
did not translate into a meaningful request win. Preserve that negative result
and leave its runtime flag default off. No fresh quality campaign is justified.

The saved decode profile instead attributes about 0.72 seconds of host time to
8,064 attention matmul calls and about 0.12 seconds to 4,032 softmax calls. These
are host intervals, not pure GPU kernel time. Test reusable output buffers and
optional CUDA replay of the SAME installed eager operations, with no SDPA or
FlashAttention substitution and no second KV buffer.

The existing lease is overwritten from K to V between score/softmax and PV.
Therefore use two separate graphs. Preserve BF16 matmul, scaling, the actual
mask dtype/values, native softmax dtype=FP32, BF16 cast, and BF16 PV. Use the
native softmax out overload, not a newly fused or differently ordered reducer.
The original stored K/V, mask/logical positions and model weights are untouched.

One fixed score/probability/query/output workspace is reused across live lengths
by contiguous flat-prefix views. Cap it at 384 KiB; charge backing storage, not
view sizes. Bind one original fixed-address KV lease; do not retain per-length
KV or score outputs. A request-private table keys graphs by actual live length.
Use the same existing common epilogue capture stream, never a fresh cuBLAS stream
for every length. Expose common backend warm allocation separately. Unexpected
post-warm opaque Tensor owners fail; only the independently checked pinned two
generator scalars may remain and they must retire with the graph table. Include
capture-time transient Tensor allocations in the 384-KiB bound and use one
private graph memory pool for all lengths; no intermediate may escape a stage.
Report that pool's reserved bytes separately rather than hiding allocator
rounding. Model integration must account for this footprint; an operator pass
does not certify a model cache-memory reduction.

The label-blind graph rule is frozen at at least 16 planned uses of a layout,
computed only from immutable layer lengths and the registered decode horizon.
Never choose it from benchmark answers. Test uniform and adaptive layouts, with
original eager, reusable-out, and amortized-graph policies for each. Include cold
graph capture, warm kernel invocations, input copies, retirement, synchronization,
five warmups and twenty randomized paired repetitions. Keep all per-trial wall,
CUDA event, capture count and allocator-peak measurements. An operator progression
requires a <=0.96 ratio for at least one tested optimization in EACH layout; the
policies and any negative graph results remain separate. Do not infer a model
latency win or select a policy by sample answers.

Before timing, require exact original probability/output words for no mask,
BF16 mask and FP32 mask at early/middle/late lengths, changed A-B-A inputs,
unchanged original K/V, preserved unused-capacity sentinels, one K-to-V transaction,
bounded unique workspace, and complete graph-owner retirement. Any mismatch blocks
timing and integration. CPU tests first verify the installed out API and strict
transaction/geometry handling. One A100 follows a full same-SHA CPU suite and
afterany terminal analysis. No model weights/datasets, independent quality N=0.
Whole-request GPU peak and paper-quality evidence remain unestablished. This
module is not wired into model inference; common model integration would need a
separate same-input 128-logit, packet, storage, and full-request validation.
