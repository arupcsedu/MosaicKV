# All-head source retirement and hybrid latency

The source palette/escape codec replaces the complete shared source stream,
not an isolated head's K/V. It covers both projections, all original positions,
and every head. The original quantizer codes decode bitwise, metadata and replay
scratch are charged, and weak-reference checks verify that replaced source owners
are retired. It does not claim a one-head KV prototype retires an all-head owner.

The initial pretrained canary is preserved at
`/scratch/djy8hg/runs/evicastkv/all-head-source-model-20261107-v1`.
Its two inputs are consumed development diagnostics. The codec saves approximately
4.1 MiB at each reported storage boundary while preserving all 128 storage-parent
logit vectors bitwise. Counter peak is decode-owned, not whole-request GPU peak.
The single-call times are not a latency acceptance result.

The `bitmap_prefix` source decoder is a separate default-off implementation.
It replaces two 512-element symbol prefix scans with prefix popcounts of the
64-byte bitmaps. It does not change the packet format, quantizer, reconstructed
codes or charged workspace. Its native 64-layer, two-input parity experiment is
`/scratch/djy8hg/runs/evicastkv/source-bitmap-prefix-20261107-v1`.
Only a passed native decision allows `source_hybrid_latency.py prepare
--bitmap-prefix --root NEW_PATH` to use it in the optimized model arms.
The common storage control receives the same optimized decoder as the hybrid.
Operator timing alone is not an end-to-end latency claim.

`--reuse-anchor-readout` tests one further default-off hybrid optimization.
The guard already evaluates the final normalization and vocabulary projection
for both candidate and parent. The outer model can consume the selected native
readout instead of evaluating it a third time. This uses no approximate Jacobian,
new attention formula, changed guard threshold or parameter update. Request-local
hooks enforce normalization-before-projection and exactly one consumption per
guarded step; pending outputs are retired on failure. Additional temporary
activation backing bytes are reported separately, never counted as KV savings.
Model-level 128-logit bitwise checks and 127 actual reuse records per request are
required before any latency interpretation.
The follow-up uses only storage-only, optimized-codec storage-only, and optimized
hybrid: six warmups and twenty-one timed trials give balanced three-arm orders.
The older factorial arms are not rerun; their saved logits remain immutable
correctness references. Six registered input/comparison pairs share familywise
alpha .05. This does not reuse old timings as if they were newly paired trials.

Two default-off optimizations retain the existing reference paths:

- `all_head_source_gpu.use(construction="pack_selected")` counts both lossless
  alternatives, obtains both sizes with one host synchronization, and packs only
  the winner. Format decisions, tie-breaking and packet bytes must match the
  original pack-both implementation exactly.
- `OverlayPolicy(anchor_counterfactual="single_head")` computes parent attention
  only for the modified head and keeps the already computed other head outputs.
  It avoids temporary writes into the full live K/V lease. Unchanged KL, logit,
  finite-value and greedy-token checks remain. This is still a parent-preservation
  guard, not a benchmark-quality improvement mechanism.

The finite `source_hybrid_latency.py prepare --root NEW_PATH` protocol runs native
packet/in-place replay and CUDA attention parity before pretrained timing. It
checks every 128-logit vector against frozen references on every trial. Five
warmups and twenty timed trials per arm/input use balanced, shuffled arm order.
The five arms are storage-only, reference/optimized common codec, and
reference/optimized guarded hybrid. Common coding is applied to the storage
control too: its savings cannot be attributed to the six adaptive mechanisms.

At most two one-A100/12-CPU workers run, under the six-A100 aggregate controller;
each has a thirty-minute limit. An `afterany` CPU analyzer records failures and
strict equal-or-faster latency bounds with Bonferroni correction over ten
registered input/comparison pairs. No new cohort, confirmation, or larger paper
campaign is authorized by this engineering test. No new quality superiority or
individual component benefit is presumed from bitwise or execution parity.
