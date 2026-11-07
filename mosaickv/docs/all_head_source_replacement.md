# All-head source replacement and the storage-only reference

The user's new reference is the frozen **EviCastKV storage-only** variant, source
`a6f689fcdfae2b9023045aa412c6d90d8ce79012`. Do not substitute the previous slow
hybrid as the performance baseline. Preserve previous negative source-format
screens. Their MSE-preserving lossy formats did not beat the common byte codec.

## First engineering correction: replacement serves every consumer

Retiring a shared projection source requires every original feature, position,
K/V projection and all 32 heads to remain represented. One-head overlays fail
this requirement. `SourceCoverage` makes the full 64-consumer obligation
explicit; incomplete heads, components, rows, feature gaps or overlaps fail.

This first replacement is **lossless all-head source coding**, not a newly
validated KV-merging algorithm. Encode the original 3-bit source codes, retaining
their original ranges, logical positions, weights and RoPE operations. A block
contains 512 symbols. Two common symbols form a palette prototype; a first
bitmap distinguishes the first symbol, a second distinguishes the second, and
the remaining exact 3-bit literals are escape residuals. Choose direct codes or
the XOR against the preceding layer by actual payload size; raw blocks remain
available. Stable ties choose the lowest symbol and direct coding. Charge all
uint32 offsets, bitmaps, residual bytes, selector bytes and replay scratch.

The first CPU experiment reconstructs every source code bitwise on both already
consumed 32-layer captures. Compare against the **existing storage-only byte-XOR
codec**, not uncompressed source alone. Both encoders reconstruct identical
bytes, so selecting an encoding does not destroy the next layer's original
delta opportunities. Keep the parent codec whenever net storage would regress.
Require strictly positive complete net CPU savings on both inputs before a
native GPU operator is worth implementing. This is a lossless engineering gate,
not a relaxed version of the earlier 10%-saving gate for lossy source formats.
CPU code-stream bytes are not GPU KV or whole-request peak measurements.

Raw blocks remain real charged owners. Only accepted compressed owners can be
retired. No packet may retain the old code owner or a decoded full source. A
native decoder must reconstruct once into the existing per-layer replay buffer
used by every head, rather than keep another source copy for unmodified heads.
GPU/native support and model timing remain false until actually verified.

## Second correction: safety is not a quality-improvement objective

Lossless code replacement does not need a per-token parent-answer guard: correct
bitwise reconstruction preserves the model computation. It does **not** improve
answers, and must be reported as quality preservation. For the adaptive hybrid,
an explicit `fullkv_draft_improvement` policy instead requires the repository's
actual original-prefill FullKV draft/readout path. It rejects the simultaneous
storage-parent greedy-token lock, which mathematically forces the same answer.
No benchmark labels are online inputs. FullKV draft improvement is only an
online proxy: independent CIDEr/PPL/ROUGE validation is still required. The
earlier FullKV-draft diagnostic is retained as a negative result, not relabeled
as success. No model-quality threshold is loosened by this configuration guard.

## Bounded execution and attribution

First: synthetic CPU coverage, codec, corruption and policy tests, then two
CPU source-stream workers (12 CPUs, 16 GiB, 30 minutes, `bii`) and an `afterany`
analyzer. No model weights, labels, fresh samples or GPU jobs are needed here.
Save hash-bound original code streams for a possible native operator check.
If the exact-stream gate passes, separately test native GPU reconstruction and
the unchanged 128-token model path on consumed inputs before timing.

Required later factorial controls are (1) frozen storage-only, (2) storage-only
with common replacement, (3) old hybrid, and (4) hybrid with replacement. Apply
the common codec identically to storage and hybrid controls; never credit a
general lossless codec gain as evidence that forecast/graph/value/selection or
uncertainty repair improves quality. A six-mechanism performance claim still
requires real source-tier execution and matched-budget component ablations.
At most six A100s in aggregate, one per worker, 12 CPUs, bounded jobs. No fresh
or paper campaign follows automatically from a codec feasibility test.
