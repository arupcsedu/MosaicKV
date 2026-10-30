# Immutable reconstruction replay — original attention arithmetic

The packed-attention successor was negative at the unchanged retention-one
model gate: all32 tokens agreed, but logit maxima 0.1875/0.25 exceeded 0.001.
Preserve it; do not loosen that gate or describe token agreement as a full
numerical pass. The complete CPU suite at c875de9 passed2656 tests.

This separately labeled successor changes storage scheduling only. All
original prompt/history K/V and logical positions must remain bitwise exact,
and HF eager QK/softmax/value arithmetic is untouched. One reusable physical
K/V workspace serves sequential layers. Actual per-layer lengths slice it;
the full allocation, including unused capacity, is charged once. Overlapping
leases fail. Drop all graphs and workspace before source-tail sealing, so
constructor workspace cannot be hidden behind an inactive leased allocation.
Drop them at request completion. No cross-request graph or storage reuse.

Capture only immutable prompt and sealed-source reconstruction. Raw tails
copy after replay. Warmup, compilation, capture and recapture costs remain
inside measured decode time; no amortization is presumed. Graph-private
allocations must be measured and independently released rather than omitted.
Whole-request GPU peak remains uncertified, as in the reference diagnostic.

Canary: same two consumed inputs only, full retention128-token legacy,
strided-workspace and graph outputs bitwise equal, then identical selected
cache/history outputs. Initial/final bytes must equal the reference; declared
decode-owned peak may not exceed it by more than5% (explicit workspace charge).
No quality threshold changes. Only a numerical/memory pass permits twelve
one-A100 method/input workers, five warmups and twenty randomized paired
technical repeats against the direct source-history reference. Include
constructor and replay cost. Apply identical policy to all six paths, preserve
LOOK-M merged K/V, and keep local PrefixKV/VL-Cache adaptation labels.

No fresh cohort, calibration or paper campaign is authorized. This cannot
establish superiority on independent benchmark examples. A CPU regression
also fixes a false alias rejection of zero-byte buffers: the null storage
address is not an actual overlapping mutable data owner; nonempty alias
checks remain enforced.

## V1 stopped safely; V2 explicitly owns graph-private memory

V1 stopped at its private-allocation guard, without promoting a model result.
V2 charges each graph's actual allocator delta (conservatively rounded by
PyTorch) in final/peak storage, then verifies an equal released allocation
after reset/destruction. A mismatch fails closed. Final equality requires zero
live private bytes, not an assumption that graph capture allocates nothing.
Synthetic GPU controls verify copied prompt bits, nonoverlapping workspace,
replay, and complete private-owner teardown before the model study.

Explicit nondefault-stream capture replaces torch.cuda.graph's Python context,
which in this pinned version invokes synchronize, gc.collect and empty_cache
at every entry. Stream dependencies remain explicit. Capture time remains
included in measured decode; no prebuilt/permanent graph is smuggled outside
the timing boundary. Error cleanup closes graph/workspace owners after the
ordinary transport has released its current layer lease.

V2's teardown test exposed shared private RNG ownership, not a new tensor
corruption: the first graph released zero bytes and the last released1024.
Installed `ATen/cuda/CUDAGeneratorImpl.h:CUDAGeneratorState` explicitly stores
`registered_graphs_`, `seed_extragraph_`, and `offset_extragraph_`; graphs can
share this state. V3 preserves the capture-delta charges, but checks the
aggregate release only after all cache graphs are destroyed. A CPU regression
models exactly the first-zero/last1024 case. The total storage and fidelity
thresholds are unchanged. V2 remains failed, not retrospectively relabeled.
