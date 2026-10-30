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
inside measured decode time; no amortization is presumed. A graph-private
PyTorch tensor allocation fails memory certification instead of being omitted.
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
