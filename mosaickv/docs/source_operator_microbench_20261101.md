# Reconstruction operator microbenchmark, no model-quality selection

Dependency: corrected source/text GPU study passes both consumed model guards
and reduces all three declared storage boundaries; compact-header CPU study
passes both frozen numerical gates. Preserve its large decode slowdown.

On the same two consumed normalized captures and all 32 layers, test only
conditional-range 3-bit per-channel source with FP16 headers, identical exact
K/V overlays and original phases. Four registered operators:

- fused, one head per output tile (reference);
- fused, two heads per tile;
- fused, four heads per tile;
- native frozen BF16 BMM with at most 256 decoded source rows in workspace.

The one-head control retains four warps/three pipeline stages. Two-head uses
four warps/two stages and four-head eight warps/one stage to bound shared
memory. These are frozen hardware choices, not selected using quality.
Compilation/resource failures remain explicitly unsupported per operator;
poisoned CUDA contexts fail the whole worker rather than returning partial
success. Verify encoded-state immutability after all repetitions.

Grouping reuses unpacked input across more output columns; it must not change
K/V bits. Keep phases within each 128-dimensional head. Native BMM may have
different reduction rounding and is not called lossless without evidence.
Its bounded workspace is counted, including during its lifetime across the
two eager attention operations; library-internal workspace remains excluded
from this tensor boundary and whole-request memory remains uncertified.

Use one physical strided layer lease with 128 history-sentinel rows. Check K
before overwriting it with V, bitwise equality, finite values, frozen operator
tolerance (.01/2^-6), and untouched history. No benchmark outputs, answers or
references enter this study. Five warmups and twenty CUDA-event repetitions
with deterministic randomized method order. Synchronize at boundaries.

Choose a grouped-head configuration only if EVERY component/input/layer is
bitwise identical and its summed per-layer median operator cost improves by
at least 10%. Otherwise retain one head. A native BMM result can only justify
a separate model-level guard, not automatic selection or a quality claim.
No full-model/end-to-end latency conclusion from this microbenchmark.

Full CPU suite first; one A100/12CPU worker, <=30min; afterany analysis. These
remain consumed diagnostics, independent N=0, no fresh/larger campaign.
