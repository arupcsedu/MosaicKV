# Reversible source-code screen after negative dictionary tests

Uniform source dictionaries failed on all 64 layer captures. A bounded,
cost-aware facility selector also admitted no dictionary on its ten fixed
layer captures. Preserve these negatives; do not build their native readers.

This next test preserves original three-bit symbols exactly, instead of
constructing replacement K/V. Binary-to-Gray mapping makes adjacent quantizer
integers differ by one bit. Applying the inverse reconstructs the original
binary symbols before any source dequantization, projection or RoPE. Source
precision, headers, feature order, logical positions and weights are unchanged.
This is a common lossless codec optimization, not a six-component quality gain.

384-byte blocks are divisible by both 3 and 128. They reduce offset overhead
versus 128-byte byte-delta blocks, and permit per-symbol inverse Gray coding
within a block. The final packed byte does not require the neighboring block.
This is important for a possible in-place native decoder: the existing replay
destination must suffice, without another full-source workspace. A native
implementation and real GPU allocation checks are still required.

The CPU screen uses the same consumed Description/MM-Vet source archives, all
32 layers. Compare actual best byte/symbol packets with one charged selector
per layer. Include an ordinary 384-byte ablation to separate larger blocks
from Gray coding. Verify every code bit; retain common coding whenever the
complete new packet is not smaller. At least 2% serialized code-stream saving
on both inputs is required before bounded native-operator work. These are
not GPU KV measurements, benchmark quality, or a whole-request peak audit.

No model weights, answers, references, fresh cohort or large sweep enter the
screen. Two four-CPU workers and an afterany analyzer preserve all outcomes.
If the gate fails, no GPU follow-up is justified. Any accepted format must
be offered to the storage-only comparator as well as the experimental hybrid.
