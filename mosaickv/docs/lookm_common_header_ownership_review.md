# CPU-only LOOK-M common-header ownership review

The larger-trace retry completed both consumed inputs and preserved all 128
logit vectors before/after header conversion. Its initial ownership result is
negative, with unknown allocation generations; do not overwrite that result.

Inspection of the saved stacks identified LOOK-M's selection/merge construction,
the common low-bit cache encoder, private native history workspaces, and four
pinned eager-attention lines. The narrow derived reviewer charges those cache
operations conservatively as KV/shared workspace. It does not classify generic
model/wrapper functions and never alters allocation start/end boundaries.
Hash-check the exact producer source before applying line rules. Keep unresolved
allocations fail-closed and independently recompute interval peaks.

No GPU inference is repeated. This is requested PyTorch KV plus algorithm/shared
workspace accounting, excluding allocator rounding/reservation, opaque vendor
memory and process HBM. These are local LOOK-M precision-variant controls, not
official reproduction, independent task-quality or clean latency evidence.
