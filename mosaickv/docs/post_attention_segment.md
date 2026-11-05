# Common, exact-operator post-attention replay

The saved joint-KV profile leaves roughly 0.97 seconds in RMSNorm dispatch and
1.63 seconds in native linear calls across the instrumented trajectory. These
are attribution totals, not isolated removable wall time. The active replay
screen must finish before this bounded successor starts.

Capture only the pinned Llama decoder epilogue: BF16 residual addition, native
FP32 RMSNorm arithmetic and BF16 cast, original gate/up/down matrix products,
SiLU, product, and final residual addition. Preserve the exact operation order;
there is no fusion, approximation, training, attention capture or selection
change. The design is based on installed Transformers LlamaDecoderLayer,
LlamaRMSNorm and LlamaMLP (Apache-2.0); it is not a new model architecture.
Single-token, private BF16 eager LLaVA only. Unsupported architectures, hooks,
training and retained hidden-state outputs fail explicitly.

Use one shared 117,764-byte activation workspace, charged in every method's
decode-owned peak. Dynamic residual/attention inputs copy to it before graph
replay. Each layer captures its own frozen weights; output lives only until the
next layer copies its residual. Capture/retirement are included in request
time. A shared CUDA stream holds no cache/tensor/request owners. Warm native
GEMM backend allocation is reported separately; opaque vendor and whole-request
GPU peaks remain uncertified. Graph capture must introduce zero unowned tensor
bytes; existing prompt-graph generator ownership checks remain unchanged.

Offer the same optimization to both PrefixKV reader controls. Six arms compare
each old/new path using two consumed inputs, all 128 vocabulary logit vectors
bitwise to frozen parents, exact packet digests and original cache positions.
Before loading weights, validate the full-width operator against original
Llama arithmetic. Require 4,064 epilogue replays, 32 captures and complete
retirement. Require unchanged initial/final cache bytes and exactly the charged
workspace difference at peak. Five warmups and three randomized paired timed
requests screen the 4% target; no formal gate follows from three trials.

Full CPU suite before two one-A100 workers (12 CPUs, 20 minutes); afterany
analysis records failures as well as successes. No fresh identities, independent
benchmark evidence (N=0), calibration or larger campaign is authorized. The
storage variant disables the six algorithmic mechanisms; an exact numerical
execution improvement does not establish complete EviCastKV quality superiority.
