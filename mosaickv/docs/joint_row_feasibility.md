# Joint cached-prompt/incoming-token projection: feasibility first

The optimized host-call profile still attributes about 2.3 seconds of self time
to 28,350 language-model linear calls over the 126 profiled decode updates.
This is instrumentation (including GPU waits), not a removable wall-time bound.
K/V incoming-token linears are two of the seven attention/MLP linears per layer.

A structural source-reader idea is to append the current normalized token to
the partially filled final 128-row cached-source GEMM, rather than execute an
additional one-row K/V matrix operation. A 600/604-token prompt leaves 88/92
rows in its final chunk; one incoming row fits without growing the fixed chunk.
This could share weight access/dispatch. It is NOT implemented as a model path.

Changing GEMM row count may change reduction algorithms or BF16 rounding.
Therefore first test exact feasibility, not latency: decode the actual frozen
first-layer source tail, append each actual incoming normalized token at decode
updates 1/64/126, and compare BOTH Torch-native and installed zero-workspace
FP32 cuBLASLt joint projections with their separate prompt projections and the
unchanged model's actual one-row K/V outputs. Any differing word rejects that
backend's same-parent integration. No tolerance relaxation, repaired benchmark
answer, replacement operator, or claimed numerical guarantee follows.

Hooks never replace outputs. Every original model operation still executes,
and all 128 own-parent vocabulary vectors must remain bitwise. No labels,
new sample, second prefill or model-weight change. The diagnostic owns two
128x4096 BF16 buffers (2 MiB); temporary linear outputs and opaque vendor
resources mean whole-request GPU peak is NOT certified. Do not mislabel this
extra workspace as production-cache reduction or instrumented latency as a win.

Finite full CPU -> two one-A100 operator workers -> CPU terminal analysis,
serialized after the constructor screen. One bounded event-driven dependency
deferral is permitted if its GPU analysis ID is not yet registered; no polling.
One GPU/12 CPUs/48 GiB/twenty minutes, at most six aggregate A100s, daily
reservation guard, scratch-only cache, clean immutable source. Only the same
two consumed inputs; independent quality N=0. Even a positive first-layer
operator result would require all-layer checks and a separately implemented
runtime before any latency test. No automatic quality campaign is authorized.
