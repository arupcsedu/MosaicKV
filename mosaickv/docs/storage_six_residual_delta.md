# Residual-delta prototype and individual-query forecast diagnostic

The matched-storage component experiment at source 4900ed8 completed 24/24
records on two consumed COCO inputs. All component arms matched actual initial,
final and decode-owned counter-peak backing storage; all 32-token parent guards
passed. Captions and CIDEr were unchanged. No all-six benefit was established.
Removing forecast or graph edges reduced FullKV logit KL on both inputs, while
the full arm had mixed effects versus the parent. These negative findings are
preserved under storage-six-component-ablation-20261106-v1.

## Evidence-driven bounded successor

The saved layer/head capture separates extra exact, absolute prototype, and
restored-member effects. Additional exact protection lowered local prompt
attention-output error on both inputs. Absolute prototypes increased error
relative to the parent, and relative to extra-exact-only. This does NOT imply
that local reconstruction error predicts answer quality: logit effects differed.

Two independently identifiable changes are frozen before GPU follow-up:

1. **Individual-query forecasting:** retain the actual prompt query samples
   through softmax, then average their attention distributions. Do not replace
   the distribution with attention at its mean query. The same core forecast
   object/statistics, graph, value utility and lazy budgeted selector remain.
2. **Parent-residual-delta prototype:** store a shared correction to the
   parent's approximate K/V, not an absolute replacement for their full values.
   Compute original-minus-parent K/V in FP32, invert actual member RoPE phases,
   average uniformly, and rotate to the real anchor. The reader adds the shared
   correction at each original phase/position. BF16 addition and all buffers
   count. CPU residuals remain the ORIGINAL full K/V, and promotion overrides
   the corrected values with those originals. No learned weights or labels.

The CPU capture study must lower both early and later prompt-half error on
both consumed inputs before the bounded GPU follow-up is submitted. That is a
local feasibility gate, not an independent quality/validation cohort. The CPU
study uses the last member as an explicit temporary phase anchor; the live
constructor uses its actual selected anchor. BF16 anchor-rounding differences
are therefore validated at model level, not claimed bitwise identical by fiat.

The GPU study compares the unchanged full overlay, individual forecast only,
delta prototype only, both, inactive matched reservation, exact members, and
controlled original restoration; native FullKV and the storage parent are
references. All produce 128 predictions with the unchanged 32-token KL<=0.02,
maximum-logit<=1.0 and greedy-token checks. Same 6713 auxiliary GPU bytes and
1024 pinned CPU residual bytes are required, with actual initial/final/counter
matching checked separately. Original dense KV owners must be retired.

## What this does NOT solve

The delta prototype replaces the absolute PROTOTYPE PAYLOAD, not the shared
encoded source. It does not yet reduce storage below the successful parent.
A one-head decision cannot retire a source row needed by other heads. No
source-retirement, whole-request KV-peak or latency superiority is implied.
The constructor temporarily reconstructs the parent layer for the delta; that
real work/allocator peak is recorded and must not be hidden. A later production
implementation could reuse an already owned reconstruction only after proving
parity and ownership; it is not assumed here.

Natural entropy-triggered repairs may remain zero. Forced restoration tests
transport/recomputation correctness, not a natural repair quality benefit.
FullKV is only a quality reference. Timing includes instrumented construction
and is not randomized repeated latency evidence. These consumed examples cannot
choose a paper candidate. All six mechanisms executing is not the criterion:
report each positive, neutral and negative effect honestly. Fresh/larger runs
remain disabled; no universal or independently confirmed superiority claim.
