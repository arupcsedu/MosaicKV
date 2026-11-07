# Source replacement and isolated precision-policy diagnostics

This is a separately labeled development variant of the validated storage
reader. The frozen 256-image campaign and its defaults remain unchanged.

## Actual replacement, not an additive prototype overlay

The source is shared across all K/V heads. A prototype for a single head
cannot delete a normalized source row still needed by other heads. Instead,
the constructor first identifies positions already protected exactly in every
head. The existing quantizer has assigned zero integer codes to those rows.
The replacement physically omits these codes, retaining the original FP32
headers and a charged int32 logical-to-physical row map. A map entry of -1
means integer code zero, NOT a zero decoded value. Original group headers
and RoPE positions still apply. No requantization, weight changes, token
eviction, or altered decode history is permitted.

Replacement is deferred until all layers are available. Use the intersection
of their removable positions so payload lengths remain aligned for the
existing lossless cross-layer codec. Evaluate original-plus-codec and
replacement-plus-codec, charging replay scratch in BOTH. Accept only a strict
net saving; otherwise retain the original codec explicitly. Raw omitted-code
bytes alone are not evidence of net storage reduction.
The common row map has ONE backing owner shared across layers. Duplicating it
per layer can outweigh small code savings; verify actual alias identity and
global storage deduplication, not merely equal map contents.
The map is classified as shared in both encoded specifications and bound
readers. Cached per-layer counters must equal independent global enumeration.
The corrected initial boundary includes all shared phase/position/map owners,
as final/peak already do. Historical initial-private-only records are retained;
they must not be reinterpreted as this corrected physical-storage boundary.

The old code owner must die before decoding. Weak references audit original
tensor retirement; the new payload must own separate, smaller backing storage.
Every source tensor and map is exposed to existing physical-byte accounting.
If the map costs more than it saves, the original body is retained explicitly.
All-omitted cases charge one dummy physical row. The codec is common and
available identically to every method using this format; its savings must not
be attributed exclusively to the six mechanisms or used to handicap baselines.

Supported readers: original fused projection, prepared projection dispatch,
native 128-row decoding, joint incoming projection, full-source decoding, and
CPU oracle. The legacy 32-column tile and row-major nibble scratch explicitly
reject compacted bodies. Unregistered source-code delta formats remain outside
this diagnostic. The disabled/default path and retention-1 reference do not
transform sources. This does not compress arbitrary prototype members or
remove the remaining shared source; it completes a lossless source-code
replacement transaction only.

## Improve policies separately, not by enabling everything

The development reference is individual-query forecasting plus phase-safe
parent-residual prototypes and the bitwise-validated segmented sparse graph.
Error-only utility, demand weighting and opportunity routing stay disabled.

Two isolated ablations are registered before execution:

1. **Signed exact-restoration utility.** For each candidate row, recompute
   per-query attention-output error with that original K/V row restored,
   including the changed key score and softmax partition function. Use signed
   improvement relative to the actual parent, normalized by parent MSE, with
   a fixed 0.25 local-utility weight. A common +1 offset keeps the contribution
   nonnegative and preserves the core utility equation; fixed-cardinality
   rankings are unaffected by that constant. Unlike value norm or error magnitude,
   restoring a row can receive a negative score. This is only local evidence,
   not a certificate of model-logit or benchmark-quality benefit.
2. **Actual-attention prototype risk.** Keep the original entropy trigger and
   add `attention_mass * sqrt(dispersion) / mean_original_member_value_norm`,
   using actual provisional decode attention from the adapter, not prompt
   demand. The fixed threshold is 0.01. Missing attention fails explicitly.
   This bounded diagnostic checks the first provisional update only; it does
   not implement arbitrary-step repair. A natural trigger is not itself a
   quality recovery, and forced restoration remains a mechanics control.

## Bounded verification and stopping

Use only the same two consumed COCO inputs, the pinned LLaVA-1.5 checkpoint,
128 predictions, BF16 eager, and the unchanged 32-token parent guard
(KL <= 0.02, max logit difference <= 1.0, greedy-token agreement 1.0).
Run CUDA operator controls before model loading. Require bitwise parent versus
replacement logits, all 32 code-owner retirements, positive net savings and
complete initial/final/decode-owned peak accounting. Compare the same simpler
hybrid with and without replacement. Precision-policy ablations use the
same replacement and auxiliary allocation, so memory is matched between them.

Record synchronized phase timings, raw predictions, exact provenance, real
repair events and negative outcomes. Single calls and two consumed images do
not establish latency or quality superiority. Whole-request peak certification,
fresh cohorts, calibration and larger campaigns remain blocked. Proceed only
on observed useful effects, never merely on a successful Slurm exit code.
