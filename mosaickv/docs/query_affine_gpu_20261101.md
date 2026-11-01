# Bounded same-input CUDA verification of query-aware affine fitting

The CPU query-weighted feasibility passed the unchanged numerical gate on both
consumed inputs. That establishes only a useful prompt-output reconstruction
signal, not answer quality. Preserve the negative unweighted fit, its component
audit, and the prior source PPL/ROUGE trade-off. Do not retune from their labels.

Authorize only the same Description and MM-Vet inputs, one A100/12 CPUs each,
30-minute limits and a dependent afterany CPU analysis. Fresh identities,
calibration, paper promotion and large campaigns remain disabled.

Implement the same two-step weighted affine refit on actual CUDA tensors. Keep
three/eight-bit policy, group128, FP32 headers, exact rows, scalar exceptions,
original logical positions and append-only history. No model weights change.
Require CPU/CUDA fixture agreement before loading model weights: bitwise codes
and decoded BF16 values, FP32 weights/headers within atol=1e-6, rtol=1e-5. No
CPU implementation may masquerade as CUDA execution. Disable TF32, as in the
existing highest-precision FP32 control; do not change model BF16 precision.
The fitter fixture uses identical input importance on both devices to isolate
solver transport. Independently computed CUDA importance is compared with the
CPU formula separately. Cover both 3/8-bit and time/feature group axes. Actual
model construction computes its importance on CUDA, not from saved CPU captures.

Fit the full-source variant with the frozen prompt-query sensitivity bound,
using actual current-prefill queries, keys, values, normalized inputs and phases.
The constructor must not receive annotations or reference-generation logits.
Refit sparse numerical transport corrections against native projection of the
same new encoded source, then apply the common lossless text representation and
already validated four-head tiling. This remains an explicit precision variant,
not the six-component exact-selection paper method.

Offer corresponding query-weighted quantization to every compared local
baseline. For post-RoPE selected/merged KV, use diagonal key sensitivity
sum_q a^2 ||V_i-O_q||^2 Q_qd^2/d and value sensitivity sum_q a^2. Do not regenerate
LOOK-M's merged values from original source vectors. Preserve protected rows and
existing exact exception values; give them zero fitting weight. Past calibration
queries with no causal retained key are ignored, never shown a future key.

Execute the old source and four FP32-header baseline controls plus their five
new refinements. Bind prior control artifacts, retain the earlier FP16-header
results, and reject any old-control logit drift. Same model/revision, prompts,
media, generation, seed, BF16/eager transport and source-cap convention apply.
These are different realized physical-memory points, **not byte-matched rows**.
The original own physical caps remain hard bounds; record every achieved byte.

Run retention-one raw bypass, complete ordinary/reinjected teacher-forcing
parity, unchanged 32-token KL<=0.02/logit<=1/token agreement gate, 128 greedy
predictions, complete 97/593-token targets, source retirement, immutable format
and replay controls. Keep PPL and FullKV-reference ROUGE-L separate. Failed and
unsupported rows remain explicit. Report generation and teacher-forcing initial,
final and temporary-layer-inclusive peak cache backing separately.

Compression time must include the parent's constructor cost, not merely the
incremental affine solve. Shared protection setup and inactive diagnostic formats
remain separately disclosed. Single fixed-order decode timings are exploratory,
not end-to-end or statistically established latency superiority. Whole-request
construction/allocated peaks are not certified. No prototype or repair claim.

This verification cannot create a paper-level superiority claim: independent
N is zero. A positive model result would still require a separately frozen,
baseline-fair independent evaluation with all negative outcomes retained.
