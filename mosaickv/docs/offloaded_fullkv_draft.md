# Actual online FullKV draft target: bounded engineering diagnostic

The conditional-head readout study completed all 32 rows, with bitwise selected
logits and the original 32-token guards intact. It is nevertheless negative:
joint mean FullKV KL was 0.00117984 versus 0.00113719 for the simpler prototype
candidate. Repair occurred, but unchanged captions are not recovered quality.
Restoring a head can remove compensating error from earlier compressed layers;
therefore its isolated full-precision target need not improve global logits.

This successor changes the target, not the thresholds or benchmark labels.
Snapshot the ORIGINAL prefill K/V into pinned CPU memory during the existing
layer-wise constructor. Run one exact, isolated autoregressive FullKV update
from those snapshots, using native eager attention and the original logical
positions. A layer receives temporary GPU K/V leases only while its attention
executes. No second prefill, evaluation reference artifact, benchmark answer,
or oracle future-query API is used. It is a two-prediction online draft rollout.

The same verified actual final-layer suffix ranks the same bounded equal-byte
plans, now against this actual FullKV forecast. Identity remains available and
greedy-token safety, mandatory rows, 6,713 auxiliary GPU bytes, source retirement
and the full 32-token guard stay unchanged. Prototype/repair and value/graph
effects must still be measured, not assumed.

## Costs and stopping controls

- The temporary CPU originals cost **301.5 MiB** for this 603-position geometry
  (a tensor-size calculation, not a measured result). Report measured backing
  bytes, pinned status, layer leases, history and forecast wall time per input.
- Original snapshot bytes are hashed before and after the draft. This diagnostic
  validation cost is real; it is not an optimized latency benchmark.
- Clear every CPU snapshot and require all 64 original snapshot tensor objects
  to retire before final compressed decoding. Keep only forecast logits and
  metadata. Do not hide the temporary CPU cost in the residual tier.
- Require two-prediction forecast logits to match ordinary FullKV bitwise in
  evaluation, without feeding the saved reference into selection.
- Require actual selected logits to match their predicted suffix bitwise.
- Record the new draft target explicitly; do not call these global forecast
  logits a conditional-head target or claim no FullKV logits were used.
- The old append-only storage reader is unchanged. Its decode-owned counter
  does not certify whole-request peak, particularly during snapshot creation.
- Use only the two consumed inputs. No fresh cohort, calibration, larger sweep,
  independent quality claim or paper promotion follows from execution alone.

If the target or isolation controls fail, repair that boundary only. If the
controls pass but fidelity does not improve, preserve the negative result;
do not loosen the guard, increase the beam or promise all-component superiority.
