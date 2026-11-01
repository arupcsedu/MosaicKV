# Global projection-source CPU feasibility (not a method-quality test)

Seal the small full-layer restoration study as negative: zero MM-Vet single
layers and zero registered combinations passed the unchanged token/logit gate.
The top-eight exact restoration lowered max logit error below one but retained
one greedy mismatch. No local restoration/source candidate is promoted. Saved
logit composition with history repair also predicts zero passing MM-Vet arms;
do not run those joint arms on GPU.

This separately registered CPU study tests GLOBAL shared source storage, not
more restorations. Measured full-source 3-bit payload lower bounds are about
33 MiB; they leave possible room under the old cap for exact exceptions and
metadata. This does not prove a complete codec fits. K and V are projections of
the same normalized input; storing that input once may retain all prompt
positions instead of independent K/V at only half the positions. This is related
to Palu/MLA, not a novel general claim about joint latent storage. No weights are
trained or changed.

Frozen CPU-only study, both already consumed captures, all 32 layers:

- Use only captured normalized inputs, original phases, memberships and the
  pinned LLaVA-1.5 checkpoint weights. No answers/references/task scores.
- Quantize the complete input at exactly 3 or 4 bits, each with feature-group
  (axis 3) or token-group (axis 2) uniform affine quantization, group size 128.
  Four arms, no hyperparameter search. Record every arm and negative.
- Reproject K/V using FP32 CPU accumulation rounded to BF16 and apply original
  phase. This is an explicitly CPU numerical oracle, NOT GPU parity evidence.
- Use the last 16 PROMPT queries, projected from the actual full normalized
  inputs. Apply causal masks at their original logical positions. No future
  generated queries or benchmark-label access.
- Compare source-reconstructed attention outputs to the same CPU full reference.
  The pruning-only control uses ORIGINAL CPU-reference BF16 selected K/V at
  the saved memberships, with no K/V quantization, not a weakened baseline.
- Report full-cohort sum squared output error normalized by full-reference
  energy, plus the p95 per-head/per-query relative squared error. These are
  numerical diagnostics, not benchmark scores, NLL or answer quality.
- CPU feasibility requires ONE registered arm on BOTH inputs to reduce relative
  MSE by at least 20% versus pruning-only, with p95 no worse, and its measured
  source-payload lower bound below the unchanged cap. This authorizes only
  consideration of implementation/testing, NOT GPU submission or promotion.
- A real format must subsequently count ALL exact overlays, maps, phase,
  history, leases and construction temporaries; satisfy the original cap;
  preserve mandatory values and positions; and pass the UNCHANGED actual
  32-token model guard before any new development cohort. Reconstruction
  improvement is never a substitute for that guard or independent quality.

CPU execution only, no new images/cohort/GPU. Clean SHA, scratch caches, complete
bound input hashes. No new paper report, superiority claim, frozen candidate,
calibration or large evaluation is authorized. All prior negatives remain.
