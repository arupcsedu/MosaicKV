# Bounded packed-source / softmax / history study

This study implements the user's November 8 sequence on the SAME TWO consumed
COCO inputs. It is not new calibration, independent quality validation, a paper
campaign, or a demonstration of all six mechanisms. The frozen storage-only
default, paper sources, original artifacts and main worktree stay unchanged.

## Frozen engineering policy

- Source: clean commit, exact common environment, cache/build/tmp outside home.
- Model: pinned LLaVA-1.5-7B, BF16/eager, original prompt/media/processor,
  603 logical positions, 128 greedy predictions. No benchmark labels enter any
  grouping or precision decision. No new sample materialization occurs.
- Original 32-token gate: greedy agreement, KL <=0.02 and logit difference <=1.0.
  Require all 128 greedy tokens too. Lossy history also must satisfy the SAME
  KL/logit thresholds over 128 predictions; this prevents delaying approximation
  until after the short guard from evading verification.
- CPU feasibility: common maps with 1/2/4/8 groups of eight source rows, selected
  by label-blind cross-layer integer-code variance. This is a bytes-only proxy,
  NOT an attention/output quality predictor. Charge actual compact-rare modulo-8
  streams, offsets, replay, shared map and original CPU residuals. Carry the
  measured unchanged noncode inventory. New transport-fit patch changes remain
  unmeasured until model execution; do NOT call CPU estimates complete measured
  model-cache costs. Expansion is screened only when GPU saving exceeds CPU
  residual storage at that frozen inventory; numerical gates still apply.
- Packed constructor: bounded direct three-bit gather, same centroids, logical
  map, changed symbols and exact CPU residual/repair semantics as the existing
  constructor. Never unpack all source rows into INT32 on GPU.
- Softmax: same original GEMMs, mask, FP32 normalization and BF16 cast. Tile
  complete rows in blocks of 2048; never tile the key reduction axis or reduce
  precision. Operator output bits AND full-model control logits must match.
  This is COMMON transport, not an EviCastKV-only advantage.
- History: the original newest sealed 16-token source and mutable tail remain
  exact, with original native checks. Older source uses INT8 per 128 features,
  not history eviction. Original positions/RoPE and frozen GEMM remain. Do NOT
  claim original history bits, a lossless method, or official KIVI. No residuals
  preserve the deliberately approximate old history. Default native history
  graphs are disabled IDENTICALLY for every arm; prompt graphs are unchanged.

## Finite automatic chain

`packed_history_study.py prepare --root NEW` submits CPU tests/cost/build, then an
`afterany` review. Only CPU success submits two one-A100 operator workers. Their
`afterany` reviewer submits two model workers only if packed equivalence,
softmax bits/finite values and INT8 oracle/sentinel checks all pass. GPU operators
use five warmups and twenty timed repetitions, CUDA events and synchronization.
Each model worker runs original control, tiled control, packed-prompt/tiled,
history/tiled and any eligible expansion. Combination runs only after both
component numerical gates pass. The final `afterany` analyzer preserves failed
rows; no fresh or larger campaign is submitted automatically.

One A100, 12 CPUs, 48 GiB and at most one hour per worker; at most two workers
at once, under the registered aggregate-six admission controller. Reservation
`bi_fox_dgx`, partition `bii-gpu`; CPU jobs use `bii`. Never tail logs continually.

## Accounting and limitations

Every model arm writes provenance, raw JSONL, 128 logits, phase timings and a
gzip time-aligned ownership ledger. Count requested PyTorch storage, source,
headers, indices, residuals, numerical patches, overlapping old/new buffers and
declared KV/shared workspaces. Report initial, final and whole-request KV peak
plus CPU residuals separately. Unknown allocations or invalid interval proofs
block certification. Allocator rounding/reservation, vendor allocations outside
PyTorch, and process HBM are not this certificate.

Model requests are instrumented single trials, NOT clean paired latency evidence.
Operator timings are operator-only evidence, not whole-request latency. Any
common storage/transport optimization must also be available to baselines.
Successful execution or two consumed-input matches cannot establish benchmark
quality preservation, equal-quality memory superiority or paper admissibility.
Advance only after meaningful measured savings and numerical controls; then
freeze a successor and separately preregister independent validation.

## Selective transport retry

The first GPU operator attempt exposed an unsupported Triton-language API; its
failure is preserved. The corrected operators passed integer/softmax/dequantizer
checks. Both full-model original/tiled controls and packed-prompt ablations then
passed, but the new INT8 history reached the sequential lease's explicit
unsupported-format guard. This is a transport integration failure, not measured
quality degradation.

The retry adds ONLY a type-specific `HistoryFormat` dispatch, retaining the
format's request-owner, phase, layout and live-engine checks. Unknown formats
still fail. A full-module AST fingerprint proves legacy transport identical
after removing that exact reviewed branch/import. CPU tests cover both dispatch
acceptance and rejection of unrelated changes. Only failed history workers are
rerun; six completed control/prompt rows and operator evidence remain bound to
their original producer SHA/manifests. Existing interval certificates are
independently verified on CPU before reuse. Prior failed attempts remain linked
and are not overwritten or relabeled as quality measurements.

## Separately frozen group-32 history diagnostic

The completed group-128 model diagnostic preserved every greedy token, but one
input failed the unchanged longer-horizon numerical gate: prediction 36 exceeded
the logit limit and prediction 46 exceeded the KL limit. This is a negative
numerical result, not an engineering failure and not permission to relax limits.
The first 32 predictions alone would miss approximation beginning at prediction
34. Both original and new checks are therefore retained.

A follow-up on only the SAME consumed inputs uses 32-feature INT8 groups instead
of 128. This spends four times the FP32 header storage to reduce within-group
dynamic range. It changes neither source membership, recent-16 exact protection,
weights, positions, thresholds nor prompts. It is still explicitly lossy, with
no assertion that finer groups guarantee better model outputs. The established
prompt quantizer is not modified; group 128 remains byte-identical by regression.
Complete header and conversion workspace costs are charged by the same ledger.

`prepare --history-feature-group 32 --reuse-cpu PREVIOUS --reuse-model V2`
freezes the new choice before execution, binds the prior six control/prompt rows,
and reruns CPU tests. Each GPU worker first validates the new dequantizer's exact
FP32-operation oracle and untouched workspace sentinels at 16/32/96 history rows.
Only then does it run the new history arm; its combined prompt/history arm is
conditional on that input's numerical success. All prior negative attempts stay
immutable. No fresh cohort or independent quality study is enabled by this retry.
