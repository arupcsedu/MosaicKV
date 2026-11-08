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

## Bounded common correction-workspace follow-up

The saved tiled/prompt ledgers identify five simultaneous 9.42 MiB temporaries
inside `shared_source_format.transport_corrections` at their whole-request peak.
The next bounded COMMON-transport test limits correction/verification predicate
spans to 262,144 scalars. It changes only scratch scheduling: identical elementwise
legacy FP32 and actual BF16 predicates, complete coverage, the same coalesced
nonzero mask, sorted patch indices, protected-row rejection and commit checks.
No tolerance, source value, history policy or model weight changes. CPU tests
compare patch words and all failure categories. Every model worker first compares
full/bounded patch words on the GPU and records five warmups/twenty CUDA-event
trials before any model request.

`prepare --predicate-span-cap 262144 --reuse-cpu V4 --reuse-model V2` runs only
common tiled control and packed-prompt/tiled under that cap on the SAME consumed
inputs. Require all 128 control logits bitwise equal to frozen storage outputs,
and all packed-prompt logits bitwise equal to frozen prototype outputs. Reuse
the six immutable original controls as explicitly different transport ablations;
do not represent these as an exclusive EviCastKV algorithm or byte-matched
cross-policy comparison. The unchanged 32-token guard, all-128 token check and
same time-aligned ownership ledger remain. This test does not expand lossy-history
tuning, authorize independent validation, or claim clean end-to-end latency.

## Completed evidence and stopping decisions (November 8)

Immutable roots are under
`/scratch/djy8hg/runs/evicastkv/packed-prototype-history-20261108-v{1,2,3,4}`.
Each contains a source/config-bound plan, job submissions, logs, CPU records and
available operator/model evidence. V1 preserves the unsupported Triton API
failure; V2 preserves the explicit history-dispatch failure and six valid model
controls; V3/V4 are numerical diagnostics, not engineering retries of failed
quality. Their `analysis/report.md` and `decision.json` preserve all outcomes.

- CPU screens cover 1/2/4/8 codec-aligned eight-row groups with a common map
  across all 32 layers. Every case has negative GPU-plus-CPU saving when the
  original pinned residuals are charged. These are frozen-inventory cost
  estimates; new transport-patch changes remain unmeasured. No expanded-group
  model execution was admitted by that screen.
- V2 packed-constructor p50: 1.357040/1.351680 ms versus
  10.618224/10.719104 ms for the original constructor. Its measured incremental
  operator allocation peak is 1.539551 versus 38.627930 MiB. Operator trials
  have five warmups and twenty CUDA-event repetitions. Code/map/residual/repair
  equivalence and the packed-prompt model's 128-logit bitwise equivalence passed.
- Tiled softmax uses 31.615234 versus 88.772461 MiB incremental operator peak,
  but takes about 0.644 versus 0.237 ms. Its bits match the original operator and
  model controls; this is a memory/latency trade-off, not a general latency win.
- V3 group-128 INT8 history preserves all 128 greedy tokens on both inputs.
  Input 1 nevertheless fails the unchanged longer-horizon thresholds:
  maximum logit difference 1.1875 and KL 0.03976334011603344. The first failure
  occurs at prediction 36, after the unchanged first-32 gate.
- V4 group-32 also preserves all 128 tokens but is negative: input 1 has maximum
  logit difference 3.1171875 (limit 1.0), despite KL 0.015658688731020755
  (limit 0.02). Smaller quantizer groups did NOT guarantee downstream stability,
  and their extra headers reduce final-storage savings. Do not tune another
  group size on these outcomes or promote either policy to fresh validation.

There is no benchmark-quality, all-baseline, strict latency, or six-component
superiority conclusion. Approximate history remains default-off. The common
correction-workspace follow-up is a bitwise transport test, not a remedy for
the negative history or residual-cost results. A fresh cohort remains blocked
until a separately frozen candidate shows meaningful storage savings with all
controls; source-group counts must not be expanded merely because packing works.
