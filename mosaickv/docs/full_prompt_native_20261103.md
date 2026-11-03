# Bounded full-prompt native reconstruction engineering protocol

EviCastKV=MosaicKV. This is an execution replacement for the frozen full-prompt
source-precision/storage VARIANT, with the six principal algorithm mechanisms
disabled. It is not evidence for the complete design. Baseline rows are local
precision/storage adaptations, not official code or byte-matched algorithm
comparisons. Use only the same two already consumed Description/MM-Vet inputs;
independent quality N=0. No labels, references or answers enter construction.

The source-launch study completed with roughly one-percent gains but did not
meet its frozen promotion rule. The safe tile-skipping audit found only 0.08%
and 0.10% fully overwritten programs. Do not retry those unpromoted policies.

## Fixed intervention

Decode the existing compressed normalized source ONCE per active layer into an
owned, contiguous BF16 workspace; use full-prompt native batched matrix products
directly into the existing padded head lease for K then V. Retain the workspace
between those two components, not across layers or requests; release after V.
Keep original weights, phases, positions, protected rows, old transport patches,
text overlays, source-code compression and lossless generated history.

Native arithmetic may round differently. Fit sparse LOSSLESS BF16 repairs
against the frozen packet decoder, not against labels or model answers. Test
the SAME repairs on 0/1/128 history padding layouts. No refit during decoding.
Choose eligible layers by ascending repair bytes then layer index under a
fixed total 3 MiB repair cap; require at least eight layers (all for small
fixtures). Decline explicitly if unavailable; never silently label fallback
as native. Count every repair index/value and the full source workspace.

The predeclared secondary ablation is `v_only`: original fused K, full-prompt
native V. It is NOT claimed to reuse the source for both K/V. It can proceed
only after independent bitwise/storage feasibility controls. No other policies
or budget changes are automatically permitted.

## Checks before latency

Full CPU suite from a clean SHA. Actual synthetic CUDA kernels cover partial
source tiles, strided destinations, original protected values, nonzero repair
arrays when present, untouched history sentinels and exactly one source decode.
The earlier dirty primitive test is exploratory only and must be repeated.

Then two one-A100/12-CPU workers run complete original and native 128-prediction
requests from one original prefill per request. Require all vocabulary logits
bitwise equal to each method's saved parent, actual native engagement, private
workspaces and no outstanding leases. Original packet hashes are checked
BEFORE lossless execution conversion. Both native policies get audited tensor
backing initial/final/peak records. Each accepted policy must be strictly smaller
at EVERY boundary than the smallest recorded baseline boundary for that input.
The full decoded source is included in the actual layer-owned inventory and
lease counter; it cannot be hidden as an uncharged projection temporary.

Native BMM may have library-internal workspaces. Record complete-request
torch.cuda.max_memory_allocated/reserved separately. These do NOT certify a
whole-request peak or make opaque library storage into decoded-cache storage.
The primary recorded peak remains the declared decode-owned tensor boundary.

## Finite autonomous decisions

afterany primitive analysis -> two-policy model feasibility -> choose KV if
both inputs pass, otherwise V-only if both pass -> paired original/native
requests. Five warmups, twenty repetitions, randomized method/mode order,
128 predictions. Include source fitting, correction construction, preprocessing,
original encoder/prefill, history construction/decoding and synchronized cleanup
in complete-request latency. Tensor audits/hashing occur only in separate
controls, never timed repetitions. Register timed counter peaks in the same
unaudited category; bind endpoints to independent same-policy audited records.

Paired promotion requires adjusted EviCastKV native/original upper ratio <=0.99
AND native/PrefixKV-adaptation upper ratio <=1.05 on BOTH inputs. If KV fails
latency (including the final all-five no-slower check), try only the already
safe V-only policy once. Numerical/code failures
stop instead of automatically bypassing guards. No promotion after rounding.

If a policy passes, run the all-five complete-request benchmark with that
representation option offered identically to every eligible format. Ineligible
baseline formats are explicit no-ops and must reproduce their frozen storage
and logits. Report final stricter <=1.0 no-slower gates against all four
adaptations with adjusted engineering-trial intervals; never call them quality
intervals over independent examples. Preserve negative outcomes and all failures.
The gate intervals conservatively adjust sixteen contrasts across both possible
policies and both inputs; the ordinary eight-comparison intervals remain visible
but are not used to select a winner after trying a second policy. Absolute
promotion thresholds stay unchanged.

At most two one-A100 workers simultaneously, 12 CPUs and 48 GiB each, one-hour
limits, six-A100 aggregate controller registration. No continuous log polling.
Callbacks write immutable engineering_closeout.json and engineering_report.md.
No fresh cohort, calibration, larger campaign, answer tuning, threshold changes,
automatic source edits or paper updates are authorized by this engineering DAG.
Even a timing win does not establish independent benchmark superiority, PPL/
ROUGE non-inferiority, or whole-request memory superiority.
