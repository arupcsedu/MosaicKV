# Direct construction of the identical frozen text cache

This is a consumed-input execution variant, not a new selection/quality claim.
EviCastKV=MosaicKV; six principal mechanisms remain disabled here. Keep the
same two Description/MM-Vet inputs, exact original/merged K/V, memberships,
positions, precision, attention arithmetic, source-code replay and history.

## Motivation and controls

The existing text-source constructor materializes the full packet, fits a
normalized text projection, builds numerical repairs, and verifies it three
times. Posthoc text-row caching then discards that projection. Replace this
wasted work by borrowing only rows already protected exactly by the parent.
For unprotected text, reconstruct the actual frozen parent once; NEVER borrow
raw model values or upgrade its approximation. The first GPU verification
correctly rejected that unsafe shortcut; preserve its negative artifacts.
Record the layers requiring parent reconstruction. Strip only redundant text overlays. Preserve the
exact codec/resident choice and per-layer membership from the completed frozen
text-cache study; no new cost profiling or policy search is allowed.

Require the complete ordered tensor SHA (shapes, dtypes and all words) BEFORE
source-code serialization to match the validated posthoc packet, plus exact
initial/final/decode-owned-peak storage, and every vocabulary word in all 128
parent predictions. The old constructor remains the default and is used for
the paired control. Request-private scope must unwind on errors. Borrowed
values must be the baseline's actual merged values where applicable.

## Finite verification

Full same-clean-SHA CPU suite, then two one-A100 numerical/storage workers
offering the same intervention to MosaicKV and all four local baseline
adaptations. Preserve all failed rows and stop on identity/storage failures.
Only after BOTH inputs pass: five warmups and twenty randomized paired full
requests for EviCastKV and PrefixKV, with posthoc/direct construction and the
previous strongest whole-layer PrefixKV control. Include construction, scratch,
copies, and synchronized cleanup. Hashing/tensor audits use separate controls.

Require adjusted upper direct/posthoc request ratio <=0.99 and direct versus
EVERY PrefixKV control <=1.05 on both inputs before finite all-five-method
timing. Report the <=1.0 gate against every measured baseline too. Use 10,000
paired timing-repetition bootstraps, maximum family 48 engineering contrasts
across both inputs and both timing stages. This is not sample-level quality
inference or a paper superiority test. No automatic default promotion.

Each worker uses one A100, twelve CPUs, 48 GiB and <=105 minutes, maximum two
workers in this chain and aggregate <=6 A100s across chains. CPU jobs use
`bii,standard`, two CPUs/16 GiB. Scratch-only caches, immutable output roots,
clean SHA and `afterany` analyzers. Do not modify the other running worktree.
No fresh cohort, calibration, paper campaign, native serving, or certified
whole-request GPU-peak claim is authorized.
