# Bounded gate repairs, localization and media audit

## Completed repair evidence

Commit `88cee84432b98778e941847e0e2532b534fad64f` initializes stable
reservations whether the prompt reference is freshly computed, already present,
or bypassed through external exact blocks. The all-routed branch legitimately
has no stable-head reservations. Partial/full-routing regressions cover both
uniform-fill settings, deterministic selection, mandatory entries and budget.

Only the original failed 8-input, 16-request full-graph pilot was retried under
a new root. Array `22104806` and analysis `22104807` completed; the gate records
16/16 completed requests. No parent rows were overwritten or new inputs chosen.

CPU reproduction `22104808` regenerated primary and mechanism statistics twice.
Only the two exact `reducer_verification.requested_cpus` / `used_cpus` paths
are compared as visible provenance rather than measured scores. Their values
must remain positive integers and appear in the reproduction report. Scores,
intervals, bootstrap family sizes, counts, missing values, verification errors
and every other field remain subject to strict comparison. No GPU inference
was repeated for this repair.

## Operator localization

`evicastkv_operator_localization.py` uses the same four consumed numerical
inputs and native eager arithmetic, not another precision-policy sweep. It
reuses the original model/dataset revisions and records the installed software
and GPU environment. The original padding and A/B thresholds remain unchanged.

Read-only hooks capture each layer's last-query normalization, Q/K/V projection,
attention output projection and MLP down-projection inputs/outputs, plus layer
output, final normalization and vocabulary projection. Padding is followed for
all 16 FullKV-forced steps. Before each step, valid cached K/V history is
compared excluding the artificial holes. This distinguishes an initially equal
history from divergence accumulated in later decoding. A/B compares the last
query of full prefill with the held-back-token decode.

Outputs are instrumented diagnostics, **not latency measurements**. A first
nonidentical operator localizes a path but does not by itself establish why its
error amplifies downstream. No hooks modify weights or return replacement
activations. Synthetic real-module tests verify hook cleanup and unchanged
outputs. A finite afterany CPU callback writes the localization summary even
when execution is incomplete; it cannot promote a cohort or loosen a gate.

## Underlying media

The CPU audit reads the pinned previews' ID/media columns and already cached
archives, without extracting or downloading more image archives. Source frames
and montages are compared by encoded SHA-256 and EXIF-normalized RGB-content
SHA-256; filenames or different encodings are not accepted as freshness proof.
The entire original four-task pool is conservatively excluded, not only the
most recent development IDs. Requested tar symlinks, oversized files, unsafe
paths, corrupt images and missing members are not silently accepted.

Official question types are checked from the leading annotation metadata
without JSON-decoding the reference-answer array. Missing archive parts/assets,
incomplete global history, and unresolved underlying video/document identities
remain explicit blockers. An observed content-disjointness result is **not**
an exhaustive freshness certificate. Neither unavailable media nor a source
group hint may be treated as verified untouched confirmation.

## Freeze boundary

`evicastkv_engineering_freeze.py` seals the existing complete 1/256 policy and
common COCO controls for **consumed-input boundary-A correctness only**. It
requires the repaired full-graph gate and identical production-runtime content,
writes canonical configuration hashes for FullKV, Top-K, the local VL-Cache port
and exact MosaicKV, and reruns only two consumed inputs / eight compressed
correctness requests. The model retention-one checks are part of that runner.
The analyzer now honors explicitly disabled automatic progression, preventing
a successful correctness check from launching scoring/profiling/measurement.

This engineering freeze does not set `human_reviewed=true`, create a passing
`frozen_candidate.yaml`, certify MileBench, authorize confirmation, or establish
superiority. The pending scientific gates cannot be bypassed by renaming the
configuration. Boundary-B remains diagnostic until its original control passes.
