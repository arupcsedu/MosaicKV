# Fused six-mechanism row application

The compact-source study preserved all 128 greedy tokens on its two consumed
inputs, but the shortlisted hybrid remained about 3% slower than same-codec
storage-only. Recorded mean decode overhead was approximately 0.15 seconds;
the construction difference was approximately 0.06 seconds. An older, explicitly
instrumented cProfile attributed about 0.098 seconds to 254 overlay fills. It
is supporting attribution, not a clean standalone latency measurement.

This default-off execution variant combines exact-row copies, existing
prototype-delta addition, the GPU promotion flag, final row stores, and the
committed-fill counter into one native operation. Forecast, graph, utilities,
budget selection, prototype values, residual policy, logical positions and
source codec are unchanged. The obsolete two-member mixed scratch is not
allocated (1,024 bytes in the registered case); all other buffers remain
charged. This is not retirement of shared all-head source storage.

The supported scope is the isolated final-layer full score-risk policy. The
Torch path stays the default; unsupported modes fail rather than fall back.
Native controls test both KV components, absolute/delta representations,
promoted/unpromoted flags, strided prefix leases, untouched heads/history,
and fill counters. These synthetic controls are not model measurements.

The two model canaries must match every one of 128 saved Torch-hybrid
vocabulary vectors bit-for-bit, and retain all storage-parent tokens. Only
then may new paired measurements run: same-codec storage, Torch score-risk
hybrid, and fused score-risk hybrid; five warmups, twenty timed repetitions,
and separate memory audits. Strict adjusted upper latency ratio remains
<=1.00 across four registered input/comparison pairs. Smaller mixed scratch
is reported as measured storage, never as an equal-byte selector comparison.

This is a consumed-input engineering diagnostic. No fresh cohort, independent
quality claim, whole-request exact KV-peak certificate or paper campaign is
authorized. The pure source-codec optimization remains common to controls.

From a clean source use `compact_hybrid_study.py prepare-fill --root NEW_ROOT
--reuse /scratch/djy8hg/runs/evicastkv/compact-escape-six-hybrid-20261108-v1`.
Source the scratch cache environment and use this worktree's package boundary.
The existing finite two-worker A100 and afterany analysis chain is reused.
