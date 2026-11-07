# Do not run and discard the exact-search branch in prototype-only fitting

The complete twenty-pair experiment at `partition-inline-paired-latency-20261107-v1`
retains all outliers: input one failed its latency gate. Saved phases localize
1.4--4.5 second spikes to the precision transaction. The configured prototype-
only branch nevertheless ran a four-row exact search, including extreme-score
direct recomputation, before always setting exact_ok false. These discarded
scores do not enter prototype weights, the baseline, or the joint veto.

An explicit default-off execution flag skips only that dead branch. Exact and
joint modes reject the flag. The baseline, prototype estimator, causal/future
prefix checks, final transaction veto, returned K/V/IDs, physical allocations,
and every threshold remain unchanged. The trace says the discarded search was
not evaluated rather than inventing its diagnostic values. CPU randomized
regressions prove returned payload and guard equality and fail if the discarded
search executes. Pretrained graph hashes and all 128 logits must also match.

Preserve the failed timing result. Do not drop slow trials or increase warmups
after inspecting them. Repeat the same five-warmup/twenty-pair registered design
only after numerical parity, in a new root with this measured code correction.
This is execution improvement, not evidence that all six mechanisms improve
independent benchmark quality. The ongoing sixteen-input panel remains frozen
on its original source and is not changed or rerun by this optimization.

The graph-parity analyzer may conditionally submit exactly one preregistered
same-two-input paired timing job. Its new output root is fixed in the plan;
failed parity writes a blocked decision. It does not launch another candidate,
new cohort, calibration or paper campaign. All jobs use afterany analysis.
