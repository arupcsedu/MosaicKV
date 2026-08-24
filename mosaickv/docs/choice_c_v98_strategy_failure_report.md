# Choice-C V98 negative calibration report

V98 is sealed as a negative, label-blind calibration result. This report uses
only aggregate mechanism telemetry and paired FullKV-output fidelity. No sample
identity, answer, reference, benchmark score, generated-token identity, or
per-sample failure identity was inspected for design decisions.

## Design

V98 replaced V97's attention-times-value surrogate admission test with the
model's actual short-horizon response. For each sample, it constructed two
same-byte exact-cache candidates: the joint group-and-donor Pareto candidate
and a scalar Prompt Top-K anchor. Both reused the original FullKV prefill and
were decoded for two teacher-forced FullKV greedy steps. Admission required:

1. the same greedy token at both steps;
2. maximum forward KL divergence at most 0.02; and
3. maximum absolute logit difference at most 1.0.

Probe states were isolated and discarded. Benchmark labels were unavailable
to the online compressor. When neither candidate passed, V98 executed the
scalar anchor so the runtime remained valid but marked the sample ineligible.

## What passed

- All 192 measured rows completed without errors.
- Active and active-plus-residual memory fell by 32.06% on DocVQA and 32.00%
  on TextVQA relative to Prompt Top-K.
- Residual storage remained zero and the implementation stayed exact-only.
- Aggregate FullKV answer agreement was not below Prompt Top-K on either task.
- Layout-group generation, Pareto filtering, and joint-donor construction were
  active.

## Why it failed

Only 3 of 64 selected caches passed all three logit criteria. Greedy tokens
matched on 49 samples, but token equality alone did not imply that the logit
distribution stayed within the preregistered KL and absolute-difference
limits. The scalar anchor was required on 61 samples and was itself unsafe at
the target memory on most of them. Therefore the failure was not caused merely
by choosing the wrong point from the two-candidate shortlist; both points were
already too compressed before the direct stability test was applied.

The 75% atomic-group reservation-utilization gate also failed on every sample.
Finally, only 6 of 8 answer folds and 4 of 8 token folds passed the unchanged
five-point noninferiority margins. These failures prohibit freezing V98 or
running held-out evaluation.

## V99 design implication

V99 must not weaken thresholds or tune from V98 failures. It should start from
FullKV and remove low-risk blocks progressively. Every accepted cumulative
eviction checkpoint must pass the same two-step token, KL, and logit gates
against the original FullKV trajectory. If the process cannot reach the
preregistered byte target, it must fall back to a valid exact cache, record the
failure, and remain ineligible. V99 must use a new disjoint calibration split.
