# V97 joint group-and-donor calibration: sealed negative result

## Scientific boundary

V97 is an immutable negative, label-blind calibration result. The analysis used
aggregate FullKV agreement, aggregate Prompt Top-K agreement, measured cache
bytes, and aggregate mechanism telemetry. It did not use benchmark references,
official task scores, answer strings, generated token identities, sample-level
failure identities, or prior-run failure identities to design a successor.

The complete dependency chain ran from clean commit
`3b1143b62b27a49a5b339eace2af7d03ce9bdffe`. All 192 measured rows completed
without errors. The missing `analysis/frozen_candidate.yaml` is intentional:
the fail-closed freeze correctly found no eligible candidate.

## What V97 changed

V95 and V96 formed each coherent OCR/layout group and then filled the remaining
route capacity primarily by prompt attention. That order made the donor set an
accidental consequence of the fill. V97 instead treated a group and its donors
as one exact-capacity counterfactual:

1. Start from the scalar, same-byte Prompt Top-K route.
2. Insert one complete two-to-four-block layout group.
3. Remove exactly the added number of positions.
4. Choose donors jointly by minimizing capped-query loss, prompt-mass loss, and
   attention-weighted pooled-value deviation.
5. Retain nondominated counterfactuals in query gain, combined risk, and bytes.
6. Use an atomic multiple-choice knapsack to choose at most one group per route.
7. Apply the existing output-impact gate to non-group cross-head changes.

This design was chosen because V96 had about one thousand Pareto candidates per
sample but almost no accepted atomic replacements. Joint donor construction was
intended to give the frontier counterfactuals whose donors were already safe,
instead of attempting to repair an unsafe prompt-only fill afterward.

## Gate results

Both tasks passed the unchanged memory requirement relative to Prompt Top-K:
DocVQA reduced active and total cache memory by 32.41%, while TextVQA reduced
both by 31.92%. Residual bytes were exactly zero and all retained storage was
exact-tier storage.

Fidelity did not pass. DocVQA FullKV exact-answer agreement was 59.38%, versus
68.75% for Prompt Top-K, a 9.38-point deficit and therefore outside the fixed
five-point margin. TextVQA reached 87.50% versus 90.63% and passed the aggregate
margin. Only five of eight answer folds and three of eight token folds passed.
The worst fold deficits were 25.00 answer-agreement points and 18.95
token-agreement points.

Atomic-group utilization also failed. The fixed target was at least 75% on
every sample. DocVQA averaged 0.282% and TextVQA averaged 0.147%; neither task
had a single sample reaching 75%. The maximum on either task was only 1.08%.

## Why the individual strategies failed

### Joint group-and-donor construction

The construction was active: it produced about 35,705 feasible route-local
counterfactuals per sample. This disproved the hypothesis that V96 failed only
because it could not find exact-capacity donor sets. Capacity feasibility was
not the bottleneck. The remaining problem was predicting which feasible donor
set preserved the model's output.

### Gain-risk-byte Pareto filtering

Approximately 1,040 DocVQA and 1,018 TextVQA candidates per sample survived as
nondominated. A large frontier did not translate into safe cache changes. The
frontier coordinates measured attention coverage, prompt loss, pooled-value
deviation, and byte cost, but not the direction or magnitude of the resulting
language-model logit perturbation.

### Atomic knapsack

The knapsack preserved group membership correctly, but the preceding surrogate
risk test starved it. It selected only 35 groups on DocVQA and 18 on TextVQA
across 64 total samples. Its reserved capacity therefore could not meet the
utilization gate. Increasing the reservation or weakening the risk threshold
after observing these results would violate the preregistered protocol.

### Output-impact stability proxy

The proxy multiplied donor prompt attention by the closest pooled-value
deviation to a recipient. It rejected an average of 32.2 DocVQA routes and 22.2
TextVQA routes per sample, so the mechanism was not inactive. Nevertheless,
token fidelity remained poor. A pooled value distance is not the downstream
attention output after normalization, output projection, residual addition,
and later transformer layers; a small local proxy can still cause a large logit
change.

### Prompt-mass preservation

The selected caches retained more than 99.98% of their scalar-anchor prompt
mass on average. That did not preserve FullKV outputs. Attention mass is a
scalar magnitude. It does not encode whether a block's value vector points in
an answer-critical direction, whether multiple blocks cancel, or whether a
small perturbation crosses a greedy-token decision boundary.

## Evidence-based implication for V98

V97 shows that another handcrafted scalar risk score is unlikely to resolve
the mismatch. V98 must keep candidate generation label blind and bounded, but
measure the model response directly. It should evaluate a small deterministic
Pareto shortlist through isolated, teacher-forced decode steps that reuse the
original FullKV prefill. A candidate is admissible only when every probed step
keeps the FullKV greedy token and remains below preregistered KL-divergence and
maximum-logit-difference thresholds. The probe state must be discarded before
the real decode, and its overhead must be reported separately.

This is still training-free: it performs no gradient update, learned fitting,
or benchmark-label lookup. The V98 calibration split must exclude V97 and all
earlier Choice-C samples. The existing 30% memory, five-point task fidelity,
every-fold answer/token fidelity, zero-residual, exact-only, and 75% atomic
utilization gates remain unchanged.
