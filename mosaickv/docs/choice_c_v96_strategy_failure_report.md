# V96 Strategy and Failure Report

## Status and evidence boundary

V96 is a negative, label-blind calibration result. Its Slurm chain completed,
but no candidate passed every preregistered gate and no frozen candidate was
created. This report uses only aggregate mechanism telemetry, FullKV agreement
rates, Prompt Top-K agreement rates, and fold summaries. It does not inspect or
name sample answers, references, generated tokens, task scores, or failure
identities.

The candidate was evaluated on 32 fresh DocVQA calibration samples and 32 fresh
TextVQA calibration samples. Every arm used the same model, model revision,
precision, processor, prompt, media, decoding configuration, output length,
attention backend, block size, mandatory-token rule, and sample ledger.

## Problem V96 attempted to solve

V95 formed coherent OCR and layout groups, but it examined only the four groups
with the highest standalone score in each layer/head route. It protected almost
no groups. That left two plausible explanations:

1. useful groups existed outside the four-item shortlist; or
2. the route-local replacement used to insert a group was intrinsically too
   risky.

V96 was designed to test the first explanation without using labels. It removed
the shortlist, evaluated all feasible groups, retained the nondominated choices,
and allocated groups jointly under the same byte budget.

## Strategies used in V96

### 1. Enumerate every feasible coherent group

**Design.** Within each layer and KV head, visual blocks were grouped only when
they shared image, page, frame, modality, and coarse layout-row metadata and were
contiguous in reading order. Groups contained two to four blocks. Every group
that could participate in an exact-capacity cache was evaluated.

**Why it was chosen.** V95's four-candidate shortlist could discard an
answer-relevant OCR row or chart region before risk was evaluated. Complete
enumeration removed that blind spot.

**What happened.** V96 produced approximately 1,014 Pareto candidates per
DocVQA sample and 1,005 per TextVQA sample. Every sample had positive frontier
activity.

**Why it failed.** Candidate scarcity was not the bottleneck. Thousands of
groups could be formed, but almost none survived the later exact-replacement
risk test. More enumeration increased search coverage without improving the
construction of a safe replacement.

### 2. Gain/risk/byte Pareto filtering

**Design.** A group counterfactual was removed only when another choice had at
least as much capped cumulative query gain, no greater combined risk, and no
greater byte cost, with at least one strict improvement. Combined risk was the
maximum of displaced prompt-mass risk and attention-weighted pooled-value
deviation.

**Why it was chosen.** A fixed scalar score can hide useful tradeoffs. Pareto
filtering preserves low-risk, high-gain, and low-byte alternatives without
using benchmark labels.

**What happened.** The frontier remained large on every sample, confirming that
the filter did not collapse the search to a small arbitrary shortlist.

**Why it failed.** Pareto optimality is relative to the available candidates.
A nondominated candidate can still be absolutely unsafe. Most frontier members
had insufficient query gain relative to the output-impact risk introduced by
their prompt-only donor selection.

### 3. Atomic multiple-choice byte knapsack

**Design.** The selector could choose at most one complete group
counterfactual per layer/head route. It maximized the sum of normalized query
gain minus 1.01 times combined risk under the preregistered group byte
reservation. Group membership could not be fragmented by final capacity fill.

**Why it was chosen.** V95 admitted groups independently and then greedily spent
bytes. A joint knapsack should choose the best compatible portfolio and use the
reserved bytes more effectively.

**What happened.** Only 9 groups were selected across 32 DocVQA samples and 7
across 32 TextVQA samples. Mean utilization was 0.071% and 0.057%; the maxima
were 0.538% and 0.496%. No sample reached the unchanged 75% utilization gate.

**Why it failed.** The knapsack received almost no admissible items. A hard
per-candidate gain/risk test ran before the global optimizer. Therefore the
knapsack could optimize only a nearly empty set and could not spend its 3.17--
3.22 MB reservation.

### 4. Prompt-mass-preserving exact-capacity fill

**Design.** After forcing an atomic group into a route, the remaining capacity
was filled with the exact subset having the greatest prompt-attention mass.
This maintained the original route capacity and byte count.

**Why it was chosen.** Prompt Top-K was the strongest nearby exact-cache
baseline. Preserving its highest-mass blocks appeared to be a conservative way
to add structured evidence without discarding important text evidence.

**What happened.** Mean prompt-mass preservation exceeded 99.98% on both tasks.

**Why it failed.** Equal prompt attention does not imply equal contribution to
the attention output. Two blocks can have similar attention mass but very
different value directions. The prompt-only fill chose donors after the group
was fixed and did not minimize the output-impact change of the complete swap.
Thus excellent prompt-mass preservation coexisted with worse answer and token
fidelity.

### 5. Output-impact stability for non-group reallocations

**Design.** Non-group cross-head reallocations were accepted only when their
capped-query gain cleared an output-impact risk based on donor prompt-attention
share multiplied by normalized pooled-value deviation to the nearest added
block.

**Why it was chosen.** V95 was dominated by ordinary cross-head reallocations
rather than atomic groups. Those changes needed the same value-aware safety
principle.

**What happened.** The mechanism was active on every sample. It audited an
average of 100.9 donors and rejected 30.4 routes per DocVQA sample; for TextVQA
it audited 68.9 donors and rejected 22.0 routes.

**Why it failed.** It correctly constrained non-group changes, but it operated
after atomic group candidates had already been constructed using prompt-only
fill. It could not make unsafe group counterfactuals safe and therefore could
not solve the reservation-utilization failure.

### 6. Unchanged exact-byte and scientific gates

**Design.** V96 retained the 28.5 MB configured active cap, common four-token
blocks, last-token mandatory policy, exact-only storage, zero residual bytes,
at least 30% active and total-memory reduction, five-point task/fold fidelity
margins, and 75% atomic-reservation utilization.

**Why it was chosen.** Changing the budget or gate after observing V95 would
make the comparison scientifically ambiguous.

**What happened.** Memory passed: V96 reduced active bytes by 31.99% on DocVQA
and 32.06% on TextVQA. DocVQA answer agreement was 6.25 points below Prompt
Top-K, outside the five-point margin. Only 5 of 8 answer folds and 4 of 8 token
folds passed.

**Why it failed.** The algorithm met its byte target, but the selected exact
blocks did not preserve output behavior robustly. This is an algorithmic
fidelity failure, not a budget-accounting or execution failure.

## Root cause

V96 optimized the wrong subproblem in the wrong order:

1. force a coherent group;
2. fill the remaining capacity using prompt mass;
3. measure output-impact risk;
4. reject the completed counterfactual if risk is too high; and
5. run a global knapsack on what remains.

The donor set was chosen without considering output impact. Consequently, the
later risk gate rejected almost every counterfactual. The large Pareto frontier
and nearly empty knapsack input show that shortlist breadth was not the primary
problem.

## V97 corrective design

V97 must choose the group and its displaced donors jointly. For each coherent
group it will solve one exact-capacity, label-blind route-local optimization:

\[
\max_X\; \Delta Q(X)
       - \lambda_p R_{prompt}(X)
       - \lambda_o R_{output}(X),
\]

subject to:

- the complete group is present;
- mandatory blocks are present;
- selected positions equal the original route-local capacity;
- prompt mass remains above the unchanged floor;
- no labels, answers, references, or task scores are used; and
- deterministic tie-breaking is applied.

Unlike V96, donor selection will explicitly minimize attention-weighted value
deviation while constructing the counterfactual. The Pareto frontier and atomic
knapsack will then operate on counterfactuals that were already optimized for
output stability. V97 will use a new disjoint calibration split and all V96
memory, fidelity, residual, utilization, and robust-fold gates unchanged.
