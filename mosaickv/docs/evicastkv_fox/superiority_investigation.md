# Investigating a stronger exact-selection policy

This is an exploratory design assessment, not a superiority result or a new
confirmation authorization. The frozen Fox-review candidate and historical raw
measurements remain unchanged. The approved confirmation scope is COCO-only;
MileBench remains development-only because source-group freshness is unresolved.

## Evidence to inspect before changing an algorithm

Run `mosaickv/scripts/evicastkv_selection_reach.py` on the passed, 64-input
COCO mechanism campaign. It checks producer hashes, complete registered request
counts, input pairing, measured initial/final/peak bytes, and activity hashes.
It reports replacement counts against the same-capacity Prompt Top-K subset,
actual graph routes, node/edge counts, edge sources, and cross-modal removal.
It does not use answers, references, sample scores, or a favorable intersection.

The separate aggregate CIDEr evidence is in:

`/scratch/djy8hg/runs/evicastkv/fox-review-20261028-v6-coco-scoped-mechanisms/analysis/mechanisms/native/mechanism_metrics.json`

That study already includes all registered routing fractions and all three
random seeds. Do not choose the best seed or treat a numerical ranking among
uncertain estimates as a validated method selection. More graph participation
is not automatically better. Identical aggregate scores also do not by
themselves prove identical predictions or selected entries.

## Specific weaknesses suggested by the implementation

1. `_uncertain_head_routes` in `backends/hf_runtime.py` ranks normalized
   attention entropy. Entropy is not the loss incurred by removing a particular
   subset. A diffuse but redundant head can rank ahead of a concentrated head
   whose few omitted entries matter. This is a hypothesis to test, not a proven
   explanation of individual answer failures.
2. Only routed heads receive non-Top-K selection. At fraction 1/256, the policy
   can differ from Top-K in very little of the total cache. Raising that fraction
   without a better router repeats an already inconclusive experiment.
3. `BudgetedObjective.evaluate` in `selection/objective.py` averages facility
   coverage uniformly over graph targets. Covering many low-demand targets can
   receive more reward than covering one high-demand target. This is the
   configured objective, not a sign bug. The signed utility is also implemented
   literally; changing coefficient signs would be an algorithm revision.
4. Semantic similarity or prompt co-activation edges must not automatically be
   described as grounded cross-modal alignment or OCR/layout evidence. Activity
   counts, deletion effects, and actual source metadata determine that claim.

## Two bounded hypotheses worth testing

### A. Route by potential improvement, not entropy alone

For each route, build the same feasible Top-K anchor under its physical capacity.
Use prompt-derived, individual-query attention distributions to estimate the
evidence missed by that anchor and whether a bounded equal-capacity replacement
can improve it. Rank routes by positive improvement relative to their own
anchor, normalized consistently within the route; never globally compare
independently normalized head scores as if they were byte-allocation utilities.

The router must distinguish *high unavoidable error* from *high improvable
error*. Keep the routed count fixed when comparing with entropy and random
controls. No cross-layer byte transfers, ragged-head padding, draft tokens,
benchmark labels, gradients, or model probes are required for this hypothesis.
Such a surrogate still may not predict generated quality.

### B. Demand-weighted, query-preserving graph coverage

Within each fixed-capacity route, replace uniform graph-target demand with
prompt-query demand:

\[
 C(S)=\sum_q w_q\sum_j a_{qj}\max_{i\in S}s_{ij},
 \qquad w_q\ge0,\quad\sum_qw_q=1.
\]

Here `a_qj` is the full-domain prompt attention probability, not a probability
renormalized over the shortlist. Similarity `s_ij` is a nonnegative, sparse,
audited edge weight; self-coverage is one. Keys/values are never replaced.
Nonnegative weighted facility coverage retains diminishing returns. Mandatory
tokens, attained capacity, source positions, and physical packing stay fixed.

Simply averaging equally weighted queries in this expression collapses to
pooled attention demand. Therefore a claim about preserving individual queries
requires an additional, explicitly specified per-query deficit constraint or
query-group coverage term; retaining a query axis in code alone is insufficient.
Test a pooled-demand control before adding such a constraint. No gain is assumed.

The intended benefit is to stop spending graph reward on low-demand regions,
not to increase graph size or perform another expensive forward-probe search.
New score normalizations and any deficit constraints must be fixed before a
new development evaluation, rather than fitted to inspected sample failures.

## Staged test and stopping rule

1. Complete the read-only selection-activity audit. If a proposed mechanism is
   inactive, repair or remove that claim before measuring quality.
2. CPU tests: deterministic ties; mandatory retention; equal cardinality and
   physical capacity; nonnegative demand; full-domain normalization; weighted
   facility diminishing returns; a synthetic case where entropy and opportunity
   disagree. These are correctness tests, not model results.
3. Same consumed inputs may test numerical/storage correctness and activity,
   but cannot become independent evidence or be mined for favorable failures.
4. Evaluate four registered exploratory arms on a new disjoint development
   cohort: frozen candidate, opportunity-router only, demand-coverage only,
   and both. Include Prompt Top-K and `vl_cache_reimpl` at **both 20/40 MiB**,
   with FullKV only as the quality reference. Keep existing prompt, precision,
   output schedule, backend, block policy, and physical accounting unchanged.
5. Stop the revision if it produces no reproducible quality benefit or only
   adds planning time. Do not repeatedly increase routing/swaps or enlarge the
   sample after an unfavorable result. Freeze once before independent testing.
6. Use a separately frozen, untouched COCO confirmation with prespecified
   sample size, multiplicity, and stopping rule. A positive point estimate or a
   CI touching zero does not establish strictly positive superiority.

Matched-budget quality superiority and lower memory at equal quality are
different claims. The latter needs registered budget curves and the smallest
quality-eligible competitor, with measured initial/final/peak storage. It cannot
be inferred from two arbitrarily chosen unequal-budget points.

## Relation to prior attempts and literature

Prompt/draft reconstruction swaps, downstream-JVP ranking, and modified-value
fitting already have negative development diagnostics in this repository. Do
not repackage them as untested solutions or silently activate them in the
prompt-only exact campaign. Larger fixed visual quotas and blind entropy
expansion also lack a validated superiority result.

Attention-output preservation is established related work, not a new general
principle invented here. Attention Matching constructs replacement K/V and
preserves attention mass; it is not equivalent to this exact-subset policy:
https://arxiv.org/abs/2602.16284

KVzip motivates context-reconstruction evidence, but uses extra scoring work
and addresses reusable query-agnostic caches; its reported results do not
establish gains for this VLM captioning protocol:
https://arxiv.org/abs/2505.23416

Neither paper licenses a promised improvement. Current evidence supports
investigation, not a numeric confidence percentage or a guaranteed win.
